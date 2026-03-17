"""Scorer modules for all three training methods.

Contains:
  - PRMScorer: PRM-only evaluation for the RL method
  - OPDScorer: Hint judge + optional PRM eval + teacher logprobs for OPD
  - CombinedScorer: Hint judge + PRM eval + teacher logprobs for Combined

Shared:
  - PRM eval prompt (used by RL, OPD eval_mode, Combined)
  - Hint judge prompt (used by OPD, Combined)
  - Parsing helpers: parse_judge_result, parse_prm_eval_score, majority_vote
  - Teacher log-prob extraction via Tinker SamplingClient

The teacher/judge model is deployed on Tinker as a base-model SamplingClient
(no LoRA), sharing the same cloud infrastructure as the policy model.
"""

from __future__ import annotations

import asyncio
import collections
import copy
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

_BOXED_RE = re.compile(r"\\boxed\{([-+]?\d)\}")
_HINT_RE = re.compile(r"\[HINT_START\](.*?)\[HINT_END\]", re.DOTALL)


# ---------------------------------------------------------------------------
# Shared prompts
# ---------------------------------------------------------------------------

def build_prm_eval_prompt(
    response_text: str, next_state_text: str, next_state_role: str = "user"
) -> list[dict]:
    """PRM eval prompt — used by RL (primary), OPD (eval_mode), Combined."""
    system = (
        "You are a process reward model (PRM) evaluating an AI assistant.\n"
        "You will see the assistant's output and the subsequent next state.\n"
        "Your task: decide whether the assistant's output **successfully fulfilled** the user's intent "
        "at that step, using the next state as evidence.\n\n"
        "## Understanding the next state's role\n"
        "- role='user': A reply from the user.\n"
        "- role='tool': The return value of a tool the assistant invoked. "
        "This content was NOT available before the assistant's action \u2014 "
        "it exists BECAUSE the assistant called the tool. "
        "A successful, non-error tool output means the assistant's action worked correctly "
        "and should be scored positively.\n\n"
        "## Scoring rules\n"
        "- \\boxed{1} (good): The next state shows the task progressed as expected \u2014 "
        "e.g. the user moves on, says thanks, the environment confirms success, "
        "or a tool returns a successful, non-error result.\n"
        "- \\boxed{-1} (bad): The next state signals the assistant's output was wrong, "
        "incomplete, or unwanted. **Key negative signals include:**\n"
        "  * The user asks the assistant to **redo, retry, or repeat** the same action "
        "(\"do it again\", \"try again\", \"one more time\").\n"
        "  * The user requests a **correction or modification** to what the assistant just did "
        "(\"change X to Y\", \"no, I meant \u2026\", \"not that, \u2026\", \"please fix \u2026\").\n"
        "  * The user **rephrases or restates** the same request, implying the assistant "
        "did not understand or execute it correctly.\n"
        "  * The environment returns an **error, failure, or unexpected result** caused "
        "by the assistant's action.\n"
        "- \\boxed{0} (neutral): The next state is ambiguous \u2014 e.g. the user gives an "
        "unrelated follow-up that neither confirms nor denies success, or there is "
        "insufficient information to judge.\n\n"
        "## Important\n"
        "A change request IS negative feedback \u2014 it means the previous output did not "
        "meet the user's need. Do NOT treat it as a neutral new instruction.\n\n"
        "Think step-by-step, then give your final score inside \\boxed{}."
    )
    user = (
        f"## Assistant output\n{response_text}\n\n"
        f"## Next state [role: {next_state_role}]\n{next_state_text}\n\n"
        "First, classify the next state: is it (a) positive progression, "
        "(b) a correction / redo / change request, or (c) ambiguous? "
        "Then assign \\boxed{1}, \\boxed{-1}, or \\boxed{0}."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_hint_judge_messages(
    response_text: str, next_state_text: str, next_state_role: str = "user"
) -> list[dict]:
    """Hint judge prompt — used by OPD and Combined methods."""
    system = (
        "You are a process reward model used for hindsight hint extraction.\n"
        "You are given:\n"
        "1) The assistant response at turn t.\n"
        "2) The next state at turn t+1, along with its **role**.\n\n"
        "## Understanding the next state's role\n"
        "- role='user': A reply from the user (follow-up, correction, new request, etc.).\n"
        "- role='tool': The return value of a tool the assistant invoked. "
        "This content was NOT available before the assistant's action \u2014 "
        "it exists BECAUSE the assistant called the tool. "
        "A successful, non-error tool output generally means the assistant's "
        "action was appropriate; do NOT treat it as information the assistant "
        "should have already known.\n\n"
        "Your goal is to decide whether the next state reveals useful hindsight information\n"
        "that could have helped improve the assistant response at turn t.\n\n"
        "Output format rules (strict):\n"
        "- You MUST include exactly one final decision token: \\boxed{1} or \\boxed{-1}.\n"
        "- If and only if decision is \\boxed{1}, provide a concise, information-dense hint in 1-3 sentences,\n"
        "  wrapped between [HINT_START] and [HINT_END].\n"
        "- If decision is \\boxed{-1}, do not provide a hint block.\n"
        "- Hint must be concrete and actionable for improving the previous response."
    )
    user = (
        f"## Assistant response (turn t)\n{response_text}\n\n"
        f"## Next state (turn t+1) [role: {next_state_role}]\n{next_state_text}\n\n"
        "Now output your decision and (if positive) the hint in the required format."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_prm_eval_score(text: str) -> Optional[int]:
    """Extract \\boxed{N} score for PRM eval (N in {+1, -1, 0})."""
    matches = _BOXED_RE.findall(text)
    if not matches:
        return None
    val = int(matches[-1])
    return val if val in (1, -1, 0) else None


def parse_judge_result(text: str) -> tuple[Optional[int], str]:
    """Extract score and hint from hint-judge output."""
    boxed = _BOXED_RE.findall(text)
    score = int(boxed[-1]) if boxed else None
    if score not in (1, -1):
        score = None
    hint_matches = _HINT_RE.findall(text)
    hint = hint_matches[-1].strip() if hint_matches else ""
    return score, hint


def majority_vote(scores: list[Optional[int]]) -> float:
    """Return majority-voted score; ties or all-None -> 0.0."""
    valid = [s for s in scores if s is not None]
    if not valid:
        return 0.0
    counter = collections.Counter(valid)
    top = counter.most_common(1)[0]
    if list(counter.values()).count(top[1]) > 1:
        return 0.0
    return float(top[0])


def select_best_hint(votes: list[dict]) -> Optional[dict]:
    """Select the longest positive hint from voting results."""
    good = [
        v for v in votes
        if v.get("score") == 1 and isinstance(v.get("hint"), str) and len(v["hint"].strip()) > 10
    ]
    return max(good, key=lambda v: len(v["hint"].strip())) if good else None


def append_hint_to_messages(messages: list[dict], hint: str) -> list[dict]:
    """Append a hindsight hint to the last user message."""
    cloned = copy.deepcopy(messages)
    if not cloned:
        return [{"role": "user", "content": f"[user's hint / instruction]\n{hint}"}]
    target_idx = None
    for i in range(len(cloned) - 1, -1, -1):
        if cloned[i].get("role") == "user":
            target_idx = i
            break
    if target_idx is None:
        target_idx = len(cloned) - 1
    content = cloned[target_idx].get("content", "")
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
        content = " ".join(parts)
    suffix = f"\n\n[user's hint / instruction]\n{hint.strip()}"
    cloned[target_idx]["content"] = (str(content) + suffix).strip()
    return cloned


# ---------------------------------------------------------------------------
# Tinker query helpers (shared across all scorers)
# ---------------------------------------------------------------------------

async def _tinker_generate(teacher_client, tokenizer, messages, temperature, max_tokens):
    """Send generation prompt to Tinker teacher model and return decoded text."""
    import tinker

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    chunk = tinker.EncodedTextChunk(tokens=list(prompt_ids), type="encoded_text")
    model_input = tinker.ModelInput(chunks=[chunk])
    sampling_params = tinker.SamplingParams(
        temperature=temperature, max_tokens=max_tokens, top_k=50, top_p=0.95,
    )

    response = await teacher_client.sample_async(
        prompt=model_input, num_samples=1, sampling_params=sampling_params,
        include_prompt_logprobs=False, topk_prompt_logprobs=0,
    )
    seq = response.sequences[0]
    return tokenizer.decode(seq.tokens, skip_special_tokens=True)


async def _tinker_teacher_logprobs(
    teacher_client, tokenizer, hint: str, turn_data: dict,
    normalize_fn, session_id: str = "", turn_num: int = 0,
) -> list[float]:
    """Query Tinker teacher model for per-token logprobs on the student's response.

    Args:
        normalize_fn: Function to normalize messages (e.g., _normalize_messages from api_server).
    """
    import tinker

    if not tokenizer:
        return [0.0] * len(turn_data["response_ids"])

    messages = turn_data.get("messages", [])
    enhanced = append_hint_to_messages(messages, hint)
    tools = turn_data.get("tools")

    norm_enhanced = normalize_fn(enhanced)
    enhanced_prompt = tokenizer.apply_chat_template(
        norm_enhanced, tools=tools, tokenize=False, add_generation_prompt=True,
    )
    full_text = enhanced_prompt + turn_data["response_text"]
    response_ids = turn_data["response_ids"]
    response_len = len(response_ids)

    try:
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        chunk = tinker.EncodedTextChunk(tokens=list(full_ids), type="encoded_text")
        model_input = tinker.ModelInput(chunks=[chunk])
        sampling_params = tinker.SamplingParams(temperature=0.0, max_tokens=1)

        response = await teacher_client.sample_async(
            prompt=model_input, num_samples=1, sampling_params=sampling_params,
            include_prompt_logprobs=True, topk_prompt_logprobs=1,
        )

        seq = response.sequences[0]
        prompt_logprobs = response.prompt_logprobs or []

        prompt_token_count = len(tokenizer.encode(enhanced_prompt, add_special_tokens=False))
        teacher_lps = [
            float(lp) if lp is not None else 0.0
            for lp in prompt_logprobs[prompt_token_count:]
        ]

        if len(teacher_lps) > response_len:
            teacher_lps = teacher_lps[:response_len]
        elif len(teacher_lps) < response_len:
            teacher_lps += [0.0] * (response_len - len(teacher_lps))

        return teacher_lps
    except Exception as e:
        logger.error("[Scorer] teacher logprob query FAILED session=%s turn=%d: %s", session_id, turn_num, e, exc_info=True)
        return [0.0] * response_len


# ===========================================================================
# PRMScorer — for RL method
# ===========================================================================

class PRMScorer:
    """Async PRM scorer for the Binary RL method.

    Evaluates assistant responses using next-state evidence via majority voting
    across m independent Tinker teacher queries.
    """

    def __init__(self, teacher_sampling_client, tokenizer,
                 prm_m: int = 3, temperature: float = 0.6, max_tokens: int = 4096):
        self._teacher_client = teacher_sampling_client
        self._tokenizer = tokenizer
        self.m = prm_m
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def evaluate(self, response_text: str, next_state_text: str,
                       next_state_role: str = "user",
                       session_id: str = "", turn_num: int = 0) -> dict:
        msgs = build_prm_eval_prompt(response_text, next_state_text, next_state_role)
        results = await asyncio.gather(
            *[self._query_once(msgs, i) for i in range(self.m)]
        )
        scores = [r[0] for r in results]
        final = majority_vote(scores)

        representative = ""
        if final != 0.0:
            for s, text in results:
                if s is not None and s == int(final):
                    representative = text
                    break

        votes_display = [s if s is not None else "fail" for s in scores]
        logger.info("[PRM] session=%s turn=%d votes=%s -> score=%.1f",
                    session_id, turn_num, votes_display, final)
        return {"score": final, "votes": votes_display, "representative": representative}

    async def _query_once(self, messages: list[dict], vote_id: int) -> tuple[Optional[int], str]:
        try:
            content = await _tinker_generate(
                self._teacher_client, self._tokenizer, messages,
                self.temperature, self.max_tokens,
            )
            return parse_prm_eval_score(content), content
        except Exception as e:
            logger.error("[PRM] query failed (vote %d): %s", vote_id, e, exc_info=True)
            return None, ""


# ===========================================================================
# OPDScorer — for OPD method
# ===========================================================================

class OPDScorer:
    """Hint judge + optional PRM eval + teacher log-probs for the OPD method."""

    def __init__(self, teacher_sampling_client, tokenizer,
                 prm_m: int = 3, temperature: float = 0.6, max_tokens: int = 4096,
                 eval_mode: bool = False):
        self._teacher_client = teacher_sampling_client
        self._tokenizer = tokenizer
        self.m = prm_m
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.eval_mode = eval_mode

    async def evaluate(self, response_text: str, next_state_text: str,
                       next_state_role: str, turn_data: dict, tokenizer,
                       normalize_fn, session_id: str = "", turn_num: int = 0) -> dict:
        # Hint judge votes
        msgs = build_hint_judge_messages(response_text, next_state_text, next_state_role)
        votes = await asyncio.gather(*[self._query_judge_once(msgs, i) for i in range(self.m)])

        # Optional PRM eval
        eval_score = None
        eval_raw = ""
        if self.eval_mode:
            eval_msgs = build_prm_eval_prompt(response_text, next_state_text, next_state_role)
            eval_results = await asyncio.gather(
                *[self._query_eval_once(eval_msgs, i) for i in range(self.m)]
            )
            eval_scores = [r[0] for r in eval_results]
            eval_raws = [r[1] for r in eval_results]
            eval_score = majority_vote(eval_scores)
            # Pick the representative raw text matching the winning vote
            for s, raw in zip(eval_scores, eval_raws):
                if s is not None and s == int(eval_score):
                    eval_raw = raw
                    break

        selected = select_best_hint(votes)
        if selected is None:
            logger.info("[OPD] session=%s turn=%d no valid hint, sample dropped", session_id, turn_num)
            return {"accepted": False, "teacher_log_probs": None, "hint": "",
                    "eval_score": eval_score, "hint_raw": "", "eval_raw": eval_raw}

        hint = selected["hint"].strip()
        teacher_lps = await _tinker_teacher_logprobs(
            self._teacher_client, tokenizer, hint, turn_data,
            normalize_fn, session_id, turn_num,
        )

        logger.info("[OPD] session=%s turn=%d accepted hint_len=%d hint=%s",
                     session_id, turn_num, len(hint), hint)
        return {"accepted": True, "teacher_log_probs": teacher_lps, "hint": hint,
                "eval_score": eval_score, "hint_raw": selected.get("raw", ""), "eval_raw": eval_raw}

    async def _query_judge_once(self, messages: list[dict], vote_id: int) -> dict:
        try:
            content = await _tinker_generate(
                self._teacher_client, self._tokenizer, messages,
                self.temperature, self.max_tokens,
            )
            score, hint = parse_judge_result(content)
            return {"vote_id": vote_id, "score": score, "hint": hint, "raw": content}
        except Exception as e:
            logger.error("[OPD] judge query failed (vote %d): %s", vote_id, e, exc_info=True)
            return {"vote_id": vote_id, "score": None, "hint": "", "raw": ""}

    async def _query_eval_once(self, messages: list[dict], vote_id: int) -> tuple[Optional[int], str]:
        try:
            content = await _tinker_generate(
                self._teacher_client, self._tokenizer, messages,
                self.temperature, self.max_tokens,
            )
            return parse_prm_eval_score(content), content
        except Exception as e:
            logger.error("[OPD] eval query failed (vote %d): %s", vote_id, e, exc_info=True)
            return None, ""


# ===========================================================================
# CombinedScorer — for Combined (OPD + RL) method
# ===========================================================================

class CombinedScorer:
    """Hint judge + PRM eval + teacher log-probs for the Combined method.

    Always runs both hint judge and PRM eval, returning both signals
    so the API server can dispatch OPD+RL, OPD-only, RL-only, or nothing.
    """

    def __init__(self, teacher_sampling_client, tokenizer,
                 prm_m: int = 3, temperature: float = 0.6, max_tokens: int = 4096):
        self._teacher_client = teacher_sampling_client
        self._tokenizer = tokenizer
        self.m = prm_m
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def evaluate(self, response_text: str, next_state_text: str,
                       next_state_role: str, turn_data: dict, tokenizer,
                       normalize_fn, session_id: str = "", turn_num: int = 0) -> dict:
        hint_msgs = build_hint_judge_messages(response_text, next_state_text, next_state_role)
        eval_msgs = build_prm_eval_prompt(response_text, next_state_text, next_state_role)

        hint_coros = [self._query_judge_once(hint_msgs, i) for i in range(self.m)]
        eval_coros = [self._query_eval_once(eval_msgs, i) for i in range(self.m)]

        all_results = await asyncio.gather(*hint_coros, *eval_coros)
        votes = list(all_results[:self.m])
        eval_results = list(all_results[self.m:])

        eval_scores = [r[0] for r in eval_results]
        eval_raws = [r[1] for r in eval_results]
        eval_score = majority_vote(eval_scores)

        eval_raw = ""
        for s, raw in zip(eval_scores, eval_raws):
            if s is not None and s == int(eval_score):
                eval_raw = raw
                break

        selected = select_best_hint(votes)
        if selected is None:
            logger.info("[Combined] session=%s turn=%d no valid hint eval_score=%.1f",
                         session_id, turn_num, eval_score)
            return {"accepted": False, "teacher_log_probs": None, "hint": "",
                    "eval_score": eval_score, "hint_raw": "", "eval_raw": eval_raw}

        hint = selected["hint"].strip()
        teacher_lps = await _tinker_teacher_logprobs(
            self._teacher_client, tokenizer, hint, turn_data,
            normalize_fn, session_id, turn_num,
        )

        logger.info("[Combined] session=%s turn=%d accepted hint_len=%d eval_score=%.1f hint=%s",
                     session_id, turn_num, len(hint), eval_score, hint)
        return {"accepted": True, "teacher_log_probs": teacher_lps, "hint": hint,
                "eval_score": eval_score, "hint_raw": selected.get("raw", ""), "eval_raw": eval_raw}

    async def _query_judge_once(self, messages: list[dict], vote_id: int) -> dict:
        try:
            content = await _tinker_generate(
                self._teacher_client, self._tokenizer, messages,
                self.temperature, self.max_tokens,
            )
            score, hint = parse_judge_result(content)
            return {"vote_id": vote_id, "score": score, "hint": hint, "raw": content}
        except Exception as e:
            logger.error("[Combined] judge query failed (vote %d): %s", vote_id, e, exc_info=True)
            return {"vote_id": vote_id, "score": None, "hint": "", "raw": ""}

    async def _query_eval_once(self, messages: list[dict], vote_id: int) -> tuple[Optional[int], str]:
        try:
            content = await _tinker_generate(
                self._teacher_client, self._tokenizer, messages,
                self.temperature, self.max_tokens,
            )
            return parse_prm_eval_score(content), content
        except Exception as e:
            logger.error("[Combined] eval query failed (vote %d): %s", vote_id, e, exc_info=True)
            return None, ""
