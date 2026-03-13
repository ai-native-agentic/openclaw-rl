import asyncio
import logging
from typing import Any

import torch
from openclaw_opd_api_server import OpenClawOPDAPIServer, generate, reward_func  # noqa: F401
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class OpenClawCombineAPIServer(OpenClawOPDAPIServer):
    """Combine OPD hint distillation and RL PRM-eval training.

    Each turn produces at most ONE sample.  When both hint-judge and
    eval-judge succeed, the sample carries *both* real teacher log-probs
    AND the RL reward so that the combined advantage has both signals:

        combined_adv = w_opd * (teacher - old) + w_rl * reward

    Three cases:
    ┌──────────────────┬──────────────────┬────────────────────┐
    │ hint accepted?   │ eval ±1?         │ result             │
    ├──────────────────┼──────────────────┼────────────────────┤
    │ yes              │ yes              │ 1 combined sample  │
    │ yes              │ no               │ 1 OPD-only sample  │
    │ no               │ yes              │ 1 RL-only sample   │
    │ no               │ no               │ nothing            │
    └──────────────────┴──────────────────┴────────────────────┘
    """

    @staticmethod
    def _is_valid_rl_score(score) -> bool:
        return score in (1, -1, 1.0, -1.0)

    # ------------------------------------------------------------------
    # OPD / combined sample: real teacher log-probs, configurable reward.
    # ------------------------------------------------------------------
    async def _submit_turn_sample(
        self,
        turn_data: dict[str, Any],
        session_id: str,
        opd_result: dict[str, Any],
        reward: float = 0.0,
    ):
        prompt_ids = turn_data["prompt_ids"]
        response_ids = turn_data["response_ids"]

        teacher_log_probs = opd_result.get("teacher_log_probs") or []
        if len(teacher_log_probs) > len(response_ids):
            teacher_log_probs = teacher_log_probs[: len(response_ids)]
        elif len(teacher_log_probs) < len(response_ids):
            teacher_log_probs = teacher_log_probs + [0.0] * (
                len(response_ids) - len(teacher_log_probs)
            )

        sample = Sample()
        sample.prompt = turn_data["prompt_text"]
        sample.response = turn_data["response_text"]
        sample.tokens = prompt_ids + response_ids
        sample.response_length = len(response_ids)
        sample.loss_mask = [1] * len(response_ids)
        sample.rollout_log_probs = turn_data["response_logprobs"]
        sample.teacher_log_probs = torch.tensor(teacher_log_probs, dtype=torch.float32)

        if self._use_topk_distillation:
            K = self.distill_topk
            topk_lp = opd_result.get("teacher_topk_log_probs") or []
            topk_idx = opd_result.get("teacher_topk_indices") or []
            if len(topk_lp) > len(response_ids):
                topk_lp = topk_lp[: len(response_ids)]
                topk_idx = topk_idx[: len(response_ids)]
            elif len(topk_lp) < len(response_ids):
                pad_len = len(response_ids) - len(topk_lp)
                topk_lp = [[0.0] * K] * pad_len + topk_lp
                topk_idx = [list(range(K))] * pad_len + topk_idx
            sample.teacher_topk_log_probs = torch.tensor(topk_lp, dtype=torch.float32)
            sample.teacher_topk_indices = torch.tensor(topk_idx, dtype=torch.long)

        sample.status = Sample.Status.COMPLETED
        sample.index = next(self._index_counter)
        sample.group_index = next(self._group_counter)
        sample.reward = {"score": reward}

        tag = "OPD+RL" if reward != 0.0 else "OPD"
        logger.info(
            "[OpenClaw-Combine] submitted %s sample session=%s index=%d "
            "reward=%.1f prompt_len=%d response_len=%d hint_len=%d",
            tag,
            session_id,
            sample.index,
            reward,
            len(prompt_ids),
            len(response_ids),
            len(opd_result.get("hint", "")),
        )
        await asyncio.to_thread(self.output_queue.put, (sample.group_index, [sample]))

    # ------------------------------------------------------------------
    # RL-only sample: no real teacher signal, reward = eval_score (±1).
    # ------------------------------------------------------------------
    async def _submit_rl_turn_sample(
        self, turn_data: dict, session_id: str, eval_score: float,
    ):
        prompt_ids = turn_data["prompt_ids"]
        response_ids = turn_data["response_ids"]
        response_logprobs = turn_data["response_logprobs"]

        if len(response_logprobs) > len(response_ids):
            response_logprobs = response_logprobs[: len(response_ids)]
        elif len(response_logprobs) < len(response_ids):
            response_logprobs = response_logprobs + [0.0] * (
                len(response_ids) - len(response_logprobs)
            )

        sample = Sample()
        sample.prompt = turn_data["prompt_text"]
        sample.response = turn_data["response_text"]
        sample.tokens = prompt_ids + response_ids
        sample.response_length = len(response_ids)
        sample.loss_mask = [1] * len(response_ids)
        sample.rollout_log_probs = response_logprobs
        sample.teacher_log_probs = torch.tensor(response_logprobs, dtype=torch.float32)
        sample.status = Sample.Status.COMPLETED
        sample.index = next(self._index_counter)
        sample.group_index = next(self._group_counter)
        sample.reward = {"score": float(eval_score)}

        logger.info(
            "[OpenClaw-Combine] submitted RL sample session=%s index=%d "
            "score=%.1f prompt_len=%d response_len=%d",
            session_id,
            sample.index,
            float(eval_score),
            len(prompt_ids),
            len(response_ids),
        )
        await asyncio.to_thread(self.output_queue.put, (sample.group_index, [sample]))

    # ------------------------------------------------------------------
    # Dispatch: ONE sample per turn, merging both signals when possible.
    # ------------------------------------------------------------------
    def _maybe_submit_ready_samples(
        self, session_id: str, force_drop_without_next_state: bool = False,
    ):
        prm_tasks = self._prm_tasks.get(session_id, {})
        pending = self._pending_turn_data.get(session_id, {})
        for turn_num in sorted(list(pending.keys())):
            td = pending[turn_num]
            task = prm_tasks.get(turn_num)

            if task is None:
                if force_drop_without_next_state:
                    pending.pop(turn_num, None)
                    if self._eval_mode:
                        with self._eval_scores_lock:
                            self._eval_scores.append(0.0)
                    logger.info(
                        "[OpenClaw-Combine] dropped session=%s turn=%d (no next_state)",
                        session_id,
                        turn_num,
                    )
                continue
            if not task.done():
                continue

            pending.pop(turn_num, None)
            prm_tasks.pop(turn_num, None)
            try:
                opd_result = task.result()
            except Exception as e:
                logger.warning(
                    "[OpenClaw-Combine] evaluation task failed session=%s turn=%d: %s",
                    session_id,
                    turn_num,
                    e,
                )
                if self._eval_mode:
                    with self._eval_scores_lock:
                        self._eval_scores.append(0.0)
                continue

            eval_score = opd_result.get("eval_score")
            if self._eval_mode and eval_score is not None:
                with self._eval_scores_lock:
                    self._eval_scores.append(eval_score)

            opd_accepted = opd_result.get("accepted")
            has_valid_rl = self._is_valid_rl_score(eval_score)

            if opd_accepted and has_valid_rl:
                self._safe_create_task(
                    self._submit_turn_sample(
                        td, session_id, opd_result, reward=float(eval_score),
                    )
                )
            elif opd_accepted:
                self._safe_create_task(
                    self._submit_turn_sample(td, session_id, opd_result, reward=0.0)
                )
            elif has_valid_rl:
                self._safe_create_task(
                    self._submit_rl_turn_sample(td, session_id, float(eval_score))
                )
