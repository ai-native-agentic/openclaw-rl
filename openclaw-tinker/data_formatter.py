"""Convert training samples to Tinker Datum format.

Supports all three methods:

  RL / OPD (sample_to_datum):
    advantage = scalar GRPO reward, optionally + KL penalty from teacher logprobs
    Used via: batch_to_datums(batch, advantages, kl_penalty_coef)

  Combined (sample_to_datum_combined):
    combined_adv = w_opd * (-kl_coef * (student_lp - teacher_lp)) + w_rl * reward
    Used via: batch_to_datums_combined(batch, w_opd, w_rl, kl_penalty_coef)

Tinker Datum convention:
  model_input   - input tokens (all but the last token of the full sequence)
  loss_fn_inputs:
    target_tokens - full sequence left-shifted by 1
    logprobs      - prompt positions = 0.0, response positions = sampled logprob
    advantages    - prompt = 0.0, response = advantage * loss_mask
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class TrainingSample:
    """One training example collected from the API proxy."""

    session_id: str
    turn_num: int
    prompt_tokens: list[int]
    response_tokens: list[int]
    response_logprobs: list[float]
    loss_mask: list[int]
    reward: float
    prompt_text: str = ""
    response_text: str = ""
    teacher_logprobs: Optional[list[float]] = None
    sample_type: str = ""  # "opd+rl", "opd", "rl", or "" (for pure RL method)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit(lst: list[float], length: int) -> list[float]:
    if len(lst) > length:
        return lst[:length]
    if len(lst) < length:
        return lst + [0.0] * (length - len(lst))
    return lst


def _sanitize(lst: list[float], name: str, session_id: str, turn_num: int) -> list[float]:
    bad = [i for i, v in enumerate(lst) if not math.isfinite(v)]
    if bad:
        logger.warning(
            "[DataFormatter] non-finite %s at %d positions for session=%s turn=%d",
            name, len(bad), session_id, turn_num,
        )
        for i in bad:
            lst[i] = 0.0
    return lst


def _build_datum(all_tokens: list[int], logprobs: list[float], advantages: list[float],
                 session_id: str, turn_num: int):
    """Build a Tinker Datum from prepared sequences."""
    import tinker
    from tinker import TensorData

    T = len(all_tokens) - 1
    if T <= 0:
        raise ValueError(
            f"Empty sequence: session={session_id} turn={turn_num} tokens={len(all_tokens)}"
        )

    target_tokens = all_tokens[1:]
    logprobs = _fit(logprobs, T)
    advantages = _fit(advantages, T)
    logprobs = _sanitize(logprobs, "logprobs", session_id, turn_num)
    advantages = _sanitize(advantages, "advantages", session_id, turn_num)

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(all_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "logprobs": TensorData.from_torch(torch.tensor(logprobs, dtype=torch.float32)),
            "advantages": TensorData.from_torch(torch.tensor(advantages, dtype=torch.float32)),
        },
    )


# ---------------------------------------------------------------------------
# RL / OPD datum conversion
# ---------------------------------------------------------------------------

def sample_to_datum(sample: TrainingSample, advantage: float, kl_penalty_coef: float = 0.0):
    """Convert one sample + scalar advantage into a Tinker Datum (RL / OPD)."""
    prompt_len = len(sample.prompt_tokens)
    all_tokens = sample.prompt_tokens + sample.response_tokens

    logprobs = [0.0] * (prompt_len - 1) + list(sample.response_logprobs)
    resp_advantages = [advantage * float(m) for m in sample.loss_mask]

    # OPD: add reverse-KL penalty to response advantages
    if sample.teacher_logprobs is not None and kl_penalty_coef > 0:
        for i in range(min(len(resp_advantages), len(sample.teacher_logprobs))):
            student_lp = sample.response_logprobs[i] if i < len(sample.response_logprobs) else 0.0
            teacher_lp = sample.teacher_logprobs[i]
            kl_i = student_lp - teacher_lp
            resp_advantages[i] += -kl_penalty_coef * kl_i * float(sample.loss_mask[i])

    advantages = [0.0] * (prompt_len - 1) + resp_advantages
    return _build_datum(all_tokens, logprobs, advantages, sample.session_id, sample.turn_num)


def batch_to_datums(
    batch: list[TrainingSample], advantages: list[float], kl_penalty_coef: float = 0.0,
) -> list:
    """Convert a batch of samples + per-sample scalar advantages to Tinker Datums."""
    datums = []
    for sample, adv in zip(batch, advantages):
        try:
            datums.append(sample_to_datum(sample, adv, kl_penalty_coef=kl_penalty_coef))
        except Exception as e:
            logger.error(
                "[DataFormatter] FAILED to convert session=%s turn=%d: %s",
                sample.session_id, sample.turn_num, e, exc_info=True,
            )
    return datums


# ---------------------------------------------------------------------------
# Combined datum conversion
# ---------------------------------------------------------------------------

def sample_to_datum_combined(
    sample: TrainingSample,
    w_opd: float = 1.0,
    w_rl: float = 1.0,
    kl_penalty_coef: float = 0.0,
):
    """Convert one sample into a Tinker Datum with combined OPD+RL advantages.

    combined_adv_i = w_opd * (-kl_coef * (student_lp_i - teacher_lp_i)) + w_rl * reward
    """
    prompt_len = len(sample.prompt_tokens)
    all_tokens = sample.prompt_tokens + sample.response_tokens

    logprobs = [0.0] * (prompt_len - 1) + list(sample.response_logprobs)

    resp_advantages = []
    for i in range(len(sample.response_tokens)):
        mask = float(sample.loss_mask[i]) if i < len(sample.loss_mask) else 0.0

        # RL component: broadcast scalar reward
        rl_adv = w_rl * sample.reward * mask

        # OPD component: reverse-KL from teacher
        opd_adv = 0.0
        if sample.teacher_logprobs is not None and kl_penalty_coef > 0 and i < len(sample.teacher_logprobs):
            student_lp = sample.response_logprobs[i] if i < len(sample.response_logprobs) else 0.0
            teacher_lp = sample.teacher_logprobs[i]
            opd_adv = w_opd * (-kl_penalty_coef * (student_lp - teacher_lp)) * mask

        resp_advantages.append(rl_adv + opd_adv)

    advantages = [0.0] * (prompt_len - 1) + resp_advantages
    return _build_datum(all_tokens, logprobs, advantages, sample.session_id, sample.turn_num)


def batch_to_datums_combined(
    batch: list[TrainingSample],
    w_opd: float = 1.0,
    w_rl: float = 1.0,
    kl_penalty_coef: float = 0.0,
) -> list:
    """Convert a batch of samples to Tinker Datums with combined advantages."""
    datums = []
    for sample in batch:
        try:
            datums.append(sample_to_datum_combined(
                sample, w_opd=w_opd, w_rl=w_rl, kl_penalty_coef=kl_penalty_coef,
            ))
        except Exception as e:
            logger.error(
                "[DataFormatter] FAILED to convert combined session=%s turn=%d: %s",
                sample.session_id, sample.turn_num, e, exc_info=True,
            )
    return datums


# ---------------------------------------------------------------------------
# GRPO advantages
# ---------------------------------------------------------------------------

def compute_grpo_advantages(batch: list[TrainingSample]) -> list[float]:
    """GRPO-style advantage: broadcast reward as advantage (no normalization).

    Matches OpenClaw-RL's --disable-rewards-normalization behavior.
    """
    return [s.reward for s in batch]
