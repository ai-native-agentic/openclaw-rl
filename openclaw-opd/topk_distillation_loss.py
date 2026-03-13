"""Top-K logits-based distillation loss with tail trick.

Usage:
    --loss-type custom_loss
    --custom-loss-function-path topk_distillation_loss.topk_distillation_loss_function
    --distill-topk 50
    --disable-compute-advantages-and-returns

Reference: SDFT (arXiv 2601.19897), SDPO (arXiv 2601.20802)
"""

from __future__ import annotations

from argparse import Namespace
from typing import Callable

import torch
import torch.nn.functional as F
from megatron.core import mpu
from slime.backends.megatron_utils.loss import get_responses
from slime.utils.ppo_utils import compute_log_probs


def topk_distillation_loss_function(
    args: Namespace,
    batch: dict,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute top-K logits-based distillation loss with tail trick.

    Computes reverse KL divergence D_KL(student || teacher) over the teacher's
    top-K vocabulary tokens plus a "tail" bin capturing remaining probability
    mass, following the SDFT/SDPO approach.

    Reads from ``teacher_topk_log_probs`` ([T, K]) and ``teacher_topk_indices``
    ([T, K]) — separate fields that do not interfere with the legacy 1D
    ``teacher_log_probs`` used by the token-level OPD path.
    """
    teacher_topk_logprobs = batch["teacher_topk_log_probs"]
    teacher_topk_indices = batch["teacher_topk_indices"]
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)
    tp_group = mpu.get_tensor_model_parallel_group()

    K = args.distill_topk

    all_student_topk_logps = []
    all_teacher_topk_logps = []
    for i, (logits_chunk, tokens_chunk) in enumerate(get_responses(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    )):
        t_logps = teacher_topk_logprobs[i]
        t_indices = teacher_topk_indices[i]

        if not t_logps.is_cuda:
            t_logps = t_logps.to(device=logits_chunk.device)
        if not t_indices.is_cuda:
            t_indices = t_indices.to(device=logits_chunk.device)

        student_logps_k = []
        need_clone = torch.is_grad_enabled()
        for k in range(K):
            logit_input = logits_chunk.clone() if need_clone else logits_chunk
            lp_k = compute_log_probs(logit_input, t_indices[:, k], tp_group)
            student_logps_k.append(lp_k.squeeze(-1))
        student_topk_logps = torch.stack(student_logps_k, dim=-1)

        all_student_topk_logps.append(student_topk_logps)
        all_teacher_topk_logps.append(t_logps)

    student_topk = torch.cat(all_student_topk_logps, dim=0)
    teacher_topk = torch.cat(all_teacher_topk_logps, dim=0)

    student_log_s = torch.logsumexp(student_topk, dim=-1, keepdim=True)
    student_log_s = torch.clamp(student_log_s, max=-1e-7)
    student_tail = torch.log(-torch.expm1(student_log_s))

    teacher_log_s = torch.logsumexp(teacher_topk, dim=-1, keepdim=True)
    teacher_log_s = torch.clamp(teacher_log_s, max=-1e-7)
    teacher_tail = torch.log(-torch.expm1(teacher_log_s))

    student_with_tail = torch.cat([student_topk, student_tail], dim=-1)
    teacher_with_tail = torch.cat([teacher_topk, teacher_tail], dim=-1)

    per_token_kl = F.kl_div(
        teacher_with_tail,
        student_with_tail,
        reduction="none",
        log_target=True,
    ).sum(dim=-1)

    kl_loss = sum_of_sample_mean(per_token_kl)

    loss = kl_loss
    entropy_loss = torch.tensor(0.0, device=logits.device)
    if args.entropy_coef != 0.0:
        student_probs = torch.exp(student_with_tail)
        entropy = -(student_probs * student_with_tail).sum(dim=-1)
        entropy_loss = sum_of_sample_mean(entropy)
        loss = loss - args.entropy_coef * entropy_loss

    if per_token_kl.numel() == 0:
        loss = loss + 0 * logits.sum()

    reported_loss = {
        "loss": loss.clone().detach(),
        "kl_loss": kl_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
    }

    return loss, reported_loss
