"""Joint loss for mixed RL (GRPO) + OPD (distillation) batches.

Both branches use the same PPO-style clipped policy gradient objective,
matching SLIME's ``policy_loss_function``, but with different advantages:

- OPD samples: advantage = teacher_logp - old_logp  (token-level distillation)
- RL  samples: advantage = reward broadcast          (GRPO-style)

OPD samples carry reward=0 so GRPO advantage=0; RL samples carry
teacher_logp ≈ rollout_logp so teacher advantage ≈ 0.  The combined
advantage is simply their sum, and each branch naturally dominates for
its own sample type.
"""

from __future__ import annotations

import os
from argparse import Namespace
from collections.abc import Callable

import torch
from slime.backends.megatron_utils.loss import get_log_probs_and_entropy
from slime.utils.ppo_utils import compute_approx_kl, compute_policy_loss


def combine_loss_function(
    args: Namespace,
    batch: dict,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # ---- pre-computed GRPO advantages (reward broadcast) ----
    grpo_advantages = torch.cat(batch["advantages"], dim=0)

    old_log_probs_list = (
        batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]
    )

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)

    # ---- forward pass: new log-probs (and optional entropy) ----
    need_entropy_for_loss = args.entropy_coef != 0.0
    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=need_entropy_for_loss,
        max_seq_lens=max_seq_lens,
    )
    new_log_probs = torch.cat(log_probs_and_entropy["log_probs"], dim=0)
    old_log_probs = torch.cat(old_log_probs_list, dim=0)

    # ---- OPD teacher advantages ----
    teacher_log_probs_list = batch.get("teacher_log_probs")
    if teacher_log_probs_list is not None:
        device = new_log_probs.device
        teacher_advantages = torch.cat(
            [
                t.to(device=device) - o.to(device=device)
                for t, o in zip(teacher_log_probs_list, old_log_probs_list)
            ],
            dim=0,
        )
    else:
        teacher_advantages = torch.zeros_like(grpo_advantages)

    # ---- combine: w_opd * (teacher - old) + w_rl * grpo_advantage ----
    w_opd = float(os.getenv("OPENCLAW_COMBINE_W_OPD", "1.0"))
    w_rl = float(os.getenv("OPENCLAW_COMBINE_W_RL", "1.0"))
    combined_advantages = w_opd * teacher_advantages + w_rl * grpo_advantages

    # ---- PPO clipped policy loss (identical to SLIME policy_loss_function) ----
    ppo_kl = old_log_probs - new_log_probs
    pg_loss, pg_clipfrac = compute_policy_loss(
        ppo_kl, combined_advantages, args.eps_clip, args.eps_clip_high,
    )
    pg_loss = sum_of_sample_mean(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl_mean = sum_of_sample_mean(ppo_kl)

    # ---- entropy ----
    if need_entropy_for_loss:
        entropy = torch.cat(log_probs_and_entropy["entropy"], dim=0)
        entropy_loss = sum_of_sample_mean(entropy)
    else:
        with torch.no_grad():
            _, ent_data = get_log_probs_and_entropy(
                logits,
                args=args,
                unconcat_tokens=batch["unconcat_tokens"],
                total_lengths=total_lengths,
                response_lengths=response_lengths,
                with_entropy=True,
                max_seq_lens=max_seq_lens,
            )
            entropy_loss = sum_of_sample_mean(torch.cat(ent_data["entropy"], dim=0))

    loss = pg_loss - args.entropy_coef * entropy_loss

    # ---- KL loss (ref model regularisation) ----
    kl_loss = torch.tensor(0.0, device=logits.device)
    if args.use_kl_loss and batch.get("ref_log_probs") is not None:
        ref_log_probs = torch.cat(batch["ref_log_probs"], dim=0)
        kl = compute_approx_kl(
            new_log_probs, ref_log_probs, kl_loss_type=args.kl_loss_type,
        )
        kl_loss = sum_of_sample_mean(kl)
        loss = loss + args.kl_loss_coef * kl_loss

    if new_log_probs.numel() == 0:
        loss = loss + 0 * logits.sum()

    # ---- rollout vs train log-prob drift (monitoring) ----
    train_rollout_logprob_abs_diff = None
    if "rollout_log_probs" in batch and batch["rollout_log_probs"]:
        rollout_lp = torch.cat(batch["rollout_log_probs"], dim=0)
        train_rollout_logprob_abs_diff = sum_of_sample_mean(
            (old_log_probs - rollout_lp).abs()
        )

    reported_loss: dict[str, torch.Tensor] = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl_mean.clone().detach(),
    }
    if train_rollout_logprob_abs_diff is not None:
        reported_loss["train_rollout_logprob_abs_diff"] = (
            train_rollout_logprob_abs_diff.clone().detach()
        )
    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    return loss, reported_loss
