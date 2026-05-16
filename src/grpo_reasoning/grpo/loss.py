# src/grpo_reasoning/grpo/loss.py
"""
PPO-style clipped policy gradient loss with KL penalty.

All tensors are [B*G, L]. The response_mask is the source of truth for which
tokens contribute to the loss.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class LossOutput:
    loss: torch.Tensor          # scalar
    pg_loss: torch.Tensor       # scalar (policy gradient component, before KL)
    kl_loss: torch.Tensor       # scalar (mean KL across response tokens)
    clip_frac: torch.Tensor     # scalar in [0,1], fraction of tokens that got clipped
    approx_kl: torch.Tensor     # scalar, used for early stopping if desired


def grpo_loss(
    log_probs_new: torch.Tensor,    # [B*G, L]
    log_probs_old: torch.Tensor,    # [B*G, L]
    log_probs_ref: torch.Tensor,    # [B*G, L]
    advantages: torch.Tensor,       # [B*G] — one scalar per rollout
    response_mask: torch.Tensor,    # [B*G, L], 1 where loss applies
    clip_eps: float = 0.2,
    kl_coef: float = 0.04,
) -> LossOutput:
    """Compute GRPO loss exactly as derived in Part A.6."""
    # Importance sampling ratio
    log_ratio = log_probs_new - log_probs_old
    ratio = torch.exp(log_ratio)                       # [B*G, L]

    # Broadcast scalar advantages to token-level
    adv = advantages.unsqueeze(1).expand_as(ratio)     # [B*G, L]

    # Clipped surrogate
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    pg_per_token = -torch.min(unclipped, clipped)      # negate because we minimize

    # Schulman k3 KL estimator
    log_ratio_ref = log_probs_ref - log_probs_new      # δ_t in our notation
    kl_per_token = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0

    # Per-token total loss
    per_token = pg_per_token + kl_coef * kl_per_token

    # Mask + average: per-response normalization (Part A.6 design choice)
    mask = response_mask.float()
    response_lens = mask.sum(dim=1).clamp(min=1.0)     # [B*G]
    per_response_loss = (per_token * mask).sum(dim=1) / response_lens  # [B*G]
    loss = per_response_loss.mean()

    # Diagnostics
    with torch.no_grad():
        pg_loss = ((pg_per_token * mask).sum(dim=1) / response_lens).mean()
        kl_loss = ((kl_per_token * mask).sum(dim=1) / response_lens).mean()
        clipped_tokens = ((ratio > 1.0 + clip_eps) | (ratio < 1.0 - clip_eps)).float()
        clip_frac = (clipped_tokens * mask).sum() / mask.sum().clamp(min=1.0)
        # Approximate forward KL for monitoring (not the same as the k3 penalty)
        approx_kl = ((-log_ratio) * mask).sum() / mask.sum().clamp(min=1.0)

    return LossOutput(loss=loss, pg_loss=pg_loss, kl_loss=kl_loss,
                      clip_frac=clip_frac, approx_kl=approx_kl)