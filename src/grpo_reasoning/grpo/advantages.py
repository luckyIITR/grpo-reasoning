# src/grpo_reasoning/grpo/advantages.py
"""
Group-relative advantage estimation — the defining feature of GRPO.
"""
from __future__ import annotations
import torch


def compute_group_advantages(
    rewards: torch.Tensor,    # [B * G]
    group_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Reshape to [B, G], normalize within each group, flatten back to [B*G].

    Returns one scalar advantage per rollout. The caller expands this to per-token
    advantages by broadcasting across the response_mask.
    """
    assert rewards.dim() == 1
    assert rewards.size(0) % group_size == 0, "rewards length not divisible by group_size"
    n_groups = rewards.size(0) // group_size

    grouped = rewards.view(n_groups, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)

    advantages = (grouped - mean) / (std + eps)
    return advantages.view(-1)  # [B*G]