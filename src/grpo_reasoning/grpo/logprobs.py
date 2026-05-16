# src/grpo_reasoning/grpo/logprobs.py
"""
Teacher-forced log-prob computation under a given model.

For a sequence of tokens x_1, x_2, ..., x_L, an autoregressive LM's output
at position t predicts x_{t+1}. So to get log p(x_t) for t in [1, L-1], we
take the model's logits at position t-1 and gather the entry for token x_t.

The result has length L-1. We pad it back to L with a zero at position 0 to
keep shape alignment with sequences/masks.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def compute_logprobs(
    model,
    sequences: torch.Tensor,        # [B, L]
    attention_mask: torch.Tensor,   # [B, L]
) -> torch.Tensor:
    """Return per-token log-probs of `sequences` under `model`. Shape [B, L].

    Position 0 is always 0 (there's no prediction for the first token). Use
    response_mask downstream to mask out anything you don't want to score.
    """
    out = model(input_ids=sequences, attention_mask=attention_mask, use_cache=False)
    logits = out.logits  # [B, L, V]

    # Shift: logits at t predict token at t+1
    logits = logits[:, :-1, :]                       # [B, L-1, V]
    targets = sequences[:, 1:]                       # [B, L-1]
    log_probs = F.log_softmax(logits, dim=-1)        # [B, L-1, V]
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, L-1]

    # Pad position 0 with zero so shapes line up with `sequences`
    pad = torch.zeros(token_log_probs.size(0), 1, device=token_log_probs.device,
                      dtype=token_log_probs.dtype)
    return torch.cat([pad, token_log_probs], dim=1)  # [B, L]