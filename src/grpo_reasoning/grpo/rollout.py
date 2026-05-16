# src/grpo_reasoning/grpo/rollout.py
"""
Generate G responses per prompt under the current policy.

Key design points:
- We use HF's .generate() with do_sample=True for top-p/temperature sampling.
- We left-pad the prompts so that all rollouts in a batch start generating
  from the same column index. This makes the response mask trivial to compute.
- We DON'T compute log-probs here. We compute them in a separate pass over
  the concatenated (prompt + response) so that we can use the same code path
  for old/ref/new log-probs.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from transformers import PreTrainedTokenizer

from grpo_reasoning.data.prompt_template import format_chat


@dataclass
class RolloutBatch:
    """One batch of rollouts. Shapes:
        sequences:     [B*G, L]        prompt tokens + response tokens, left-padded
        prompt_mask:   [B*G, L]        1 where token is in the prompt, 0 elsewhere
        response_mask: [B*G, L]        1 where token is in the response (and not pad)
        attention_mask:[B*G, L]        1 for any non-pad token
        prompts:       list[str]       original prompt strings (len B*G, repeated G times)
        responses:     list[str]       decoded responses (len B*G)
    """
    sequences: torch.Tensor
    prompt_mask: torch.Tensor
    response_mask: torch.Tensor
    attention_mask: torch.Tensor
    prompts: list[str]
    responses: list[str]


@torch.no_grad()
def generate_rollouts(
    model,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    group_size: int,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> RolloutBatch:
    """Generate `group_size` responses per prompt.

    `prompts` is a list of B raw problem strings. The output has B*G sequences,
    with rollouts grouped contiguously: rollouts 0..G-1 are from prompt 0, etc.
    """
    model.eval()
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"  # critical for batched generation

    # Format with chat template, then repeat each prompt G times
    formatted = [format_chat(p, tokenizer) for p in prompts]
    expanded = [p for p in formatted for _ in range(group_size)]
    raw_expanded = [p for p in prompts for _ in range(group_size)]

    enc = tokenizer(
        expanded, return_tensors="pt", padding=True, truncation=True, max_length=512,
    ).to(device)
    prompt_len = enc["input_ids"].shape[1]

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )
    sequences = out.sequences  # [B*G, prompt_len + new_tokens]

    # Build masks
    # prompt_mask: 1 in the prompt region (left of prompt_len), 0 after
    # but ALSO 0 for actual padding tokens (left-pad is part of prompt visually,
    # we still want to exclude those from any mask we use for loss)
    full_attention = (sequences != tokenizer.pad_token_id).long()
    # The original attention_mask covers the prompt region's real tokens.
    # For positions >= prompt_len, treat anything != pad as a valid response token.
    prompt_mask = torch.zeros_like(sequences)
    prompt_mask[:, :prompt_len] = enc["attention_mask"]  # real prompt tokens only
    response_mask = full_attention.clone()
    response_mask[:, :prompt_len] = 0  # zero out the prompt region

    # Decode just the response portion for reward computation
    response_token_ids = sequences[:, prompt_len:]
    responses = tokenizer.batch_decode(response_token_ids, skip_special_tokens=True)

    return RolloutBatch(
        sequences=sequences,
        prompt_mask=prompt_mask,
        response_mask=response_mask,
        attention_mask=full_attention,
        prompts=raw_expanded,
        responses=responses,
    )