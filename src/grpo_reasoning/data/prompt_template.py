# src/grpo_reasoning/data/prompt_template.py
"""
Single source of truth for prompt formatting. Both SFT data and GRPO rollouts
must format prompts through these functions. Never inline a format string.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a careful logical reasoner. For each problem, think step by step "
    "inside <think>...</think> tags, then give your final answer inside "
    "<answer>...</answer> tags. The final answer must be exactly one word."
)


def format_chat(problem: str, tokenizer) -> str:
    """Render the problem as a chat prompt using the tokenizer's chat template.

    Uses add_generation_prompt=True so the prompt ends right where the model
    should start generating (i.e., at the assistant turn opener).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def format_sft_example(problem: str, completion: str, tokenizer) -> str:
    """Render a full SFT example: prompt + completion + EOS.

    The completion should already contain the <think>...</think><answer>...</answer>
    structure as generated in Phase 1.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": completion},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )