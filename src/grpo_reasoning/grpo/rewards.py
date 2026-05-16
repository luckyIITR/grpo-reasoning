# src/grpo_reasoning/grpo/rewards.py
"""
Reward functions. All rewards are pure functions of (prompt, response, ground_truth).
"""
from __future__ import annotations
import re
from dataclasses import dataclass


ANSWER_RE = re.compile(r"<answer>\s*(\w+)\s*</answer>", re.IGNORECASE)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


@dataclass
class RewardComponents:
    total: float
    correctness: float  # 0 or 1
    format_bonus: float  # 0 or small positive


def compute_reward(
    response: str,
    ground_truth: str,
    correctness_weight: float = 1.0,
    format_weight: float = 0.1,
) -> RewardComponents:
    """Reward = correctness_weight * is_correct + format_weight * is_well_formatted.

    Format bonus is small relative to correctness; it just shapes early training
    when the policy doesn't yet produce parseable answers.
    """
    has_think = bool(THINK_RE.search(response))
    ans_match = ANSWER_RE.search(response)
    has_answer = ans_match is not None
    well_formatted = has_think and has_answer

    is_correct = 0.0
    if has_answer:
        predicted = ans_match.group(1).strip().lower()
        if predicted == ground_truth.strip().lower():
            is_correct = 1.0

    return RewardComponents(
        total=correctness_weight * is_correct + format_weight * float(well_formatted),
        correctness=is_correct,
        format_bonus=float(well_formatted),
    )


def batch_rewards(
    responses: list[str],
    ground_truths: list[str],
    **kwargs,
) -> list[RewardComponents]:
    return [compute_reward(r, g, **kwargs) for r, g in zip(responses, ground_truths)]