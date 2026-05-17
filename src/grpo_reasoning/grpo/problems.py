# src/grpo_reasoning/grpo/problems.py
"""
Infinite problem stream for GRPO training. Each call to next_batch()
returns a batch of fresh (prompt, ground_truth) pairs.
"""
from __future__ import annotations
import random
from grpo_reasoning.data.syllogisms import generate_syllogism
from grpo_reasoning.data.propositional import generate_prop_problem


class ProblemStream:
    def __init__(self, seed: int = 0, syll_fraction: float = 0.5):
        self.rng = random.Random(seed)
        self.syll_fraction = syll_fraction

    def next_batch(self, batch_size: int) -> tuple[list[str], list[str]]:
        prompts: list[str] = []
        gts: list[str] = []
        for _ in range(batch_size):
            if self.rng.random() < self.syll_fraction:
                s = generate_syllogism(self.rng)
                prompts.append(s.to_prompt())
                gts.append(s.answer)
            else:
                p = generate_prop_problem(self.rng)
                prompts.append(p.to_prompt())
                gts.append(p.answer)
        return prompts, gts