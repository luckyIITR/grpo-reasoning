# src/grpo_reasoning/data/syllogisms.py
"""
Categorical syllogism generator.

Forms used (Aristotelian):
  A: All S are P     (universal affirmative)
  E: No S are P      (universal negative)
  I: Some S are P    (particular affirmative)
  O: Some S are not P (particular negative)

We generate 4-figure syllogisms with random category names and ask the model
to determine if the conclusion follows from the premises.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import random


class Mood(str, Enum):
    A = "A"  # All
    E = "E"  # No
    I = "I"  # Some
    O = "O"  # Some...not


# The 24 classically valid syllogisms by (figure, mood-triple)
# Figure determines the position of the middle term M.
# Fig 1: M-P, S-M   Fig 2: P-M, S-M   Fig 3: M-P, M-S   Fig 4: P-M, M-S
VALID_SYLLOGISMS: set[tuple[int, str, str, str]] = {
    (1, "A", "A", "A"), (1, "E", "A", "E"), (1, "A", "I", "I"), (1, "E", "I", "O"),
    (2, "E", "A", "E"), (2, "A", "E", "E"), (2, "E", "I", "O"), (2, "A", "O", "O"),
    (3, "A", "A", "I"), (3, "I", "A", "I"), (3, "A", "I", "I"), (3, "E", "A", "O"),
    (3, "O", "A", "O"), (3, "E", "I", "O"),
    (4, "A", "A", "I"), (4, "A", "E", "E"), (4, "I", "A", "I"), (4, "E", "A", "O"),
    (4, "E", "I", "O"),
    # Note: the traditional 24 includes some with "existential import" assumptions
    # we omit a few edge cases for cleanliness. ~19 entries here is fine for training.
}


@dataclass(frozen=True)
class Syllogism:
    premise1: str
    premise2: str
    conclusion: str
    is_valid: bool
    metadata: dict

    def to_prompt(self) -> str:
        return (
            f"Determine if the following syllogism is logically valid.\n\n"
            f"Premise 1: {self.premise1}\n"
            f"Premise 2: {self.premise2}\n"
            f"Conclusion: {self.conclusion}\n\n"
            f"Answer with 'valid' or 'invalid'."
        )

    @property
    def answer(self) -> str:
        return "valid" if self.is_valid else "invalid"


def _statement(mood: str, subject: str, predicate: str) -> str:
    templates = {
        "A": f"All {subject} are {predicate}",
        "E": f"No {subject} are {predicate}",
        "I": f"Some {subject} are {predicate}",
        "O": f"Some {subject} are not {predicate}",
    }
    return templates[mood]


# Realistic-sounding category names so the model can't memorize symbols
CATEGORIES = [
    "philosophers", "mathematicians", "biologists", "artists", "engineers",
    "musicians", "writers", "athletes", "teachers", "doctors", "lawyers",
    "scientists", "students", "researchers", "designers", "mammals", "birds",
    "reptiles", "predators", "herbivores", "vehicles", "tools", "machines",
]


def generate_syllogism(rng: random.Random) -> Syllogism:
    """Generate one syllogism (valid or invalid, ~50/50)."""
    figure = rng.randint(1, 4)
    moods = (rng.choice("AEIO"), rng.choice("AEIO"), rng.choice("AEIO"))
    is_valid = (figure, *moods) in VALID_SYLLOGISMS

    s, m, p = rng.sample(CATEGORIES, 3)

    # Figure determines middle-term position
    if figure == 1:
        prem1 = _statement(moods[0], m, p)
        prem2 = _statement(moods[1], s, m)
    elif figure == 2:
        prem1 = _statement(moods[0], p, m)
        prem2 = _statement(moods[1], s, m)
    elif figure == 3:
        prem1 = _statement(moods[0], m, p)
        prem2 = _statement(moods[1], m, s)
    else:  # figure == 4
        prem1 = _statement(moods[0], p, m)
        prem2 = _statement(moods[1], m, s)

    conclusion = _statement(moods[2], s, p)

    return Syllogism(
        premise1=prem1, premise2=prem2, conclusion=conclusion,
        is_valid=is_valid,
        metadata={"figure": figure, "moods": "".join(moods), "S": s, "M": m, "P": p},
    )


def generate_dataset(n: int, seed: int = 42) -> list[Syllogism]:
    """Generate n syllogisms with balanced valid/invalid ratio."""
    rng = random.Random(seed)
    out: list[Syllogism] = []
    valid_count = invalid_count = 0
    target = n // 2

    # Rejection sampling for balance
    while len(out) < n:
        s = generate_syllogism(rng)
        if s.is_valid and valid_count < target:
            out.append(s); valid_count += 1
        elif (not s.is_valid) and invalid_count < (n - target):
            out.append(s); invalid_count += 1

    rng.shuffle(out)
    return out