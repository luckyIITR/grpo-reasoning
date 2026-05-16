# src/grpo_reasoning/data/propositional.py
"""
Propositional logic problem generator.

Problem shape: given a set of premises about propositional variables P, Q, R, S
and a target proposition, determine if the target follows.

We restrict to a fragment that's decidable by short proofs (depth <=3) so the
ground truth is mechanical and the chain of thought has a known answer.
"""
from __future__ import annotations
from dataclasses import dataclass
import random
from itertools import product


@dataclass(frozen=True)
class PropProblem:
    premises: tuple[str, ...]
    target: str
    is_entailed: bool
    metadata: dict

    def to_prompt(self) -> str:
        prem_block = "\n".join(f"- {p}" for p in self.premises)
        return (
            f"Given the premises:\n{prem_block}\n\n"
            f"Does it logically follow that: {self.target}?\n\n"
            f"Answer with 'yes' or 'no'."
        )

    @property
    def answer(self) -> str:
        return "yes" if self.is_entailed else "no"


# Variables we'll use
VARS = ["P", "Q", "R", "S"]


def _eval_clause(clause: str, assignment: dict[str, bool]) -> bool:
    """Evaluate one of our restricted clause forms under an assignment."""
    # Forms: "P", "not P", "P implies Q", "P and Q", "P or Q"
    c = clause.strip()
    if c.startswith("not "):
        return not assignment[c[4:].strip()]
    for op, fn in [
        (" implies ", lambda a, b: (not a) or b),
        (" and ", lambda a, b: a and b),
        (" or ", lambda a, b: a or b),
    ]:
        if op in c:
            left, right = c.split(op)
            lv = assignment[left.strip()] if left.strip() in VARS else not assignment[left.strip()[4:]]
            rv = assignment[right.strip()] if right.strip() in VARS else not assignment[right.strip()[4:]]
            return fn(lv, rv)
    return assignment[c]


def _check_entailment(premises: tuple[str, ...], target: str) -> bool:
    """Brute-force entailment by enumerating all 2^n assignments."""
    for vals in product([False, True], repeat=len(VARS)):
        assignment = dict(zip(VARS, vals))
        try:
            if all(_eval_clause(p, assignment) for p in premises):
                if not _eval_clause(target, assignment):
                    return False  # countermodel found
        except KeyError:
            continue
    return True


def generate_prop_problem(rng: random.Random) -> PropProblem:
    """Generate one propositional problem with controlled difficulty."""
    n_premises = rng.randint(2, 4)
    used_vars = rng.sample(VARS, rng.randint(2, 4))

    premises: list[str] = []
    for _ in range(n_premises):
        form = rng.choice(["atom", "neg", "implies", "and", "or"])
        if form == "atom":
            premises.append(rng.choice(used_vars))
        elif form == "neg":
            premises.append(f"not {rng.choice(used_vars)}")
        else:
            a, b = rng.sample(used_vars, 2)
            premises.append(f"{a} {form} {b}")

    # Target: either a variable, its negation, or a simple compound
    target_form = rng.choice(["atom", "neg", "implies"])
    if target_form == "atom":
        target = rng.choice(used_vars)
    elif target_form == "neg":
        target = f"not {rng.choice(used_vars)}"
    else:
        a, b = rng.sample(used_vars, 2)
        target = f"{a} implies {b}"

    is_entailed = _check_entailment(tuple(premises), target)
    return PropProblem(
        premises=tuple(premises),
        target=target,
        is_entailed=is_entailed,
        metadata={"n_premises": n_premises, "vars": used_vars},
    )


def generate_dataset(n: int, seed: int = 42) -> list[PropProblem]:
    rng = random.Random(seed)
    out: list[PropProblem] = []
    yes_count = no_count = 0
    target = n // 2
    while len(out) < n:
        p = generate_prop_problem(rng)
        if p.is_entailed and yes_count < target:
            out.append(p); yes_count += 1
        elif (not p.is_entailed) and no_count < (n - target):
            out.append(p); no_count += 1
    rng.shuffle(out)
    return out