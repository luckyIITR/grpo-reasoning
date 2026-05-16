# tests/test_data_generation.py
import random
from grpo_reasoning.data.syllogisms import generate_dataset as gen_syll, VALID_SYLLOGISMS
from grpo_reasoning.data.propositional import generate_dataset as gen_prop, _check_entailment

def test_syllogism_balance():
    ds = gen_syll(n=100, seed=0)
    n_valid = sum(1 for s in ds if s.is_valid)
    assert 40 <= n_valid <= 60, f"Expected ~50/50 split, got {n_valid}/100 valid"

def test_syllogism_determinism():
    ds1 = gen_syll(n=50, seed=123)
    ds2 = gen_syll(n=50, seed=123)
    assert [s.conclusion for s in ds1] == [s.conclusion for s in ds2]

def test_syllogism_metadata_matches_validity():
    """The is_valid flag must agree with the figure/mood lookup."""
    ds = gen_syll(n=200, seed=7)
    for s in ds:
        fig = s.metadata["figure"]
        moods = s.metadata["moods"]
        expected = (fig, moods[0], moods[1], moods[2]) in VALID_SYLLOGISMS
        assert s.is_valid == expected

def test_prop_entailment_modus_ponens():
    """Classic check: P, P implies Q  ⊨  Q."""
    assert _check_entailment(("P", "P implies Q"), "Q") is True

def test_prop_entailment_affirming_consequent():
    """Q, P implies Q  does NOT entail P (fallacy)."""
    assert _check_entailment(("Q", "P implies Q"), "P") is False

def test_prop_balance():
    ds = gen_prop(n=100, seed=0)
    n_yes = sum(1 for p in ds if p.is_entailed)
    assert 40 <= n_yes <= 60
    
def test_prop_premises_always_satisfiable():
    from grpo_reasoning.data.propositional import _premises_satisfiable
    ds = gen_prop(n=50, seed=11)
    for p in ds:
        assert _premises_satisfiable(p.premises), f"Inconsistent premises: {p.premises}"

def test_prop_target_not_in_premises():
    ds = gen_prop(n=50, seed=13)
    for p in ds:
        assert p.target not in p.premises