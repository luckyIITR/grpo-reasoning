# tests/test_rewards.py
from grpo_reasoning.grpo.rewards import compute_reward

def test_correct_well_formatted():
    r = compute_reward("<think>so it follows</think><answer>yes</answer>", "yes")
    assert r.correctness == 1.0
    assert r.format_bonus == 1.0
    assert r.total == 1.1

def test_wrong_well_formatted():
    r = compute_reward("<think>nope</think><answer>no</answer>", "yes")
    assert r.correctness == 0.0
    assert r.format_bonus == 1.0
    assert r.total == 0.1

def test_correct_no_format():
    r = compute_reward("the answer is yes", "yes")
    assert r.correctness == 0.0  # we ONLY reward parsed answers
    assert r.format_bonus == 0.0

def test_case_insensitivity():
    r = compute_reward("<think>x</think><answer>YES</answer>", "yes")
    assert r.correctness == 1.0