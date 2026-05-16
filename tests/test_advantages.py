# tests/test_advantages.py
import torch
from grpo_reasoning.grpo.advantages import compute_group_advantages

def test_zero_variance_group():
    """All identical rewards → all zero advantages."""
    r = torch.tensor([0.5, 0.5, 0.5, 0.5])
    a = compute_group_advantages(r, group_size=4)
    assert torch.allclose(a, torch.zeros_like(a), atol=1e-6)

def test_group_mean_is_baseline():
    """Advantage of the mean reward is exactly zero."""
    r = torch.tensor([0.0, 0.5, 1.0, 0.5])  # mean = 0.5
    a = compute_group_advantages(r, group_size=4)
    # The two 0.5 rollouts should have advantage 0
    assert abs(a[1].item()) < 1e-4
    assert abs(a[3].item()) < 1e-4
    # The 0.0 should be negative, the 1.0 positive
    assert a[0].item() < 0
    assert a[2].item() > 0

def test_multiple_groups_independent():
    """Each group normalizes independently."""
    # Group 1: [0, 1], Group 2: [0.4, 0.6]
    r = torch.tensor([0.0, 1.0, 0.4, 0.6])
    a = compute_group_advantages(r, group_size=2)
    # Group 1: ±1/std, Group 2: small ±
    assert abs(a[0].item() + a[1].item()) < 1e-4  # zero-mean within group
    assert abs(a[2].item() + a[3].item()) < 1e-4
    # Group 1 has bigger spread
    assert abs(a[0].item()) > abs(a[2].item())