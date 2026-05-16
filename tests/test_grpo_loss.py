# tests/test_grpo_loss.py
import torch
from grpo_reasoning.grpo.loss import grpo_loss

def _setup(B=2, L=4):
    torch.manual_seed(0)
    lp_old = torch.randn(B, L) * 0.1 - 1.0
    lp_new = lp_old.clone()  # start identical → ratio = 1
    lp_ref = lp_old.clone()
    adv = torch.tensor([1.0, -1.0])
    mask = torch.ones(B, L)
    return lp_new, lp_old, lp_ref, adv, mask

def test_initial_loss_equals_negative_advantage():
    """When π_θ == π_old, ratio=1, KL term=0, so loss = -mean(adv)."""
    lp_new, lp_old, lp_ref, adv, mask = _setup()
    out = grpo_loss(lp_new, lp_old, lp_ref, adv, mask, kl_coef=0.0)
    # advantages are [1, -1], per-response loss is [-1, 1], mean is 0
    assert abs(out.loss.item() - 0.0) < 1e-5

def test_loss_with_positive_advantage_only():
    """Single positive advantage → loss should be -advantage at ratio=1."""
    lp = torch.zeros(1, 3)
    adv = torch.tensor([0.5])
    mask = torch.ones(1, 3)
    out = grpo_loss(lp, lp, lp, adv, mask, kl_coef=0.0)
    assert abs(out.loss.item() + 0.5) < 1e-5

def test_kl_nonnegative():
    """k3 KL estimator is always ≥ 0."""
    torch.manual_seed(42)
    lp_new = torch.randn(4, 8)
    lp_old = torch.randn(4, 8)
    lp_ref = torch.randn(4, 8)
    adv = torch.randn(4)
    mask = torch.ones(4, 8)
    out = grpo_loss(lp_new, lp_old, lp_ref, adv, mask, kl_coef=1.0)
    assert out.kl_loss.item() >= -1e-6  # allow numerical slack

def test_clipping_caps_positive_advantage_gain():
    """If advantage > 0 and ratio is forced very high, the clipped path wins
    (smaller value), so loss saturates at -(1+eps)*A."""
    lp_old = torch.zeros(1, 1)
    lp_new = torch.tensor([[5.0]])  # ratio = e^5, huge
    lp_ref = torch.zeros(1, 1)
    adv = torch.tensor([1.0])
    mask = torch.ones(1, 1)
    out = grpo_loss(lp_new, lp_old, lp_ref, adv, mask, clip_eps=0.2, kl_coef=0.0)
    # Loss should be -(1.2) * 1.0 = -1.2, plus tiny KL contribution
    assert abs(out.loss.item() - (-1.2)) < 1e-3
    assert out.clip_frac.item() == 1.0

def test_mask_excludes_padding():
    """Padding tokens shouldn't contribute to the loss."""
    lp_new = torch.zeros(1, 4)
    lp_old = torch.zeros(1, 4)
    lp_ref = torch.zeros(1, 4)
    adv = torch.tensor([1.0])
    mask_full = torch.ones(1, 4)
    mask_partial = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    out_full = grpo_loss(lp_new, lp_old, lp_ref, adv, mask_full, kl_coef=0.0)
    out_partial = grpo_loss(lp_new, lp_old, lp_ref, adv, mask_partial, kl_coef=0.0)
    # Per-response normalization → both should equal -1.0
    assert abs(out_full.loss.item() - out_partial.loss.item()) < 1e-5