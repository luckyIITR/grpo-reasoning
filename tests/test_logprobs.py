# tests/test_logprobs.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from grpo_reasoning.grpo.logprobs import compute_logprobs

def test_logprob_shape_and_position_zero():
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
    model.eval()
    ids = tok("Hello world, this is a test.", return_tensors="pt")
    lp = compute_logprobs(model, ids["input_ids"], ids["attention_mask"])
    assert lp.shape == ids["input_ids"].shape
    assert lp[0, 0].item() == 0.0  # first position is unpredictable

def test_logprob_matches_manual_gather():
    """Cross-check against a manual computation on a tiny input."""
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
    model.eval()
    ids = tok("The cat sat.", return_tensors="pt")["input_ids"]
    am = torch.ones_like(ids)

    with torch.no_grad():
        lp = compute_logprobs(model, ids, am)
        # Manual: log_softmax of logits at position 0, gather token at position 1
        logits = model(ids).logits[0, 0]
        expected = torch.log_softmax(logits, dim=-1)[ids[0, 1]].item()
    assert abs(lp[0, 1].item() - expected) < 1e-4