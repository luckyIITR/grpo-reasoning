# scripts/02b_eval_sft.py
import torch
from grpo_reasoning.models.loader import ModelConfig, load_model_with_adapter, load_tokenizer
from grpo_reasoning.eval.benchmarks import evaluate

if __name__ == "__main__":
    model_cfg = ModelConfig(model_name="HuggingFaceTB/SmolLM-135M-Instruct")
    tok = load_tokenizer(model_cfg.model_name)

    # Baseline (no adapter)
    print("=== Baseline ===")
    base = load_model_with_adapter(model_cfg, adapter_path=None, trainable=False, device="cuda" if torch.cuda.is_available() else "cpu")
    r0 = evaluate(base, tok, n_syll=50, n_prop=50)
    print(f"acc={r0.accuracy:.3f}  format={r0.format_rate:.3f}  syll={r0.syll_acc:.3f}  prop={r0.prop_acc:.3f}")
    
    for ex in r0.examples:                                       # NEW
        print("---")
        print("PROMPT:", ex["prompt"][:200])
        print("GEN:", repr(ex["generation"][:300]))
        print("GOLD:", ex["gold"], "PRED:", ex["predicted"])
    del base

    # SFT
    print("=== SFT ===")
    sft = load_model_with_adapter(model_cfg, adapter_path="checkpoints/sft", trainable=False, device="cuda" if torch.cuda.is_available() else "cpu")
    r1 = evaluate(sft, tok, n_syll=50, n_prop=50)
    print(f"acc={r1.accuracy:.3f}  format={r1.format_rate:.3f}  syll={r1.syll_acc:.3f}  prop={r1.prop_acc:.3f}")
    print(f"Lift: {r1.accuracy - r0.accuracy:+.3f}")

    for ex in r1.examples[:3]:
        print("---"); print(ex["prompt"]); print("GEN:", ex["generation"]); print("GOLD:", ex["gold"])