# scripts/02_train_sft.py
from grpo_reasoning.sft.train_sft import SFTConfig, train_sft
from grpo_reasoning.models.loader import ModelConfig

if __name__ == "__main__":
    model_cfg = ModelConfig(model_name="HuggingFaceTB/SmolLM-135M-Instruct")
    sft_cfg = SFTConfig(epochs=4, batch_size=8, grad_accum_steps=2)
    path = train_sft(sft_cfg, model_cfg)
    print(f"SFT adapter saved to {path}")