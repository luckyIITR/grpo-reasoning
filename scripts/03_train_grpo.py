# scripts/03_train_grpo.py
from grpo_reasoning.grpo.trainer import train_grpo
from grpo_reasoning.grpo.config import GRPOConfig
from grpo_reasoning.models.loader import ModelConfig

if __name__ == "__main__":
    model_cfg = ModelConfig(model_name="HuggingFaceTB/SmolLM-135M-Instruct")
    cfg = GRPOConfig(
        sft_adapter_path="checkpoints/sft",
        output_dir="checkpoints/grpo",
        n_iterations=200,
        batch_size=8,
        group_size=8,
        ppo_epochs=2,
        lr=1e-6,
        kl_coef=0.04,
        eval_every=20,
    )
    path = train_grpo(cfg, model_cfg)
    print(f"GRPO checkpoint saved to {path}")