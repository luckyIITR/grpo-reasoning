# scripts/03_smoke_grpo.py
from grpo_reasoning.grpo.trainer import train_grpo
from grpo_reasoning.grpo.config import GRPOConfig
from grpo_reasoning.models.loader import ModelConfig

if __name__ == "__main__":
    model_cfg = ModelConfig(model_name="HuggingFaceTB/SmolLM-135M-Instruct")
    cfg = GRPOConfig(
        sft_adapter_path="checkpoints/sft",
        output_dir="checkpoints/grpo_smoke",
        n_iterations=3,
        batch_size=4,    # smaller for speed
        group_size=4,
        ppo_epochs=1,
        eval_every=999,  # skip eval during smoke
        save_every=999,
        wandb_enabled=False,
    )
    train_grpo(cfg, model_cfg)