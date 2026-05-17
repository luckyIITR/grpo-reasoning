# src/grpo_reasoning/grpo/config.py
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class GRPOConfig:
    # Model
    model_name: str = "HuggingFaceTB/SmolLM-135M-Instruct"
    sft_adapter_path: str = "checkpoints/sft"
    output_dir: str = "checkpoints/grpo"

    # Data
    n_train_problems: int = 5000   # total problems we'll generate on the fly
    n_eval_syll: int = 50
    n_eval_prop: int = 50

    # Rollout
    group_size: int = 8            # G — responses per prompt
    batch_size: int = 8            # B — prompts per iteration; total rollouts = B*G
    max_new_tokens: int = 200
    temperature: float = 1.0
    top_p: float = 0.95

    # Optimization
    n_iterations: int = 200
    ppo_epochs: int = 2            # inner epochs per rollout batch
    minibatch_size: int = 8        # micro-batch for grad computation (rollouts)
    lr: float = 1e-6               # MUCH lower than SFT — RL is fragile
    clip_eps: float = 0.2
    kl_coef: float = 0.04
    max_grad_norm: float = 1.0
    correctness_weight: float = 1.0
    format_weight: float = 0.1

    # Eval / checkpointing
    eval_every: int = 20           # iterations between evals
    save_every: int = 50

    # Misc
    seed: int = 42
    wandb_project: str = "grpo-reasoning"
    wandb_enabled: bool = True