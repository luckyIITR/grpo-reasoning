# src/grpo_reasoning/grpo/trainer.py
"""
GRPO training loop. Orchestrates rollout, advantage computation, and policy updates.
"""
from __future__ import annotations
import math
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from grpo_reasoning.grpo.config import GRPOConfig
from grpo_reasoning.grpo.problems import ProblemStream
from grpo_reasoning.grpo.rollout import generate_rollouts
from grpo_reasoning.grpo.logprobs import compute_logprobs
from grpo_reasoning.grpo.rewards import batch_rewards
from grpo_reasoning.grpo.advantages import compute_group_advantages
from grpo_reasoning.grpo.loss import grpo_loss
from grpo_reasoning.models.loader import (
    ModelConfig, load_tokenizer, load_model_with_adapter,
)
from grpo_reasoning.eval.benchmarks import evaluate
from grpo_reasoning.utils.seed import set_seed
from grpo_reasoning.utils.logging import MetricLogger, setup_logging


def _cosine_with_warmup(optimizer, num_warmup, num_total):
    def lr_lambda(step):
        if step < num_warmup:
            return step / max(1, num_warmup)
        progress = (step - num_warmup) / max(1, num_total - num_warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def train_grpo(cfg: GRPOConfig, model_cfg: ModelConfig) -> str:
    setup_logging()
    set_seed(cfg.seed)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # ---- Load tokenizer ----
    tokenizer = load_tokenizer(model_cfg.model_name)

    # ---- Load policy (trainable) and reference (frozen) ----
    # Both start from the SFT-bootstrapped checkpoint.
    print("Loading policy model...")
    policy = load_model_with_adapter(
        model_cfg, adapter_path=cfg.sft_adapter_path, trainable=True, device="cuda",
    )
    policy.train()

    print("Loading reference model...")
    reference = load_model_with_adapter(
        model_cfg, adapter_path=cfg.sft_adapter_path, trainable=False, device="cuda",
    )
    reference.eval()

    # ---- Optimizer + scheduler ----
    optimizer = AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0,
    )
    total_optimizer_steps = cfg.n_iterations * cfg.ppo_epochs * max(
        1, (cfg.batch_size * cfg.group_size) // cfg.minibatch_size
    )
    scheduler = _cosine_with_warmup(
        optimizer, num_warmup=max(10, total_optimizer_steps // 20),
        num_total=total_optimizer_steps,
    )

    # ---- Logger ----
    mlog = MetricLogger(
        project=cfg.wandb_project,
        run_name=f"grpo-{model_cfg.model_name.split('/')[-1]}",
        config={**cfg.__dict__, **model_cfg.__dict__},
        enabled=cfg.wandb_enabled,
    )

    # ---- Problem stream ----
    problems = ProblemStream(seed=cfg.seed)

    optimizer_step = 0

    # ============================================================
    # Main training loop
    # ============================================================
    for iteration in range(cfg.n_iterations):
        # ---- Phase A: rollout & advantage computation (no grad) ----
        policy.eval()  # generate in eval mode
        prompt_strs, ground_truths = problems.next_batch(cfg.batch_size)
        # Repeat ground_truths G times to align with B*G rollouts
        gts_expanded = [g for g in ground_truths for _ in range(cfg.group_size)]

        rollouts = generate_rollouts(
            policy, tokenizer, prompt_strs,
            group_size=cfg.group_size,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature, top_p=cfg.top_p,
        )
        device = rollouts.sequences.device

        # Compute rewards
        reward_components = batch_rewards(
            rollouts.responses, gts_expanded,
            correctness_weight=cfg.correctness_weight,
            format_weight=cfg.format_weight,
        )
        rewards = torch.tensor([rc.total for rc in reward_components], device=device)
        correctness = torch.tensor([rc.correctness for rc in reward_components], device=device)

        # Group-relative advantages
        advantages = compute_group_advantages(rewards, group_size=cfg.group_size)

        # Old & reference log-probs (no grad)
        with torch.no_grad():
            log_probs_old = compute_logprobs(
                policy, rollouts.sequences, rollouts.attention_mask,
            )
            log_probs_ref = compute_logprobs(
                reference, rollouts.sequences, rollouts.attention_mask,
            )

        # ---- Phase B: optimize ----
        policy.train()
        n_rollouts = rollouts.sequences.size(0)

        # Diagnostic flags accumulated across PPO epochs/minibatches
        iter_loss = iter_pg = iter_kl = iter_clip = 0.0
        n_updates = 0

        for ppo_epoch in range(cfg.ppo_epochs):
            # Shuffle indices each PPO epoch
            perm = torch.randperm(n_rollouts, device=device)
            for start in range(0, n_rollouts, cfg.minibatch_size):
                idx = perm[start : start + cfg.minibatch_size]
                mb_seq = rollouts.sequences[idx]
                mb_attn = rollouts.attention_mask[idx]
                mb_mask = rollouts.response_mask[idx]
                mb_lp_old = log_probs_old[idx]
                mb_lp_ref = log_probs_ref[idx]
                mb_adv = advantages[idx]

                # Forward pass with gradients
                log_probs_new = compute_logprobs(policy, mb_seq, mb_attn)

                loss_out = grpo_loss(
                    log_probs_new=log_probs_new,
                    log_probs_old=mb_lp_old,
                    log_probs_ref=mb_lp_ref,
                    advantages=mb_adv,
                    response_mask=mb_mask,
                    clip_eps=cfg.clip_eps,
                    kl_coef=cfg.kl_coef,
                )

                optimizer.zero_grad()
                loss_out.loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in policy.parameters() if p.requires_grad],
                    max_norm=cfg.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer_step += 1

                iter_loss += loss_out.loss.item()
                iter_pg += loss_out.pg_loss.item()
                iter_kl += loss_out.kl_loss.item()
                iter_clip += loss_out.clip_frac.item()
                n_updates += 1

        # ---- Per-iteration metrics ----
        avg_reward = rewards.mean().item()
        avg_correct = correctness.mean().item()
        # Mean group reward variance (high = good learning signal)
        group_std = rewards.view(cfg.batch_size, cfg.group_size).std(dim=1).mean().item()
        # Mean response length in tokens (watch for collapse)
        resp_len = rollouts.response_mask.sum(dim=1).float().mean().item()

        mlog.log({
            "train/iteration": iteration,
            "train/loss": iter_loss / max(1, n_updates),
            "train/pg_loss": iter_pg / max(1, n_updates),
            "train/kl_loss": iter_kl / max(1, n_updates),
            "train/clip_frac": iter_clip / max(1, n_updates),
            "train/reward_mean": avg_reward,
            "train/correctness_mean": avg_correct,
            "train/group_reward_std": group_std,
            "train/response_length": resp_len,
            "train/lr": scheduler.get_last_lr()[0],
        }, step=iteration)

        print(
            f"[iter {iteration:3d}] reward={avg_reward:.3f} "
            f"correct={avg_correct:.3f} group_std={group_std:.3f} "
            f"loss={iter_loss / max(1, n_updates):+.4f} "
            f"kl={iter_kl / max(1, n_updates):.4f} "
            f"clip={iter_clip / max(1, n_updates):.3f} "
            f"len={resp_len:.0f}"
        )

        # ---- Eval ----
        if (iteration + 1) % cfg.eval_every == 0:
            print(f"  running eval...")
            policy.eval()
            eval_result = evaluate(
                policy, tokenizer,
                n_syll=cfg.n_eval_syll, n_prop=cfg.n_eval_prop,
            )
            mlog.log({
                "eval/accuracy": eval_result.accuracy,
                "eval/format_rate": eval_result.format_rate,
                "eval/syll_acc": eval_result.syll_acc,
                "eval/prop_acc": eval_result.prop_acc,
            }, step=iteration)
            print(
                f"  eval: acc={eval_result.accuracy:.3f} "
                f"format={eval_result.format_rate:.3f} "
                f"syll={eval_result.syll_acc:.3f} prop={eval_result.prop_acc:.3f}"
            )
            policy.train()

        # ---- Checkpoint ----
        if (iteration + 1) % cfg.save_every == 0:
            ckpt_dir = Path(cfg.output_dir) / f"iter_{iteration + 1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(ckpt_dir)
            print(f"  saved checkpoint to {ckpt_dir}")

    # Final save
    policy.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    mlog.finish()
    return cfg.output_dir