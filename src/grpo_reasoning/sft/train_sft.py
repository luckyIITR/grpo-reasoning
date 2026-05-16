# src/grpo_reasoning/sft/train_sft.py
"""
LoRA SFT trainer. Pure PyTorch — no Trainer abstraction.

Why not HF Trainer? Because the GRPO loop is custom and you'll benefit from
having the SFT loop also be explicit. The patterns transfer.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from grpo_reasoning.models.loader import ModelConfig, load_model_for_sft, load_tokenizer
from grpo_reasoning.data.dataset import SFTDataset, sft_collate
from grpo_reasoning.utils.seed import set_seed
from grpo_reasoning.utils.logging import MetricLogger, setup_logging


@dataclass
class SFTConfig:
    data_path: str = "data/processed/sft.jsonl"
    output_dir: str = "checkpoints/sft"
    epochs: int = 3
    batch_size: int = 8
    grad_accum_steps: int = 2  # effective batch 16
    lr: float = 2e-4           # higher than full-FT — LoRA tolerates this
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1
    max_length: int = 512
    eval_every_steps: int = 25
    seed: int = 42


def _cosine_schedule_with_warmup(optimizer, num_warmup, num_total):
    def lr_lambda(step):
        if step < num_warmup:
            return step / max(1, num_warmup)
        progress = (step - num_warmup) / max(1, num_total - num_warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def train_sft(cfg: SFTConfig, model_cfg: ModelConfig) -> str:
    setup_logging()
    set_seed(cfg.seed)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(model_cfg.model_name)
    tokenizer.padding_side = "right"  # SFT uses right padding
    model = load_model_for_sft(model_cfg, device="cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device

    ds = SFTDataset(cfg.data_path, tokenizer, max_length=cfg.max_length)
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=lambda b: sft_collate(b, tokenizer.pad_token_id),
    )

    steps_per_epoch = math.ceil(len(loader) / cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
    )
    scheduler = _cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    mlog = MetricLogger(
        project="grpo-reasoning", run_name=f"sft-{model_cfg.model_name.split('/')[-1]}",
        config={**cfg.__dict__, **model_cfg.__dict__},
    )

    model.train()
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(cfg.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for micro_step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / cfg.grad_accum_steps
            loss.backward()

            if (micro_step + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                mlog.log(
                    {
                        "sft/loss": out.loss.item(),
                        "sft/lr": scheduler.get_last_lr()[0],
                        "sft/epoch": epoch,
                    },
                    step=global_step,
                )
                pbar.set_postfix(loss=out.loss.item())

    # Save LoRA adapter only — small file, easy to share.
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    mlog.finish()
    return cfg.output_dir