# src/grpo_reasoning/models/loader.py
"""
Load SmolLM/Qwen base models with optional LoRA adapters.

For both SFT and GRPO we use the same loader. The GRPO trainer will load
two copies of the model with the SFT-LoRA merged in: one trainable (policy)
and one frozen (reference for KL).
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel


@dataclass
class ModelConfig:
    model_name: str = "HuggingFaceTB/SmolLM-135M-Instruct"
    dtype: str = "bfloat16"            # SmolLM trains fine in bf16 on A100/L4
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Target modules — these names are correct for both SmolLM (Llama-style)
    # and Qwen2 architectures.
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


def _torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def load_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # For generation we'll need left-padding; for SFT we want right-padding.
    # The caller sets this after loading.
    return tok


def load_model_for_sft(cfg: ModelConfig, device: str = "cuda"):
    """Load base model + fresh LoRA adapter for SFT training."""
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=_torch_dtype(cfg.dtype),
        device_map=device,
    )
    base.config.use_cache = False  # incompatible with gradient checkpointing
    base.gradient_checkpointing_enable()

    if cfg.use_lora:
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora_cfg)
        model.print_trainable_parameters()
    else:
        model = base
    return model


def load_model_with_adapter(
    cfg: ModelConfig,
    adapter_path: str | None,
    trainable: bool,
    device: str = "cuda",
):
    """Load base + optional SFT adapter. Used by GRPO to load both
    the policy (trainable=True) and the reference (trainable=False)."""
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=_torch_dtype(cfg.dtype),
        device_map=device,
    )

    if adapter_path is not None:
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=trainable)
    else:
        model = base

    if not trainable:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    return model