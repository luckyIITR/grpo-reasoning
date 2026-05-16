# src/grpo_reasoning/data/dataset.py
"""
PyTorch Dataset for SFT. The key behavior: we mask the prompt tokens out of
the loss so the model only learns to predict the assistant's response.
"""
from __future__ import annotations
import json
import torch
from torch.utils.data import Dataset

from grpo_reasoning.data.prompt_template import format_chat, format_sft_example


class SFTDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512):
        self.examples = []
        with open(jsonl_path) as f:
            for line in f:
                self.examples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        prompt_text = format_chat(ex["prompt"], self.tokenizer)
        full_text = format_sft_example(ex["prompt"], ex["completion"], self.tokenizer)

        # Tokenize both so we know where the prompt ends
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]

        # Truncate from the right (keep prompt intact)
        full_ids = full_ids[: self.max_length]
        prompt_len = min(len(prompt_ids), len(full_ids))

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[:prompt_len] = -100  # mask prompt tokens from loss

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        }


def sft_collate(batch: list[dict], pad_token_id: int) -> dict:
    """Right-pad a batch to max length within the batch."""
    max_len = max(b["input_ids"].size(0) for b in batch)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for b in batch:
        n = b["input_ids"].size(0)
        pad = max_len - n
        out["input_ids"].append(torch.cat([b["input_ids"], torch.full((pad,), pad_token_id, dtype=torch.long)]))
        out["labels"].append(torch.cat([b["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        out["attention_mask"].append(torch.cat([b["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
    return {k: torch.stack(v) for k, v in out.items()}