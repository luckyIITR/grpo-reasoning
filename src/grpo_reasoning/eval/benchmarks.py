# src/grpo_reasoning/eval/benchmarks.py
"""
Evaluation against held-out problems. Used after SFT and after GRPO.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from grpo_reasoning.data.syllogisms import generate_dataset as gen_syll
from grpo_reasoning.data.propositional import generate_dataset as gen_prop
from grpo_reasoning.data.prompt_template import format_chat


ANSWER_RE = re.compile(r"<answer>\s*(\w+)\s*</answer>", re.IGNORECASE)


@dataclass
class EvalResult:
    accuracy: float
    format_rate: float
    n_total: int
    syll_acc: float
    prop_acc: float
    examples: list[dict]  # first 5 for qualitative inspection


def extract_answer(text: str) -> str | None:
    m = ANSWER_RE.search(text)
    return m.group(1).lower().strip() if m else None


@torch.no_grad()
def evaluate(model, tokenizer, n_syll: int = 100, n_prop: int = 100,
             max_new_tokens: int = 256, eval_seed: int = 9999) -> EvalResult:
    """Use eval_seed != training seeds to ensure held-out problems."""
    model.eval()
    tokenizer.padding_side = "left"

    items = []
    for s in gen_syll(n_syll, seed=eval_seed):
        items.append({"prompt": s.to_prompt(), "answer": s.answer, "task": "syllogism"})
    for p in gen_prop(n_prop, seed=eval_seed + 1):
        items.append({"prompt": p.to_prompt(), "answer": p.answer, "task": "propositional"})

    correct, formatted = 0, 0
    syll_correct = prop_correct = 0
    syll_total = prop_total = 0
    examples = []

    batch_size = 16  # tune for your GPU memory
    for i in tqdm(range(0, len(items), batch_size), desc="eval"):
        batch = items[i:i + batch_size]
        prompts_formatted = [format_chat(it["prompt"], tokenizer) for it in batch]
        inputs = tokenizer(
            prompts_formatted, return_tensors="pt", padding=True, truncation=True,
        ).to(model.device)
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        # Decode just the new tokens for each item
        prompt_lens = inputs["attention_mask"].sum(dim=1)
        for j, it in enumerate(batch):
            gen_ids = out[j, inputs["input_ids"].shape[1]:]
            gen = tokenizer.decode(gen_ids, skip_special_tokens=True)
            pred = extract_answer(gen)
            is_formatted = pred is not None
            is_correct = is_formatted and pred == it["answer"].lower()
            formatted += int(is_formatted)
            correct += int(is_correct)
            if it["task"] == "syllogism":
                syll_total += 1; syll_correct += int(is_correct)
            else:
                prop_total += 1; prop_correct += int(is_correct)
            if len(examples) < 5:
                examples.append({"prompt": it["prompt"], "gold": it["answer"],
                                "generation": gen, "predicted": pred})

    return EvalResult(
        accuracy=correct / len(items),
        format_rate=formatted / len(items),
        n_total=len(items),
        syll_acc=syll_correct / max(1, syll_total),
        prop_acc=prop_correct / max(1, prop_total),
        examples=examples,
    )