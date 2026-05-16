# src/grpo_reasoning/data/generate_synthetic.py
"""
Generate chain-of-thought solutions for SFT bootstrap using GPT-4.1-mini.

We give GPT the problem and the ground-truth answer, and ask it to produce
a reasoning trace ending in that answer. This guarantees the SFT data is
correctly labeled (we already know the answer; GPT just writes the trace).
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
import re
    
from grpo_reasoning.utils.env import require_env
from grpo_reasoning.data.syllogisms import generate_dataset as gen_syll
from grpo_reasoning.data.propositional import generate_dataset as gen_prop


CLIENT = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYSTEM = """You write concise step-by-step reasoning for logic problems. Strict rules:

Format (no exceptions):
<think>
Step 1: [one short sentence — state what's given]
Step 2: [one short sentence — make a single inference]
Step 3: [one short sentence — state the conclusion]
</think>
<answer>X</answer>

Constraints:
- Use 2-4 numbered steps. Never more than 4.
- Each step must say something NEW. Do not restate previous steps.
- Do not write "Therefore X. Therefore X." patterns.
- The <answer> must be exactly one word matching the expected answer.
- Do not write anything outside the tags.
- Do not mention being told the answer.
"""

USER_TEMPLATE = """Problem:
{problem}

The correct answer is: {answer}

Write a reasoning trace that arrives at this answer."""


def make_cot(problem_text: str, answer: str) -> str:
    resp = CLIENT.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TEMPLATE.format(problem=problem_text, answer=answer)},
        ],
        temperature=0.5,    # lower than before — less variance, tighter format
        max_tokens=200,     # forces brevity
    )
    return resp.choices[0].message.content.strip()


def build_sft_corpus(n_syll: int = 100, n_prop: int = 100, out_path: str = "data/processed/sft.jsonl") -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    examples = []

    for s in tqdm(gen_syll(n_syll, seed=42), desc="syllogisms"):
        try:
            completion = make_cot(s.to_prompt(), s.answer)
            examples.append({"prompt": s.to_prompt(), "completion": completion,
                             "ground_truth": s.answer, "task": "syllogism"})
        except Exception as e:
            print(f"skip: {e}")

    for p in tqdm(gen_prop(n_prop, seed=43), desc="propositional"):
        try:
            completion = make_cot(p.to_prompt(), p.answer)
            examples.append({"prompt": p.to_prompt(), "completion": completion,
                             "ground_truth": p.answer, "task": "propositional"})
        except Exception as e:
            print(f"skip: {e}")

    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(examples)} SFT examples to {out_path}")
    

# Validate the generated CoT data
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

def validate_example(completion: str, ground_truth: str) -> bool:
    think_match = THINK_RE.search(completion)
    ans_match = ANSWER_RE.search(completion)
    if not (think_match and ans_match):
        return False
    if ans_match.group(1).strip().lower() != ground_truth.lower():
        return False
    # NEW: reject overly long traces (likely repetitive)
    think_content = think_match.group(1)
    n_steps = think_content.count("Step ")
    if n_steps > 5 or n_steps < 2:
        return False
    if len(completion) > 800:  # rough length cap
        return False
    return True