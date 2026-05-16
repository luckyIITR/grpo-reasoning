# scripts/01_generate_data.py
from grpo_reasoning.data.generate_synthetic import build_sft_corpus
from grpo_reasoning.utils.seed import set_seed

if __name__ == "__main__":
    set_seed(42)
    build_sft_corpus(n_syll=120, n_prop=120, out_path="data/processed/sft.jsonl")