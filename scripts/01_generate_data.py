# scripts/01_generate_data.py
from grpo_reasoning.data.generate_synthetic import build_sft_corpus
from grpo_reasoning.utils.seed import set_seed

if __name__ == "__main__":
    set_seed(42)
    # Generate more, expecting to filter ~15-20%
    build_sft_corpus(n_syll=200, n_prop=200, out_path="data/processed/sft.jsonl")