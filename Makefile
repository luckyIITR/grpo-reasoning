.PHONY: install test lint format data sft grpo eval clean

install:
	uv sync --all-extras

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

data:
	python scripts/01_generate_data.py

sft:
	python scripts/02_train_sft.py --config-name=sft

grpo:
	python scripts/03_train_grpo.py --config-name=grpo_smollm

eval:
	python scripts/04_evaluate.py

reproduce: data sft grpo eval
	@echo "Full pipeline complete. See outputs/results.md"

clean:
	rm -rf outputs/ wandb/ .pytest_cache/ .ruff_cache/