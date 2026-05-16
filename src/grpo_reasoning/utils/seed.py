# src/grpo_reasoning/utils/seed.py
import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Set all relevant seeds for reproducibility.

    Note: full determinism on GPU also requires
    `torch.use_deterministic_algorithms(True)` and the env var
    `CUBLAS_WORKSPACE_CONFIG=:4096:8`, but these slow training and
    are usually only enabled for debugging.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)