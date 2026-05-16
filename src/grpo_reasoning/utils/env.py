# src/grpo_reasoning/utils/env.py
"""Centralized environment loading. Import this at the top of every entry-point script."""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

# Walk up from this file to find the project root (where .env lives)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ENV_PATH = _PROJECT_ROOT / ".env"

load_dotenv(_ENV_PATH, override=False)


def require_env(key: str) -> str:
    """Fetch an env var or raise a clear error. Use this instead of os.environ[key]."""
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(
            f"Missing required env var: {key}. "
            f"Add it to {_ENV_PATH} (see .env.example)."
        )
    return val