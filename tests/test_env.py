# tests/test_env.py
import os
from grpo_reasoning.utils.env import require_env

def test_require_env_raises_clearly(monkeypatch):
    monkeypatch.delenv("DEFINITELY_NOT_SET_XYZ", raising=False)
    try:
        require_env("DEFINITELY_NOT_SET_XYZ")
    except RuntimeError as e:
        assert "DEFINITELY_NOT_SET_XYZ" in str(e)
        return
    raise AssertionError("Expected RuntimeError")