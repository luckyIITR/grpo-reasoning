# tests/test_smoke.py
def test_imports():
    """Smoke test: package imports cleanly."""
    import grpo_reasoning
    from grpo_reasoning.utils.seed import set_seed
    from grpo_reasoning.utils.logging import MetricLogger
    set_seed(42)
    assert True