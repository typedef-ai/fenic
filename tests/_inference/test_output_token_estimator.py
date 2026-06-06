import threading

from fenic._inference.output_token_estimator import OutputTokenEstimator


def _estimator(**kw):
    # small min_samples keeps tests fast
    return OutputTokenEstimator(enabled=True, safety_margin=1.0, min_samples=5, window=256, **kw)


def test_cold_start_returns_static_ceiling():
    est = _estimator()
    # fewer than min_samples observations -> fall back to the static ceiling
    for _ in range(4):
        est.observe(("p", 512), 10)
    assert est.reserve(("p", 512), static_ceiling=8704, reasoning=False) == 8704


def test_warm_reserves_below_ceiling():
    est = _estimator()
    for _ in range(50):
        est.observe(("p", 512), 100)
    # p95 of constant 100 * margin 1.0 == 100, clamped below the 8704 ceiling
    assert est.reserve(("p", 512), static_ceiling=8704, reasoning=False) == 100


def test_reserve_clamps_to_static_ceiling():
    est = _estimator()
    for _ in range(50):
        est.observe(("p", 512), 100000)  # huge actuals
    # learned value would exceed the ceiling, so it is clamped down
    assert est.reserve(("p", 512), static_ceiling=512, reasoning=False) == 512


def test_safety_margin_applied():
    est = OutputTokenEstimator(enabled=True, safety_margin=1.5, min_samples=5, window=256)
    for _ in range(50):
        est.observe(("p", 512), 100)
    assert est.reserve(("p", 512), static_ceiling=8704, reasoning=False) == 150


def test_reasoning_uses_higher_quantile():
    est = _estimator()
    # right-skewed: mostly small, a few large -> p99 > p95
    for _ in range(99):
        est.observe(("p", 512), 100)
    est.observe(("p", 512), 5000)
    p95 = est.reserve(("p", 512), static_ceiling=10000, reasoning=False)
    p99 = est.reserve(("p", 512), static_ceiling=10000, reasoning=True)
    assert p99 > p95


def test_disabled_always_returns_ceiling():
    est = OutputTokenEstimator(enabled=False, safety_margin=1.0, min_samples=5, window=256)
    for _ in range(50):
        est.observe(("p", 512), 100)
    assert est.reserve(("p", 512), static_ceiling=8704, reasoning=False) == 8704


def test_keys_are_isolated():
    est = _estimator()
    for _ in range(50):
        est.observe(("p", 512), 100)
    # a different key has no samples -> ceiling
    assert est.reserve(("p", 1024), static_ceiling=4096, reasoning=False) == 4096


def test_concurrent_observe_and_reserve_is_safe():
    est = _estimator()

    def writer():
        for _ in range(2000):
            est.observe(("p", 512), 100)

    threads = [threading.Thread(target=writer) for _ in range(4)]
    for t in threads:
        t.start()
    for _ in range(2000):
        est.reserve(("p", 512), static_ceiling=8704, reasoning=False)
    for t in threads:
        t.join()
    assert 1 <= est.reserve(("p", 512), static_ceiling=8704, reasoning=False) <= 8704
