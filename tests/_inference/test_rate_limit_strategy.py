import typing as _t

import pytest

from fenic._inference.rate_limit_strategy import (
    AdaptiveBackoffRateLimitStrategy,
    TokenEstimate,
)


class FakeClock:
    def __init__(self, start: float = 1_000_000.0):
        self.current_time = float(start)

    def now(self) -> float:
        return self.current_time

    def advance(self, seconds: _t.SupportsFloat) -> None:
        self.current_time += float(seconds)


@pytest.fixture()
def fake_clock(monkeypatch):
    clock = FakeClock()
    import fenic._inference.rate_limit_strategy as mod

    monkeypatch.setattr(mod.time, "time", clock.now)
    return clock


def consume_one_request(strategy: AdaptiveBackoffRateLimitStrategy, clock: FakeClock) -> bool:
    return strategy.check_and_consume_rate_limit(TokenEstimate())


def test_backoff_reduces_rpm_and_clears_burst(fake_clock):
    strategy = AdaptiveBackoffRateLimitStrategy(rpm=200, min_rpm=50, backoff_multiplier=0.5)
    assert consume_one_request(strategy, fake_clock)
    strategy.backoff(fake_clock.now())
    assert strategy.rpm == 100
    assert not consume_one_request(strategy, fake_clock)
    fake_clock.advance(60.0)
    assert consume_one_request(strategy, fake_clock)


def test_cooldown_gate_blocks_until_reset(fake_clock):
    strategy = AdaptiveBackoffRateLimitStrategy(rpm=100, min_rpm=50)
    strategy.register_rate_limit_hint(rpm_hint=40, retry_at_epoch_seconds=fake_clock.now() + 5.0)
    assert not consume_one_request(strategy, fake_clock)
    fake_clock.advance(5.1)
    assert consume_one_request(strategy, fake_clock)


def test_rpm_hint_clamps_and_disables_growth(fake_clock):
    strategy = AdaptiveBackoffRateLimitStrategy(rpm=200, min_rpm=50, additive_increment=20, increase_after_successes=2)
    strategy.register_rate_limit_hint(rpm_hint=120, retry_at_epoch_seconds=None)
    for _ in range(10):
        assert consume_one_request(strategy, fake_clock)
        fake_clock.advance(6.0)
    assert strategy.rpm <= 120


def test_additive_increase_without_hint(fake_clock):
    strategy = AdaptiveBackoffRateLimitStrategy(rpm=100, min_rpm=50, max_rpm=120, additive_increment=5, increase_after_successes=3)
    for _ in range(3):
        assert consume_one_request(strategy, fake_clock)
        fake_clock.advance(10.0)
    assert strategy.rpm == 105
    for _ in range(3):
        assert consume_one_request(strategy, fake_clock)
        fake_clock.advance(10.0)
    assert strategy.rpm == 110
    for _ in range(12):
        assert consume_one_request(strategy, fake_clock)
        fake_clock.advance(10.0)
    assert strategy.rpm <= 120

