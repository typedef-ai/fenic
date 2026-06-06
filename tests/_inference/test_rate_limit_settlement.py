import time

from fenic._inference.rate_limit_strategy import (
    AdaptiveBackoffRateLimitStrategy,
    SeparatedTokenRateLimitStrategy,
    TokenEstimate,
    UnifiedTokenRateLimitStrategy,
)


def test_unified_settle_refunds_over_reservation():
    s = UnifiedTokenRateLimitStrategy(rpm=100, tpm=1000)
    reserved = TokenEstimate(input_tokens=200, output_tokens=300)  # total 500
    assert s.check_and_consume_rate_limit(reserved)               # bucket: 1000 -> 500
    actual = TokenEstimate(input_tokens=200, output_tokens=50)     # total 250
    s.settle(reserved, actual)                                     # refund 250 -> ~750
    avail = s.unified_tokens_bucket._get_available_capacity(time.time())
    assert 745 <= avail <= 755


def test_unified_settle_debits_under_reservation():
    s = UnifiedTokenRateLimitStrategy(rpm=100, tpm=1000)
    reserved = TokenEstimate(input_tokens=100, output_tokens=100)  # total 200
    assert s.check_and_consume_rate_limit(reserved)               # bucket: 1000 -> 800
    actual = TokenEstimate(input_tokens=100, output_tokens=400)    # total 500 (used MORE)
    s.settle(reserved, actual)                                     # extra debit 300 -> ~500
    avail = s.unified_tokens_bucket._get_available_capacity(time.time())
    assert 495 <= avail <= 505


def test_unified_settle_clamps_to_capacity():
    s = UnifiedTokenRateLimitStrategy(rpm=100, tpm=1000)
    reserved = TokenEstimate(input_tokens=0, output_tokens=10)
    # never consumed; bucket is full at 1000; a refund must not exceed tpm
    s.settle(reserved, TokenEstimate(input_tokens=0, output_tokens=0))
    avail = s.unified_tokens_bucket._get_available_capacity(time.time())
    assert avail == 1000


def test_separated_settle_refunds_each_bucket():
    s = SeparatedTokenRateLimitStrategy(rpm=100, input_tpm=1000, output_tpm=1000)
    reserved = TokenEstimate(input_tokens=300, output_tokens=400)
    assert s.check_and_consume_rate_limit(reserved)  # in: 700, out: 600
    actual = TokenEstimate(input_tokens=250, output_tokens=50)
    s.settle(reserved, actual)  # refund in 50 -> 750, out 350 -> 950
    now = time.time()
    assert 745 <= s.input_tokens_bucket._get_available_capacity(now) <= 755
    assert 945 <= s.output_tokens_bucket._get_available_capacity(now) <= 955


def test_adaptive_settle_is_noop():
    s = AdaptiveBackoffRateLimitStrategy(rpm=100)
    # no token accounting; must not raise
    s.settle(TokenEstimate(input_tokens=10, output_tokens=10),
             TokenEstimate(input_tokens=5, output_tokens=5))


def test_unified_settle_clamps_to_zero():
    s = UnifiedTokenRateLimitStrategy(rpm=100, tpm=1000)
    # massive under-reservation drives available below zero -> clamped to 0
    s.settle(TokenEstimate(input_tokens=0, output_tokens=0),
             TokenEstimate(input_tokens=2000, output_tokens=0))
    assert s.unified_tokens_bucket._get_available_capacity(time.time()) == 0


def test_separated_settle_clamps_to_zero():
    s = SeparatedTokenRateLimitStrategy(rpm=100, input_tpm=100, output_tpm=100)
    s.settle(TokenEstimate(input_tokens=0, output_tokens=0),
             TokenEstimate(input_tokens=500, output_tokens=500))
    now = time.time()
    assert s.input_tokens_bucket._get_available_capacity(now) == 0
    assert s.output_tokens_bucket._get_available_capacity(now) == 0
