from fenic.core.metrics import LMMetrics


def test_reserved_field_defaults_zero_and_adds():
    a = LMMetrics(num_output_tokens=10, num_reserved_output_tokens=100)
    b = LMMetrics(num_output_tokens=5, num_reserved_output_tokens=40)
    c = a + b
    assert c.num_output_tokens == 15
    assert c.num_reserved_output_tokens == 140


def test_reserved_field_default_is_zero():
    assert LMMetrics().num_reserved_output_tokens == 0
