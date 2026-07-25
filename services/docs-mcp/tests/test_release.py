"""Tests for release version safeguards."""

import pytest
from fenic_mcp.release import normalize_fenic_version


@pytest.mark.parametrize(
    ("value", "expected"),
    [("0.10.0", "0.10.0"), ("v1.2.3", "1.2.3")],
)
def test_normalize_fenic_version(value, expected):
    """Accept stable release strings and strip tag prefixes."""
    assert normalize_fenic_version(value) == expected  # nosec B101


@pytest.mark.parametrize("value", ["main", "v1", "1.2", "1.2.3; echo bad"])
def test_normalize_fenic_version_rejects_invalid_values(value):
    """Reject values unsafe or ambiguous for release dependency pinning."""
    with pytest.raises(ValueError):
        normalize_fenic_version(value)
