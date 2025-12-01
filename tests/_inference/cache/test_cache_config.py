"""Unit tests for LLMResponseCacheConfig."""


import pytest

from fenic.api.session.config import LLMResponseCacheConfig
from fenic.core.types.enums import CacheBackend


class TestLLMResponseCacheConfig:
    """Test suite for LLMResponseCacheConfig validation and functionality."""

    def test_default_ttl(self):
        """Test that default TTL is 1 hour."""
        config = LLMResponseCacheConfig()
        assert config.ttl == "1h"
        assert config.ttl_seconds() == 3600

    def test_duration_parsing_seconds(self):
        """Test TTL parsing for seconds."""
        config = LLMResponseCacheConfig(ttl="30s")
        assert config.ttl_seconds() == 30

        config = LLMResponseCacheConfig(ttl="1s")
        assert config.ttl_seconds() == 1

    def test_duration_parsing_minutes(self):
        """Test TTL parsing for minutes."""
        config = LLMResponseCacheConfig(ttl="15m")
        assert config.ttl_seconds() == 900

        config = LLMResponseCacheConfig(ttl="1m")
        assert config.ttl_seconds() == 60

    def test_duration_parsing_hours(self):
        """Test TTL parsing for hours."""
        config = LLMResponseCacheConfig(ttl="2h")
        assert config.ttl_seconds() == 7200

        config = LLMResponseCacheConfig(ttl="1h")
        assert config.ttl_seconds() == 3600

    def test_duration_parsing_days(self):
        """Test TTL parsing for days."""
        config = LLMResponseCacheConfig(ttl="7d")
        assert config.ttl_seconds() == 604800

        config = LLMResponseCacheConfig(ttl="1d")
        assert config.ttl_seconds() == 86400

    def test_duration_parsing_case_insensitive(self):
        """Test that TTL parsing is case-insensitive."""
        config1 = LLMResponseCacheConfig(ttl="1H")
        config2 = LLMResponseCacheConfig(ttl="1h")
        assert config1.ttl_seconds() == config2.ttl_seconds()

        config3 = LLMResponseCacheConfig(ttl="30M")
        config4 = LLMResponseCacheConfig(ttl="30m")
        assert config3.ttl_seconds() == config4.ttl_seconds()

    def test_invalid_ttl_format_no_unit(self):
        """Test that TTL without unit raises ValueError."""
        with pytest.raises(ValueError, match="Invalid TTL format"):
            LLMResponseCacheConfig(ttl="123")

    def test_invalid_ttl_format_wrong_unit(self):
        """Test that TTL with invalid unit raises ValueError."""
        with pytest.raises(ValueError, match="Invalid TTL format"):
            LLMResponseCacheConfig(ttl="1x")

        with pytest.raises(ValueError, match="Invalid TTL format"):
            LLMResponseCacheConfig(ttl="1w")

    def test_invalid_ttl_format_no_number(self):
        """Test that TTL without number raises ValueError."""
        with pytest.raises(ValueError, match="Invalid TTL format"):
            LLMResponseCacheConfig(ttl="h")

        with pytest.raises(ValueError, match="Invalid TTL format"):
            LLMResponseCacheConfig(ttl="s")

    def test_invalid_ttl_format_mixed(self):
        """Test that mixed format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid TTL format"):
            LLMResponseCacheConfig(ttl="1h30m")

    def test_ttl_range_validation_min_seconds(self):
        """Test that minimum TTL validation works."""
        with pytest.raises(ValueError, match="must be at least 1 second"):
            LLMResponseCacheConfig(ttl="0s")

    def test_ttl_range_validation_max_hours(self):
        """Test that maximum hours validation works."""
        with pytest.raises(ValueError, match="cannot exceed 720 hours"):
            LLMResponseCacheConfig(ttl="721h")

    def test_ttl_range_validation_max_days(self):
        """Test that maximum days validation works."""
        with pytest.raises(ValueError, match="cannot exceed 30 days"):
            LLMResponseCacheConfig(ttl="31d")

    def test_ttl_range_validation_edge_cases(self):
        """Test edge cases for TTL range validation."""
        # Should work: exactly 30 days
        config = LLMResponseCacheConfig(ttl="30d")
        assert config.ttl_seconds() == 30 * 86400

        # Should work: exactly 720 hours
        config = LLMResponseCacheConfig(ttl="720h")
        assert config.ttl_seconds() == 720 * 3600

        # Should work: 1 second
        config = LLMResponseCacheConfig(ttl="1s")
        assert config.ttl_seconds() == 1

    def test_default_config(self):
        """Test default configuration values."""
        config = LLMResponseCacheConfig()
        assert config.backend == CacheBackend.LOCAL
        assert config.ttl == "1h"
        assert config.max_size_mb == 128
        assert config.namespace == "default"


    def test_max_size_validation(self):
        """Test that max_size_mb validation works."""
        # Should work
        config = LLMResponseCacheConfig(max_size_mb=1)
        assert config.max_size_mb == 1

        config = LLMResponseCacheConfig(max_size_mb=100000)
        assert config.max_size_mb == 100000

        # Should fail: too small
        with pytest.raises(ValueError):
            LLMResponseCacheConfig(max_size_mb=0)

        # Should fail: too large
        with pytest.raises(ValueError):
            LLMResponseCacheConfig(max_size_mb=100001)
