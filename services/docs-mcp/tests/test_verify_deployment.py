"""Tests for the production deployment probe."""

from unittest.mock import AsyncMock

import pytest
from fenic_mcp import verify_deployment


@pytest.mark.asyncio
async def test_verify_deployment_retries_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale Modal revision is retried before the probe succeeds."""
    verify_once = AsyncMock(side_effect=[RuntimeError("old revision"), None])
    sleep = AsyncMock()
    monkeypatch.setattr(verify_deployment, "_verify_once", verify_once)
    monkeypatch.setattr(verify_deployment.asyncio, "sleep", sleep)

    await verify_deployment.verify_deployment(
        "https://mcp.fenic.ai/",
        attempts=2,
        retry_delay_seconds=5,
    )

    assert verify_once.await_count == 2  # nosec B101
    sleep.assert_awaited_once_with(5)


@pytest.mark.asyncio
async def test_verify_deployment_raises_final_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final verification failure is reported after retries are exhausted."""
    verify_once = AsyncMock(side_effect=RuntimeError("old revision"))
    sleep = AsyncMock()
    monkeypatch.setattr(verify_deployment, "_verify_once", verify_once)
    monkeypatch.setattr(verify_deployment.asyncio, "sleep", sleep)

    with pytest.raises(RuntimeError, match="old revision"):
        await verify_deployment.verify_deployment(
            "https://mcp.fenic.ai/",
            attempts=2,
            retry_delay_seconds=0,
        )

    assert verify_once.await_count == 2  # nosec B101
    sleep.assert_awaited_once_with(0)


@pytest.mark.asyncio
async def test_verify_deployment_rejects_zero_attempts() -> None:
    """At least one deployment verification attempt is required."""
    with pytest.raises(ValueError, match="attempts must be at least 1"):
        await verify_deployment.verify_deployment(
            "https://mcp.fenic.ai/",
            attempts=0,
        )
