"""Probe the public Fenic documentation MCP endpoint after deployment."""

import argparse
import asyncio
import logging

from fastmcp import Client

logger = logging.getLogger(__name__)

EXPECTED_TOOLS = {
    "get_api_tree",
    "get_entities",
    "get_entity",
    "get_project_overview",
    "search_fenic_api",
}


async def _verify_once(url: str) -> None:
    """Verify discovery, search, and representative entity lookup."""
    async with Client(url) as client:
        tools = await client.list_tools()
        actual_tools = {tool.name for tool in tools}
        if actual_tools != EXPECTED_TOOLS:
            raise RuntimeError(
                f"Unexpected tool contract: expected {EXPECTED_TOOLS}, got {actual_tools}"
            )

        search = await client.call_tool(
            "search_fenic_api",
            {
                "query": "recursive word chunks",
                "element_type": "function",
                "table_format": "structured",
            },
        )
        rows = search.structured_content["rows"]
        if not rows:
            raise RuntimeError("Representative API search returned no results")

        qualified_name = rows[0]["qualified_name"]
        entity = await client.call_tool(
            "get_entity",
            {
                "qualified_name": qualified_name,
                "table_format": "structured",
            },
        )
        if not entity.structured_content["rows"]:
            raise RuntimeError(f"Entity lookup failed for {qualified_name}")


async def verify_deployment(
    url: str,
    *,
    attempts: int = 12,
    retry_delay_seconds: float = 5,
) -> None:
    """Verify a deployment, allowing time for the new Modal revision to propagate."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    for attempt in range(1, attempts + 1):
        try:
            await _verify_once(url)
            return
        except Exception:
            if attempt == attempts:
                raise
            logger.warning(
                "Deployment verification failed (attempt %d/%d); retrying in %.1fs",
                attempt,
                attempts,
                retry_delay_seconds,
                exc_info=True,
            )
            await asyncio.sleep(retry_delay_seconds)


def main() -> None:
    """Run the production MCP verification probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://mcp.fenic.ai/")
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--retry-delay-seconds", type=float, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(
        verify_deployment(
            args.url,
            attempts=args.attempts,
            retry_delay_seconds=args.retry_delay_seconds,
        )
    )


if __name__ == "__main__":
    main()
