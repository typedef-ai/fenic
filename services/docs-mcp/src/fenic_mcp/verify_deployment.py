"""Probe the public Fenic documentation MCP endpoint after deployment."""

import argparse
import asyncio

from fastmcp import Client

EXPECTED_TOOLS = {
    "get_api_tree",
    "get_entities",
    "get_entity",
    "get_project_overview",
    "search_fenic_api",
}


async def verify_deployment(url: str) -> None:
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


def main() -> None:
    """Run the production MCP verification probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://mcp.fenic.ai/")
    args = parser.parse_args()
    asyncio.run(verify_deployment(args.url))


if __name__ == "__main__":
    main()
