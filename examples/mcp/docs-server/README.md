# Fenic Documentation MCP Server

An MCP (Model Context Protocol) server that provides AI assistants with tools to search and explore Fenic's API documentation.

## Features

- **Search Fenic API** - Search for functions, classes, methods across the codebase
- **Project Overview** - Get high-level understanding of Fenic's architecture
- **API Tree Navigation** - Explore the hierarchical structure of the public API
- **Learning Storage** - Store and retrieve Q&A pairs from interactions

## Setup

1. **(Optional) Set environment variables:**

   ```bash
   # Optional: Set custom data directory (defaults to ~/.fenic)
   export FENIC_WORK_DIR="/path/to/custom/directory"
   ```

2. **Populate the documentation database:**

   ```bash
   cd /path/to/fenic/mcp/docs-server
   python populate_tables.py
   ```

3. **Configure Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):

   ```json
   {
     "mcpServers": {
       "fenic-docs": {
         "command": "/path/to/fenic/examples/mcp/docs-server/.venv/bin/python",
         "args": ["/path/to/fenic/examples/mcp/docs-server/server.py"]
       }
     }
   }
   ```

4. **Restart Claude Desktop** and the server will be available.

## Tools Provided

### search(query, max_results=30)

Search the Fenic codebase and stored learnings for relevant information. Supports regex patterns.

### get_project_overview()

Get a comprehensive overview of the Fenic project with its API structure.

### get_api_tree()

Get the hierarchical tree structure of Fenic's public API.

### store_learning(question, answer, ...)

Store Q&A pairs from interactions for future search retrieval.

## Database Location

The server stores its data in a DuckDB database at:

- **Default**: `~/.fenic/docs.duckdb`
- **Custom**: `$FENIC_WORK_DIR/docs.duckdb` (if `FENIC_WORK_DIR` is set)

You can access this database directly with DuckDB tools for inspection and analysis.

## Troubleshooting

**Server fails to start with "Missing required tables" error:**

- Run `python populate_tables.py` to create the documentation database
- Ensure you have the required API keys set as environment variables

**Permission errors:**

- Check that the data directory (`~/.fenic` by default) is writable
- Or set `FENIC_WORK_DIR` to a directory you have write access to
