# fenic Documentation MCP Server

**Dogfooding fenic to teach AI new APIs.**

This MCP server lets AI assistants explore fenic’s API docs and source like a power user, searching symbols, traversing the public API tree, and capturing “learnings” as Q&A for reuse. We built it with fenic itself, so the pipeline that organizes the docs is the same engine we use to build AI apps.

**Why it matters:** when you’re developing in a codebase the model hasn’t “seen,” you either hand-craft prompts or burn tokens asking vague questions.

Here, the assistant gets a curated, queryable view of the repo and evolves a private memory as you work. Because fenic blends batch semantic processing with cheap non-semantic tools (regex, AST, indexes), you get better answers with fewer tokens and lower latency.

## Why this exists

General models are great at Typescript and Python, not your private APIs. This server gives them a grounded, evolving knowledge base of fenic, so they answer with the right symbols, types, and patterns, instead of guessing. It turns “where do I start?” into “jump to fc.text.recursive_word_chunk and show usage.”

## How it works

1. Ingest fenic docs + source
2. Build a fenic knowledge store with fenic pipelines (batch semantic where needed, non-semantic everywhere else)
3. Expose tools via MCP (search, get_api_tree, etc.)
4. Your assistant queries, learns, and stores Q&A for next time.

## Features

- **Search fenic API** - Search for functions, classes, methods across the codebase
- **Project Overview** - Get high-level understanding of fenic's architecture
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

3. **Configure Claude Desktop** :

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

Search the fenic codebase and stored learnings for relevant information. Supports regex patterns.

### get_project_overview()

Get a comprehensive overview of the fenic project with its API structure.

### get_api_tree()

Get the hierarchical tree structure of fenic's public API.

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
