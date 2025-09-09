# Fenic Documentation MCP Server

**Using Fenic's native MCP capabilities to serve API documentation.**

This MCP server demonstrates Fenic's built-in MCP support, creating parameterized tools from DataFrame queries. The server exposes Fenic's API documentation through searchable, queryable tools that AI assistants can use to understand and work with the Fenic framework.

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

- **Native MCP Support** - Uses Fenic's built-in `create_mcp_server` and parameterized tools
- **Search Fenic API** - Search for functions, classes, methods using regex patterns
- **Get Entity Details** - Fetch detailed documentation for specific API elements
- **Project Overview** - Get high-level understanding of Fenic's architecture
- **API Tree Navigation** - Explore the hierarchical structure of the public API
- **Type-based Search** - Search for specific types of API elements

## Quick Start with Hosted Server

The easiest way to get started is using our hosted MCP server at https://mcp.fenic.ai. No setup required!

### Using with Claude Code

Simply add the fenic documentation server to Claude Code:

```bash
claude mcp add -t http fenic-docs https://mcp.fenic.ai
```

That's it! The fenic documentation tools are now available in your Claude Code session.

### Prompt Starter Pack

Try these prompts to explore fenic's capabilities:

#### Fit and Feasibility
1. **"What is fenic?"** - Get an overview of the framework and its purpose
2. **"What are some problems I can solve with fenic?"** - Explore use cases and applications
3. **"How does fenic compare to langchain?"** - Understand the differences and when to use each

#### Install and Initialize
1. **"How do I install and use fenic?"** - Get started with installation and basic usage

#### Concepts & Mental Model
1. **"What are the core concepts and abstractions of fenic?"** - Understand the fundamental building blocks
2. **"What is the purpose of the markdown data type in fenic?"** - Learn about unstructured data handling

#### Recipes / How-to Guides
1. **"How can I extract all the sections from a markdown document with fenic?"** - Practical example of document processing
2. **"Can I use fenic to generate synthetic data to then fine tune a model with unsloth?"** - Explore advanced workflows

#### Errors / Debugging
1. **"I'm getting the following error on my fenic script: Candidate generation for request d79ff1db67 was truncated for stop reason MAX_TOKENS, what's the issue?"** - Get help with common errors

## Self-Hosted Setup

If you prefer to run the server locally:

1. **Set environment variables:**

```bash
  export GEMINI_API_KEY="your-gemini-api-key"
  # Optional: Set custom data directory (defaults to ~/.fenic)
  export FENIC_WORK_DIR="/path/to/custom/directory"
```

2. **Install dependencies and populate the documentation database:**

```bash
  cd /path/to/fenic/examples/mcp/docs-server
  uv venv
  source .venv/bin/activate  # On Windows: .venv\Scripts\activate
  uv pip install fenic[mcp,google]
  python populate_tables.py
```

3. **Configure Claude Desktop**:

   Edit your Claude Desktop configuration file:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - Or specify your custom path: `/path/to/claude_desktop_config.json`

   ```json
   {
     "mcpServers": {
       "fenic-docs": {
         "command": "/path/to/fenic/examples/mcp/docs-server/.venv/bin/python",
         "args": ["/path/to/fenic/examples/mcp/docs-server/server.py"],
         "env": {
           "GEMINI_API_KEY": "your-gemini-api-key",
           "FENIC_WORK_DIR": "/path/to/custom/directory"
         }
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
- Ensure you have `GEMINI_API_KEY` set as an environment variable
- Check that the script completed successfully (should show "Successfully created all required tables")

**API key errors:**

- Ensure `GEMINI_API_KEY` is set with a valid Google AI Studio API key
- Get your API key from: https://aistudio.google.com/apikey

**Permission errors:**

- Check that the data directory (`~/.fenic` by default) is writable
- Or set `FENIC_WORK_DIR` to a directory you have write access to

**Import errors:**

- Ensure you've activated the virtual environment: `source .venv/bin/activate`
- Run `uv pip install -e .` to install all dependencies
