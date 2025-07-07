# Fenic MCP Tools

This directory contains MCP (Model Context Protocol) servers and utilities for integrating Fenic with AI assistants and development tools.

## Overview

The Model Context Protocol (MCP) is a standard for enabling AI assistants to interact with external tools and data sources. This directory is organized to support multiple MCP servers, each serving a specific purpose.

## Available MCP Servers

### docs-server

A documentation server that provides tools for searching and exploring Fenic's API documentation. See [docs-server/README.md](docs-server/README.md) for details.

## Structure

```
mcp/
├── README.md                 # This file
├── docs-server/             # Documentation MCP server
│   ├── pyproject.toml      # uv/pip package configuration
│   ├── README.md           # Server-specific documentation
│   └── fenic_docs_mcp/     # Python package
│       ├── server.py       # MCP server implementation
│       ├── populate_tables.py  # Table population script
│       └── utils/          # Utility modules
└── (future servers...)      # Additional MCP servers can be added here
```

## Adding New MCP Servers

To add a new MCP server:

1. Create a new directory under `mcp/` (e.g., `mcp/code-generator/`)
2. Add a `pyproject.toml` with the server's dependencies
3. Create a Python package with your server implementation
4. Include a README.md with usage instructions
5. Follow the uv package structure for consistency

Example structure for a new server:

```
mcp/new-server/
├── pyproject.toml
├── README.md
└── fenic_new_mcp/
    ├── __init__.py
    ├── server.py
    └── utils/
```

## Development Guidelines

- Each MCP server should be a self-contained package
- Use uv for dependency management
- Include comprehensive documentation
- Follow Fenic's coding standards
- Add appropriate tests

## Installation

Each MCP server can be installed independently using uv:

```bash
cd mcp/<server-name>
uv pip install -e .
```

This allows for isolated dependencies and independent versioning of each MCP server.
