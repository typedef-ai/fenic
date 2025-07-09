# Product Overview

Fenic is an opinionated, PySpark-inspired DataFrame framework designed specifically for building production AI and agentic applications. It transforms structured and unstructured data into insights using familiar DataFrame operations enhanced with semantic intelligence.

## Core Value Proposition

- **Purpose-built for LLM inference**: Query engine designed from scratch for AI workloads, not retrofitted from traditional data tools
- **Semantic operators as first-class citizens**: Built-in operations like `semantic.classify`, `semantic.extract`, `semantic.join` that leverage LLMs for data transformation
- **Native unstructured data support**: First-class support for markdown, transcripts, JSON with specialized parsing and processing
- **Production-ready**: Multi-provider LLM support (OpenAI, Anthropic, Google), automatic batching, retry logic, cost tracking

## Key Features

- Familiar DataFrame API compatible with PySpark patterns
- Lazy evaluation with query optimization
- Batch inference optimization for efficient LLM API usage
- Native support for markdown, transcripts (SRT), and complex JSON
- Semantic operations for classification, extraction, grouping, and joining
- Multi-backend execution (local with Polars/DuckDB, cloud with Typedef)
- Comprehensive error handling and logging with Pydantic integration

## Target Use Cases

- AI and agentic application development
- Unstructured data processing and transformation
- Semantic analysis and classification
- Document processing and extraction
- Meeting transcript analysis
- News and content analysis
- Feedback clustering and analysis
