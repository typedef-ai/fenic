# fenic: the dataframe (re)built for LLM inference

Canonical HTML: https://docs.fenic.ai/latest/

---

fenic is an opinionated, PySpark-inspired DataFrame framework from typedef.ai for building AI and agentic applications. Transform unstructured and structured data into insights using familiar DataFrame operations enhanced with semantic intelligence. With first-class support for markdown, transcripts, and semantic operators, plus efficient batch inference across any model provider.

## Quick Start with AI-Guided Learning & Development

fenic provides an MCP server that gives AI assistants deep understanding of the fenic API. This enables AI tools to provide accurate, context-aware assistance with:

- Learning fenic's API and features
- Understanding usage patterns and best practices
- Writing code using the correct functions and patterns
- Debugging issues with real knowledge of the codebase

### Connect Your AI Assistant

The easiest way to get started is using our hosted MCP server at <https://mcp.fenic.ai>.

**Example with Claude Code:**

```
claude mcp add -t http fenic-docs https://mcp.fenic.ai
```

Once connected, you can ask questions like:

- "How do I use semantic.extract() to parse JSON from text?"
- "Show me how to implement a custom async UDF"
- "What's the difference between semantic.map() and semantic.filter()?"
- "How do I set up batch inference with multiple LLM providers?"

The AI assistant will have direct access to fenic's complete API documentation and architectural details to provide accurate, helpful responses specific to fenic rather than generic Python advice.

For self-hosting, see the [docs-server example](https://github.com/typedef-ai/fenic/tree/main/examples/mcp_server/docs-server/).

## Install

fenic supports Python `[3.10, 3.11, 3.12]`

```
pip install fenic
```

Install optional feature extras when you need operators with heavier dependencies:

```
pip install "fenic[pdf]"       # semantic.parse_pdf and PDF metadata loading
pip install "fenic[cluster]"   # DataFrame.semantic.with_cluster_labels
pip install "fenic[sim-join]"  # semantic.sim_join
```

Extras can be combined with model provider extras:

```
pip install "fenic[google,pdf,cluster,sim-join]"
```

### LLM Provider Setup

fenic requires an API key from at least one LLM provider. Set the appropriate environment variable for your chosen provider:

```
# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# For Google
export GOOGLE_API_KEY="your-google-api-key"

# For Cohere
export COHERE_API_KEY="your-cohere-api-key"
```

## Quickstart

The fastest way to learn about fenic is by checking the examples.

Below is a quick list of the examples in this repo:

| Example | Description | Colab |
| --- | --- | --- |
| [Hello World!](https://github.com/typedef-ai/fenic/tree/main/examples/hello_world) | Introduction to semantic extraction and classification using fenic's core operators through error log analysis. |  |
| [Enrichment](https://github.com/typedef-ai/fenic/tree/main/examples/enrichment) | Multi-stage DataFrames with template-based text extraction, joins, and LLM-powered transformations demonstrated via log enrichment. |  |
| [Meeting Transcript Processing](https://github.com/typedef-ai/fenic/tree/main/examples/meeting_transcript_processing) | Native transcript parsing, Pydantic schema integration, and complex aggregations shown through meeting analysis. |  |
| [News Analysis](https://github.com/typedef-ai/fenic/tree/main/examples/news_analysis) | Analyze and extract insights from news articles using semantic operators and structured data processing. |  |
| [Podcast Summarization](https://github.com/typedef-ai/fenic/tree/main/examples/podcast_summarization) | Process and summarize podcast transcripts with speaker-aware analysis and key point extraction. |  |
| [Semantic Join](https://github.com/typedef-ai/fenic/tree/main/examples/semantic_joins) | Instead of simple fuzzy matching, use fenic's powerful semantic join functionality to match data across tables. |  |
| [Named Entity Recognition](https://github.com/typedef-ai/fenic/tree/main/examples/named_entity_recognition) | Extract and classify named entities from text using semantic extraction and classification. |  |
| [Markdown Processing](https://github.com/typedef-ai/fenic/tree/main/examples/markdown_processing) | Process and transform markdown documents with structured data extraction and formatting. |  |
| [JSON Processing](https://github.com/typedef-ai/fenic/tree/main/examples/json_processing) | Handle complex JSON data structures with semantic operations and schema validation. |  |
| [Feedback Clustering](https://github.com/typedef-ai/fenic/tree/main/examples/feedback_clustering) | Group and analyze feedback using semantic similarity and clustering operations. |  |
| [Document Extraction](https://github.com/typedef-ai/fenic/tree/main/examples/document_extraction) | Extract structured information from various document formats using semantic operators. |  |

(Feel free to click any example above to jump right to its folder.)

## Why use fenic?

fenic is an opinionated, PySpark-inspired DataFrame framework for building production AI and agentic applications.

Unlike traditional data tools retrofitted for LLMs, fenic's query engine is built from the ground up with inference in mind.

Transform structured and unstructured data into insights using familiar DataFrame operations enhanced with semantic intelligence. With first-class support for markdown, transcripts, and semantic operators, plus efficient batch inference across any model provider.

fenic brings the reliability of traditional data pipelines to AI workloads.

### Key Features

#### Purpose-Built for LLM Inference

- Query engine designed from scratch for AI workloads, not retrofitted
- Automatic batch optimization for API calls
- Built-in retry logic and rate limiting
- Token counting and cost tracking

#### Semantic Operators as First-Class Citizens

- `semantic.analyze_sentiment` - Built-in sentiment analysis
- `semantic.classify` - Categorize text with few-shot examples
- `semantic.extract` - Transform unstructured text into structured data with schemas
- `semantic.group_by` - Group data by semantic similarity
- `semantic.join` - Join DataFrames on meaning, not just values
- `semantic.map` - Apply natural language transformations
- `semantic.predicate` - Create predicates using natural language to filter rows
- `semantic.reduce` - Aggregate grouped data with LLM operations

#### Native Unstructured Data Support

Goes beyond typical multimodal data types (audio, images) by creating specialized types for text-heavy workloads:

- Markdown parsing and extraction as a first-class data type
- Transcript processing (SRT, generic formats) with speaker and timestamp awareness
- JSON manipulation with JQ expressions for nested data
- Automatic text chunking with configurable overlap for long documents

#### Production-Ready Infrastructure

- Multi-provider support (OpenAI, Anthropic, Gemini)
- Local and cloud execution backends
- Comprehensive error handling and logging
- Pydantic integration for type safety

#### Familiar DataFrame API

- PySpark-compatible operations
- Lazy evaluation and query optimization
- SQL support for complex queries
- Seamless integration with existing data pipelines

### Why DataFrames for LLM and Agentic Applications?

AI and agentic applications are fundamentally pipelines and workflows - exactly what DataFrame APIs were designed to handle. Rather than reinventing patterns for data transformation, filtering, and aggregation, fenic leverages decades of proven engineering practices.

#### Decoupled Architecture for Better Agents

fenic creates a clear separation between heavy inference tasks and real-time agent interactions. By moving batch processing out of the agent runtime, you get:

- More predictable and responsive agents
- Better resource utilization with batched LLM calls
- Cleaner separation between planning/orchestration and execution

#### Built for All Engineers

DataFrames aren't just for data practitioners. The fluent, composable API design makes it accessible to any engineer:

- Chain operations naturally: `df.filter(...).semantic.group_by(...)`
- Mix imperative and declarative styles seamlessly
- Get started quickly with familiar patterns from pandas/PySpark or SQL

## Support

Join our community on [Discord](https://discord.gg/Enfa5Kgxtc) where you can connect with other users, ask questions, and get help with your fenic projects. Our community is always happy to welcome newcomers!

If you find fenic useful, consider giving us a ⭐ at the top of [our repository](https://github.com/typedef-ai/fenic). Your support helps us grow and improve the framework for everyone!

## Contributing

We welcome contributions of all kinds! Whether you're interested in writing code, improving documentation, testing features, or proposing new ideas, your help is valuable to us.

For developers planning to submit code changes, we encourage you to first open an issue to discuss your ideas before creating a Pull Request. This helps ensure alignment with the project's direction and prevents duplicate efforts.

Please refer to our [contribution guidelines](https://docs.fenic.ai/latest/CONTRIBUTING/index.md) for detailed information about the development process and project setup.
