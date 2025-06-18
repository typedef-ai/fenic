# Roadmap

## Inference

### Model Cascading

- Implement model cascading for `semantic.predicate()`, `semantic.join()`, `semantic.classify()`, and `semantic.analyze_sentiment()`
- Enable use of cheaper models for simpler tasks with confidence thresholds based on log probabilities

### Model Provider Support

- **Anthropic**: Bedrock integration, Snowflake Cortex
- **OpenAI**: Azure OpenAI integration
- **Google**: Gemini via GCP Vertex
- **Compatibility**: Simple to add support for OpenAI API-compatible models
- **Local**: Local model support
- Integration with batch inference APIs for single LLM stage pipelines

## Query Engine

### Core Architecture

- Migrate to out-of-core execution to enable processing of larger datasets
- Add standard query engine optimizations:
- Predicate pushdown
- Projection pushdown
- Common subexpression elimination
- And many more
- Implement first-class SQL planning instead of treating SQL execution as a black box

### Missing Operators

- `EXCEPT` operator
- Non-equi join support
- Window functions
- Date/DateTime/Timestamp support

### Column Functions

- Enhanced ArrayType ergonomics (mapping/filtering/aggregating over lists)
- Math functions
- Implicit casting for improved developer experience
- More StringType functions

### Multi-modal DataType Support

- Support for logical types:
- Filesystem files
- HTML documents
- Images
- Videos

## Catalog

### Persistence & Views

- Persistent view support
- Expose curated tables/views via MCP (Model Context Protocol) to agents
- Save query metrics to managed system tables for cost tracking
- Save fenic managed tables as Parquet files managed by DuckLake instead of .duckdb native files

### Indexing

- Full-text search indices on fenic managed tables
- Semantic search indices on fenic managed tables

### AI Observability Integration

- Add support for emitting traces of LLM interactions to AI observability backends, possibly standardizing on OTEL traces

## Ingestion/IO

### First-Class Integrations

- **Mistral OCR**: Document to Markdown conversion
- **Whisper/WhisperX**: Audio/video to JSON/Markdown/SRT with diarization support

### Data Sources/Sinks

- Utilities for treating local file directories as first class DataFrame sources/sinks
- Hugging Face dataset integration

### Output Utilities

- Serialize query results as Markdown/JSON blobs for downstream LLM processing
