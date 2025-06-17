# Markdown Processing with Fenic

This example demonstrates how to process structured markdown documents using fenic's specialized markdown functions, JSON processing, and text extraction capabilities. We use the "Attention Is All You Need" paper, the foundational work that introduced the Transformer architecture, which powers modern large language models (LLMs).

## What This Example Shows

Learn how to process structured markdown documents using fenic's comprehensive capabilities:

- **`markdown.generate_toc()`** - Generate automatic table of contents from document headings
- **`markdown.extract_header_chunks()`** - Extract and structure document sections into DataFrame rows
- **`markdown.to_json()`** - Convert markdown to structured JSON for complex querying
- **`json.jq()`** - Navigate complex document structures with powerful jq queries
- **`text.extract()`** - Parse structured text using templates for field extraction
- **DataFrame operations** - Filter, explode, unnest, and transform document data
- **Text processing** - Split and parse content using fenic's templating functionality

## Files

- `attention_is_all_you_need.md` - OCR output of the "Attention Is All You Need" paper
- `markdown_processing.py` - Python script demonstrating markdown processing workflow

## Running the Example

```bash
cd examples/markdown_processing
uv run python markdown_processing.py
```

Make sure you have your OpenAI API key set in your environment:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## What You'll Learn

This example showcases how fenic transforms unstructured markdown into structured, queryable data:

### 1. Document Structure Analysis

- Load markdown documents into fenic DataFrames
- Cast content to `MarkdownType` for specialized processing
- Generate automatic table of contents from document hierarchy

### 2. Section Extraction and Structuring

- Extract document sections using `markdown.extract_header_chunks()`
- Convert nested arrays to DataFrame rows with `explode()` and `unnest()`
- Work with structured section data (headings, content, hierarchy paths)

### 3. Traditional Text Processing

- Filter DataFrames to find specific sections (e.g., References)
- Parse structured content using text splitting with regex patterns
- Handle academic citation formats and numbered references

### 4. JSON-Based Document Processing

- Convert markdown to structured JSON with `markdown.to_json()`
- Navigate complex nested document structures using `json.jq()`
- Handle hierarchical content with powerful jq query language
- Extract data from deeply nested JSON structures

### 5. Template-Based Text Extraction

- Use `text.extract()` with templates for structured data parsing
- Extract multiple fields from text in a single operation
- Parse academic citations into separate reference numbers and content
- Handle structured text formats with template patterns

### 6. Advanced DataFrame Operations

- Combine multiple markdown functions in a single select statement
- Chain operations for complex data transformations
- Process hierarchical document structures efficiently
- Cast between different data types (JsonType ↔ StringType)

Perfect for building academic paper analysis pipelines, research document processing, citation extraction systems, or preparing structured content for downstream analysis.
