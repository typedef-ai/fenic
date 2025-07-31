# Getting Started with Fenic

## What is Fenic?

Fenic is an opinionated, PySpark-inspired DataFrame framework for building production AI and agentic applications. Unlike traditional data tools retrofitted for LLMs, fenic's query engine is built from the ground up with inference in mind. It transforms structured and unstructured data into insights using familiar DataFrame operations enhanced with semantic intelligence.

### Key Features

**AI-First Design**

- Query engine designed from scratch for AI workloads, not retrofitted
- Automatic batch optimization for API calls
- Built-in retry logic and rate limiting
- Token counting and cost tracking

**Semantic Operations**

- `semantic.analyze_sentiment` - Built-in sentiment analysis
- `semantic.classify` - Categorize text with few-shot examples
- `semantic.extract` - Transform unstructured text into structured data with schemas
- `semantic.group_by` - Group data by semantic similarity
- `semantic.join` - Join DataFrames on meaning, not just values
- `semantic.map` - Apply natural language transformations
- `semantic.predicate` - Create predicates using natural language to filter rows
- `semantic.reduce` - Aggregate grouped data with LLM operations

**AI-Native Data Types**

- Markdown parsing and extraction as a first-class data type
- Transcript processing (SRT, generic formats) with speaker and timestamp awareness
- JSON manipulation with JQ expressions for nested data
- Automatic text chunking with configurable overlap for long documents

## Installation

### Prerequisites

Fenic supports Python [3.10, 3.11, 3.12]

### Install Fenic

```bash
pip install fenic
```

### Set Up API Keys

Fenic requires an API key from at least one LLM provider. Set the appropriate environment variable for your chosen provider:

```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# For Google
export GEMINI_API_KEY="your-google-api-key"
```

## Quick Start

### Basic Imports

```python
import fenic as fc
from fenic import DataFrame
```

### Creating Your First DataFrame

```python
# Create a DataFrame from a dictionary
data = {
    "text": ["This is great!", "I hate this product", "It's okay, could be better"],
    "product": ["Widget A", "Widget B", "Widget C"]
}

df = fc.DataFrame(data)
print(df)
```

### Reading Data

```python
# Read from CSV
df = fc.read_csv("data.csv")

# Read from JSON
df = fc.read_json("data.json")

# Read from Parquet
df = fc.read_parquet("data.parquet")
```

## Core Operations

### Traditional DataFrame Operations

Fenic supports all the familiar DataFrame operations you know from pandas and PySpark:

```python
# Filtering
filtered_df = df.filter(fc.col("product").contains("Widget"))

# Selecting columns
selected_df = df.select("text", "product")

# Adding new columns
df_with_length = df.with_column("text_length", fc.length(fc.col("text")))

# Grouping and aggregation
grouped = df.group_by("product").agg(fc.count("*").alias("count"))
```

### Semantic Operations

This is where Fenic truly shines - adding AI-powered operations to your data pipeline:

#### Sentiment Analysis

```python
# Analyze sentiment of text data
df_with_sentiment = df.with_column(
    "sentiment",
    fc.semantic.analyze_sentiment(fc.col("text"))
)
```

#### Text Classification

```python
# Classify text into categories
df_classified = df.with_column(
    "category",
    fc.semantic.classify(
        fc.col("text"),
        categories=["positive", "negative", "neutral"]
    )
)
```

#### Semantic Extraction

```python
# Extract structured information from unstructured text
from pydantic import BaseModel

class ProductReview(BaseModel):
    rating: int
    summary: str
    recommendation: bool

df_extracted = df.with_column(
    "review_data",
    fc.semantic.extract(fc.col("text"), ProductReview)
)
```

#### Semantic Filtering

```python
# Filter rows using natural language predicates
positive_reviews = df.filter(
    fc.semantic.predicate("Is this review positive? {text}")
)
```

#### Semantic Joins

```python
# Join DataFrames based on semantic similarity, not exact matches
products_df = fc.DataFrame({
    "name": ["Super Widget", "Amazing Tool", "Great Gadget"],
    "description": ["A fantastic widget for all your needs", "The best tool ever made", "Perfect gadget for everyday use"]
})

reviews_df = fc.DataFrame({
    "review": ["Love my new widget!", "This tool is incredible", "Great little device"],
    "rating": [5, 5, 4]
})

# Join based on semantic similarity between product names and review content
joined = reviews_df.semantic.join(
    products_df,
    "Does this review match this product? Review: {review:left} Product: {name:right}"
)
```

## Working with AI-Native Data Types

### Markdown Processing

```python
# Process markdown content
markdown_df = fc.DataFrame({
    "content": ["# Title\n## Section\nContent here", "# Another Title\nMore content"]
})

processed = (markdown_df
    .with_column("parsed_md", fc.col("content").cast(fc.MarkdownType))
    .with_column("chunks", fc.markdown.extract_header_chunks("parsed_md"))
    .explode("chunks")
)
```

### JSON Processing

```python
# Work with JSON data using JQ expressions
json_df = fc.DataFrame({
    "data": ['{"name": "John", "age": 30}', '{"name": "Jane", "age": 25}']
})

extracted = json_df.with_column(
    "name",
    fc.json.jq("data", ".name")
)
```

### Transcript Processing

```python
# Process transcript data with speaker awareness
transcript_df = fc.DataFrame({
    "transcript": ["[Speaker A] Hello there", "[Speaker B] How are you?"]
})

processed_transcript = transcript_df.with_column(
    "speaker",
    fc.transcript.extract_speaker("transcript")
)
```

## Advanced Examples

### Content Analysis Pipeline

```python
# Complete content analysis pipeline
content_df = (fc.read_csv("articles.csv")
    .with_column("word_count", fc.length(fc.split(fc.col("content"))))
    .filter(fc.col("word_count") > 100)
    .with_column("sentiment", fc.semantic.analyze_sentiment(fc.col("content")))
    .with_column("topics", fc.semantic.extract(fc.col("content"), ["technology", "business", "health"]))
    .with_column("summary", fc.semantic.map(fc.col("content"), "Summarize this article in one sentence"))
)
```

### Multi-Modal Data Processing

```python
# Process different data types in a unified pipeline
mixed_data = fc.DataFrame({
    "text": ["Product review text...", "Another review..."],
    "metadata": ['{"category": "electronics"}', '{"category": "books"}'],
    "markdown_content": ["# Review\nGreat product!", "# Analysis\nNot bad."]
})

processed = (mixed_data
    .with_column("category", fc.json.jq("metadata", ".category"))
    .with_column("md_parsed", fc.col("markdown_content").cast(fc.MarkdownType))
    .with_column("sentiment", fc.semantic.analyze_sentiment(fc.col("text")))
    .group_by("category")
    .agg(
        fc.avg("sentiment").alias("avg_sentiment"),
        fc.count("*").alias("review_count")
    )
)
```

## Best Practices

### Performance Optimization

1. **Batch Operations**: Fenic automatically batches LLM calls for efficiency
2. **Lazy Evaluation**: Operations are optimized and executed only when needed
3. **Caching**: Results are cached to avoid repeated API calls for the same inputs

### Error Handling

```python
# Fenic provides built-in error handling and retry logic
try:
    result = df.semantic.extract(fc.col("text"), schema)
except fc.FenicError as e:
    print(f"Processing error: {e}")
    # Handle gracefully
```

### Cost Management

```python
# Monitor token usage and costs
session = fc.get_session()
metrics = session.get_metrics()
print(f"Tokens used: {metrics.token_count}")
print(f"Estimated cost: {metrics.estimated_cost}")
```

## Use Cases and Examples

The fastest way to learn about fenic is by checking the examples. Here are some common use cases:

### 1. Log Enrichment

Multi-stage DataFrames with template-based text extraction, joins, and LLM-powered transformations for analyzing error logs.

### 2. Meeting Transcript Processing

Native transcript parsing, Pydantic schema integration, and complex aggregations for meeting analysis.

### 3. News Analysis

Analyze and extract insights from news articles using semantic operators and structured data processing.

### 4. Document Extraction

Extract structured information from various document formats using semantic operators.

### 5. Feedback Clustering

Group and analyze feedback using semantic similarity and clustering operations.

## Configuration and Session Management

```python
# Configure Fenic session
config = fc.SessionConfig(
    default_model="gpt-4",
    max_retries=3,
    timeout=30
)

session = fc.FenicSession(config)
fc.set_session(session)
```

## Next Steps

1. **Explore Examples**: Check out the [GitHub repository](https://github.com/typedef-ai/fenic) for detailed examples
2. **Join the Community**: Connect with other users on [Discord](https://discord.gg/aAvsqRW3)
3. **Read the Documentation**: Visit [docs.fenic.ai](https://docs.fenic.ai/) for comprehensive API documentation
4. **Experiment**: Start with simple semantic operations and gradually build more complex pipelines

## Why Choose Fenic?

AI and agentic applications are fundamentally pipelines and workflows - exactly what DataFrame APIs were designed to handle. Fenic provides:

- **Familiar API**: If you know pandas or PySpark, you already know Fenic
- **Production Ready**: Built-in reliability features for production AI workloads
- **Composable**: Chain operations naturally with fluent API design
- **Separation of Concerns**: Clear separation between heavy inference tasks and real-time agent interactions

Start building your AI-powered data pipelines with Fenic today!
