# Document Metadata Extraction with Fenic

[View in Github](https://github.com/typedef-ai/fenic/blob/main/examples/document_extraction/README.md)

This example demonstrates how to extract structured metadata from unstructured text data, using Fenic's semantic extraction capabilities. It compares two different approaches for defining extraction schemas.

## Overview

Document metadata extraction is a common use case for LLMs, allowing you to automatically parse and structure information from various document types including research papers, product announcements, meeting notes, news articles, and technical documentation.

This example showcases:

- **Two Schema Approaches**: ExtractSchema vs Pydantic models
- **Structured Extraction**: Converting unstructured text to structured metadata
- **Zero-Shot Extraction**: No examples required

## Schema Approaches Compared

### ExtractSchema Approach

```python
doc_metadata_schema = fc.ExtractSchema([
    fc.ExtractSchemaField(
        name="keywords",
        data_type=fc.ExtractSchemaList(element_type=fc.StringType),
        description="List of key topics and terms"
    )
])
```

**Advantages:**

- ✅ Supports complex data types (lists, nested structures)
- ✅ Native Fenic schema definition
- ✅ Type-safe with proper list handling

### Pydantic Model Approach

```python
from typing import Literal

class DocumentMetadata(BaseModel):
    keywords: str = Field(..., description="Comma-separated list of key topics and terms")
    document_type: Literal["research paper", "product announcement", "meeting notes", "news article", "technical documentation"] = Field(..., description="Type of document")
```

**Advantages:**

- ✅ Familiar Python class syntax
- ✅ Leverages existing Pydantic knowledge
- ✅ Supports Literal strings for constraining output values
- ❌ Limited to simple data types (str, int, float, bool, Literal)

## Sample Data

The example processes 5 diverse document types:

1. **Research Paper** - Academic abstract with technical terms
2. **Product Announcement** - Marketing content with features and pricing
3. **Meeting Notes** - Internal documentation with decisions and action items
4. **News Article** - Breaking news with facts and impact
5. **Technical Documentation** - API reference with specifications

## Extracted Metadata Fields

- **Title**: Main subject or heading of the document
- **Document Type**: Classification (research paper, product announcement, etc.)
- **Date**: Any relevant date mentioned (publication, meeting, etc.)
- **Keywords**: Key topics and terms (list vs comma-separated string)
- **Summary**: One-sentence overview of the document's purpose

## Key Differences in Output

**ExtractSchema Keywords:**

```json
["machine learning", "climate modeling", "neural networks"]
```

**Pydantic Keywords:**

```bash
"machine learning, climate modeling, neural networks"
```

## Usage

```bash
# Ensure you have OpenAI API key configured
export OPENAI_API_KEY="your-api-key"

# Run the extraction example
python document_extraction.py
```

## Expected Output

The script will display:

1. **Sample Documents** - Overview of loaded documents with text lengths
2. **ExtractSchema Results** - Structured extraction using native Fenic schema
3. **Pydantic Results** - Same extraction using Pydantic model
4. **Method Comparison** - Side-by-side analysis of both approaches

## When to Use Each Approach

**Choose ExtractSchema when:**

- You need complex data types (lists, nested objects)
- Working primarily within Fenic ecosystem
- Type safety is important for downstream processing

**Choose Pydantic when:**

- You prefer familiar Python class syntax
- Your team has existing Pydantic expertise
- Simple data types meet your requirements

## Learning Outcomes

This example teaches:

- How to perform zero-shot semantic extraction
- Differences between schema definition approaches

Perfect for understanding Fenic's semantic extraction capabilities and choosing the right schema approach for your use case.
