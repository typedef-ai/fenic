# Document Metadata Extraction with Fenic

[View in Github](https://github.com/typedef-ai/fenic/blob/main/examples/document_extraction/README.md)

This example demonstrates how to extract structured metadata from unstructured text data, using Fenic's semantic extraction capabilities.

## Overview

Document metadata extraction is a common use case for LLMs, allowing you to automatically parse and structure information from various document types including research papers, product announcements, meeting notes, news articles, and technical documentation.

This example showcases:

- **Structured Extraction**: Converting unstructured text to structured metadata
- **Zero-Shot Extraction**: No examples required

## How it works

1. **Schema Definition with Pydantic**

   Define a Pydantic model to represent the structure of the data you want to extract. Each field must include a natural language description. This schema drives prompt generation and model output parsing.

2. **LLM Orchestration**

   Fenic uses the model provider of your choice to call the LLM with a structured output or tool-calling interface. The LLM returns data that conforms to the schema you defined.

3. **Data Structuring**

   The extracted data is represented as a struct column in a DataFrame with native Fenic struct fields. From there, it can be:

   - Unnested into individual columns
   - Exploded if it contains arrays
   - Processed in place as nested data

Because Fenic maps Pydantic models to a strongly typed, columnar data model, certain Python types are not currently supported:

- **Non-Optional Union types**: Not expressible in Fenic's type system
- **Dictionaries**: Fenic does not yet support map types (future support via a JsonType is planned)
- **Custom classes / dataclasses**: These are stateful or logic-heavy constructs that don't fit the declarative nature of Fenic's data model

Despite these constraints, you can define complex extraction schemas using nested Pydantic models, optional fields, and lists—enabling robust and expressive structured extraction pipelines.

### Defining a Pydantic Model

```python
from typing import Literal

class DocumentMetadata(BaseModel):
    keywords: str = Field(..., description="Comma-separated list of key topics and terms")
    document_type: Literal["research paper", "product announcement", "meeting notes", "news article", "technical documentation"] = Field(..., description="Type of document")
```

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
2. **Results** - Extraction results
