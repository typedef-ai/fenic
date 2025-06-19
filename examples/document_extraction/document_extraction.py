"""Document metadata extraction example using fenic semantic operations.

This example demonstrates how to extract structured metadata from unstructured
document text using both ExtractSchema and Pydantic model approaches.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

import fenic as fc


def main(config: Optional[fc.SessionConfig] = None):
    """Extract metadata from document excerpts using semantic operations."""
    # Configure session with semantic capabilities
    config = config or fc.SessionConfig(
        app_name="document_extraction",
        semantic=fc.SemanticConfig(
            language_models={
                "mini": fc.OpenAIModelConfig(
                    model_name="gpt-4o-mini",
                    rpm=500,
                    tpm=200_000,
                )
            }
        ),
    )

    # Create session
    session = fc.Session.get_or_create(config)

    print("Document Metadata Extraction Example")
    print("=" * 50)
    print("Comparing ExtractSchema vs Pydantic model approaches")
    print()

    # Sample document data - diverse text types for metadata extraction
    documents_data = [
        {
            "id": "doc_001",
            "text": "Neural Networks for Climate Prediction: A Comprehensive Study. Published March 15, 2024. This research presents a novel deep learning approach for predicting climate patterns using multi-layered neural networks. Our methodology combines satellite imagery data with ground-based sensor readings to achieve 94% accuracy in temperature forecasting. The study was conducted over 18 months across 12 research stations. Keywords: machine learning, climate modeling, neural networks, environmental science."
        },
        {
            "id": "doc_002",
            "text": "Introducing CloudSync Pro - Next-Generation File Synchronization. Release Date: January 8, 2024. CloudSync Pro revolutionizes how teams collaborate with real-time file synchronization across unlimited devices. Features include end-to-end encryption, automatic conflict resolution, and integration with over 50 productivity tools. Pricing starts at $12/month per user with enterprise discounts available. Contact our sales team for a personalized demo."
        },
        {
            "id": "doc_003",
            "text": "Weekly Engineering Standup - December 4, 2023. Attendees: Sarah Chen (Lead), Marcus Rodriguez (Backend), Lisa Park (Frontend), James Wilson (DevOps). Key decisions: Migration to Kubernetes approved for Q1 2024, new CI/CD pipeline reduces deployment time by 60%, API rate limiting implementation scheduled for next sprint. Action items: Sarah to finalize container specifications, Marcus to document database migration plan."
        },
        {
            "id": "doc_004",
            "text": "Breaking: Major Data Breach Affects 2.3 Million Users. December 12, 2023 - TechCorp announced today that unauthorized access to customer databases occurred between November 28-30, 2023. Compromised data includes email addresses, encrypted passwords, and partial payment information. The company has implemented additional security measures and is offering free credit monitoring to affected users. Stock prices dropped 8% in after-hours trading."
        },
        {
            "id": "doc_005",
            "text": "API Reference: Authentication Service v2.1. Last updated: February 20, 2024. The Authentication Service provides secure user login and session management for distributed applications. Supports OAuth 2.0, SAML, and multi-factor authentication. Rate limits: 1000 requests per hour for standard accounts, 10000 for premium. Available endpoints include /auth/login, /auth/refresh, /auth/logout. Response format: JSON with standardized error codes."
        }
    ]

    # Create DataFrame
    docs_df = session.create_dataframe(documents_data)

    print(f"Loaded {docs_df.count()} sample documents:")
    docs_df.select("id", fc.text.length("text").alias("text_length")).show()
    print()

    # Method 1: Using ExtractSchema
    print("Method 1: ExtractSchema Approach")
    print("-" * 40)

    # Define schema for document metadata extraction
    doc_metadata_schema = fc.ExtractSchema([
        fc.ExtractSchemaField(
            name="title",
            data_type=fc.StringType,
            description="The main title or subject of the document"
        ),
        fc.ExtractSchemaField(
            name="document_type",
            data_type=fc.StringType,
            description="Type of document (e.g., research paper, product announcement, meeting notes, news article, technical documentation)"
        ),
        fc.ExtractSchemaField(
            name="date",
            data_type=fc.StringType,
            description="Any date mentioned in the document (publication date, meeting date, etc.)"
        ),
        fc.ExtractSchemaField(
            name="keywords",
            data_type=fc.ExtractSchemaList(element_type=fc.StringType),
            description="List of key topics, technologies, or important terms mentioned in the document"
        ),
        fc.ExtractSchemaField(
            name="summary",
            data_type=fc.StringType,
            description="Brief one-sentence summary of the document's main purpose or content"
        )
    ])

    # Apply extraction using ExtractSchema
    extracted_df = docs_df.select(
        "id",
        fc.semantic.extract("text", doc_metadata_schema).alias("metadata")
    )

    # Flatten the extracted metadata into separate columns
    extract_schema_results = extracted_df.select(
        "id",
        extracted_df.metadata.title.alias("title"),
        extracted_df.metadata.document_type.alias("document_type"),
        extracted_df.metadata.date.alias("date"),
        extracted_df.metadata.keywords.alias("keywords"),
        extracted_df.metadata.summary.alias("summary")
    )

    print("ExtractSchema Results:")
    extract_schema_results.show()
    print()

    # Method 2: Using Pydantic Model
    print("Method 2: Pydantic Model Approach")
    print("-" * 40)

    # Define Pydantic model for document metadata
    # Note: Pydantic models for extraction support simple data types (str, int, float, bool, Literal)
    # Complex types like lists must be represented as strings (e.g., comma-separated values)
    class DocumentMetadata(BaseModel):
        """Pydantic model for document metadata extraction."""
        title: str = Field(..., description="The main title or subject of the document")
        document_type: Literal["research paper", "product announcement", "meeting notes", "news article", "technical documentation", "other"] = Field(..., description="Type of document")
        date: str = Field(..., description="Any date mentioned in the document (publication date, meeting date, etc.)")
        keywords: str = Field(..., description="Comma-separated list of key topics, technologies, or important terms mentioned in the document")
        summary: str = Field(..., description="Brief one-sentence summary of the document's main purpose or content")

    # Apply extraction using Pydantic model
    pydantic_extracted_df = docs_df.select(
        "id",
        fc.semantic.extract("text", DocumentMetadata).alias("metadata")
    )

    # Flatten the extracted metadata into separate columns
    pydantic_results = pydantic_extracted_df.select(
        "id",
        pydantic_extracted_df.metadata.title.alias("title"),
        pydantic_extracted_df.metadata.document_type.alias("document_type"),
        pydantic_extracted_df.metadata.date.alias("date"),
        pydantic_extracted_df.metadata.keywords.alias("keywords"),
        pydantic_extracted_df.metadata.summary.alias("summary")
    )

    print("Pydantic Model Results:")
    pydantic_results.show()
    print()

    # Method Comparison
    print("Key Differences Between Approaches:")
    print("-" * 50)
    print("ExtractSchema:")
    print("  ✓ Supports complex data types (lists, nested structures)")
    print("  ✓ Native Fenic schema definition")
    print("  ✓ Type-safe with proper list handling")
    print()
    print("Pydantic Model:")
    print("  ✓ Familiar Python class syntax")
    print("  ✓ Leverages existing Pydantic knowledge")
    print("  ✓ Supports Literal types for constraining string values")
    print("  ✗ Limited to simple data types (str, int, float, bool, Literal)")
    print("  → Lists must be comma-separated strings")
    print()

    # Show the difference in keywords output
    print("Notice the keywords field difference:")
    print("ExtractSchema: Returns actual list → ['item1', 'item2', ...]")
    print("Pydantic Model: Returns comma-separated string → 'item1, item2, ...'")
    print()

    # Clean up
    session.stop()
    print("Session complete!")


if __name__ == "__main__":
    main()
