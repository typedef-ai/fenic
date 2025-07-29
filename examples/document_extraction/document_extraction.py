"""Document metadata extraction example using fenic semantic operations.

This example demonstrates how to extract structured metadata from unstructured document text
using Fenic’s Pydantic model integration for schema definitions.
"""

from typing import List, Literal, Optional

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

    # Define Pydantic model for document metadata
    class DocumentMetadata(BaseModel):
        """Pydantic model for document metadata extraction."""
        title: str = Field(description="The main title or subject of the document")
        document_type: Literal["research paper", "product announcement", "meeting notes", "news article", "technical documentation", "other"] = Field(description="Type of document")
        date: str = Field(description="Any date mentioned in the document (publication date, meeting date, etc.)")
        keywords: List[str] = Field(description="List of key topics, technologies, or important terms mentioned in the document")
        summary: str = Field(description="Brief one-sentence summary of the document's main purpose or content")

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

    print("Extraction Results:")
    pydantic_results.show()
    print()

    # Clean up
    session.stop()
    print("Session complete!")


if __name__ == "__main__":
    main()
