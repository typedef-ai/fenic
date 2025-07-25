from typing import List, Literal

from jambo import SchemaConverter
from pydantic import BaseModel, Field


def test_jambo():
    class DocumentMetadata(BaseModel):
        """Pydantic model for document metadata extraction."""
        title: str = Field(description="The main title or subject of the document")
        document_type: Literal["research paper", "product announcement", "meeting notes", "news article", "technical documentation", "other"] = Field(description="Type of document")
        date: str = Field(description="Any date mentioned in the document (publication date, meeting date, etc.)")
        keywords: List[str] = Field(description="List of key topics, technologies, or important terms mentioned in the document")
        summary: str = Field(description="Brief one-sentence summary of the document's main purpose or content")

    json_schema = DocumentMetadata.model_json_schema()

    reconstituted_model = SchemaConverter.build(json_schema)
    reconstituted_model_instance = reconstituted_model(title="test", document_type="research paper", date="2021-01-01", keywords=["test"], summary="test")
    assert hasattr(reconstituted_model_instance, "title")
    assert hasattr(reconstituted_model_instance, "document_type")
    assert hasattr(reconstituted_model_instance, "date")
    assert hasattr(reconstituted_model_instance, "keywords")
    assert hasattr(reconstituted_model_instance, "summary")
