from typing import List, Literal

from pydantic import BaseModel, Field

from fenic.core._logical_plan.resolved_types import ResolvedResponseFormat
from fenic.core._serde.proto.serde_context import SerdeContext


def test_resolved_response_format_serde():

    class Participant(BaseModel):
        name: str = Field(..., description="Name of participant")
        age: int = Field(..., description="Age of participant")
        is_active: bool = Field(..., description="Is participant active?")
        tags: List[str] = Field(..., description="Tags associated with participant")
        address: str = Field(..., description="Address associated with participant")

    class DocumentMetadata(BaseModel):
        """Pydantic model for document metadata extraction."""
        title: str = Field(description="The main title or subject of the document")
        document_type: Literal["research paper", "product announcement", "meeting notes", "news article", "technical documentation", "other"] = Field(description="Type of document")
        date: str = Field(description="Any date mentioned in the document (publication date, meeting date, etc.)")
        keywords: List[str] = Field(description="List of key topics, technologies, or important terms mentioned in the document")
        summary: str = Field(description="Brief one-sentence summary of the document's main purpose or content")
        participants: List[Participant] = Field(description="List of participants in the document")

    resolved_response_format = ResolvedResponseFormat.from_pydantic_model(DocumentMetadata)
    assert resolved_response_format.schema_fingerprint == ResolvedResponseFormat.from_pydantic_model(DocumentMetadata).schema_fingerprint
    serde_context = SerdeContext()
    serialized_resolved_response_format = serde_context.serialize_resolved_response_format("response_format", resolved_response_format)
    deserialized_resolved_response_format = serde_context.deserialize_resolved_response_format("response_format", serialized_resolved_response_format)
    assert deserialized_resolved_response_format == resolved_response_format
