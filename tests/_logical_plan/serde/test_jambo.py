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
    original_model_instance = DocumentMetadata(title="test", document_type="research paper", date="2021-01-01", keywords=["test"], summary="test")
    reconstituted_model_instance = reconstituted_model(title="test", document_type="research paper", date="2021-01-01", keywords=["test"], summary="test")
    for field_name, field_info in DocumentMetadata.model_fields.items():
        if field_name != "document_type": # document_type is a literal, so it will be an enum in the serialized form.
            assert getattr(reconstituted_model_instance, field_name) == getattr(original_model_instance, field_name)
        assert field_info.description == reconstituted_model.model_fields[field_name].description
    reconstituted_model_instance_json = reconstituted_model_instance.model_dump_json()
    original_model_instance_json = original_model_instance.model_dump_json()
    assert reconstituted_model_instance_json == original_model_instance_json
    reconstituted_based_on_original_json = reconstituted_model.model_validate_json(original_model_instance_json)
    assert reconstituted_based_on_original_json
