"""Schema definitions for MCP documentation tables."""

import fenic as fc


def get_learnings_schema(include_embeddings: bool = True) -> fc.Schema:
    """Get the schema for the learnings table.
    
    Args:
        include_embeddings: Whether to include embedding columns
        
    Returns:
        Schema for the learnings table
    """
    schema_fields = [
        fc.ColumnField('id', fc.StringType),
        fc.ColumnField('question', fc.StringType),
        fc.ColumnField('answer', fc.StringType),
        fc.ColumnField('learning_type', fc.StringType),
        fc.ColumnField('keywords', fc.ArrayType(fc.StringType)),
        fc.ColumnField('related_functions', fc.ArrayType(fc.StringType)),
        fc.ColumnField('created_at', fc.StringType)
    ]
    
    if include_embeddings:
        # Using text-embedding-3-large with 3072 dimensions
        embedding_type = fc.EmbeddingType(
            dimensions=3072, 
            embedding_model="openai/text-embedding-3-large"
        )
        schema_fields.extend([
            fc.ColumnField('question_embedding', embedding_type),
            fc.ColumnField('answer_embedding', embedding_type),
            fc.ColumnField('combined_embedding', embedding_type)
        ])
    
    return fc.Schema(schema_fields)