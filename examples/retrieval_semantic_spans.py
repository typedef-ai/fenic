"""Retrieval — Semantic Spans.

This example demonstrates how to break long documents into overlapping spans,
embed once, and serve semantic top-K retrieval.

Tools exposed: `docs_neighbors(query, k)`, plus system tools

Note: This example requires PDF files in a 'sample_pdfs/' directory.
"""

import fenic as fc
from fenic import SemanticConfig, OpenRouterLanguageModel, OpenAIEmbeddingModel
from fenic.api.mcp._tool_generation_utils import auto_generate_system_tools_from_tables
from fenic.core.mcp._server import FenicMCPServer
from fenic.core.mcp.types import SystemTool

def main():
    session = fc.Session.get_or_create(
        fc.SessionConfig(
            app_name="docs",
            semantic=SemanticConfig(
                language_models={
                    "gpt4": OpenRouterLanguageModel(
                        model_name="openai/gpt-4.1-nano",
                        profiles={
                            "default": OpenRouterLanguageModel.Profile(
                                parsing_engine="pdf-text",
                            ),
                        },
                        default_profile="default",
                    ),
                },
                embedding_models={
                    "embed": OpenAIEmbeddingModel(
                        model_name="text-embedding-3-small", rpm=100, tpm=100_000
                    )
                },
                default_embedding_model="embed",
            ),
        )
    )

    exploded = (
        session.read.pdf_metadata("sample_pdfs/**/*.pdf")
        .select(
            fc.col("file_path").alias("source"),
            fc.semantic.parse_pdf(fc.col("file_path")).alias("content"),
        )
        .select(
            fc.col("source"),
            fc.text.recursive_word_chunk(
                fc.col("content").cast(fc.StringType),
                chunk_size=500,
                chunk_overlap_percentage=10,
            ).alias("chunks"),
        )
        .explode("chunks")
    )

    # Use SQL to generate unique chunk IDs with ROW_NUMBER()
    chunks = session.sql(
        "SELECT ROW_NUMBER() OVER () as chunk_id, * FROM {df}", df=exploded
    ).select(
        "chunk_id",
        "source",
        fc.col("chunks").alias("text"),
        fc.semantic.embed(fc.col("chunks")).alias("embedding"),
    )
    chunks.write.save_as_table("chunks", mode="overwrite")
    session.catalog.set_table_description(
        "chunks", "Document chunks with embeddings for semantic retrieval"
    )

    async def docs_neighbors(query: str, k: int = 3):
        q = session.create_dataframe([{"q": query}])
        res = q.semantic.sim_join(
            session.table("chunks"),
            left_on=fc.semantic.embed(fc.col("q")),
            right_on=fc.col("embedding"),
            k=k,
            similarity_score_column="relevance",
        ).select("chunk_id", "source", "text", "relevance")
        return res._plan

    generated_system_tools = auto_generate_system_tools_from_tables(
        ["chunks"], session, tool_namespace="docs", max_result_limit=100
    )

    server = FenicMCPServer(
        session._session_state,
        user_defined_tools=[],
        system_tools=[
            SystemTool(
                name="docs_neighbors",
                description="Semantic chunk retrieval",
                max_result_limit=100,
                func=docs_neighbors,
            ),
            *generated_system_tools,
        ],
        server_name="Docs",
    )

    print("Retrieval Semantic Spans example completed.")
    print("Table created: chunks")
    print("MCP server configured with docs_neighbors tool and system tools.")
    return server


if __name__ == "__main__":
    main()
