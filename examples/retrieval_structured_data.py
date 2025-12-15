"""Retrieval — From unstructured to structured data.

This example demonstrates how to turn unstructured sources into typed rows
(Q&A, policies, products), pre-embed, and retrieve with citations.

Tools exposed: `qa_neighbors(query, k)` with citations, plus system tools

Note: This example requires PDF files in a 'sample_pdfs/' directory.
"""

from pydantic import BaseModel, Field

import fenic as fc
from fenic import SemanticConfig, OpenRouterLanguageModel, OpenAIEmbeddingModel
from fenic.api.mcp._tool_generation_utils import auto_generate_system_tools_from_tables
from fenic.core.mcp._server import FenicMCPServer
from fenic.core.mcp.types import SystemTool


class QAPair(BaseModel):
    question: str = Field(description="A question extracted from the document")
    answer: str = Field(description="The answer to the question")


def main():
    session = fc.Session.get_or_create(
        fc.SessionConfig(
            app_name="policy_qa",
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

    qa_pairs = (
        session.read.pdf_metadata("sample_pdfs/**/*.pdf")
        .select(
            fc.col("file_path").alias("source"),
            fc.semantic.parse_pdf(fc.col("file_path")).alias("content"),
        )
        .select(
            fc.col("source"),
            fc.semantic.extract(fc.col("content").cast(fc.StringType), QAPair).alias("qa"),
        )
        .unnest("qa")
        .select(
            "source",
            "question",
            "answer",
            fc.semantic.embed(fc.col("question")).alias("embedding"),
        )
    )
    qa_pairs.write.save_as_table("policy_qa", mode="overwrite")
    session.catalog.set_table_description(
        "policy_qa", "Q&A pairs extracted from policy documents with embeddings"
    )

    async def qa_neighbors(query: str, k: int = 3):
        q = session.create_dataframe([{"q": query}])
        res = q.semantic.sim_join(
            session.table("policy_qa"),
            left_on=fc.semantic.embed(fc.col("q")),
            right_on=fc.col("embedding"),
            k=k,
            similarity_score_column="relevance",
        ).select("question", "answer", "source", "relevance")
        return res._plan

    generated_system_tools = auto_generate_system_tools_from_tables(
        ["policy_qa"], session, tool_namespace="qa", max_result_limit=50
    )

    server = FenicMCPServer(
        session._session_state,
        user_defined_tools=[],
        system_tools=[
            SystemTool(
                name="qa_neighbors",
                description="Semantic Q/A retrieval",
                max_result_limit=50,
                func=qa_neighbors,
            ),
            *generated_system_tools,
        ],
        server_name="Policy QA",
    )

    print("Retrieval Structured Data example completed.")
    print("Table created: policy_qa")
    print("MCP server configured with qa_neighbors tool and system tools.")
    return server


if __name__ == "__main__":
    main()
