"""Retrieval — From unstructured to structured data.

This example demonstrates how to turn unstructured sources into typed rows
(Q&A, policies, products), pre-embed, and retrieve with citations.

Tools exposed: `qa_neighbors(query, k)` with citations, plus system tools

Note: This example requires PDF files in a 'policies/' directory.
"""

import fenic as fc
from fenic import SemanticConfig, OpenAILanguageModel, OpenAIEmbeddingModel
from pydantic import BaseModel
from fenic.core.mcp.types import SystemTool


class QAPair(BaseModel):
    question: str
    answer: str


def main():
    session = fc.Session.get_or_create(
        fc.SessionConfig(
            app_name="policy_qa",
            semantic=SemanticConfig(
                language_models={
                    "gpt4": OpenAILanguageModel(
                        model_name="gpt-4.1-nano", rpm=100, tpm=100_000
                    )
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
        session.read.pdf_metadata("policies/*.pdf")
        .select(
            fc.col("file_path").alias("source"),
            fc.semantic.parse_pdf(fc.col("file_path")).alias("content"),
        )
        .select(
            fc.col("source"),
            fc.semantic.extract(fc.col("content"), QAPair).alias("qa"),
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

    server = fc.create_mcp_server(
        session,
        "Policy QA",
        user_defined_tools=[
            SystemTool(
                name="qa_neighbors",
                description="Semantic Q/A retrieval",
                fn=qa_neighbors,
            )
        ],
        system_tools=fc.SystemToolConfig(
            table_names=["policy_qa"], tool_namespace="qa", max_result_rows=50
        ),
    )

    print("Retrieval Structured Data example completed.")
    print("Table created: policy_qa")
    print("MCP server configured with qa_neighbors tool and system tools.")
    return server


if __name__ == "__main__":
    main()
