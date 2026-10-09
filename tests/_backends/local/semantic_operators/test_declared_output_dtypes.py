from unittest.mock import MagicMock

import polars as pl
import pytest

from fenic._backends.local.semantic_operators.cluster import Cluster
from fenic._backends.local.semantic_operators.parse_pdf import ParsePDF
from fenic._backends.local.semantic_operators.reduce import DATA_COLUMN_NAME, Reduce
from fenic._backends.local.semantic_operators.sim_join import (
    DISTANCE_COL_NAME,
    LEFT_ON_COL_NAME,
    RIGHT_ON_COL_NAME,
    SimJoin,
)
from fenic._inference.types import LMRequestFile
from fenic.core._inference.model_catalog import ModelProvider
from fenic.core._logical_plan.plans import CentroidInfo


@pytest.mark.parametrize("documents", [[], ["document"], ["first", "second"]])
@pytest.mark.parametrize("grouped", [False, True])
def test_reduce_empty_and_none_completions_are_strings(documents, grouped):
    group = pl.Series(
        [{DATA_COLUMN_NAME: value} for value in documents],
        dtype=pl.Struct({DATA_COLUMN_NAME: pl.String}),
    )
    input_series = (
        pl.Series([group, group], dtype=pl.List(group.dtype)) if grouped else group
    )
    model = MagicMock()
    model.count_tokens.return_value = 1
    model.max_context_window_length = 3000
    model.model_parameters.max_output_tokens = 100
    model.get_completions.side_effect = lambda messages, **kwargs: [
        MagicMock(completion=None) for _ in messages
    ]
    result = Reduce(
        input=input_series,
        user_instruction="Summarize",
        model=model,
        max_tokens=100,
        temperature=0,
        descending=[],
        nulls_last=[],
    ).execute()

    assert result.dtype == pl.String
    assert result.to_list() == [None] * (2 if grouped else 1)
    if not documents:
        model.get_completions.assert_not_called()


@pytest.mark.parametrize("height", [0, 2])
@pytest.mark.parametrize("with_centroids", [False, True])
def test_cluster_empty_and_none_embeddings_keep_declared_types(height, with_centroids):
    df = pl.DataFrame(
        {"embedding": [None] * height},
        schema={"embedding": pl.Array(pl.Float32, 2)},
    )
    result = Cluster(
        input=df,
        embedding_column_name="embedding",
        num_clusters=1,
        max_iter=10,
        num_init=1,
        label_column="cluster",
        centroid_info=CentroidInfo("centroid", 2) if with_centroids else None,
    ).execute()

    assert result["cluster"].dtype == pl.Int32
    assert result["cluster"].to_list() == [None] * height
    if with_centroids:
        assert result["centroid"].dtype == pl.Array(pl.Float32, 2)
        assert result["centroid"].to_list() == [None] * height


def test_cluster_valid_embeddings_preserve_int32_labels():
    df = pl.DataFrame(
        {"embedding": [[1.0, 0.0], [0.0, 1.0]]},
        schema={"embedding": pl.Array(pl.Float32, 2)},
    )
    result = Cluster(
        input=df,
        embedding_column_name="embedding",
        num_clusters=1,
        max_iter=10,
        num_init=1,
        label_column="cluster",
        centroid_info=None,
    ).execute()

    assert result["cluster"].dtype == pl.Int32
    assert result["cluster"].to_list() == [0, 0]


@pytest.mark.parametrize("paths", [[], [None, None], ["document.pdf"]])
def test_parse_pdf_empty_and_none_responses_are_strings(paths, monkeypatch, tmp_path):
    if paths == ["document.pdf"]:
        path = tmp_path / "document.pdf"
        path.touch()
        paths = [str(path)]
    model = MagicMock(provider=ModelProvider.OPENAI)
    operator = ParsePDF(input=pl.Series(paths, dtype=pl.String), model=model)
    monkeypatch.setattr(
        operator,
        "_get_file_chunks",
        lambda path: [LMRequestFile(path=path, page_range=(0, 0))],
    )
    operator.request_sender.send_requests = MagicMock(
        side_effect=lambda prompts: [None] * len(prompts)
    )

    result = operator.execute()

    assert result.dtype == pl.String
    assert result.to_list() == [None] * len(paths)


def test_parse_pdf_failed_chunk_nulls_only_its_document():
    operator = ParsePDF(
        input=pl.Series([], dtype=pl.String),
        model=MagicMock(provider=ModelProvider.OPENAI),
        page_separator="Page {page}",
    )
    assert operator.postprocess(["first", None, "last"], [[1, 1], [1]]) == [
        None,
        "last",
    ]
    assert operator.postprocess(["first", "second"], [[1, 1]]) == [
        "first\nPage 1\nsecond"
    ]


@pytest.mark.parametrize("left_height,right_height", [(0, 2), (2, 0), (0, 0), (2, 2)])
@pytest.mark.parametrize("include_embeddings", [False, True])
def test_sim_join_empty_and_none_embeddings_keep_declared_types(
    left_height, right_height, include_embeddings
):
    embedding_dtype = pl.Array(pl.Float32, 2)
    left = pl.DataFrame(
        {"left": ["left"] * left_height, LEFT_ON_COL_NAME: [None] * left_height},
        schema={"left": pl.String, LEFT_ON_COL_NAME: embedding_dtype},
    )
    right = pl.DataFrame(
        {"right": ["right"] * right_height, RIGHT_ON_COL_NAME: [None] * right_height},
        schema={"right": pl.String, RIGHT_ON_COL_NAME: embedding_dtype},
    )

    result = SimJoin(
        left=left,
        right=right,
        k=1,
        similarity_metric="cosine",
        include_left_on=include_embeddings,
        include_right_on=include_embeddings,
    ).execute()

    expected = {"left": pl.String}
    if include_embeddings:
        expected[LEFT_ON_COL_NAME] = embedding_dtype
    expected["right"] = pl.String
    if include_embeddings:
        expected[RIGHT_ON_COL_NAME] = embedding_dtype
    expected[DISTANCE_COL_NAME] = pl.Float64
    assert result.schema == expected
    assert result.is_empty()
