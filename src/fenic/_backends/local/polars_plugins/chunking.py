from pathlib import Path
from typing import List, Optional

import polars as pl
from polars._typing import IntoExpr
from polars.plugins import register_plugin_function

from fenic.core._logical_plan.expressions import ChunkCharacterSet, ChunkLengthFunction

PLUGIN_PATH = Path(__file__).parents[3]



def chunk_text(
    expr: IntoExpr,
    desired_chunk_size: int,
    chunk_overlap_percentage: int,
    chunk_length_function_name: ChunkLengthFunction,
    chunking_character_set_name: ChunkCharacterSet,
    chunking_character_set_custom_characters: Optional[List[str]] = None,
) -> pl.Expr:
    chunk_overlap = round(desired_chunk_size * (chunk_overlap_percentage / 100.0))
    chunk_kwargs = {
        "desired_chunk_size": desired_chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_length_function_name": chunk_length_function_name.value,
        "chunking_character_set_name": chunking_character_set_name.value,
        "chunking_character_set_custom_characters": chunking_character_set_custom_characters,
    }
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="text_chunk_expr",
        args=expr,
        kwargs=chunk_kwargs,
        is_elementwise=True,
    )


@pl.api.register_expr_namespace("chunking")
class TextChunker:
    def __init__(self, expr: pl.Expr) -> None:
        """Initialize a TextChunker with a Polars expression.

        Args:
            expr: A Polars expression containing text to be chunked.
        """
        self.expr = expr

    def recursive(
        self,
        desired_chunk_size: int,
        chunk_overlap_percentage: int,
        chunk_length_function_name: ChunkLengthFunction = ChunkLengthFunction.WORD,
        chunking_character_set_name: ChunkCharacterSet = ChunkCharacterSet.ASCII,
        chunking_character_set_custom_characters: Optional[List[str]] = None,
    ) -> pl.Expr:
        """Split text into chunks recursively based on specified parameters.

        Args:
            desired_chunk_size: The target length for each chunk.
            chunk_overlap_percentage: Percentage of overlap between consecutive chunks.
            chunk_length_function_name: Method to measure chunk length ("CHARACTER", "WORD", or "TOKEN").
            chunking_character_set_name: Character set to use for chunking ("CUSTOM", "ASCII", or "UNICODE").
            chunking_character_set_custom_characters: Custom characters to use when chunking_character_set_name is "CUSTOM".

        Returns:
            A Polars expression that chunks the text according to the specified parameters.
        """
        return chunk_text(
            self.expr,
            desired_chunk_size,
            chunk_overlap_percentage,
            chunk_length_function_name,
            chunking_character_set_name,
            chunking_character_set_custom_characters,
        )
