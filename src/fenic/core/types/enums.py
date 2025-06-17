
"""Enums used in the DataFrame API."""

from typing import Literal

SemanticSimilarityMetric = Literal["cosine", "l2", "dot"]
"""
Type alias representing supported semantic similarity metrics.

Valid values:

- "cosine": Cosine similarity, measures the cosine of the angle between two vectors.
- "l2": Euclidean (L2) distance, measures the straight-line distance between two vectors.
- "dot": Dot product similarity, the raw inner product of two vectors.

These metrics are commonly used for comparing embedding vectors in semantic search
and other similarity-based applications.
"""


BranchSide = Literal["left", "right"]
"""
Type alias representing the side of a branch in a lineage graph.

Valid values:

- "left": The left branch of a join.
- "right": The right branch of a join.
"""

TranscriptFormatType = Literal["srt", "generic"]
"""
Type alias representing supported transcript formats.

Valid values:

- "srt": SubRip Subtitle format with indexed entries and timestamp ranges
- "generic": Conversation transcript format with speaker names and timestamps

Both formats are parsed into a unified schema with fields: index, speaker,
start_time, end_time, duration, content, format.
"""

JoinType = Literal["inner", "full", "left", "right", "cross"]
"""
Type alias representing supported join types.

Valid values:

- "inner": Inner join, returns only rows that have matching values in both tables.
- "outer": Outer join, returns all rows from both tables, filling missing values with nulls.
- "left": Left join, returns all rows from the left table and matching rows from the right table.
- "right": Right join, returns all rows from the right table and matching rows from the left table.
- "cross": Cross join, returns the Cartesian product of the two tables.
"""
