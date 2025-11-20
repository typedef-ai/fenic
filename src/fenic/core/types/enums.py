
"""Enums used in the DataFrame API."""
from __future__ import annotations

from enum import Enum
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

TranscriptFormatType = Literal["srt", "generic", "webvtt"]
"""
Type alias representing supported transcript formats.

Valid values:

- "srt": SubRip Subtitle format with indexed entries and timestamp ranges
- "generic": Conversation transcript format with speaker names and timestamps
- "webvtt": Web Video Text Tracks format with speaker names and timestamps

All formats are parsed into a unified schema with fields: index, speaker,
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

FuzzySimilarityMethod = Literal["indel", "levenshtein", "damerau_levenshtein", "jaro_winkler", "jaro", "hamming"]
"""
Type alias representing the supported fuzzy string similarity algorithms.

These algorithms quantify the similarity or difference between two strings using various distance or similarity metrics:

- "indel":
  Computes the Indel (Insertion-Deletion) distance, which counts only insertions and deletions needed to transform one string into another, excluding substitutions. This is equivalent to the Longest Common Subsequence (LCS) problem. Useful when character substitutions should not be considered as valid operations (e.g., DNA sequence alignment where only insertions/deletions occur).
- "levenshtein":
  Computes the Levenshtein distance, which is the minimum number of single-character edits (insertions, deletions, or substitutions) required to transform one string into another. Suitable for general-purpose fuzzy matching where transpositions do not matter.
- "damerau_levenshtein":
  An extension of Levenshtein distance that also accounts for transpositions of adjacent characters (e.g., "ab" → "ba"). This metric is more accurate for real-world typos and keyboard errors.
- "jaro":
  Measures similarity based on the number and order of common characters between two strings. It is particularly effective for short strings such as names. Returns a normalized score between 0 (no similarity) and 1 (exact match).
- "jaro_winkler":
  A variant of the Jaro distance that gives more weight to common prefixes. Designed to improve accuracy on strings with shared beginnings (e.g., first names, surnames).
- "hamming":
  Measures the number of differing characters between two strings of equal length. Only valid when both strings are the same length. It does not support insertions or deletions—only substitutions.

Choose the method based on the type of expected variation (e.g., typos, transpositions, or structural changes).
"""

StringCasingType = Literal["lower", "upper", "title"]
"""
Type alias representing the type of string casing.

Valid values:

- "lower": Convert to lowercase.
- "upper": Convert to uppercase.
- "title": Convert to title case.
"""

StripCharsSide = Literal["left", "right", "both"]
"""
Type alias representing the side of a string to strip characters from.

Valid values:

- "left": Strip characters from the left side.
- "right": Strip characters from the right side.
- "both": Strip characters from both sides.
"""

DocContentType = Literal["markdown", "json", "pdf"]
"""
Type alias representing the type of document content.

Valid values:

- "markdown": Markdown format.
- "json": JSON format.
- "pdf": PDF format.
"""

DateTimeUnit = Literal["year", "month", "day", "hour", "minute", "second", "millisecond"]
"""
Type alias representing the supported date part or time unit to be used in date functions.

Valid values:

- "year": Year.
- "month": Month.
- "day": Day.
- "hour": Hour.
- "minute": Minute.
- "second": Second.
- "millisecond": Millisecond.
"""


class CacheBackend(str, Enum):
    """Cache backend implementations.

    Attributes:
        LOCAL: Local persistent cache backend.
    """

    LOCAL = "local"
