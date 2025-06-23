"""Summary structures for different formats of summaries."""

from dataclasses import dataclass


class SummaryFormat:
    """Base class for summary formats."""
    pass

@dataclass
class KeyPoints(SummaryFormat):
    """Summary as a concise bulleted list."""

    max_points: int = 5

    def __str__(self):
        """Return a description of the summary format for KeyPoints."""
        return "The summary should be presented as a concise bulleted list, with each bullet capturing a distinct and essential idea, and the total number of points not exceeding " + str(self.max_points) + " points."

    def max_tokens(self) -> int:
        """Calculate the maximum number of tokens for the summary based on the number of key points."""
        return self.max_points * 75

@dataclass
class Paragraph(SummaryFormat):
    """Summary as a cohesive narrative."""
    max_words: int = 120

    def __str__(self):
        """Return a description of the summary format for Paragraph."""
        return "The summary should be written as a cohesive narrative that flows naturally and does not exceed the max_words limit of " + str(self.max_words)

    def max_tokens(self) -> int:
        """Calculate the maximum number of tokens for the summary based on the number of words."""
        return int(self.max_words * 1.5)