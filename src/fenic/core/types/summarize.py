"""Summary structures for different formats of summaries."""

from pydantic import BaseModel, Field


class KeyPoints(BaseModel):
    """Summary as a concise bulleted list.
    
    Each bullet should capture a distinct and essential idea, with a maximum number of points specified.

    Attributes:
        max_points: The maximum number of key points to include in the summary.
    """

    max_points: int = Field(default=5, gt=0)

    def __str__(self):
        """Return a description of the summary format for KeyPoints."""
        return (
            f"The summary should be presented as a concise bulleted list, "
            f"with each bullet capturing a distinct and essential idea, and the total "
            f"number of points not exceeding {self.max_points} points."
        )

    def max_tokens(self) -> int:
        """Calculate the maximum number of tokens for the summary based on the number of key points."""
        return self.max_points * 75

class Paragraph(BaseModel):
    """Summary as a cohesive narrative.
    
    The summary should flow naturally and not exceed a specified maximum word count.

    Attributes:
        max_words: The maximum number of words allowed in the summary.
    """

    max_words: int = Field(default=120, gt=10)

    def __str__(self):
        """Return a description of the summary format for Paragraph."""
        return (
            f"The summary should be written as a cohesive narrative that flows naturally and does not exceed the max_words limit of {self.max_words}."
        )

    def max_tokens(self) -> int:
        """Calculate the maximum number of tokens for the summary based on the number of words."""
        return int(self.max_words * 1.5)