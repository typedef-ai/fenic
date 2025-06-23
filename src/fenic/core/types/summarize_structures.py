from dataclasses import dataclass


class SummaryFormat:
    pass

@dataclass
class KeyPoints(SummaryFormat):
    max_points: int = 5

    def __str__(self):
        return "The summary should be presented as a concise bulleted list, with each bullet capturing a distinct and essential idea, and the total number of points not exceeding " + str(self.max_points) + " points."

    def max_tokens(self) -> int:
        return self.max_points * 75

@dataclass
class Paragraph(SummaryFormat):
    max_words: int = 120

    def __str__(self):
        return "The summary should be written as a cohesive narrative that flows naturally and does not exceed the max_words limit of " + str(self.max_words)

    def max_tokens(self) -> int:
        return int(self.max_words * 1.5)