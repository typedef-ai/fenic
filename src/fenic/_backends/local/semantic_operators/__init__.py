from fenic._backends.local.semantic_operators.analyze_sentiment import (
    AnalyzeSentiment,
)
from fenic._backends.local.semantic_operators.classify import Classify
from fenic._backends.local.semantic_operators.extract import Extract
from fenic._backends.local.semantic_operators.join import Join
from fenic._backends.local.semantic_operators.map import Map
from fenic._backends.local.semantic_operators.predicate import Predicate
from fenic._backends.local.semantic_operators.reduce import Reduce
from fenic._backends.local.semantic_operators.summarize import Summarize


def __getattr__(name: str):
    if name == "Cluster":
        from fenic._backends.local.semantic_operators.cluster import Cluster

        return Cluster
    if name == "ParsePDF":
        from fenic._backends.local.semantic_operators.parse_pdf import ParsePDF

        return ParsePDF
    if name == "SimJoin":
        from fenic._backends.local.semantic_operators.sim_join import SimJoin

        return SimJoin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Classify",
    "Extract",
    "Predicate",
    "Cluster",
    "Join",
    "Map",
    "AnalyzeSentiment",
    "ParsePDF",
    "SimJoin",
    "Reduce",
    "Summarize",
]
