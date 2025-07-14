"""Types for resolved instruction templates."""

from typing import Union

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from fenic.core._logical_plan.expressions import AliasExpr, ColumnExpr

NamedExpr = Union[AliasExpr, ColumnExpr]

@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ResolvedInstructionTemplate:
    """A resolved instruction template containing the instruction string and resolved expressions.

    This is used internally by the expression layer to avoid circular imports with api.types.InstructionTemplate.
    The api layer resolves InstructionTemplate.resolve_children() and passes the result here.
    """
    instruction: str
    exprs: list[NamedExpr]
