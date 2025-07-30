from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ResolvedModelAlias:
    """A resolved model alias with an optional profile name.

    Attributes:
        name: The name of the model.
        profile: The optional name of a profile configuration to use for the model.
    """
    name: str
    profile: Optional[str] = None


@dataclass
class ResolvedClassDefinition:
    label: str
    description: Optional[str] = None
