
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fenic._backends.local.physical_plan import PhysicalPlan
    from fenic.core._logical_plan import (
        LogicalPlan,
    )


class BaseTranspiler(ABC):
    @abstractmethod
    def transpile(self, logical_plan: LogicalPlan) -> PhysicalPlan:
        pass
