"""Plan serialization/deserialization modules.

This module organizes plan serde functions according to the same structure
as the logical plan modules in _logical_plan/plans.
"""

from fenic.core._serde.proto.plans import (  # noqa: F401
    aggregate,
    join,
    sink,
    source,
    transform,
)
