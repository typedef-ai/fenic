"""Main API for logical plan and expression serialization/deserialization."""

import zstandard as zstd

from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core._serde.proto.serde_context import create_serde_context
from fenic.core._serde.proto.types import LogicalPlanProto
from fenic.core._serde.serde_protocol import SupportsLogicalPlanSerde


class ProtoSerde(SupportsLogicalPlanSerde):
    """Proto Serde implementation.

    This implementation uses the Protobuf specs defined in the `protos` package to serialize
    and deserialize logical plans. Provides the main API for converting between LogicalPlan
    objects and their binary protobuf representation.
    """

    _zstd_compressor = zstd.ZstdCompressor()
    _zstd_decompressor = zstd.ZstdDecompressor()

    @classmethod
    def serialize(cls, logical_plan: LogicalPlan) -> bytes:
        """Serialize a logical plan to binary protobuf format.

        Args:
            logical_plan: The logical plan to serialize.

        Returns:
            Binary protobuf representation of the logical plan.
        """
        context = create_serde_context()
        logical_plan_proto = context.serialize_logical_plan("root", logical_plan)
        return cls._zstd_compressor.compress(logical_plan_proto.SerializeToString())

    @classmethod
    def deserialize(
        cls,
        data: bytes,
    ) -> LogicalPlan:
        """Deserialize a logical plan from binary protobuf format.

        Args:
            data: Binary protobuf data to deserialize.

        Returns:
            The deserialized logical plan.
        """
        context = create_serde_context()
        logical_plan_proto = LogicalPlanProto.FromString(cls._zstd_decompressor.decompress(data))
        logical_plan = context.deserialize_logical_plan("root", logical_plan_proto)
        return logical_plan