import cloudpickle  # nosec: B403

from fenic.core._logical_plan.plans.base import LogicalPlan


class LogicalPlanSerde:
    @staticmethod
    def serialize(plan: LogicalPlan) -> bytes:
        """Serialize a LogicalPlan to bytes using pickle.

        Args:
            plan: The LogicalPlan to serialize

        Returns:
            bytes: The serialized plan
        """
        return cloudpickle.dumps(plan)

    @staticmethod
    def deserialize(data: bytes) -> LogicalPlan:
        """Deserialize bytes back into a LogicalPlan using pickle.

        Args:
            data: The serialized plan data

        Returns:
            The deserialized plan
        """
        return cloudpickle.loads(data)  # nosec: B301
