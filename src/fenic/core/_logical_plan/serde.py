import cloudpickle  # nosec: B403

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.plans.base import LogicalPlan

from substrait.gen.proto.plan_pb2 import Plan, PlanRel
from substrait.gen.proto.algebra_pb2 import ReadRel, Rel, RelRoot
from substrait.gen.proto.type_pb2 import NamedStruct, Type

class LogicalPlanSerde2():
    @staticmethod
    def serialize(plan: LogicalPlan) -> bytes:
        """Serialize the logical plan using substrait.
        """
        # plan_proto = Plan(
        #     relations=[
        #         ReadRel(
        #             base_schema=NamedStruct(
        #                 names=["a", "b"],
        #             )
        #         )
        #     ]
        # )
        plan_proto = Plan()
        plan_proto.version.major_number = 1
        plan_proto.version.minor_number = 0
        plan_proto.version.patch_number = 0
        plan_proto.version.git_hash = "1234567890"
        plan_proto.version.producer = "Fenic"
        print(f"Plan: {plan_proto}")
        return bytes(str(plan), 'utf-8')
    
    @staticmethod
    def tst_serialize_plan(plan: LogicalPlan) -> bytes:
        """Serialize the logical plan using substrait.
        """
        # expr_schema = expr.schema()
        rel = PlanRel(
            root=RelRoot(
                input=Rel(
                    read=ReadRel(
                        base_schema=NamedStruct(
                            names=["a", "b"],
                            struct=Type.Struct(types=[Type(i32=Type.I32()), Type(string=Type.String())])
                        )
                    )
                ),
            )
        )
        plan_proto = Plan(relations=[rel])
        # plan_proto = Plan(
        #     relations=[
        #         Rel(
        #             read=ReadRel(
        #                 base_schema=NamedStruct(
        #                     names=["a", "b"],
        #                     struct=Type.Struct()
        #                 )
        #             )
        #         )
        #     ]
        # )
        # plan_proto = Plan()
        plan_proto.version.major_number = 1
        plan_proto.version.minor_number = 0
        plan_proto.version.patch_number = 0
        plan_proto.version.git_hash = "1234567890"
        plan_proto.version.producer = "Fenic"

        print(f"Plan: {plan_proto}")
        return bytes(str(plan), 'utf-8')
    
    @staticmethod
    def deserialize(data: bytes) -> LogicalPlan:
        """Deserialize bytes back into a LogicalPlan using pickle.

        Args:
            data: The serialized plan data

        Returns:
            The deserialized plan
        """
        return cloudpickle.loads(data)  # nosec: B301

class LogicalPlanSerde:
    @staticmethod
    def serialize(plan: LogicalPlan) -> bytes:
        """Serialize a LogicalPlan to bytes using pickle.

        Removes any local session state refs from the plan.

        Args:
            plan: The LogicalPlan to serialize

        Returns:
            bytes: The serialized plan
        """
        # For now, we need to copy the plan in a bottom-up manner, and then walk it again top-down to remove the session state.
        # We can't nullify the session state during the bottom-up traversal because some plan nodes rely on their children's
        # session state during initialization. Clearing it too early can break this initialization logic.
        # TODO(rohitrastogi): Decouple plan construction logic from plan validation logic.
        def copy_plan(plan: LogicalPlan) -> LogicalPlan:
            new_children = []
            for child in plan.children():
                new_children.append(copy_plan(child))
            return plan.with_children(new_children)

        def remove_session_state(plan: LogicalPlan) -> LogicalPlan:
            plan.session_state = None
            for child in plan.children():
                remove_session_state(child)
            return plan

        copied_plan = copy_plan(plan)
        remove_session_state(copied_plan)
        return cloudpickle.dumps(copied_plan)

    @staticmethod
    def deserialize(data: bytes) -> LogicalPlan:
        """Deserialize bytes back into a LogicalPlan using pickle.

        Args:
            data: The serialized plan data

        Returns:
            The deserialized plan
        """
        return cloudpickle.loads(data)  # nosec: B301

    @staticmethod
    def build_logical_plan_with_session_state(
        plan: LogicalPlan, session: BaseSessionState
    ) -> LogicalPlan:
        """Build a LogicalPlan with the session state.

        Args:
            plan: The LogicalPlan to build
            session: The session state
        """
        # TODO(DY): replace pickle with substrait so we don't need this step
        new_children = []
        for child in plan.children():
            new_children.append(
                LogicalPlanSerde.build_logical_plan_with_session_state(child, session)
            )
        plan.session_state = session
        return plan.with_children(new_children)
