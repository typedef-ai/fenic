from fenic import DataFrame
from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan import LogicalPlan
from fenic.core._logical_plan.serde import LogicalPlanSerde


def _test_df_serialization(df: DataFrame, session: BaseSessionState) -> DataFrame:
    """Helper method to test serialization/deserialization of a DataFrame."""
    plan = df._logical_plan
    deserialized_df = _test_plan_serialization(plan, session)
    return deserialized_df

def _test_plan_serialization(
    plan: LogicalPlan, session: BaseSessionState
) -> LogicalPlan:
    """Helper method to test serialization/deserialization of a plan.

    TODO: Add special checking for subclass fields
    """
    # Serialize and deserialize
    serialized = LogicalPlanSerde.serialize(plan)
    deserialized = LogicalPlanSerde.deserialize(serialized)
    deserialized_with_session_state = (
        LogicalPlanSerde.build_logical_plan_with_session_state(deserialized, session)
    )
    deserialized_df = DataFrame._from_logical_plan(
        deserialized_with_session_state
    )
    deserialized_with_session_state._build_schema()
    plan._build_schema()

    # Test equivalence
    assert isinstance(deserialized_with_session_state, type(plan))
    assert plan._repr() == deserialized_with_session_state._repr()
    assert str(plan._build_schema()) == str(deserialized_with_session_state._build_schema())

    # Test children if any
    assert len(plan.children()) == len(deserialized_with_session_state.children())
    # for orig_child, deser_child in zip(plan.children(), deserialized.children()):
    #    assert orig_child._repr() == deser_child._repr()

    return deserialized_df