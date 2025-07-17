"""Common operations for execution."""

import logging
from typing import Literal, Optional, Tuple

from fenic.core._interfaces.execution import BaseExecution
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core.error import PlanError
from fenic.core.metrics import QueryMetrics

logger = logging.getLogger(__name__)


class CommonExecution(BaseExecution):
    """Common class for execution operations."""
    def _validate_table_existance(
            self,
            logical_plan: LogicalPlan,
            table_name: str,
            mode: Literal["error", "append", "overwrite", "ignore"],
    ) -> Tuple[bool, Optional[QueryMetrics]]:
        if self.session_state.catalog.does_table_exist(table_name):
            if mode == "error":
                raise PlanError(
                    f"Cannot save to table '{table_name}' - it already exists and mode is 'error'. "
                    f"Choose a different approach: "
                    f"1) Use mode='overwrite' to replace the existing table, "
                    f"2) Use mode='append' to add data to the existing table, "
                    f"3) Use mode='ignore' to skip saving if table exists, "
                    f"4) Use a different table name.")
            if mode == "ignore":
                logger.warning(f"Table {table_name} already exists, ignoring write.")
                return True, QueryMetrics()
            if mode == "append":
                saved_schema = self.session_state.catalog.describe_table(table_name)
                plan_schema = logical_plan.schema()
                if saved_schema != plan_schema:
                    raise PlanError(
                        f"Cannot append to table '{table_name}' - schema mismatch detected. "
                        f"The existing table has a different schema than your DataFrame. "
                        f"Existing schema: {saved_schema} "
                        f"Your DataFrame schema: {plan_schema} "
                        f"To fix this: "
                        f"1) Use mode='overwrite' to replace the table with your DataFrame's schema, "
                        f"2) Modify your DataFrame to match the existing table's schema, "
                        f"3) Use a different table name.")
                else:
                    return True, None
            if mode == "overwrite":
                return True, None
        else:
            return False, None