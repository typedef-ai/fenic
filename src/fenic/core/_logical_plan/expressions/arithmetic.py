from fenic.core._logical_plan.expressions.base import LogicalExpr, Operator
from fenic.core._logical_plan.signatures.scalar_function import ScalarFunction


class ArithmeticExpr(ScalarFunction):
    function_name = "arithmetic"

    def __init__(self, left: LogicalExpr, right: LogicalExpr, op: Operator):
        self.left = left
        self.right = right
        self.op = op

        super().__init__(left, right)

    def __str__(self):
        return f"({self.left} {self.op.value} {self.right})"
