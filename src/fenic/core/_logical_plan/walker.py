"""Utilities for walking logical plans and searching expressions.

These helpers provide traversal to find `LogicalExpr` instances contained in a
`LogicalPlan` tree. They do not mutate plans or expressions; they only collect
references that match a predicate.

Traversal strategy:
- For each plan node, introspect attributes to find root LogicalExpr instances
- For each root expression, traverse its expression tree via expr.children()
- Recurse into child plan nodes via plan.children()
"""

from __future__ import annotations

from typing import Callable, Generator, List, Set, Tuple

from fenic.core._logical_plan.expressions.base import LogicalExpr
from fenic.core._logical_plan.plans.base import LogicalPlan


def iter_plan_expressions(plan: LogicalPlan, leaf_exprs: bool = True) -> Generator[Tuple[LogicalPlan, LogicalExpr], None, None]:
    """Yield (plan_node, expr) pairs for every LogicalExpr reachable from the plan.

    For each plan node:
    - find attribute-attached expression roots
    - traverse each root's expression tree
    - recurse into child plan nodes
    """
    # Expressions attached to this plan node
    for expr in plan.exprs():
        if not expr.children() or not leaf_exprs:
            yield (plan, expr)
        else:
            for child_expr in _iter_expr_tree(expr):
                yield (plan, child_expr)

    # Recurse into children
    for child in plan.children():
        yield from iter_plan_expressions(child)


def find_expressions(plan: LogicalPlan, predicate: Callable[[LogicalExpr], bool], leaf_exprs: bool = True) -> List[LogicalExpr]:
    """Collect expressions in the plan tree that satisfy the predicate."""
    matches: List[LogicalExpr] = []
    for _, expr in iter_plan_expressions(plan, leaf_exprs):
        if predicate(expr):
            matches.append(expr)
    return matches


def _iter_expr_tree(expr: LogicalExpr) -> Generator[LogicalExpr, None, None]:
    """Depth-first traversal of an expression tree using expr.children()."""
    stack: List[LogicalExpr] = [expr]
    visited: Set[int] = set()
    while stack:
        current = stack.pop()
        obj_id = id(current)
        if obj_id in visited:
            continue
        visited.add(obj_id)
        yield current
        for child in current.children():
            stack.append(child)
