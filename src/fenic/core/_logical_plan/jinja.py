from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from jinja2 import Environment, nodes
from jinja2.exceptions import TemplateSyntaxError

from fenic.core.error import ValidationError

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

# Define which Jinja AST node types we allow in templates
ALLOWED_JINJA_NODES = (
    nodes.Template,
    nodes.Output,
    nodes.Name,
    nodes.Getattr,
    nodes.Getitem,
    nodes.If,
    nodes.For,
    nodes.Not,
    nodes.TemplateData,
    nodes.Const,
)

# =============================================================================
# DATA MODELS
# =============================================================================

class TypeRequirement(Enum):
    """Expected data type for a variable based on how it's used in the template."""
    BOOLEAN = "boolean"
    ARRAY = "array"
    STRUCT = "struct"

class VariableAccessContext(Enum):
    """How a variable is used in the template."""
    OUTPUT = "output"
    CONDITION = "condition"
    ITERATION = "iteration"

@dataclass
class VariableNode:
    """A variable in the Jinja template with its expected data type."""
    requirement: Optional[TypeRequirement] = None
    children: dict[str, VariableNode] = field(default_factory=dict)
    line_no: str = '?'

    def set_requirement(self, req: TypeRequirement, line_no: str = '?') -> None:
        """Set the type requirement, validating consistency with previous uses."""
        if self.requirement is not None and self.requirement != req:
            raise ValidationError(
                f"Variable used inconsistently across the jinja template:\n"
                f"  - Used as {self.requirement.value} (line {self.line_no})\n"
                f"  - Used as {req.value} (line {line_no})\n"
                f"Each variable must have a consistent type (e.g., struct, array, boolean)."
            )
        self.requirement = req
        self.line_no = line_no

    def get_or_create_child(self, name: str) -> VariableNode:
        """Get or create a child node for nested fields like user.name or items[*]."""
        if name not in self.children:
            self.children[name] = VariableNode()
        return self.children[name]

@dataclass
class VariableTree:
    """Root of the schema tree containing all top-level variables."""
    variables: dict[str, VariableNode] = field(default_factory=dict)

    def get_or_create_variable(self, name: str) -> VariableNode:
        """Get or create a top-level variable node."""
        if name not in self.variables:
            self.variables[name] = VariableNode()
        return self.variables[name]

@dataclass
class LoopDefinition:
    """A loop variable definition with its array source."""
    var_name: str
    array_path: List[str]

@dataclass
class VariableAccess:
    """A resolved variable access with context about how it's used."""
    path: List[str]
    context: VariableAccessContext
    line_no: str = '?'

# =============================================================================
# SCOPE MANAGEMENT
# =============================================================================

class LoopStack:
    """Manages loop variable scoping during template traversal."""

    def __init__(self):
        self._frames: List[LoopDefinition] = []

    def push_loop_var(self, var_name: str, array_path: List[str]) -> None:
        """Push a new loop variable (may shadow existing ones with the same name)."""
        self._frames.append(LoopDefinition(var_name, array_path))

    def resolve_variable(self, var_name: str) -> Optional[List[str]]:
        """Resolve a variable name to its array path, or None if not a loop variable."""
        # Search backwards for most recent definition (handles shadowing)
        for frame in reversed(self._frames):
            if frame.var_name == var_name:
                return frame.array_path + ["*"]
        return None  # Not a loop variable

# =============================================================================
# MAIN PUBLIC API
# =============================================================================

def validate_and_parse_jinja_template(template: str) -> VariableTree:
    """Validate a Jinja template and extract its variable schema.

    Analyzes a Jinja template to determine what data structure it expects,
    including variable types (arrays, objects, booleans) and nested field requirements.
    Enforces security restrictions by only allowing safe template constructs.

    Args:
        template: A Jinja template string to validate and analyze.
                 Example: "Hello {{ user.name }}! {% for item in products %}{{ item.price }}{% endfor %}"

    Returns:
        VariableTree: Tree structure describing required variables and their types.
                     For the example above, this indicates:
                     - user: object with 'name' field
                     - products: array of objects with 'price' field

    Raises:
        ValidationError: If the template contains:
                        - Invalid Jinja syntax
                        - Unsupported constructs (complex expressions, dynamic indexing, etc.)
                        - Inconsistent variable usage (e.g., using same variable as both array and object)
    """
    # Parse template into AST
    ast = _parse_template(template)

    # Validate only allowed constructs are used
    _validate_template(ast)

    # Collect all variable accesses with proper scoping
    all_accesses = _collect_variable_accesses(ast)

    # Filter to only variables used in output (plus their dependencies)
    relevant_accesses = _filter_to_output_variables(all_accesses)

    # Build the final schema tree
    return _build_schema_tree(relevant_accesses)

# =============================================================================
# PARSING & VALIDATION
# =============================================================================

def _parse_template(template: str) -> nodes.Node:
    """Parse template string into AST with parent references."""
    # trunk-ignore(bandit/B701): Templates generate plain text, not HTML, so no risk of XSS.
    env = Environment(autoescape=False)
    try:
        ast = env.parse(template)
    except TemplateSyntaxError as e:
        raise ValidationError(f"Jinja template syntax error on line {e.lineno}: {e.message}") from e

    # Add parent references needed for validation
    _annotate_parents(ast)
    return ast

def _annotate_parents(node: nodes.Node, parent: Optional[nodes.Node] = None) -> None:
    """Recursively add parent references to AST nodes for validation."""
    node.parent = parent  # type: ignore[attr-defined]
    for child in node.iter_child_nodes():
        _annotate_parents(child, parent=node)

def _validate_template(ast: nodes.Node) -> None:
    """Validate that template only contains allowed, safe constructs."""
    def validate_node(node: nodes.Node) -> None:
        line_no = getattr(node, "lineno", "?")

        # Check node type is allowed
        if not isinstance(node, ALLOWED_JINJA_NODES):
            raise ValidationError(
                f"Unsupported Jinja template syntax on line {line_no}: "
                f"Only basic variables, if statements, and for loops are allowed."
            )

        # Specific validation rules
        if isinstance(node, nodes.Name) and node.name == "loop":
            raise ValidationError(
                f"Unsupported Jinja template syntax on line {line_no}: "
                f"The special 'loop' variable (e.g., 'loop.index') is not supported. "
                f"Please avoid using 'loop' inside your template expressions."
            )

        if isinstance(node, nodes.Getitem):
            if not isinstance(node.arg, nodes.Const):
                raise ValidationError(
                    f"Unsupported Jinja template syntax on line {line_no}: "
                    f"Array and object access must use fixed values like [0] or ['key']. "
                    f"Variables inside brackets are not allowed."
                )
            if type(node.arg.value) not in (int, str):
                raise ValidationError(
                    f"Unsupported Jinja template syntax on line {line_no}: "
                    f"Index must be a number or text string. Example: myarray[0] or myobject['key']"
                )

        if isinstance(node, nodes.Const):
            if not isinstance(getattr(node, 'parent', None), nodes.Getitem):
                raise ValidationError(
                    f"Unsupported Jinja template syntax on line {line_no}: "
                    f"Literal values are not allowed directly in expressions. Use variables instead."
                )

        # Recursively validate children
        for child in node.iter_child_nodes():
            validate_node(child)

    validate_node(ast)

# =============================================================================
# VARIABLE ACCESS COLLECTION
# =============================================================================

def _collect_variable_accesses(ast: nodes.Node) -> List[VariableAccess]:
    """Collect all variable accesses with proper scoping resolution."""
    scope = LoopStack()
    return _traverse_and_collect(ast, scope)

def _traverse_and_collect(node: nodes.Node, scope: LoopStack) -> List[VariableAccess]:
    """Traverse AST and collect variable accesses with resolved paths."""
    accesses = []
    line_no = getattr(node, "lineno", "?")

    if isinstance(node, nodes.For):
        accesses.extend(_handle_for_loop(node, scope, line_no))
    elif isinstance(node, nodes.If):
        accesses.extend(_handle_conditional(node, scope, line_no))
    elif isinstance(node, nodes.Output):
        accesses.extend(_handle_output(node, scope, line_no))
    else:
        # Continue traversal for other node types
        for child in node.iter_child_nodes():
            accesses.extend(_traverse_and_collect(child, scope))

    return accesses

def _handle_for_loop(node: nodes.For, scope: LoopStack, line_no: str) -> List[VariableAccess]:
    """Handle for loop: record iteration and update scope."""
    accesses = []

    # Record the array being iterated over
    array_path = _extract_variable_path(node.iter)
    if array_path:
        resolved_array_path = _resolve_variable_path(array_path, scope)
        accesses.append(VariableAccess(resolved_array_path, VariableAccessContext.ITERATION, line_no))

        # Push loop variable into scope for the loop body
        if isinstance(node.target, nodes.Name):
            scope.push_loop_var(node.target.name, resolved_array_path)

    # Process loop body with updated scope
    for child in node.iter_child_nodes():
        accesses.extend(_traverse_and_collect(child, scope))

    return accesses

def _handle_conditional(node: nodes.If, scope: LoopStack, line_no: str) -> List[VariableAccess]:
    """Handle if statement: record condition variable."""
    accesses = []

    # Record the condition variable (should be boolean)
    condition_path = _extract_variable_path(node.test)
    if condition_path:
        resolved_condition = _resolve_variable_path(condition_path, scope)
        accesses.append(VariableAccess(resolved_condition, VariableAccessContext.CONDITION, line_no))

    # Process if body and else body
    for child in node.iter_child_nodes():
        accesses.extend(_traverse_and_collect(child, scope))

    return accesses

def _handle_output(node: nodes.Output, scope: LoopStack, line_no: str) -> List[VariableAccess]:
    """Handle output expressions: record variables being displayed."""
    accesses = []

    for output_node in node.nodes:
        if isinstance(output_node, (nodes.Name, nodes.Getattr, nodes.Getitem)):
            output_path = _extract_variable_path(output_node)
            if output_path:
                resolved_output = _resolve_variable_path(output_path, scope)
                accesses.append(VariableAccess(resolved_output, VariableAccessContext.OUTPUT, line_no))

    return accesses

# =============================================================================
# PATH EXTRACTION & RESOLUTION
# =============================================================================

def _extract_variable_path(node: nodes.Node) -> Optional[List[str]]:
    """Extract variable path as list of keys/fields from AST node."""
    if isinstance(node, nodes.Name):
        return [node.name]
    elif isinstance(node, nodes.Getattr):
        base_path = _extract_variable_path(node.node)
        if base_path:
            return base_path + [node.attr]
    elif isinstance(node, nodes.Getitem):
        base_path = _extract_variable_path(node.node)
        if base_path and isinstance(node.arg, nodes.Const):
            if isinstance(node.arg.value, int):
                return base_path + ["*"]  # Integer index becomes wildcard
            else:
                return base_path + [node.arg.value]  # String key preserved
    return None

def _resolve_variable_path(path: List[str], scope: LoopStack) -> List[str]:
    """Resolve variable path using current loop scope."""
    if not path:
        return path

    # Check if first part is a loop variable
    resolved_root = scope.resolve_variable(path[0])
    if resolved_root is not None:
        # Replace loop variable with its resolved array path
        return resolved_root + path[1:]
    else:
        # Not a loop variable, use as-is
        return path

# =============================================================================
# FILTERING & TREE BUILDING
# =============================================================================

def _filter_to_output_variables(all_accesses: List[VariableAccess]) -> List[VariableAccess]:
    """Filter to only variables used in output (plus their control flow dependencies)."""
    output_accesses = [a for a in all_accesses if a.context == VariableAccessContext.OUTPUT]
    control_accesses = [a for a in all_accesses if a.context != VariableAccessContext.OUTPUT]

    # Get root variables that have actual output usage
    output_roots = {access.path[0] for access in output_accesses}

    # Only include control flow accesses for variables that are also used in output
    filtered_control = [a for a in control_accesses if a.path[0] in output_roots]

    return output_accesses + filtered_control

def _build_schema_tree(accesses: List[VariableAccess]) -> VariableTree:
    """Build the final schema tree from filtered variable accesses."""
    tree = VariableTree()

    for access in accesses:
        # Determine type requirement based on context
        leaf_requirement = None
        if access.context == VariableAccessContext.ITERATION:
            leaf_requirement = TypeRequirement.ARRAY
        elif access.context == VariableAccessContext.CONDITION:
            leaf_requirement = TypeRequirement.BOOLEAN

        _add_path_to_tree(tree, access.path, leaf_requirement, access.line_no)

    return tree

def _add_path_to_tree(
    tree: VariableTree,
    path: List[str],
    leaf_requirement: Optional[TypeRequirement] = None,
    line_no: str = '?'
) -> None:
    """Add a resolved path to the schema tree with proper type requirements."""
    if not path:
        return

    current = tree.get_or_create_variable(path[0])

    # Handle single-element path
    if len(path) == 1:
        if leaf_requirement:
            current.set_requirement(leaf_requirement, line_no)
        return

    # Walk the path and set parent requirements
    for i, part in enumerate(path[1:], 1):
        if part == "*":
            current.set_requirement(TypeRequirement.ARRAY, line_no)
            key = "*"
        else:
            current.set_requirement(TypeRequirement.STRUCT, line_no)
            key = part

        child = current.get_or_create_child(key)

        # Set leaf requirement if this is the last part
        if i == len(path) - 1 and leaf_requirement:
            child.set_requirement(leaf_requirement, line_no)

        current = child
