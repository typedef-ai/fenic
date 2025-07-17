from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from jinja2 import Environment, nodes
from jinja2.exceptions import TemplateSyntaxError

from fenic.core.error import ValidationError

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

class TypeRequirement(Enum):
    """Represents the expected data type for a variable based on how it's used in the template."""
    BOOLEAN = "boolean"
    ARRAY = "array"
    STRUCT = "struct"

@dataclass
class VariableNode:
    """Represents a variable in the Jinja template and its expected data type."""
    requirement: Optional[TypeRequirement] = None
    children: dict[str, VariableNode] = field(default_factory=dict)
    line_no: str = '?'

    def set_requirement(self, req: TypeRequirement, line_no: str = '?') -> None:
        """Sets the requirement for the variable and validates that it's consistent with previous uses. Errors if inconsistent."""
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
        """Get or create a child node (for nested fields like user.name or items[*])."""
        if name not in self.children:
            self.children[name] = VariableNode()
        return self.children[name]

@dataclass
class VariableTree:
    """The root of our schema tree. Contains all top-level variables."""
    variables: dict[str, VariableNode] = field(default_factory=dict)

    def get_or_create_variable(self, name: str) -> VariableNode:
        """Get or create a top-level variable node."""
        if name not in self.variables:
            self.variables[name] = VariableNode()
        return self.variables[name]

class VariableAccessContext(Enum):
    """Represents how a variable is used in the template."""
    OUTPUT = "output"
    CONDITION = "condition"
    ITERATION = "iteration"

@dataclass
class VariableAccess:
    """Represents one instance of a variable access in the template."""
    path: List[str]
    context: VariableAccessContext
    node: nodes.Node

@dataclass
class LoopDefinition:
    """Represents a for loop definition in the template."""
    var_name: str
    array_path: List[str]
    loop_node: nodes.For

def _annotate_parents(node: nodes.Node, parent: Optional[nodes.Node] = None) -> None:
    """Recursively add `parent` references to each AST node since Jinja includes them in the AST. This is used for loop resolution."""
    node.parent = parent  # type: ignore[attr-defined]
    for child in node.iter_child_nodes():
        _annotate_parents(child, parent=node)

def _extract_variable_path(node: nodes.Node) -> Optional[List[str]]:
    """Extract the full variable path as a list of keys/fields from AST node."""
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
                return base_path + ["*"]
            else:
                return base_path + [node.arg.value]
        return None

# Phase 1: Validate the AST
def _validate_ast(ast: nodes.Node) -> None:
    """Validate that the AST only contains allowed constructs."""
    def validate_node(node: nodes.Node) -> None:
        line_no = getattr(node, "lineno", "?")

        if not isinstance(node, ALLOWED_JINJA_NODES):
            raise ValidationError(f"Unsupported Jinja template syntax on line {line_no}: Only basic variables, if statements, and for loops are allowed.")

        if isinstance(node, nodes.Name) and node.name == "loop":
            raise ValidationError(
                f"Unsupported Jinja template syntax on line {line_no}: The special 'loop' variable (e.g., 'loop.index') is not supported. Please avoid using 'loop' inside your template expressions."
            )

        if isinstance(node, nodes.Getitem):
            if not isinstance(node.arg, nodes.Const):
                raise ValidationError(f"Unsupported Jinja template syntax on line {line_no}: Array and object access must use fixed values like [0] or ['key']. Variables inside brackets are not allowed.")
            if type(node.arg.value) not in (int, str):
                raise ValidationError(
                    f"Unsupported Jinja template syntax on line {line_no}: Index must be a number or text string. Example: myarray[0] or myobject['key']"
            )

        if isinstance(node, nodes.Const):
            if not isinstance(getattr(node, 'parent', None), nodes.Getitem):
                raise ValidationError(
                    f"Unsupported Jinja template syntax on line {line_no}: Literal values are not allowed directly in expressions. Use variables instead."
                )

        for child in node.iter_child_nodes():
            validate_node(child)


    validate_node(ast)

# Phase 2: Collect raw data from AST
def _collect_raw_data(ast: nodes.Node) -> tuple[List[VariableAccess], List[LoopDefinition]]:
    """Extract all variable accesses and loop definitions from AST."""
    accesses = []
    loops = []

    def walk_node(node: nodes.Node) -> None:
        if isinstance(node, nodes.For):
            # Collect loop definition
            if isinstance(node.target, nodes.Name):
                array_path = _extract_variable_path(node.iter)
                if array_path:
                    loops.append(LoopDefinition(node.target.name, array_path, node))
                    # Also add the iteration as an access
                    accesses.append(VariableAccess(array_path, VariableAccessContext.ITERATION, node.iter))

        elif isinstance(node, nodes.If):
            # Collect condition access
            path = _extract_variable_path(node.test)
            if path:
                accesses.append(VariableAccess(path, VariableAccessContext.CONDITION, node.test))

        elif isinstance(node, nodes.Output):
            # Collect output accesses
            for output_node in node.nodes:
                if isinstance(output_node, (nodes.Name, nodes.Getattr, nodes.Getitem)):
                    path = _extract_variable_path(output_node)
                    if path:
                        accesses.append(VariableAccess(path, VariableAccessContext.OUTPUT, output_node))

        for child in node.iter_child_nodes():
            walk_node(child)

    walk_node(ast)
    return accesses, loops

# Phase 3: Resolve loop variables to their array sources
def _resolve_loop_variables(accesses: List[VariableAccess], loops: List[LoopDefinition]) -> List[VariableAccess]:
    """Phase 2: Convert loop variable accesses to array element accesses."""
    resolved = []

    # Create a map of variable names to loop definitions for quick lookup
    loop_vars = {loop.var_name: loop for loop in loops}

    def resolve_path(path: List[str], node: nodes.Node) -> List[str]:
        """Recursively resolve loop variables in a path."""
        if not path or path[0] not in loop_vars:
            return path  # No loop variable, return as-is

        # Find the defining loop for this variable by walking up the AST
        defining_loop = _find_defining_loop(node, path[0], loops)
        if not defining_loop:
            return path  # Shouldn't happen, but safe fallback

        # Recursively resolve the array path in case it contains loop variables too
        resolved_array_path = resolve_path(defining_loop.array_path, defining_loop.loop_node)

        # Convert: item.name -> items[*].name
        if len(path) > 1:
            return resolved_array_path + ["*"] + path[1:]
        else:
            return resolved_array_path

    for access in accesses:
        resolved_path = resolve_path(access.path, access.node)
        resolved.append(VariableAccess(resolved_path, access.context, access.node))

    return resolved

def _find_defining_loop(node: nodes.Node, var_name: str, loops: List[LoopDefinition]) -> Optional[LoopDefinition]:
    """Find the innermost loop that defines the given variable name by walking up the AST until we find the loop that defines it."""
    current = getattr(node, 'parent', None)
    while current:
        for loop in loops:
            if loop.loop_node == current and loop.var_name == var_name:
                return loop
        current = getattr(current, 'parent', None)
    return None


# Phase 4: Build the schema tree
def _build_schema_tree(resolved_accesses: List[VariableAccess]) -> VariableTree:
    """Build the final schema tree from resolved accesses. Filters out variables that are not used in output."""
    tree = VariableTree()

    # Filter: only include variables that are actually used in output
    output_accesses = [a for a in resolved_accesses if a.context == VariableAccessContext.OUTPUT]
    control_accesses = [a for a in resolved_accesses if a.context != VariableAccessContext.OUTPUT]

    # Get root variables that have actual output usage
    output_roots = {access.path[0] for access in output_accesses}

    # Only include control flow accesses for variables that are also used in output
    filtered_control = [a for a in control_accesses if a.path[0] in output_roots]

    # Build tree from output + filtered control accesses
    all_relevant_accesses = output_accesses + filtered_control

    def add_path_to_tree(path: List[str], leaf_requirement: Optional[TypeRequirement] = None, line_no: str = '?') -> None:
        if not path:
            return

        current = tree.get_or_create_variable(path[0])

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
                child.set_requirement(leaf_requirement)

            current = child

    # Process all relevant accesses
    for access in all_relevant_accesses:
        line_no = getattr(access.node, 'lineno', '?')
        if access.context == VariableAccessContext.ITERATION:
            add_path_to_tree(access.path, TypeRequirement.ARRAY, line_no)
        elif access.context == VariableAccessContext.CONDITION:
            add_path_to_tree(access.path, TypeRequirement.BOOLEAN, line_no)
        else:  # output
            add_path_to_tree(access.path, line_no=line_no)  # No leaf requirement, just structure

    return tree


def validate_and_parse_jinja_template(template: str) -> VariableTree:
    """Validates a Jinja template and extracts the variable schema it requires.

    This function analyzes a Jinja template to determine what data structure it expects,
    including variable types (arrays, objects, booleans) and nested field requirements.
    It also enforces security restrictions by only allowing safe template constructs.

    Args:
        template: A Jinja template string to validate and analyze.
                 Example: "Hello {{ user.name }}! {% for item in products %}{{ item.price }}{% endfor %}"

    Returns:
        VariableTree: A tree structure describing the required variables and their types.
                     For the example above, this would indicate:
                     - user: object with 'name' field
                     - products: array of objects with 'price' field

    Raises:
        ValidationError: If the template contains:
                        - Invalid Jinja syntax
                        - Unsupported constructs (complex expressions, dynamic indexing, etc.)
                        - Inconsistent variable usage (e.g., using same variable as both array and object)
    """
    # trunk-ignore(bandit/B701): Templates generate plain text, not HTML, so no risk of XSS. In fact, we may want to allow HTML in the template.
    env = Environment(autoescape=False)
    try:
        ast = env.parse(template)
    except TemplateSyntaxError as e:
        raise ValidationError(f"Jinja template syntax error on line {e.lineno}: {e.message}") from e

    # Add parent references for loop resolution
    _annotate_parents(ast)

    # Phase 1: Validate allowed constructs
    _validate_ast(ast)

    # Phase 2: Collect raw data
    accesses, loops = _collect_raw_data(ast)

    # Phase 3: Resolve loop variables
    resolved_accesses = _resolve_loop_variables(accesses, loops)

    # Phase 4: Build schema tree
    tree = _build_schema_tree(resolved_accesses)
    return tree
