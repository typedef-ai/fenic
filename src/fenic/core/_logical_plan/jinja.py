from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from jinja2 import Environment, nodes
from jinja2.exceptions import TemplateSyntaxError

from fenic.core.error import InternalError, TypeMismatchError, ValidationError
from fenic.core.types import ArrayType, BooleanType, DataType, StructType

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
class LoopDefinition:
    """A loop variable definition with its array source."""
    var_name: str
    array_path: List[str]

class VariableAccessContext(Enum):
    """How a variable is used in the template."""
    OUTPUT = "output"
    CONDITION = "condition"
    ITERATION = "iteration"

@dataclass
class VariableAccess:
    """A resolved variable access with context about how it's used."""
    path: List[str]
    context: VariableAccessContext
    line_no: str = '?'

@dataclass
class AccessWithDeps:
    """A variable access with its control flow dependencies."""
    access: VariableAccess
    control_dependencies: List[VariableAccess]

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

@dataclass
class VariableTree:
    """Root of the schema tree containing all top-level variables."""
    variables: dict[str, VariableNode] = field(default_factory=dict)

    @classmethod
    def from_jinja_template(cls, template: str) -> VariableTree:
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
        ast = cls._parse_template(template)

        # Validate only allowed constructs are used
        cls._validate_template(ast)

        # Collect all variable accesses with proper scoping
        all_accesses = cls._collect_variable_accesses(ast)

        # Filter to only variables used in output (plus their dependencies)
        relevant_accesses = cls._extract_output_dependencies(all_accesses)

        # Build the final schema tree
        return cls._build_schema_tree(relevant_accesses)

    def validate_jinja_variable(
        self,
        variable_name: str,
        data_type: DataType
    ) -> None:
        """Recursively validates that the structure and type requirements of a Jinja template variable match the actual column schema.

        This ensures that:
          - Boolean requirements are matched by BooleanType columns.
          - For-loop variables are backed by ArrayType columns.
          - Struct field access is valid and only used on StructType columns.

        Args:
            variable_name: The name of the top-level variable used in the Jinja template.
            data_type: The corresponding DataType from the input schema.

        Raises:
            TypeMismatchError: If the variable's usage does not match its actual type.
            ValidationError: If a struct field is accessed that does not exist.
            InternalError: If an unexpected or invalid requirement is encountered.
        """
        def validate_helper(variable_node: VariableNode, data_type: DataType, path: List[str]) -> None:
            formatted_path = _format_path(path)

            if not variable_node.requirement:
                return

            if variable_node.requirement == TypeRequirement.BOOLEAN:
                if not data_type == BooleanType:
                    raise TypeMismatchError.from_message(
                        f"Column '{formatted_path}' used in Jinja template must be a BooleanType, but found {data_type}. "
                        f"This variable is used in a conditional expression and must evaluate to a boolean."
                    )

            elif variable_node.requirement == TypeRequirement.ARRAY:
                if not isinstance(data_type, ArrayType):
                    raise TypeMismatchError.from_message(
                        f"Column '{formatted_path}' used in Jinja template must be an ArrayType, but found {data_type}. "
                        f"This variable is used in a for-loop and must be an array column."
                    )
                validate_helper(variable_node.children["*"], data_type.element_type, path + ["*"])

            elif variable_node.requirement == TypeRequirement.STRUCT:
                if not isinstance(data_type, StructType):
                    raise TypeMismatchError.from_message(
                        f"Column '{formatted_path}' used in Jinja template must be a StructType, but found {data_type}. "
                        f"This variable is accessed using field notation (e.g., {formatted_path}.fieldname) and must be a struct column."
                    )

                struct_field_map = {field.name: field.data_type for field in data_type.struct_fields}
                available_fields = sorted(struct_field_map.keys())

                for child_name in variable_node.children.keys():
                    if child_name not in struct_field_map:
                        raise ValidationError(
                            f"Field '{child_name}' in Jinja template does not exist in StructType at '{formatted_path}'. "
                            f"Available StructFields: {', '.join(available_fields)}. "
                            f"Please check for typos or confirm the struct schema."
                        )
                    validate_helper(variable_node.children[child_name], struct_field_map[child_name], path + [child_name])

            else:
                raise InternalError(
                    f"Unexpected variable requirement '{variable_node.requirement}' "
                    f"for variable '{formatted_path}'. This indicates a bug in the type resolution logic."
                )
        validate_helper(self.variables[variable_name], data_type, [variable_name])


    def _get_or_create_variable(self, name: str) -> VariableNode:
        """Get or create a top-level variable node."""
        if name not in self.variables:
            self.variables[name] = VariableNode()
        return self.variables[name]



    # =============================================================================
    # PARSING & VALIDATION
    # =============================================================================
    @staticmethod
    def _parse_template(template: str) -> nodes.Node:
        """Parse template string into AST with parent references."""
        # trunk-ignore(bandit/B701): Templates generate plain text, not HTML, so no risk of XSS.
        env = Environment(autoescape=False)
        try:
            ast = env.parse(template)
        except TemplateSyntaxError as e:
            raise ValidationError(f"Jinja template syntax error on line {e.lineno}: {e.message}") from e

        # Add parent references needed for validation
        VariableTree._annotate_parents(ast)
        return ast

    @staticmethod
    def _annotate_parents(node: nodes.Node, parent: Optional[nodes.Node] = None) -> None:
        """Recursively add parent references to AST nodes for validation."""
        node.parent = parent  # type: ignore[attr-defined]
        for child in node.iter_child_nodes():
            VariableTree._annotate_parents(child, parent=node)

    @staticmethod
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
    @staticmethod
    def _collect_variable_accesses(ast: nodes.Node) -> List[AccessWithDeps]:
        """Collect all variable accesses with proper scoping resolution."""
        scope = LoopStack()
        return VariableTree._traverse_and_collect(ast, scope, [])

    @staticmethod
    def _traverse_and_collect(node: nodes.Node, scope: LoopStack, control_context: List[VariableAccess]) -> List[AccessWithDeps]:
        """Traverse AST and collect variable accesses with resolved paths."""
        accesses = []
        line_no = getattr(node, "lineno", "?")

        if isinstance(node, nodes.For):
            accesses.extend(VariableTree._handle_for_loop(node, scope, control_context, line_no))
        elif isinstance(node, nodes.If):
            accesses.extend(VariableTree._handle_conditional(node, scope, control_context, line_no))
        elif isinstance(node, nodes.Output):
            accesses.extend(VariableTree._handle_output(node, scope, control_context, line_no))
        else:
            # Continue traversal for other node types
            for child in node.iter_child_nodes():
                accesses.extend(VariableTree._traverse_and_collect(child, scope, control_context))

        return accesses

    @staticmethod
    def _handle_for_loop(node: nodes.For, scope: LoopStack, control_context: List[VariableAccess], line_no: str) -> List[AccessWithDeps]:
        """Handle for loop: record iteration and update scope."""
        accesses = []

        # Record the array being iterated over
        array_path = VariableTree._extract_variable_path(node.iter)
        if array_path:
            resolved_array_path = VariableTree._resolve_variable_path(array_path, scope)
            iter_access = VariableAccess(resolved_array_path, VariableAccessContext.ITERATION, line_no)

            # Push loop variable into scope for the loop body
            if isinstance(node.target, nodes.Name):
                scope.push_loop_var(node.target.name, resolved_array_path)
            else:
                raise ValidationError(
                    f"Unsupported Jinja template syntax on line {line_no}: "
                    f"Loop target must be a simple variable name (e.g., 'item'), not a tuple or destructuring expression.\n"
                    "Example of valid syntax: {% for item in products %}"
                )

        new_control_context = control_context + [iter_access]
        # Process loop body with updated scope
        for child in node.iter_child_nodes():
            accesses.extend(VariableTree._traverse_and_collect(child, scope, new_control_context))

        return accesses

    @staticmethod
    def _handle_conditional(node: nodes.If, scope: LoopStack, control_context: List[VariableAccess], line_no: str) -> List[AccessWithDeps]:
        """Handle if statement: record condition variable."""
        accesses = []

        # Record the condition variable (should be boolean)
        condition_path = VariableTree._extract_variable_path(node.test)
        if condition_path:
            resolved_condition = VariableTree._resolve_variable_path(condition_path, scope)
            cond_access = VariableAccess(resolved_condition, VariableAccessContext.CONDITION, line_no)
            new_control_context = control_context + [cond_access]

        # Process if body and else body
        for child in node.iter_child_nodes():
            accesses.extend(VariableTree._traverse_and_collect(child, scope, new_control_context))

        return accesses

    @staticmethod
    def _handle_output(node: nodes.Output, scope: LoopStack, control_context: List[VariableAccess], line_no: str) -> List[AccessWithDeps]:
        """Handle output expressions: record variables being displayed."""
        accesses = []

        for output_node in node.nodes:
            if isinstance(output_node, (nodes.Name, nodes.Getattr, nodes.Getitem)):
                output_path = VariableTree._extract_variable_path(output_node)
                if output_path:
                    resolved_output = VariableTree._resolve_variable_path(output_path, scope)
                    output_access = VariableAccess(resolved_output, VariableAccessContext.OUTPUT, line_no)
                    accesses.append(AccessWithDeps(output_access, control_context))

        return accesses

    # =============================================================================
    # PATH EXTRACTION & RESOLUTION
    # =============================================================================

    @staticmethod
    def _extract_variable_path(node: nodes.Node) -> Optional[List[str]]:
        """Extract variable path as list of keys/fields from AST node."""
        if isinstance(node, nodes.Name):
            return [node.name]
        elif isinstance(node, nodes.Getattr):
            base_path = VariableTree._extract_variable_path(node.node)
            if base_path:
                return base_path + [node.attr]
        elif isinstance(node, nodes.Getitem):
            base_path = VariableTree._extract_variable_path(node.node)
            if base_path and isinstance(node.arg, nodes.Const):
                if isinstance(node.arg.value, int):
                    return base_path + ["*"]  # Integer index becomes wildcard
                else:
                    return base_path + [node.arg.value]  # String key preserved
        return None

    @staticmethod
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

    @staticmethod
    def _extract_output_dependencies(all_accesses: List[AccessWithDeps]) -> List[VariableAccess]:
        """Remove all variables that are not required to evaluate the output."""
        outputs = [a.access for a in all_accesses if a.access.context == VariableAccessContext.OUTPUT]

        # Flatten all dependencies from outputs
        control_deps: List[VariableAccess] = []
        for a in all_accesses:
            if a.access.context == VariableAccessContext.OUTPUT:
                control_deps.extend(a.control_dependencies)

        # Deduplicate
        unique_control_deps = {tuple(dep.path): dep for dep in control_deps}.values()

        return outputs + list(unique_control_deps)

    @classmethod
    def _build_schema_tree(cls, accesses: List[VariableAccess]) -> VariableTree:
        """Build the final schema tree from filtered variable accesses."""
        tree = cls()

        for access in accesses:
            # Determine type requirement based on context
            leaf_requirement = None
            if access.context == VariableAccessContext.ITERATION:
                leaf_requirement = TypeRequirement.ARRAY
            elif access.context == VariableAccessContext.CONDITION:
                leaf_requirement = TypeRequirement.BOOLEAN

            cls._add_path_to_tree(tree, access.path, leaf_requirement, access.line_no)

        return tree

    @staticmethod
    def _add_path_to_tree(
        tree: VariableTree,
        path: List[str],
        leaf_requirement: Optional[TypeRequirement] = None,
        line_no: str = '?'
    ) -> None:
        """Add a resolved path to the schema tree with proper type requirements."""
        if not path:
            return

        current = tree._get_or_create_variable(path[0])

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

def _format_path(path: List[str]) -> str:
    result = []
    for part in path:
        if part == "*":
            result[-1] += "[*]"
        else:
            result.append(part)
    return ".".join(result)
