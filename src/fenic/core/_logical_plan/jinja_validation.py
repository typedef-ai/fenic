from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Union

from jinja2 import Environment, nodes
from jinja2.exceptions import TemplateSyntaxError

from fenic.core.error import InternalError, TypeMismatchError, ValidationError
from fenic.core.types import ArrayType, DataType, StructType

if TYPE_CHECKING:
    from fenic.core._logical_plan.expressions import AliasExpr, ColumnExpr

logger = logging.getLogger(__name__)

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

VALID_LOOP_ATTRIBUTES = {
    'index', 'index0', 'revindex', 'revindex0',
    'first', 'last', 'length', 'depth', 'depth0',
    'previtem', 'nextitem'
}

# =============================================================================
# DATA MODELS
# =============================================================================

class TypeRequirement(Enum):
    """Expected data type for a variable based on how it's used in the template."""
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
    """A loop variable definition mapping the variable name to the array being iterated."""
    var_name: str
    array_path: List[str]

class ReferenceAccessContext(Enum):
    """How a variable is used in the template."""
    OUTPUT = "output"
    CONDITION = "condition"
    ITERATION = "iteration"

@dataclass
class TemplateReference:
    """A resolved variable access or output marker in the template.

    Attributes:
        var_path: Variable path as a list (e.g., ["user", "name"] for user.name).
              Empty list [] indicates output without schema variable access (constants, loop.index).
        context: How this is used - OUTPUT (displayed), CONDITION (if statement),
                 or ITERATION (for loop).
        line_no: Line number in template for error reporting.
    """
    var_path: List[str]
    context: ReferenceAccessContext
    line_no: str = '?'

@dataclass
class ReferenceWithDeps:
    """A variable access with its control flow dependencies."""
    access: TemplateReference
    control_dependencies: List[TemplateReference]

class LoopStack:
    """Manages loop variable scoping during template traversal."""

    def __init__(self):
        self._frames: List[LoopDefinition] = []

    def push_loop_var(self, var_name: str, array_path: List[str]) -> None:
        """Push a new loop variable (may shadow existing ones with the same name)."""
        self._frames.append(LoopDefinition(var_name, array_path))

    def pop_loop_var(self) -> None:
        """Pop the most recent loop variable."""
        self._frames.pop()

    def resolve_variable(self, var_name: str) -> Optional[List[str]]:
        """Resolve a variable name to its array path, or None if not a loop variable."""
        # Search backwards over the loop stack
        for frame in reversed(self._frames):
            if frame.var_name == var_name:
                # Append "*" to indicate we're accessing an element of the array
                # For example: if "item" refers to "products", this returns ["products", "*"]
                # meaning we're accessing products[*] (an element of the products array)
                return frame.array_path + ["*"]
        return None  # Not a loop variable

    def is_in_loop(self) -> bool:
        """Check if a variable is defined in any loop."""
        return len(self._frames) > 0

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

        # We only include variables in the schema if they are *rendered* in the output
        # or are required to evaluate control structures (like loops/if) that *lead* to output.
        # Variables used in conditions or loops that produce no output are discarded.
        # This ensures the inferred schema reflects only what affects the rendered result.
        relevant_accesses = cls._extract_output_dependencies(all_accesses)

        # Build the final schema tree
        return cls._build_schema_tree(relevant_accesses)

    def filter_used_expressions(self, exprs: List[Union[ColumnExpr, AliasExpr]]) -> List[Union[ColumnExpr, AliasExpr]]:
        """Filters expressions to only those used in the template, validating all template variables are defined.

        Args:
            exprs: List of expression objects to filter

        Returns:
            List of expressions that are actually used in the template, in order of appearance

        Raises:
            ValidationError: If a template variable doesn't have a corresponding expression
        """
        expr_names = {expr.name: expr for expr in exprs}
        available_columns = sorted(expr_names.keys())
        used_exprs = []

        # Validate that all template variables have expressions
        for variable_name in self.variables.keys():
            if variable_name not in expr_names:
                raise ValidationError(
                    f"Template variable '{variable_name}' is not defined. "
                    f"Available columns: {', '.join(available_columns) if available_columns else 'none'}. "
                    f"Either provide a column expression for '{variable_name}' or "
                    f"modify the template to use an available column."
                )
            used_exprs.append(expr_names[variable_name])

        # Warn about unused expressions
        used_variables = set(self.variables.keys())
        for column_name in expr_names.keys():
            if column_name not in used_variables:
                logger.warning(
                    f"Column '{column_name}' is defined but not referenced in the template. "
                    f"To use this column, reference it in the template as {{{{ {column_name} }}}}. "
                    f"To remove this warning, exclude unused columns from the expression list."
                )

        return used_exprs

    def validate_jinja_variable(
        self,
        variable_name: str,
        data_type: DataType
    ) -> None:
        """Recursively validates that the structure and type requirements of a Jinja template variable match the actual column schema.

        This ensures that:
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

            if variable_node.requirement == TypeRequirement.ARRAY:
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

        return ast

    @staticmethod
    def _validate_template(ast: nodes.Node) -> None:
        """Validate that template only contains allowed, safe constructs."""
        def validate_node(node: nodes.Node) -> None:
            line_no = getattr(node, "lineno", "?")

            # Check node type is allowed
            if not isinstance(node, ALLOWED_JINJA_NODES):
                node_type = type(node).__name__
                raise ValidationError(
                    f"Unsupported Jinja template syntax on line {line_no}: "
                    f"'{node_type}' is not allowed. "
                    f"Only basic variables, if statements, and for loops are allowed."
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

            # Recursively validate children
            for child in node.iter_child_nodes():
                validate_node(child)

        validate_node(ast)

    # =============================================================================
    # VARIABLE ACCESS COLLECTION
    # =============================================================================
    @staticmethod
    def _collect_variable_accesses(ast: nodes.Node) -> List[ReferenceWithDeps]:
        """Collect all variable accesses with proper scoping resolution."""
        scope = LoopStack()
        return VariableTree._traverse_and_collect(ast, scope, [])

    @staticmethod
    def _traverse_and_collect(node: nodes.Node, scope: LoopStack, control_context: List[TemplateReference]) -> List[ReferenceWithDeps]:
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
    def _handle_for_loop(node: nodes.For, scope: LoopStack, control_context: List[TemplateReference], line_no: str) -> List[ReferenceWithDeps]:
        """Handle for loop: record iteration and update scope."""
        accesses = []

        # Record the array being iterated over
        array_path = VariableTree._extract_variable_path(node.iter, scope)
        if array_path:
            resolved_array_path = VariableTree._resolve_variable_path(array_path, scope)
            iter_access = TemplateReference(resolved_array_path, ReferenceAccessContext.ITERATION, line_no)

            accesses.append(ReferenceWithDeps(iter_access, control_context))

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

        scope.pop_loop_var()
        return accesses

    @staticmethod
    def _handle_conditional(node: nodes.If, scope: LoopStack, control_context: List[TemplateReference], line_no: str) -> List[ReferenceWithDeps]:
        """Handle if statement: record condition variable."""
        accesses = []
        new_control_context = control_context
        # Record the condition variable
        condition_path = VariableTree._extract_variable_path(node.test, scope)
        if condition_path:
            resolved_condition = VariableTree._resolve_variable_path(condition_path, scope)
            cond_access = TemplateReference(resolved_condition, ReferenceAccessContext.CONDITION, line_no)
            new_control_context = new_control_context + [cond_access]

        # Process if body and else body
        for child in node.iter_child_nodes():
            accesses.extend(VariableTree._traverse_and_collect(child, scope, new_control_context))

        return accesses

    @staticmethod
    def _handle_output(node: nodes.Output, scope: LoopStack, control_context: List[TemplateReference], line_no: str) -> List[ReferenceWithDeps]:
        """Handle output expressions: record variables being displayed."""
        accesses = []

        for output_node in node.nodes:
            # Check if this is actual output (not just an empty template data)
            if isinstance(output_node, nodes.TemplateData) and output_node.data == "":
                continue

            # For any other node type (Name, Getattr, Getitem, etc.)
            # try to extract a variable path
            output_path = VariableTree._extract_variable_path(output_node, scope)
            if output_path:
                resolved_output = VariableTree._resolve_variable_path(output_path, scope)
                output_access = TemplateReference(resolved_output, ReferenceAccessContext.OUTPUT, line_no)
                accesses.append(ReferenceWithDeps(output_access, control_context))
            elif control_context:
                # Special handling for output that doesn't access schema variables
                # Examples: {{ loop.index }}, {{ 42 }}, {{ "constant" }}
                #
                # We create a TemplateReference with empty path as a marker to indicate
                # "output happened here". This ensures control dependencies (the loops/conditions
                # that govern this output) are preserved in the final schema.
                #
                # The empty path [] is filtered out in _build_schema_tree but its control
                # dependencies are kept via _extract_output_dependencies.
                output_marker = TemplateReference(
                    var_path=[],  # Empty path = no schema variable accessed
                    context=ReferenceAccessContext.OUTPUT,
                    line_no=line_no
                )
                accesses.append(ReferenceWithDeps(output_marker, control_context))

        return accesses

    # =============================================================================
    # PATH EXTRACTION & RESOLUTION
    # =============================================================================

    @staticmethod
    def _extract_variable_path(node: nodes.Node, scope: LoopStack) -> Optional[List[str]]:
        """Extract variable path as list of keys/fields from AST node."""
        if isinstance(node, nodes.Name):
            if node.name == "loop" and scope.is_in_loop():
                return None
            else:
                return [node.name]
        elif isinstance(node, nodes.Getattr):
            # Special handling for loop.attribute
            if isinstance(node.node, nodes.Name) and node.node.name == "loop":
                # Are we in a loop context?
                if scope.is_in_loop():
                    # Yes - validate the attribute
                    if node.attr not in VALID_LOOP_ATTRIBUTES:
                        line_no = getattr(node, "lineno", "?")
                        raise ValidationError(
                            f"Invalid loop attribute '{node.attr}' on line {line_no}. "
                            f"Valid attributes are: {', '.join(sorted(VALID_LOOP_ATTRIBUTES))}"
                        )
                    return None  # Valid loop attribute in loop context, ignore
                # else: Not in a loop, treat 'loop' as a regular variable

            base_path = VariableTree._extract_variable_path(node.node, scope)
            if base_path:
                return base_path + [node.attr]
        elif isinstance(node, nodes.Getitem):
            base_path = VariableTree._extract_variable_path(node.node, scope)
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
    # TREE BUILDING
    # =============================================================================

    @staticmethod
    def _extract_output_dependencies(all_accesses: List[ReferenceWithDeps]) -> List[TemplateReference]:
        """Remove all variables that are not required to evaluate the output."""
        outputs = [a.access for a in all_accesses if a.access.context == ReferenceAccessContext.OUTPUT]

        # Flatten all dependencies from outputs
        control_deps: List[TemplateReference] = []
        for a in all_accesses:
            if a.access.context == ReferenceAccessContext.OUTPUT:
                control_deps.extend(a.control_dependencies)

        # Deduplicate
        unique_control_deps = {tuple(dep.var_path): dep for dep in control_deps}.values()

        return outputs + list(unique_control_deps)

    @classmethod
    def _build_schema_tree(cls, accesses: List[TemplateReference]) -> VariableTree:
        """Build the final schema tree from filtered variable accesses."""
        tree = cls()

        for access in accesses:
            # Skip placeholder accesses that were created to preserve control dependencies
            # These have empty paths and represent outputs like {{ loop.index }} or {{ "constant" }}
            # Their purpose was to ensure their control dependencies (loops/conditions) were included
            # in the filtered accesses, but they don't represent actual schema requirements
            if not access.var_path:
                continue
            # Determine type requirement based on context
            leaf_requirement = None
            if access.context == ReferenceAccessContext.ITERATION:
                leaf_requirement = TypeRequirement.ARRAY

            cls._add_path_to_tree(tree, access.var_path, leaf_requirement, access.line_no)

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

        # Walk the path and set parent requirements
        for part in path[1:]:
            if part == "*":
                current.set_requirement(TypeRequirement.ARRAY, line_no)
                key = "*"
            else:
                current.set_requirement(TypeRequirement.STRUCT, line_no)
                key = part

            child = current.get_or_create_child(key)
            current = child

        # Set leaf requirement if provided
        if leaf_requirement:
            current.set_requirement(leaf_requirement, line_no)
            if leaf_requirement == TypeRequirement.ARRAY:
                current.get_or_create_child("*")

def _format_path(path: List[str]) -> str:
    result = []
    for part in path:
        if part == "*":
            result[-1] += "[*]"
        else:
            result.append(part)
    return ".".join(result)
