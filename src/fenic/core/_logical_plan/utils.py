from typing import List, Optional, Set, Union

from jinja2 import Environment, nodes
from jinja2.exceptions import TemplateSyntaxError

from fenic._inference.model_catalog import (
    CompletionModelParameters,
    ModelProvider,
    model_catalog,
)
from fenic.core._resolved_session_config import (
    ResolvedGoogleModelConfig,
    ResolvedOpenAIModelConfig,
    ResolvedSessionConfig,
)
from fenic.core.error import ValidationError

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

def validate_completion_parameters(
    model_alias: Optional[str],
    resolved_session_config: ResolvedSessionConfig,
    temperature: float,
    max_tokens: Optional[int] = None,
):
    """Validates that the provided temperature and max_tokens are within the limits allowed by the specified language model.

    If no model alias is provided, the session's default language model is used.

    Parameters:
        model_alias (Optional[str]):
            Alias of the language model to validate. Defaults to the session's
            default if not provided.
        resolved_session_config (ResolvedSessionConfig):
            The resolved session config containing model definitions.
        temperature (float):
            Sampling temperature. Must be within the model's supported range.
        max_tokens (Optional[int]):
            Maximum number of tokens to generate. Must not exceed the model's limit.

    Raises:
        ValidationError: If temperature or max_tokens are out of bounds for the model.
    """
    if model_alias is None:
        model_alias = resolved_session_config.semantic.default_language_model
    if model_alias not in resolved_session_config.semantic.language_models:
        raise ValidationError(
            f"Language model alias '{model_alias}' not found in SessionConfig. "
            f"Available models: {', '.join(resolved_session_config.semantic.language_models.keys()) or 'none'}"
        )
    model_config = resolved_session_config.semantic.language_models[model_alias]
    if isinstance(model_config, ResolvedOpenAIModelConfig):
        model_provider = ModelProvider.OPENAI
    elif isinstance(model_config, ResolvedGoogleModelConfig):
        model_provider = ModelProvider.GOOGLE_GLA
    else:
        model_provider = ModelProvider.ANTHROPIC
    completion_parameters: CompletionModelParameters = model_catalog.get_completion_model_parameters(model_provider, model_config.model_name)
    if max_tokens is not None and max_tokens > completion_parameters.max_output_tokens:
        raise ValidationError(f"[{model_provider.value}:{model_config.model_name}] max_output_tokens must be a positive integer less than or equal to {completion_parameters.max_output_tokens}")
    if temperature is not None and (temperature < 0 or temperature > completion_parameters.max_temperature):
        raise ValidationError(f"[{model_provider.value}:{model_config.model_name}] temperature must be between 0 and {completion_parameters.max_temperature}")

def validate_and_parse_jinja_template(template: str) -> List[str]:
    """Validates that the provided template is a valid Jinja template and extracts the variables used in the template.

    Parameters:
        template (str): The Jinja template to validate and extract variables from.

    Raises:
        ValidationError: If the template is invalid.

    Returns:
        List[str]: A sorted list of top-level variable names used in the template.
    """
    # trunk-ignore(bandit/B701): Templates generate plain text, not HTML, so no risk of XSS. In fact, we may want to allow HTML in the template.
    env = Environment(autoescape=False)
    try:
        ast = env.parse(template)
    except TemplateSyntaxError as e:
        raise ValidationError(f"Jinja template syntax error: {e.message}") from e

    _annotate_parents(ast)

    loop_variables = set()
    for node in ast.find_all(nodes.For):
        if isinstance(node.target, nodes.Name):
            loop_variables.add(node.target.name)

    top_level_vars = set()

    for node in ast.find_all(nodes.Node):
        _validate_node(node, loop_variables)

        if isinstance(node, nodes.Name):
            if getattr(node, 'ctx', None) == 'load' and node.name not in loop_variables and node.name != "loop":
                top_level_vars.add(node.name)

    return sorted(top_level_vars)


def _annotate_parents(node: nodes.Node, parent: Optional[nodes.Node] = None) -> None:
    """Recursively add `parent` references to each AST node since Jinja doesn't have them."""
    node.parent = parent # type: ignore[attr-defined]
    for child in node.iter_child_nodes():
        _annotate_parents(child, parent=node)


def _validate_node(node: nodes.Node, loop_variables: Set[str]) -> None:
    """Validate a single AST node against allowed syntax rules."""
    line_no = getattr(node, 'lineno', '?')

    if not isinstance(node, ALLOWED_JINJA_NODES):
        raise ValidationError(
            f"Unsupported template feature used on line {line_no}. Only conditional logic and for loops over collections are allowed in templates. Use Fenic expressions for data processing within the template instead."
        )

    if isinstance(node, nodes.Name):
        _validate_name_node(node, loop_variables, line_no)
    elif isinstance(node, nodes.Getitem):
        _validate_getitem_node(node, line_no)
    elif isinstance(node, nodes.Const):
        if not isinstance(getattr(node, 'parent', None), nodes.Getitem):
            raise ValidationError(
                f"Jinja template error: Literal values are not allowed directly in expressions (line {line_no}). Use variables instead."
            )

def _validate_name_node(node: nodes.Name, loop_variables: Set[str], line_no: Union[int, str]) -> None:
    """Validate variable names to reject loop vars or special names."""
    if hasattr(node, 'ctx') and node.ctx == 'load':
        if node.name in loop_variables:
            # Check if we're inside the corresponding For node
            parent = node.parent
            while parent and not isinstance(parent, nodes.For):
                parent = getattr(parent, 'parent', None)
            if not isinstance(parent, nodes.For) or parent.target.name == node.name:
                return  # This usage is inside a valid loop

            # If not inside its own loop body, it's invalid
            raise ValidationError(
                f"Jinja template error: Cannot use loop variable '{node.name}' in expressions (line {line_no}). Loop variables are only allowed in loop contexts."
            )
        elif node.name == "loop":
            raise ValidationError(
                f"Jinja template error: The 'loop' variable is not allowed in expressions (line {line_no}). Use regular variables instead."
            )

def _validate_getitem_node(node: nodes.Getitem, line_no: Union[int, str]) -> None:
    """Validate that index access is static and type-safe."""
    if not isinstance(node.arg, nodes.Const):
        raise ValidationError(
            f"Jinja template error: Array/object access must use fixed indices like [0] or ['key'] (line {line_no}). Dynamic indices using variables are not allowed."
        )
    if type(node.arg.value) not in (int, str):
        raise ValidationError(
            f"Jinja template error: Index must be a number or text string (line {line_no}). Example: myarray[0] or myobject['key']"
        )
