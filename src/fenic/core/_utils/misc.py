import re
import uuid
from typing import Dict, List, Optional, Tuple

from fenic._constants import SQL_PLACEHOLDER_RE
from fenic.core.error import InternalError


def get_content_hash(content: str) -> str:
    """Generate a short, consistent hash for a string.

    This uses UUIDv5 (namespaced UUID) to generate a deterministic hash
    of the content string, and returns the first 8 characters for brevity.

    Args:
        content: The input string to hash.

    Returns:
        A short string representing the hash of the input.

    Example:
        >>> get_content_hash("hello")
        'aaf4c61d'  # (your output will vary depending on namespace and content)
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, content))[:8]


def generate_unique_arrow_view_name() -> str:
    """Generate a unique temporary view name for an Arrow table.

    This is useful for assigning a one-off name to a view or table when
    working with in-memory or temporary datasets.

    Returns:
        A string representing a unique temporary view name.

    Example:
        >>> generate_unique_arrow_view_name()
        'temp_arrow_view_1a2b3c4d5e6f...'
    """
    return f"temp_arrow_view_{uuid.uuid4().hex}"


def replace_sql_query_placeholders(templated_query: str, template_variable_names: List[str], generated_view_names: Optional[List[str]] = None) -> Tuple[str, Dict[str, str]]:
    """Replace query placeholders with view names.

    Args:
        templated_query: The templated query to replace placeholders in.
        template_variable_names: List of template variable names in order.
        generated_view_names: Optional list of view names to use for the placeholders.
    """
    if generated_view_names is not None:
        if len(generated_view_names) != len(template_variable_names):
            raise InternalError(f"Number of view names ({len(generated_view_names)}) must match number of template variables ({len(template_variable_names)})")
        template_name_to_view_name = dict(zip(template_variable_names, generated_view_names))
    else:
        template_name_to_view_name = {}

    def replace_placeholder(match: re.Match) -> str:
        placeholder = match.group(1)
        if placeholder not in template_name_to_view_name:
            if generated_view_names is None:
                view_name = generate_unique_arrow_view_name()
                template_name_to_view_name[placeholder] = view_name
            else:
                raise InternalError(f"Placeholder '{placeholder}' not found in template_variable_names")
        return template_name_to_view_name[placeholder]

    replaced_sql = SQL_PLACEHOLDER_RE.sub(replace_placeholder, templated_query)
    view_names = [template_name_to_view_name[name] for name in template_variable_names]
    return replaced_sql, view_names
