import re
import uuid


def parse_instruction(text: str) -> list[str]:
    """Extract variable names from instruction text that are wrapped in single curly braces.

    This function uses regex to find all variables in the format {variable_name} that are not
    escaped by double braces (e.g., {{variable_name}}).

    Args:
        text: The instruction text containing variables in curly braces.

    Returns:
        A list of variable names found in the instruction text.

    Example:
        >>> parse_instruction("Classify the sentiment of {text} from {source}")
        ['text', 'source']
        >>> parse_instruction("Use {{escaped}} and {not_escaped}")
        ['not_escaped']
    """
    pattern = r"(?<!\{)\{(?!\{)(.*?)(?<!\})\}(?!\})"
    return re.findall(pattern, text)


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
