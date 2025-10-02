import re
import uuid


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


def generate_unique_view_name() -> str:
    """Generate a unique temporary view name for an Arrow table.

    This is useful for assigning a one-off name to a view or table when
    working with in-memory or temporary datasets.

    Returns:
        A string representing a unique temporary view name.

    Example:
        >>> generate_unique_view_name()
        'temp_view_1a2b3c4d5e6f...'
    """
    return f"temp_view_{uuid.uuid4().hex}"

def to_snake_case(name: str) -> str:
    result = name
    return "_".join(
        re.sub(
            "([A-Z][a-z]+)",
            r" \1",
            re.sub("([A-Z]+)", r" \1", result.replace("-", " ")),
        ).split()
    ).lower()
