import re

from fastmcp.exceptions import ValidationError

MAX_REGEX_LENGTH = 256
MAX_ALTERNATIONS = 20
MAX_QUANTIFIER_VALUE = 1000


def _is_balanced(s: str, open_char: str, close_char: str) -> bool:
    depth = 0
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == open_char:
            depth += 1
        elif c == close_char:
            depth -= 1
            if depth < 0:
                return False
        i += 1
    return depth == 0


def _strip_slash_delimiters(pattern: str) -> tuple[str, set[str]]:
    """Support /pattern/flags syntax; return (pattern, flags).
    Only recognize i,m,s,x flags; others are rejected later.
    """
    if len(pattern) >= 2 and pattern.startswith("/") and pattern.rfind("/") > 0:
        last = pattern.rfind("/")
        core = pattern[1:last]
        flags = set(pattern[last + 1 :].lower())
        return core, flags
    return pattern, set()


def validate_and_sanitize_regex(user_query: str) -> str:
    r"""Validate user regex and return a sanitized pattern suitable for rlike.

    Rules:
    - Non-empty, max length
    - Balanced (), [], {}
    - Quantifiers {m,n} limited to reasonable bounds
    - Limit number of alternations '|'
    - Disallow backreferences (\\1, \\2, ...)
    - Disallow lookbehind and exotic inline constructs except non-capturing (?:)
    - Strip /pattern/flags form and ignore unsupported flags; case-insensitive handled upstream
    - Strip leading inline flags like (?i) to avoid duplication
    """
    if user_query is None:
        raise ValidationError("Query must not be null")

    query = user_query.strip()
    if not query:
        raise ValidationError("Query must not be empty")

    if len(query) > MAX_REGEX_LENGTH:
        raise ValidationError(f"Regex too long (>{MAX_REGEX_LENGTH} characters)")

    # Support /pattern/flags and capture flags
    query, flags = _strip_slash_delimiters(query)
    unsupported_flags = {f for f in flags if f not in {"i", "m", "s", "x"}}
    if unsupported_flags:
        raise ValidationError(
            f"Unsupported regex flags: {''.join(sorted(unsupported_flags))}"
        )

    # Strip inline flags at start like (?i), (?m), combined, to avoid duplication
    query = re.sub(r"^\(\?[aiLmsux]+\)", "", query)

    # Basic balance checks
    if not _is_balanced(query, "(", ")"):
        raise ValidationError("Unbalanced parentheses")
    if not _is_balanced(query, "[", "]"):
        raise ValidationError("Unbalanced character class brackets")
    if not _is_balanced(query, "{", "}"):
        raise ValidationError("Unbalanced curly braces")

    # Validate quantifiers {m} or {m,n}
    for m, n in re.findall(r"\{\s*(\d+)\s*(?:,\s*(\d*)\s*)?\}", query):
        try:
            m_val = int(m)
            n_val = int(n) if n else m_val
        except ValueError:
            raise ValidationError("Invalid quantifier bounds") from None
        if m_val > MAX_QUANTIFIER_VALUE or n_val > MAX_QUANTIFIER_VALUE:
            raise ValidationError("Quantifier bounds too large")
        if n and n_val < m_val:
            raise ValidationError("Quantifier upper bound less than lower bound")

    # Limit alternations
    if query.count("|") > MAX_ALTERNATIONS:
        raise ValidationError("Too many alternations in regex")

    # Disallow backreferences
    if re.search(r"\\[1-9]\\d*", query):
        raise ValidationError("Backreferences are not supported")

    # Disallow lookbehind and other exotic constructs; allow only non-capturing (?:)
    if re.search(r"\(\?(?!(?:[:]))", query):
        raise ValidationError("Unsupported inline regex construct")

    # Heuristic ReDoS guard: forbid nested quantifiers like (.+)+, (.*)+, (?:.+){2,}
    if re.search(r"\((?:[^()]*[+*])[^()]*\)\s*[+*]", query):
        raise ValidationError("Nested quantifiers are not allowed")
    if re.search(r"(\.\*){2,}|(\.\+){2,}", query):
        raise ValidationError("Excessive greedy wildcards are not allowed")

    # Ensure it compiles in Python as a basic sanity check
    try:
        re.compile(query)
    except re.error as err:
        raise ValidationError(f"Invalid regex syntax: {err}") from None

    return query