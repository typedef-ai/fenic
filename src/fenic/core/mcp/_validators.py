import re
from typing import Dict, List, Optional, Protocol, Union, runtime_checkable

from fenic.core.error import (
    ValidationError,
)
from fenic.core.types.datatypes import DataType, StringType

MAX_REGEX_LENGTH = 256
MAX_ALTERNATIONS = 20
MAX_QUANTIFIER_VALUE = 1000

@runtime_checkable
class ParamValidator(Protocol):

    def name(self) -> str:
        """The name of the validator."""
        ...

    def data_types(self) -> List[DataType]:
        """The data types that the validator operates on."""
        ...

    def validate(self, value: Union[str, int, float, bool, list, dict]) -> bool:
        """Validate an argument value.

        Args:
            value: The value to validate.

        Raises:
            ValidationError: If the value did not pass validation.
        """
        ...


class RegexValidator(ParamValidator):
    def name(self) -> str:
        return "regex"
    
    def data_types(self) -> List[DataType]:
        return [StringType]

    def validate(self, user_query: str):
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
        if not user_query:
            return
        query = user_query.strip()
        if not query and len(query) < len(user_query):
            raise ValidationError("Query must not be empty")

        if len(query) > MAX_REGEX_LENGTH:
            raise ValidationError(f"Regex too long (>{MAX_REGEX_LENGTH} characters)")

        # Support /pattern/flags and capture flags
        query, flags = self._strip_slash_delimiters(query)
        unsupported_flags = {f for f in flags if f not in {"i", "m", "s", "x"}}
        if unsupported_flags:
            raise ValidationError(
                f"Unsupported regex flags: {''.join(sorted(unsupported_flags))}"
            )

        # Strip inline flags at start like (?i), (?m), combined, to avoid duplication
        query = re.sub(r"^\(\?[aiLmsux]+\)", "", query)

        # Basic balance checks
        if not self._is_balanced(query, "(", ")"):
            raise ValidationError("Unbalanced parentheses")
        if not self._is_balanced(query, "[", "]"):
            raise ValidationError("Unbalanced character class brackets")
        if not self._is_balanced(query, "{", "}"):
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

        # Disallow backreferences (simple and robust detection)
        if any(f"\\{d}" in query for d in "123456789"):
            raise ValidationError("Backreferences are not supported")

        # Disallow lookbehind and other exotic constructs; allow only non-capturing (?:)
        if re.search(r"\(\?(?!(?:[:]))", query):
            raise ValidationError("Unsupported inline regex construct")

        # Heuristic ReDoS guard: forbid nested quantifiers like (.+)+, (.*)+, (?:.+){2,}
        if re.search(r"\((?:[^()]*[+*])[^()]*\)\s*[+*]", query):
            raise ValidationError("Nested quantifiers are not allowed")
        if re.search(r"(\.\*){2,}|(\.\+){2,}", query):
            raise ValidationError("Excessive greedy wildcards are not allowed")
        # Also forbid group with .+ or .* followed by bounded quantifier {m} or {m,n}
        if re.search(r"\([^)]*\.[+*][^)]*\)\s*\{\s*\d+(?:\s*,\s*\d+)?\s*\}", query):
            raise ValidationError("Nested bounded quantifiers are not allowed")
        # Explicitly catch common non-capturing form
        if re.search(r"(?:\(\?:\.\+\)|\(\?:\.\*\))\s*\{", query):
            raise ValidationError("Nested bounded quantifiers are not allowed")

        # Reject invalid quantifier forms with multiple commas like {1,2,3}
        if re.search(r"\{\s*\d+\s*,\s*\d+\s*,", query):
            raise ValidationError("Invalid quantifier syntax")

        # Ensure it compiles in Python as a basic sanity check
        try:
            re.compile(query)
        except re.error as err:
            raise ValidationError(f"Invalid regex syntax: {err}") from None

        return

    def _is_balanced(self, s: str, open_char: str, close_char: str) -> bool:
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


    def _strip_slash_delimiters(self, pattern: str) -> tuple[str, set[str]]:
        """Support /pattern/flags syntax; return (pattern, flags).

        Only recognize i,m,s,x flags; others are rejected later.
        """
        if len(pattern) >= 2 and pattern.startswith("/") and pattern.rfind("/") > 0:
            last = pattern.rfind("/")
            core = pattern[1:last]
            flags = set(pattern[last + 1 :].lower())
            return core, flags
        return pattern, set()


# -- Registry for reusable ParamValidators --
_PARAM_VALIDATOR_REGISTRY: Dict[str, ParamValidator] = {}


def register_param_validator(name: str, validator: ParamValidator):
    """Register a ParamValidator by name for later reference in ToolParam.

    Args:
        name: Unique name for the validator.
        validator: Instance of ParamValidator.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("Validator name must be a non-empty string")
    if name in _PARAM_VALIDATOR_REGISTRY:
        raise ValueError(f"Validator '{name}' is already registered")
    _PARAM_VALIDATOR_REGISTRY[name] = validator


def get_param_validator(name: str) -> ParamValidator:
    """Lookup a registered ParamValidator by name."""
    try:
        return _PARAM_VALIDATOR_REGISTRY[name]
    except KeyError as err:
        raise KeyError(f"No ParamValidator registered under name '{name}'") from err


def maybe_get_param_validator(name: Optional[str]) -> Optional[ParamValidator]:
    if name is None:
        return None
    return get_param_validator(name)


# Pre-register common validators
register_param_validator("regex", RegexValidator())