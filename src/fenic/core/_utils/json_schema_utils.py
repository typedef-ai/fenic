from __future__ import annotations

import copy
from typing import Any, Dict


def unwrap_optional_union(node: Dict[str, Any]) -> Dict[str, Any]:
    """If node is an anyOf/oneOf with a null branch, return the first non-null branch.

    Otherwise return node unchanged.
    """
    for key in ("anyOf", "oneOf"):
        alts = node.get(key)
        if isinstance(alts, list):
            for alt in alts:
                if isinstance(alt, dict) and alt.get("type") != "null":
                    return alt
    return node

def to_strict_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Return a strict JSON Schema suitable for a Model Provider.

    All Model Providers have different requirements for supported features and formats of JSON schemas.
    This function is a best-effort attempt to generate a JSON schema that is strict enough for the most inflexible model provider:
    OpenAI. Other model provider implementations may need to apply additional transformations to the schema before passing it to the model provider.

    Rules (adapted from OpenAI's internal logic):
    - Set additionalProperties: false for all objects (when not present)
    - Set required to all keys in properties
    - Recurse into properties, items, anyOf, allOf (flatten allOf of length 1)
    - Strip default when it's None
    - If a node has $ref alongside other keys, inline the ref target and re-ensure strictness

    This function deep-copies the input once and then mutates the copy in-place. This performs
    all the operations that would have been performed if a Pydantic Model were to be passed to the
    LLM Model Provider.
    """
    root = copy.deepcopy(schema)
    def has_more_than_n_keys(obj: Dict[str, Any], n: int) -> bool:
        count = 0
        for _ in obj.keys():
            count += 1
            if count > n:
                return True
        return False

    def resolve_ref(root_schema: Dict[str, Any], ref: str) -> Any:
        if not isinstance(ref, str) or not ref.startswith("#/"):
            raise ValueError(f"Unexpected $ref format {ref!r}; Does not start with #/")
        path = ref[2:].split("/")
        resolved: Any = root_schema
        for key in path:
            resolved = resolved[key]
            if not isinstance(resolved, dict):
                raise ValueError(f"Encountered non-dict while resolving {ref}: {resolved}")
        return resolved

    def ensure(node: Any, path: tuple[str, ...]) -> Dict[str, Any]:
        if not isinstance(node, dict):
            raise TypeError(f"Expected dict at path={path}, got {type(node)}")

        # Recurse into $defs/definitions first
        defs = node.get("$defs")
        if isinstance(defs, dict):
            for k, v in list(defs.items()):
                defs[k] = ensure(v, (*path, "$defs", k))

        definitions = node.get("definitions")
        if isinstance(definitions, dict):
            for k, v in list(definitions.items()):
                definitions[k] = ensure(v, (*path, "definitions", k))

        # Objects: apply additionalProperties and required
        typ = node.get("type")
        if typ == "object" and "additionalProperties" not in node:
            node["additionalProperties"] = False

        properties = node.get("properties")
        if isinstance(properties, dict):
            node["required"] = [k for k in properties.keys()]
            for k, v in list(properties.items()):
                properties[k] = ensure(v, (*path, "properties", k))

        # Arrays: recurse into items
        items = node.get("items")
        if isinstance(items, dict):
            node["items"] = ensure(items, (*path, "items"))

        # anyOf/oneOf/allOf
        any_of = node.get("anyOf")
        if isinstance(any_of, list):
            node["anyOf"] = [ensure(v, (*path, "anyOf", str(i))) for i, v in enumerate(any_of)]

        all_of = node.get("allOf")
        if isinstance(all_of, list):
            if len(all_of) == 1:
                # Flatten single-entry allOf
                strict_child = ensure(all_of[0], (*path, "allOf", "0"))
                node.pop("allOf", None)
                # Merge strict_child into node
                for k, v in strict_child.items():
                    node[k] = v
            else:
                node["allOf"] = [ensure(v, (*path, "allOf", str(i))) for i, v in enumerate(all_of)]

        # Remove default if it's explicitly None
        if node.get("default", object()) is None:
            node.pop("default", None)

        # Inline $ref when combined with other keys
        ref = node.get("$ref")
        if isinstance(ref, str) and has_more_than_n_keys(node, 1):
            resolved = resolve_ref(root, ref)
            if not isinstance(resolved, dict):
                raise ValueError(f"$ref {ref} did not resolve to a dict: {resolved}")
            node.update({**resolved, **node})
            node.pop("$ref", None)
            # Re-ensure after expansion
            return ensure(node, path)

        return node

    return ensure(root, ())
