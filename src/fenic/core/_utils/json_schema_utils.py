from __future__ import annotations

import json
from typing import Any, Dict


def deep_copy_json(obj: Any) -> Any:
    """Deep copy using JSON round-trip to avoid shared references in nested dicts/lists."""
    return json.loads(json.dumps(obj))


def make_nullable(node: Dict[str, Any]) -> Dict[str, Any]:
    """Return a schema node that allows null in addition to its existing type.

    - If node["type"] is a string, convert to [type, "null"]
    - If node["type"] is a list, append "null" if not present
    - If node has anyOf/oneOf without null, wrap with an additional null alternative
    - If node is a $ref or lacks type/combiners, wrap in anyOf with null
    """
    t = node.get("type")
    if isinstance(t, list):
        if "null" in t:
            return node
        node["type"] = t + ["null"]
        return node
    if isinstance(t, str):
        if t == "null":
            return node
        node["type"] = [t, "null"]
        return node
    for key in ("anyOf", "oneOf"):
        alts = node.get(key)
        if isinstance(alts, list):
            if any(isinstance(s, dict) and s.get("type") == "null" for s in alts):
                return node
            wrapped = {key: [node, {"type": "null"}]}
            # Preserve top-level descriptive metadata on wrapper
            for meta_key in ("description", "title", "examples"):
                if meta_key in node:
                    wrapped[meta_key] = node[meta_key]
            return wrapped
    if "$ref" in node:
        wrapped = {"anyOf": [node, {"type": "null"}]}
        for meta_key in ("description", "title", "examples"):
            if meta_key in node:
                wrapped[meta_key] = node[meta_key]
        return wrapped
    wrapped = {"anyOf": [node, {"type": "null"}]}
    for meta_key in ("description", "title", "examples"):
        if meta_key in node:
            wrapped[meta_key] = node[meta_key]
    return wrapped


def unwrap_optional_union(node: Dict[str, Any]) -> Dict[str, Any]:
    """If node is an anyOf/oneOf with a null branch, return a copy of the non-null branch
    while preserving top-level descriptive metadata (description/title/examples) from the wrapper.
    Otherwise return node unchanged.
    """
    for key in ("anyOf", "oneOf"):
        alts = node.get(key)
        if isinstance(alts, list):
            non_null = next((s for s in alts if not (isinstance(s, dict) and s.get("type") == "null")), None)
            if isinstance(non_null, dict):
                # Preserve wrapper metadata on the non-null branch if not present
                for meta_key in ("description", "title", "examples"):
                    if meta_key in node and meta_key not in non_null:
                        non_null[meta_key] = node[meta_key]
                return non_null
    return node


def strip_defaults_in_place(schema: Dict[str, Any]) -> None:
    """Remove all default keys recursively in-place."""
    def walk(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("default", None)
            for v in node.values():
                if isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(node, list):
            for x in node:
                if isinstance(x, (dict, list)):
                    walk(x)
    walk(schema)


def strip_schema_metadata_in_place(schema: Dict[str, Any]) -> None:
    """Remove non-semantic metadata like $id and $schema recursively in-place."""
    def walk(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("$id", None)
            node.pop("$schema", None)
            for v in node.values():
                if isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(node, list):
            for x in node:
                if isinstance(x, (dict, list)):
                    walk(x)
    walk(schema)


def strip_defaults(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-copy and remove all default keys recursively."""
    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            node = dict(node)
            node.pop("default", None)
            for k, v in list(node.items()):
                if isinstance(v, (dict, list)):
                    node[k] = walk(v)
            return node
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(deep_copy_json(schema))


def strip_schema_metadata(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Remove non-semantic metadata like $id and $schema recursively (deep copy)."""
    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            node = dict(node)
            node.pop("$id", None)
            node.pop("$schema", None)
            for k, v in list(node.items()):
                if isinstance(v, (dict, list)):
                    node[k] = walk(v)
            return node
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(deep_copy_json(schema))


