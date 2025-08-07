from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, Optional

from pydantic import BaseModel

from fenic.core._utils.schema import convert_pydantic_type_to_custom_struct_type
from fenic.core.types import StructType


@dataclass
class ResolvedModelAlias:
    """A resolved model alias with an optional profile name.

    Attributes:
        name: The name of the model.
        profile: The optional name of a profile configuration to use for the model.
    """
    name: str
    profile: Optional[str] = None


@dataclass
class ResolvedClassDefinition:
    label: str
    description: Optional[str] = None


@dataclass
class ResolvedResponseFormat:
    """Internal representation of a JSON schema for structured output.

    This class wraps a JSON schema dictionary to make it clear that this is
    the resolved format used for model client communication, as opposed to
    the original Pydantic model type.
    """
    schema: Dict[str, Any]
    struct_type: Optional[StructType] = None

    @classmethod
    def from_pydantic_model(
        cls,
        model: type[BaseModel],
    ) -> "ResolvedResponseFormat":
        """Create a ResolvedResponseFormat from a Pydantic model."""
        return cls(schema=model.model_json_schema(), struct_type=convert_pydantic_type_to_custom_struct_type(model))

    # Backwards compatibility with earlier API name
    @classmethod
    def from_pydantic_model_with_descriptions(
        cls, model: type[BaseModel]
    ) -> "ResolvedResponseFormat":
        return cls.from_pydantic_model(model)


    def __eq__(self, other: "ResolvedResponseFormat") -> bool:
        if not isinstance(other, ResolvedResponseFormat):
            return False
        return self.schema_fingerprint() == other.schema_fingerprint()

    def __hash__(self) -> int:
        return hash(self.schema_fingerprint())

    # === Helpers for schema normalization and provider payloads ===
    def _normalized_schema(self) -> Dict[str, Any]:
        """Return a deep-copied, normalized JSON Schema suitable for provider APIs.

        - Ensures additionalProperties: false on every object (incl. nested, $defs, items)
        - Strips volatile metadata keys (title, $id, $schema) for stable comparison
        """
        def deep_copy(obj: Any) -> Any:
            return json.loads(json.dumps(obj))

        def strip_metadata(s: Dict[str, Any]) -> None:
            for k in ("title", "$id", "$schema"):
                if k in s:
                    del s[k]

        def ensure_no_additional_props(s: Dict[str, Any]) -> None:
            t = s.get("type")
            if t == "object":
                # enforce additionalProperties: false
                s.setdefault("additionalProperties", False)
                props = s.get("properties", {})
                if isinstance(props, dict):
                    for prop_schema in props.values():
                        if isinstance(prop_schema, dict):
                            ensure_no_additional_props(prop_schema)
            if t == "array":
                items = s.get("items")
                if isinstance(items, dict):
                    ensure_no_additional_props(items)
            # Traverse composition and defs
            for key in ("allOf", "anyOf", "oneOf"):
                if key in s and isinstance(s[key], list):
                    for sub in s[key]:
                        if isinstance(sub, dict):
                            ensure_no_additional_props(sub)
            for defs_key in ("$defs", "definitions"):
                defs = s.get(defs_key)
                if isinstance(defs, dict):
                    for sub in defs.values():
                        if isinstance(sub, dict):
                            ensure_no_additional_props(sub)

        normalized = deep_copy(self.schema)
        if isinstance(normalized, dict):
            strip_metadata(normalized)
            ensure_no_additional_props(normalized)
        return normalized

    @cached_property
    def schema_fingerprint(self) -> str:
        """Stable string fingerprint for equality and hashing."""
        return json.dumps(self._normalized_schema(), sort_keys=True, separators=(",", ":"))

    def to_openai_response_format(self, name: str = "fenic_response") -> Dict[str, Any]:
        """Build OpenAI parse API response_format payload from the normalized schema."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "schema": self._normalized_schema_for_openai(),
                "strict": True,
            },
        }

    # Internal: stricter normalization for OpenAI parse API
    def _normalized_schema_for_openai(self) -> Dict[str, Any]:
        """Return normalized schema with OpenAI parse expectations.

        - additionalProperties: false for all objects
        - required lists include every property key
        - properties not originally required are made nullable (allow null)
        """
        def deep_copy(obj: Any) -> Any:
            return json.loads(json.dumps(obj))

        def make_nullable(schema_node: Dict[str, Any]) -> Dict[str, Any]:
            # If already allows null, return as-is
            if "type" in schema_node:
                t = schema_node["type"]
                if isinstance(t, list):
                    if "null" in t:
                        return schema_node
                    return {**schema_node, "type": t + ["null"]}
                elif isinstance(t, str):
                    if t == "null":
                        return schema_node
                    new_node = deep_copy(schema_node)
                    new_node["type"] = [t, "null"]
                    return new_node
            if "anyOf" in schema_node and isinstance(schema_node["anyOf"], list):
                anyof = schema_node["anyOf"]
                if any(isinstance(s, dict) and s.get("type") == "null" for s in anyof):
                    return schema_node
                return {"anyOf": [deep_copy(schema_node), {"type": "null"}]}
            if "$ref" in schema_node:
                return {"anyOf": [deep_copy(schema_node), {"type": "null"}]}
            # Fallback: wrap
            return {"anyOf": [deep_copy(schema_node), {"type": "null"}]}

        def ensure_openai_strict(node: Any) -> Any:
            if not isinstance(node, dict):
                return node
            t = node.get("type")
            # Enforce object rules
            if t == "object":
                node.setdefault("additionalProperties", False)
                props = node.get("properties", {})
                if isinstance(props, dict):
                    # Record original required
                    original_required = set(node.get("required", []))
                    # Recurse into properties
                    for k, v in list(props.items()):
                        props[k] = ensure_openai_strict(v)
                    all_prop_keys = list(props.keys())
                    # Set required to all keys
                    node["required"] = all_prop_keys
                    # Make previously optional properties nullable
                    for k in all_prop_keys:
                        if k not in original_required:
                            props[k] = make_nullable(props[k])

            elif t == "array":
                items = node.get("items")
                if isinstance(items, dict):
                    node["items"] = ensure_openai_strict(items)

            # Traverse composition and defs
            for key in ("allOf", "anyOf", "oneOf"):
                if key in node and isinstance(node[key], list):
                    node[key] = [ensure_openai_strict(s) for s in node[key]]
            for defs_key in ("$defs", "definitions"):
                defs = node.get(defs_key)
                if isinstance(defs, dict):
                    for dk, dv in list(defs.items()):
                        defs[dk] = ensure_openai_strict(dv)
            return node

        base = self._normalized_schema()
        return ensure_openai_strict(base)
