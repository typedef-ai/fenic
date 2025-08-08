from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, Optional, Union

from jsonschema import validate
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import BaseModel

from fenic.core._utils.schema import convert_pydantic_type_to_custom_struct_type
from fenic.core._utils.structured_outputs import (
    convert_pydantic_model_to_key_descriptions,
)
from fenic.core.types import StructType

logger = logging.getLogger(__name__)


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

    Attributes:
        schema: The JSON schema dictionary.
        struct_type: The StructType of the model.
            Only generated as required. This is only needed if the Operator returns the struct type itself (e.g. semantic.map, semantic.extract).
            In cases like semantic.classify, the struct type is not returned, only the class labels.
        prompt_schema_definition: The description of the schema that will be used in the prompt. Only generated if struct_type is generated.

    """
    schema: Dict[str, Any]
    struct_type: Optional[StructType] = None
    prompt_schema_definition: Optional[str] = None

    @classmethod
    def from_pydantic_model(
        cls,
        model: type[BaseModel],
        generate_struct_type: bool = True,
    ) -> "ResolvedResponseFormat":
        """Create a ResolvedResponseFormat from a Pydantic model."""
        return cls(
            schema=model.model_json_schema(),
            struct_type=convert_pydantic_type_to_custom_struct_type(model) if generate_struct_type else None,
            prompt_schema_definition=convert_pydantic_model_to_key_descriptions(model) if generate_struct_type else None,
        )


    def __eq__(self, other: "ResolvedResponseFormat") -> bool:
        if not isinstance(other, ResolvedResponseFormat):
            return False
        return self.schema_fingerprint == other.schema_fingerprint

    def __hash__(self) -> int:
        return hash(self.schema_fingerprint)

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

    # Internal: stricter normalization for OpenAI parse API. This is required -- previously
    # we were passing the pydantic model directly to the SDK, where all of this was handled internally.
    # Now we are passing the schema directly to the SDK, so we need to normalize it to the OpenAI parse API expectations.
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

    def validate_structured_response(
        self,
        json_resp: Union[str, dict[str, Any]],
    ):
        """Validate and parse a structured JSON response using ResolvedResponseFormat's json schema."""
        if isinstance(json_resp, str):
            json_resp = json.loads(json_resp)
        validate(instance=json_resp, schema=self.schema)

    def parse_structured_response(
        self,
        json_resp: Optional[Union[str, dict[str, Any]]],
        operator_name: str
    ) -> Optional[Dict[str, Any]]:
        """Validate and parse a structured JSON response using ResolvedResponseFormat's json schema.

        Args:
            json_resp: The JSON response string from the LLM (can be None)
            operator_name: Name of the operation (for logging purposes)

        Returns:
            Validated dictionary representation of the model, or None if validation fails
        """
        if json_resp is None:
            return None
        if isinstance(json_resp, str):
            json_resp = json.loads(json_resp)

        try:
            self.validate_structured_response(json_resp)
            return json_resp
        except json.JSONDecodeError as e:
            logger.warning(
                f"Invalid JSON in model output: {json_resp} for {operator_name}: {e}",
                exc_info=True,
            )
            return None
        except JsonSchemaValidationError as e:
            logger.warning(
                f"JSON schema validation failed for {operator_name}: {e.message} at {e.path}",
                exc_info=True,
            )
            return None
        except Exception as e:
            logger.warning(
                f"Unexpected error validating model output: {json_resp} for {operator_name}: {e}",
                exc_info=True,
            )
            return None
