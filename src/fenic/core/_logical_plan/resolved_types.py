from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, Optional, Union

from jsonschema import validate
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel, create_model

from fenic.core._utils.schema import convert_pydantic_type_to_custom_struct_type
from fenic.core._utils.json_schema_utils import (
    deep_copy_json,
    make_nullable as util_make_nullable,
    unwrap_optional_union as util_unwrap_optional_union,
    strip_schema_metadata,
)
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
    strict_schema: Dict[str, Any]
    prompt_schema_definition: str
    struct_type: Optional[StructType] = None

    @classmethod
    def from_pydantic_model(
        cls,
        model: type[BaseModel],
        generate_struct_type: bool = True,
    ) -> "ResolvedResponseFormat":
        """Create a ResolvedResponseFormat from a Pydantic model."""
        schema = model.model_json_schema()
        strict_schema = to_strict_json_schema(model)
        prompt_schema_definition = convert_pydantic_model_to_key_descriptions(model)
        struct_type = convert_pydantic_type_to_custom_struct_type(model) if generate_struct_type else None
        return cls(
            schema=schema,
            strict_schema=strict_schema,
            prompt_schema_definition=prompt_schema_definition,
            struct_type=struct_type,
        )


    def __eq__(self, other: "ResolvedResponseFormat") -> bool:
        if not isinstance(other, ResolvedResponseFormat):
            return False
        return self.schema_fingerprint == other.schema_fingerprint

    def __hash__(self) -> int:
        return hash(self.schema_fingerprint)

    # === Helpers for schema normalization and provider payloads ===
    @cached_property
    def canonical_schema(self) -> Dict[str, Any]:
        """Return a minimal canonical JSON Schema for fingerprinting (no $id/$schema)."""
        return strip_schema_metadata(self.schema)

    @cached_property
    def schema_fingerprint(self) -> str:
        """Stable string fingerprint for equality and hashing."""
        return json.dumps(self.canonical_schema, sort_keys=True, separators=(",", ":"))

    # Access canonical schema via the `canonical_schema` property

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
            # Apply defaults from schema to ensure missing optionals become nulls and shapes are consistent
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

    # === Utilities ===
    @staticmethod
    def _add_null_defaults_to_optionals(schema: dict[str, Any]) -> dict[str, Any]:
        """Ensure all optional properties are nullable throughout the schema (recursively)."""

        def make_nullable(node: dict[str, Any]) -> dict[str, Any]:
            return util_make_nullable(node)

        def unwrap_optional(node: dict[str, Any]) -> dict[str, Any]:
            return util_unwrap_optional_union(node)

        def walk(node: Any) -> Any:
            if not isinstance(node, dict):
                return node
            t = node.get("type")
            if t == "object":
                props = node.get("properties", {})
                required = set(node.get("required", []))
                for key, prop_schema in list(props.items()):
                    # Determine if optional by required list
                    is_optional = key not in required
                    # Unwrap optional branch for traversal (if any)
                    base_schema = unwrap_optional(prop_schema)
                    # Recurse into the non-null branch
                    walked_base = walk(base_schema)
                    if is_optional:
                        # Re-wrap to allow null for local validation
                        props[key] = make_nullable(walked_base)
                    else:
                        props[key] = walked_base
                # Recurse into definitions and compositions too
            elif t == "array":
                items = node.get("items")
                if isinstance(items, dict):
                    node["items"] = walk(items)
            for k in ("allOf", "anyOf", "oneOf"):
                if isinstance(node.get(k), list):
                    node[k] = [walk(s) for s in node[k]]
            for defs_key in ("$defs", "definitions"):
                defs = node.get(defs_key)
                if isinstance(defs, dict):
                    for dk, dv in list(defs.items()):
                        defs[dk] = walk(dv)
            return node

        # One deep copy at the top, then mutate in place during traversal
        cp = deep_copy_json(schema)
        return walk(cp)

    @staticmethod
    def _apply_defaults(
        schema: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply default values from schema into data recursively (for objects and arrays)."""
        def walk(obj_schema: Any, obj: Any) -> Any:
            if not isinstance(obj_schema, dict):
                return obj
            schema_type = obj_schema.get("type")
            if schema_type == "object":
                props = obj_schema.get("properties", {})
                required = set(obj_schema.get("required", []))
                result = {} if not isinstance(obj, dict) else dict(obj)
                for key, subschema in props.items():
                    subschema = util_unwrap_optional_union(subschema)
                    if key in result:
                        result[key] = walk(subschema, result[key])
                    else:
                        # If default present, set it; else if optional, set None
                        if "default" in subschema:
                            result[key] = subschema.get("default")
                        elif key not in required:
                            result[key] = None
                return result
            if schema_type == "array":
                items_schema = obj_schema.get("items")
                if isinstance(obj, list) and isinstance(items_schema, dict):
                    return [walk(items_schema, it) for it in obj]
                return obj
            # For union via anyOf/oneOf at non-object levels, leave as-is
            return obj

        # Start from the non-optional root subschema for objects
        # Avoid extra copies; unwrap on the provided schema directly
        root_schema = util_unwrap_optional_union(schema)
        return walk(root_schema, data)
