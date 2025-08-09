from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, Optional, Union

from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from jsonschema.protocols import Validator
from jsonschema.validators import validator_for
from pydantic import BaseModel

from fenic.core._utils.json_schema_utils import (
    to_strict_json_schema,
    unwrap_optional_union,
)
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
    schema_validator: Validator
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
        strict_schema = to_strict_json_schema(schema)
        validator = cls._create_validator(schema)
        prompt_schema_definition = convert_pydantic_model_to_key_descriptions(model)
        struct_type = convert_pydantic_type_to_custom_struct_type(model) if generate_struct_type else None
        return cls(
            schema=schema,
            strict_schema=strict_schema,
            schema_validator=validator,
            prompt_schema_definition=prompt_schema_definition,
            struct_type=struct_type,
        )

    @classmethod
    def _create_validator(cls, schema: Dict[str, Any]) -> Validator:
        validator_cls = validator_for(schema)
        validator_cls.check_schema(schema)
        return validator_cls(schema)

    def __eq__(self, other: "ResolvedResponseFormat") -> bool:
        if not isinstance(other, ResolvedResponseFormat):
            return False
        return self.schema_fingerprint == other.schema_fingerprint

    def __hash__(self) -> int:
        return hash(self.schema_fingerprint)

    @cached_property
    def schema_fingerprint(self) -> str:
        """Stable string fingerprint for equality and hashing."""
        return json.dumps(self.schema, sort_keys=True, separators=(",", ":"))

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
        try:
            if json_resp is None:
                return None
            if isinstance(json_resp, str):
                json_resp = json.loads(json_resp)
            self.schema_validator.validate(json_resp)
            # Apply defaults from schema to ensure missing optionals become nulls and shapes are consistent
            return self._apply_defaults(self.schema, json_resp)
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

    def _apply_defaults(
        self, schema: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply default values from schema into data recursively (for objects and arrays)."""
        def walk(obj_schema: Any, obj: Any) -> Any:
            if not isinstance(obj_schema, dict):
                return obj
            obj_schema = unwrap_optional_union(obj_schema)

            # Object case
            if isinstance(obj_schema.get("properties"), dict):
                props: dict[str, Any] = obj_schema["properties"]
                required: set[str] = set(obj_schema.get("required", []))
                result: dict[str, Any] = {} if not isinstance(obj, dict) else dict(obj)
                for key, subschema in props.items():
                    if key in result:
                        result[key] = walk(subschema, result[key])
                    else:
                        if "default" in subschema:
                            result[key] = subschema["default"]
                        elif key not in required:
                            result[key] = None
                return result

            # Array case
            items_schema = obj_schema.get("items")
            if isinstance(items_schema, dict) and isinstance(obj, list):
                return [walk(items_schema, it) for it in obj]

            # Primitive or union-of-primitives – nothing to apply
            return obj

        return walk(schema, data)
