from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, Optional, Union

from json_schema_to_pydantic import create_model as create_pydantic_model
from pydantic import BaseModel, ValidationError

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
        pydantic_model: The Pydantic Model that defines the resolved format.
        json_schema: The raw JSON schema dictionary from the pydantic model.
        struct_type: The StructType of the model.
            Only generated as required. This is only needed if the Operator returns the struct type itself (e.g. semantic.map, semantic.extract).
            In cases like semantic.classify, the struct type is not returned, only the class labels.
        prompt_schema_definition: The description of the schema that will be used in the prompt. Only generated if struct_type is generated.

    """
    pydantic_model: type[BaseModel]
    json_schema: Dict[str, Any]
    prompt_schema_definition: str
    struct_type: Optional[StructType] = None

    @classmethod
    def from_json_schema(
        cls,
        raw_schema: Dict[str, Any],
        prompt_schema_definition: str,
        struct_type: Optional[StructType] = None,
    ) -> "ResolvedResponseFormat":
        """Create a ResolvedResponseFormat from a Pydantic model."""
        pydantic_model = create_pydantic_model(raw_schema)
        return cls(
            pydantic_model=pydantic_model,
            json_schema=raw_schema,
            prompt_schema_definition=prompt_schema_definition,
            struct_type=struct_type,
        )

    @classmethod
    def from_pydantic_model(
        cls,
        model: type[BaseModel],
        generate_struct_type: bool = True,
    ) -> "ResolvedResponseFormat":
        """Create a ResolvedResponseFormat from a Pydantic model."""
        raw_schema = model.model_json_schema()
        prompt_schema_definition = convert_pydantic_model_to_key_descriptions(model)
        struct_type = convert_pydantic_type_to_custom_struct_type(model) if generate_struct_type else None

        return cls(
            pydantic_model=model,
            json_schema=raw_schema,
            prompt_schema_definition=prompt_schema_definition,
            struct_type=struct_type,
        )

    def __eq__(self, other: "ResolvedResponseFormat") -> bool:
        if not isinstance(other, ResolvedResponseFormat):
            return False
        return self.schema_fingerprint == other.schema_fingerprint

    def __hash__(self) -> int:
        return self.schema_fingerprint_hash

    @cached_property
    def schema_fingerprint_hash(self) -> int:
        return hash(self.schema_fingerprint)

    @cached_property
    def schema_fingerprint(self) -> str:
        """Stable string fingerprint for equality and hashing."""
        return json.dumps(self._drop_non_structural_keys(self.json_schema), sort_keys=True, separators=(",", ":"))

    def validate_structured_response(self, response: Dict[str, Any]) -> None:
        self.pydantic_model.model_validate(response)

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
            return self.pydantic_model.model_validate(json_resp).model_dump()
        except json.JSONDecodeError as e:
            logger.warning(
                f"Invalid JSON in model output: {json_resp} for {operator_name}: {e}",
                exc_info=True,
            )
            return None
        except ValidationError as e:
            logger.warning(
                f"Pydantic validation failed for {json_resp} for {operator_name}: {e}",
                exc_info=True,
            )
            return None
        except Exception as e:
            logger.warning(
                f"Unexpected error validating model output: {json_resp} for {operator_name}: {e}",
                exc_info=True,
            )
            return None

    def _drop_non_structural_keys(self, node: object) -> object:
        if isinstance(node, dict):
            ignore = {"title", "description", "$id"}
            return {k: self._drop_non_structural_keys(v) for k, v in node.items() if k not in ignore}
        if isinstance(node, list):
            return [self._drop_non_structural_keys(v) for v in node]
        return node
