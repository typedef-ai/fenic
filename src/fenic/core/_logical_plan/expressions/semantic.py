from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Union

from pydantic import BaseModel

from fenic.core._logical_plan.utils import validate_completion_parameters
from fenic.core.types import (
    ClassifyExampleCollection,
    KeyPoints,
    MapExampleCollection,
    Paragraph,
    PredicateExampleCollection,
)

if TYPE_CHECKING:
    from fenic.core._logical_plan import LogicalPlan
import fenic.core._utils.misc as utils
from fenic._inference.model_catalog import (
    ModelProvider,
    model_catalog,
)
from fenic.core._logical_plan.expressions.base import (
    AggregateExpr,
    LogicalExpr,
    SemanticExpr,
    ValidatedDynamicSignature,
    ValidatedSignature,
)
from fenic.core._logical_plan.expressions.basic import ColumnExpr
from fenic.core._logical_plan.signatures.signature_validator import SignatureValidator
from fenic.core._utils.schema import convert_pydantic_type_to_custom_struct_type
from fenic.core.error import ValidationError
from fenic.core.types import (
    DataType,
    EmbeddingType,
)
from fenic.core.types.schema import ColumnField


class SemanticMapExpr(ValidatedDynamicSignature, SemanticExpr):
    function_name = "semantic.map"

    def __init__(
        self,
        instruction: str,
        max_tokens: int,
        temperature: float,
        model_alias: Optional[str] = None,
        response_format: Optional[type[BaseModel]] = None,
        examples: Optional[MapExampleCollection] = None,
    ):
        self.instruction = instruction
        self.exprs = [
            ColumnExpr(parsed_col)
            for parsed_col in utils.parse_instruction(instruction)
        ]
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_alias = model_alias
        self.response_format = response_format
        if not self.exprs:
            raise ValidationError(
                "semantic.map instruction requires at least one templated column."
            )
        self.examples = None
        if examples:
            examples._validate_with_instruction(instruction)
            self.examples = examples

        # Initialize validator for composition-based type validation
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        """Return the validator instance."""
        return self._validator

    def children(self) -> List[LogicalExpr]:
        """Return the child expressions."""
        return self.exprs

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        """Handle signature validation and completion parameter validation."""
        # Common validation for all semantic functions
        self._validate_completion_parameters(plan)
        # Use mixin's implementation with dynamic return type
        return super().to_column_field(plan)

    def _validate_completion_parameters(self, plan: LogicalPlan):
        """Validate completion parameters."""
        validate_completion_parameters(
            self.model_alias,
            plan.session_state.session_config,
            self.temperature,
            self.max_tokens
        )

    def __str__(self):
        instruction_hash = utils.get_content_hash(self.instruction)
        exprs_str = ", ".join(str(expr) for expr in self.exprs)
        return f"semantic.map_{instruction_hash}({exprs_str})"


class SemanticExtractExpr(ValidatedDynamicSignature, SemanticExpr):
    function_name = "semantic.extract"

    def __init__(
        self,
        expr: LogicalExpr,
        schema: type[BaseModel],
        max_tokens: int,
        temperature: float,
        model_alias: Optional[str] = None,
    ):
        self.expr = expr
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_alias = model_alias
        self.schema = schema

        # Initialize validator for composition-based type validation
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        """Return the validator instance."""
        return self._validator

    def children(self) -> List[LogicalExpr]:
        """Return the child expressions."""
        return [self.expr]

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        """Handle signature validation and completion parameter validation."""
        # Common validation for all semantic functions
        self._validate_completion_parameters(plan)
        # Use mixin's implementation with dynamic return type
        return super().to_column_field(plan)

    def __str__(self):
        schema_hash = utils.get_content_hash(str(self.schema))
        expr_str = str(self.expr)
        return f"semantic.extract_{schema_hash}({expr_str})"

    def _validate_completion_parameters(self, plan: LogicalPlan):
        """Validate completion parameters."""
        validate_completion_parameters(
            self.model_alias,
            plan.session_state.session_config,
            self.temperature,
            self.max_tokens
        )

    def _infer_dynamic_return_type(self, arg_types: List[DataType], plan: LogicalPlan) -> DataType:
        """Return StructType based on the schema."""
        return convert_pydantic_type_to_custom_struct_type(self.schema)


class SemanticPredExpr(ValidatedSignature, SemanticExpr):
    function_name = "semantic.predicate"

    def __init__(
        self,
        instruction: str,
        temperature: float,
        model_alias: Optional[str] = None,
        examples: Optional[PredicateExampleCollection] = None,
    ):
        self.instruction = instruction
        self.exprs = [
            ColumnExpr(parsed_col)
            for parsed_col in utils.parse_instruction(instruction)
        ]
        if not self.exprs:
            raise ValueError(
                "semantic.predicate instruction requires at least one templated column."
            )
        self.examples = None
        if examples:
            examples._validate_with_instruction(instruction)
            self.examples = examples
        self.temperature = temperature
        self.model_alias = model_alias

        # Initialize validator for composition-based type validation
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        """Return the validator instance."""
        return self._validator

    def children(self) -> List[LogicalExpr]:
        """Return the child expressions."""
        return self.exprs

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        """Handle signature validation and completion parameter validation."""
        # Common validation for all semantic functions
        self._validate_completion_parameters(plan)
        # Use mixin's implementation
        return super().to_column_field(plan)

    def __str__(self):
        instruction_hash = utils.get_content_hash(self.instruction)
        exprs_str = ", ".join(str(expr) for expr in self.exprs)
        return f"semantic.predicate_{instruction_hash}({exprs_str})"

    def _validate_completion_parameters(self, plan: LogicalPlan):
        """Validate completion parameters (no max_tokens for predicate)."""
        validate_completion_parameters(self.model_alias, plan.session_state.session_config, self.temperature)


class SemanticReduceExpr(ValidatedSignature, SemanticExpr, AggregateExpr):
    function_name = "semantic.reduce"

    def __init__(
        self,
        instruction: str,
        max_tokens: int,
        temperature: float,
        model_alias: Optional[str] = None,
    ):
        self.instruction = instruction
        self.exprs = [
            ColumnExpr(parsed_col)
            for parsed_col in utils.parse_instruction(instruction)
        ]
        if not self.exprs:
            raise ValueError(
                "semantic.reduce instruction requires at least one templated column."
            )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_alias = model_alias

        # Initialize validator for composition-based type validation
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        """Return the validator instance."""
        return self._validator

    def children(self) -> List[LogicalExpr]:
        """Return the child expressions."""
        return self.exprs

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        """Handle signature validation and completion parameter validation."""
        # Common validation for all semantic functions
        self._validate_completion_parameters(plan)
        # Use mixin's implementation
        return super().to_column_field(plan)

    def _validate_completion_parameters(self, plan: LogicalPlan):
        """Validate completion parameters."""
        validate_completion_parameters(
            self.model_alias,
            plan.session_state.session_config,
            self.temperature,
            self.max_tokens
        )

    def __str__(self):
        instruction_hash = utils.get_content_hash(self.instruction)
        exprs_str = ", ".join(str(expr) for expr in self.exprs)
        return f"semantic.reduce_{instruction_hash}({exprs_str})"


class SemanticClassifyExpr(ValidatedSignature, SemanticExpr):
    function_name = "semantic.classify"

    def __init__(
        self,
        expr: LogicalExpr,
        labels: List[str] | type[Enum],
        temperature: float,
        examples: Optional[ClassifyExampleCollection] = None,
        model_alias: Optional[str] = None,
    ):
        self.expr = expr
        self.labels = labels
        self.examples = None
        if examples:
            labels_enum = (
                SemanticClassifyExpr.transform_labels_list_into_enum(labels)
                if isinstance(labels, list)
                else labels
            )
            examples._validate_with_enum(labels_enum)
            self.examples = examples
        self.temperature = temperature
        self.model_alias = model_alias

        # Initialize validator for composition-based type validation
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        """Return the validator instance."""
        return self._validator

    def children(self) -> List[LogicalExpr]:
        """Return the child expressions."""
        return [self.expr]

    def __str__(self):
        formatted_labels = "[" + ", ".join(f"'{label}'" for label in self.labels) + "]"
        return f"semantic.classify({self.expr}, {formatted_labels})"

    def _validate_completion_parameters(self, plan: LogicalPlan):
        """Validate completion parameters (called after signature validation)."""
        validate_completion_parameters(self.model_alias, plan.session_state.session_config, self.temperature)

    def _validate_labels(self, plan: LogicalPlan):
        """Validate labels format."""
        if not isinstance(self.labels, List) and not isinstance(
                next(iter(self.labels)).value, str
        ):
            raise TypeError(
                f"Type mismatch: Cannot apply semantic.classify to an enum that is not a string. "
                f"Type: {self.expr.to_column_field(plan).data_type}. "
                f"Only string enums are supported."
            )

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        """Handle signature validation and completion parameter validation."""
        self._validate_labels(plan)
        # Common validation for all semantic functions
        self._validate_completion_parameters(plan)
        # Use mixin's implementation
        return super().to_column_field(plan)

    @staticmethod
    def transform_labels_list_into_enum(labels: list[str]) -> type[Enum]:
        """Transforms a list of labels into an Enum."""
        label_enum_values = []
        for label_str in labels:
            label_enum_values.append(
                (
                    SemanticClassifyExpr.transform_value_into_enum_name(label_str),
                    label_str,
                )
            )

        return Enum("Label", label_enum_values)

    @staticmethod
    def transform_value_into_enum_name(label_value: str) -> str:
        """Transforms a label value into an enum name.

        >>> SemClassifyDataFrame._trasnform_value_into_enum_name("General Inquiry")
         ... "GENERAL_INQUIRY".
        """
        return label_value.upper().replace(" ", "_")


class AnalyzeSentimentExpr(ValidatedSignature, SemanticExpr):
    function_name = "semantic.analyze_sentiment"

    def __init__(
        self,
        expr: LogicalExpr,
        temperature: float,
        model_alias: Optional[str] = None,
    ):
        self.expr = expr
        self.temperature = temperature
        self.model_alias = model_alias

        # Initialize validator for composition-based type validation
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        """Return the validator instance."""
        return self._validator

    def children(self) -> List[LogicalExpr]:
        """Return the child expressions."""
        return [self.expr]

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        """Handle signature validation and completion parameter validation."""
        # Common validation for all semantic functions
        self._validate_completion_parameters(plan)
        # Use mixin's implementation
        return super().to_column_field(plan)

    def __str__(self):
        return f"semantic.analyze_sentiment({self.expr})"

    def _validate_completion_parameters(self, plan: LogicalPlan):
        """Validate completion parameters (no max_tokens for analyze_sentiment)."""
        validate_completion_parameters(self.model_alias, plan.session_state.session_config, self.temperature)


class EmbeddingsExpr(ValidatedDynamicSignature, SemanticExpr):
    """Expression for generating embeddings for a string column.

    This expression creates a new column of embeddings for each value in the input string column.
    The embeddings are a list of floats generated using RM
    """

    function_name = "semantic.embed"

    def __init__(self, expr: LogicalExpr, model_alias: Optional[str] = None):
        self.expr = expr
        self.model_alias = model_alias
        self.dimensions = None

        # Initialize validator for composition-based type validation
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        """Return the validator instance."""
        return self._validator

    def children(self) -> List[LogicalExpr]:
        """Return the child expressions."""
        return [self.expr]

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        """Handle signature validation and completion parameter validation."""
        # Common validation for all semantic functions
        self._validate_completion_parameters(plan)
        # Use mixin's implementation with dynamic return type
        return super().to_column_field(plan)

    def __str__(self) -> str:
        return f"semantic.embed({self.expr}, {self.model_alias})"

    def _infer_dynamic_return_type(self, arg_types: List[DataType], plan: LogicalPlan) -> DataType:
        """Return EmbeddingType with specific dimensions based on model."""
        return_type = self._validate_model_config(plan)
        return return_type

    def _validate_model_config(self, plan: LogicalPlan) -> EmbeddingType:
        """Validate model configuration and return the correct EmbeddingType."""
        session_config = plan.session_state.session_config
        semantic_config = session_config.semantic

        model_alias = self.model_alias or semantic_config.default_embedding_model
        if model_alias not in semantic_config.embedding_models:
            available = ', '.join(semantic_config.embedding_models.keys()) or 'none'
            raise ValidationError(
                f"Embedding model alias '{model_alias}' not found in SessionConfig. "
                f"Available models: {available}"
            )

        model_config = semantic_config.embedding_models[model_alias]
        model_provider = ModelProvider.OPENAI
        model_name = model_config.model_name
        embedding_params = model_catalog.get_embedding_model_parameters(model_provider, model_name)
        self.dimensions = embedding_params.output_dimensions
        return EmbeddingType(
            embedding_model=f"{model_provider.value}/{model_name}",
            dimensions=embedding_params.output_dimensions
        )

    def _validate_completion_parameters(self, plan: LogicalPlan):
        """Embeddings don't use completion parameters."""
        pass


class SemanticSummarizeExpr(ValidatedSignature, SemanticExpr):
    function_name = "semantic.summarize"

    def __init__(
        self,
        expr: LogicalExpr,
        format: Union[KeyPoints, Paragraph],
        temperature: float,
        model_alias: Optional[str] = None
    ):
        self.expr = expr
        self.format = format
        self.temperature = temperature
        self.model_alias = model_alias

        # Initialize validator for composition-based type validation
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        """Return the validator instance."""
        return self._validator

    def children(self) -> List[LogicalExpr]:
        """Return the child expressions."""
        return [self.expr]

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        """Handle signature validation and completion parameter validation."""
        # Common validation for all semantic functions
        self._validate_completion_parameters(plan)
        # Use mixin's implementation
        return super().to_column_field(plan)

    def _validate_completion_parameters(self, plan: LogicalPlan):
        """Validate completion parameters."""
        validate_completion_parameters(self.model_alias, plan.session_state.session_config, self.temperature)

    def __str__(self) -> str:
        return f"semantic.summarize({self.expr})"
