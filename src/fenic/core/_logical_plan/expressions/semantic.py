from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

from pydantic import BaseModel

from fenic.core._logical_plan.resolved_types import (
    ResolvedClassDefinition,
    ResolvedModelAlias,
)
from fenic.core._logical_plan.utils import validate_completion_parameters
from fenic.core._resolved_session_config import (
    ResolvedGoogleModelConfig,
)
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
from fenic.core._inference.model_catalog import (
    model_catalog,
)
from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.expressions.base import (
    AggregateExpr,
    LogicalExpr,
    SemanticExpr,
    ValidatedDynamicSignature,
    ValidatedSignature,
)
from fenic.core._logical_plan.expressions.basic import ColumnExpr
from fenic.core._logical_plan.jinja_validation import VariableTree
from fenic.core._logical_plan.signatures.signature_validator import SignatureValidator
from fenic.core._utils.schema import convert_pydantic_type_to_custom_struct_type
from fenic.core.error import InvalidExampleCollectionError, ValidationError
from fenic.core.types import (
    BooleanType,
    DataType,
    EmbeddingType,
    StringType,
)
from fenic.core.types.schema import ColumnField
from fenic.core.types.semantic import ModelAlias


class SemanticMapExpr(ValidatedDynamicSignature, SemanticExpr):
    function_name = "semantic.map"

    def __init__(
        self,
        jinja_template: str,
        exprs: List[Union[ColumnExpr, LogicalExpr]],
        max_tokens: int,
        temperature: float,
        model_alias: Optional[ResolvedModelAlias] = None,
        response_format: Optional[type[BaseModel]] = None,
        examples: Optional[MapExampleCollection] = None,
    ):
        self.template = jinja_template
        self.variable_tree = VariableTree.from_jinja_template(jinja_template)
        self.exprs = self.variable_tree.filter_used_expressions(exprs)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_alias = model_alias
        self.response_format = response_format
        self.examples = None
        if examples:
            self._validate_example_response_format(examples)
            examples._validate_input_column_names([expr.name for expr in self.exprs])
            self.examples = examples
        self.struct_type = convert_pydantic_type_to_custom_struct_type(response_format) if response_format else None

    def children(self) -> List[LogicalExpr]:
        """Return the child expressions."""
        return self.exprs

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        """Handle signature validation and completion parameter validation."""
        # Common validation for all semantic functions
        self._validate_completion_parameters(plan, session_state)
        for expr in self.exprs:
            data_type = expr.to_column_field(plan, session_state).data_type
            self.variable_tree.validate_jinja_variable(expr.name, data_type)

        return ColumnField(
            name=str(self),
            data_type=self.struct_type if self.struct_type else StringType,
        )

    def _validate_example_response_format(self, example_collection: MapExampleCollection):
        """Validate that all examples have outputs matching the expected response format."""
        for example in example_collection.examples:
            if self.response_format is None:
                if not isinstance(example.output, str):
                    raise InvalidExampleCollectionError(
                        "When no schema is provided to `semantic.map`, "
                        "all example outputs must be strings."
                    )
            else:
                if not isinstance(example.output, self.response_format):
                    raise InvalidExampleCollectionError(
                        f"When a schema BaseModel is provided to `semantic.map`, "
                        f"all example outputs must be instances of {self.response_format}."
                )

    def _validate_completion_parameters(self, plan: LogicalPlan, session_state: BaseSessionState):
        """Validate completion parameters."""
        validate_completion_parameters(
            self.model_alias,
            session_state.session_config,
            self.temperature,
            self.max_tokens,
        )

    def __str__(self):
        instruction_hash = utils.get_content_hash(self.template)
        exprs_str = ", ".join(str(expr) for expr in self.exprs)
        return f"semantic.map_{instruction_hash}({exprs_str})"

    def _eq_specific(self, other: SemanticMapExpr) -> bool:
        return self.max_tokens == other.max_tokens and self.temperature == other.temperature and self.model_alias == other.model_alias and self.response_format is other.response_format and self.examples == other.examples and self.template == other.template


class SemanticExtractExpr(ValidatedDynamicSignature, SemanticExpr):
    function_name = "semantic.extract"

    def __init__(
        self,
        expr: LogicalExpr,
        schema: type[BaseModel],
        max_tokens: int,
        temperature: float,
        model_alias: Optional[ResolvedModelAlias] = None,
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

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        """Handle signature validation and completion parameter validation."""
        # Common validation for all semantic functions
        self._validate_completion_parameters(plan, session_state)
        # Use mixin's implementation with dynamic return type
        return super().to_column_field(plan, session_state)

    def __str__(self):
        schema_hash = utils.get_content_hash(str(self.schema))
        expr_str = str(self.expr)
        return f"semantic.extract_{schema_hash}({expr_str})"

    def _validate_completion_parameters(self, plan: LogicalPlan, session_state: BaseSessionState):
        """Validate completion parameters."""
        validate_completion_parameters(
            self.model_alias,
            session_state.session_config,
            self.temperature,
            self.max_tokens,
        )

    def _infer_dynamic_return_type(
        self, arg_types: List[DataType], plan: LogicalPlan, session_state: BaseSessionState) -> DataType:
        """Return StructType based on the schema."""
        return convert_pydantic_type_to_custom_struct_type(self.schema)

    def _eq_specific(self, other: SemanticExtractExpr) -> bool:
        return self.schema is other.schema and self.max_tokens == other.max_tokens and self.temperature == other.temperature and self.model_alias == other.model_alias


class SemanticPredExpr(ValidatedSignature, SemanticExpr):
    function_name = "semantic.predicate"

    def __init__(
        self,
        jinja_template: str,
        exprs: List[Union[ColumnExpr, LogicalExpr]],
        temperature: float,
        model_alias: Optional[ResolvedModelAlias] = None,
        examples: Optional[PredicateExampleCollection] = None,
    ):
        self.template = jinja_template
        self.variable_tree = VariableTree.from_jinja_template(jinja_template)
        self.exprs = self.variable_tree.filter_used_expressions(exprs)
        self.examples = None
        if examples:
            examples._validate_input_column_names([expr.name for expr in self.exprs])
            self.examples = examples
        self.temperature = temperature
        self.model_alias = model_alias

    def children(self) -> List[LogicalExpr]:
        """Return the child expressions."""
        return self.exprs

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        """Handle signature validation and completion parameter validation."""
        # Common validation for all semantic functions
        self._validate_completion_parameters(plan, session_state)
        for expr in self.exprs:
            data_type = expr.to_column_field(plan, session_state).data_type
            self.variable_tree.validate_jinja_variable(expr.name, data_type)

        return ColumnField(
            name=str(self),
            data_type=BooleanType,
        )

    def __str__(self):
        instruction_hash = utils.get_content_hash(self.template)
        exprs_str = ", ".join(str(expr) for expr in self.exprs)
        return f"semantic.predicate_{instruction_hash}({exprs_str})"

    def _validate_completion_parameters(self, plan: LogicalPlan, session_state: BaseSessionState):
        """Validate completion parameters (no max_tokens for predicate)."""
        validate_completion_parameters(
            self.model_alias, session_state.session_config, self.temperature
        )

    def _eq_specific(self, other: SemanticPredExpr) -> bool:
        return self.temperature == other.temperature and self.model_alias == other.model_alias and self.examples == other.examples and self.template == other.template


class SemanticReduceExpr(ValidatedSignature, SemanticExpr, AggregateExpr):
    function_name = "semantic.reduce"

    def __init__(
        self,
        instruction: str,
        expr: LogicalExpr,
        max_tokens: int,
        temperature: float,
        model_alias: Optional[ResolvedModelAlias] = None,
    ):
        self.instruction = instruction
        self.expr = expr
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
        return [self.expr]

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        """Handle signature validation and completion parameter validation."""
        # Common validation for all semantic functions
        self._validate_completion_parameters(plan, session_state)
        # Use mixin's implementation
        return super().to_column_field(plan, session_state)

    def _validate_completion_parameters(self, plan: LogicalPlan, session_state: BaseSessionState):
        """Validate completion parameters."""
        validate_completion_parameters(
            self.model_alias,
            session_state.session_config,
            self.temperature,
            self.max_tokens,
        )

    def __str__(self):
        instruction_hash = utils.get_content_hash(self.instruction)
        return f"semantic.reduce_{instruction_hash}({self.expr})"

    def _eq_specific(self, other: SemanticReduceExpr) -> bool:
        return self.temperature == other.temperature and self.model_alias == other.model_alias and self.instruction == other.instruction and self.max_tokens == other.max_tokens


class SemanticClassifyExpr(ValidatedSignature, SemanticExpr):
    function_name = "semantic.classify"

    def __init__(
        self,
        expr: LogicalExpr,
        classes: List[ResolvedClassDefinition],
        temperature: float,
        examples: Optional[ClassifyExampleCollection] = None,
        model_alias: Optional[ResolvedModelAlias] = None,
    ):
        self.expr = expr
        self.classes = classes
        self.examples = None

        if examples:
            # Validate examples against class labels
            valid_labels = {class_def.label for class_def in classes}
            examples._validate_with_labels(valid_labels)
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
        labels_str = ", ".join(f"'{class_def.label}'" for class_def in self.classes)
        return f"semantic.classify({self.expr}, [{labels_str}])"

    def _validate_completion_parameters(self, plan: LogicalPlan, session_state: BaseSessionState):
        """Validate completion parameters (called after signature validation)."""
        validate_completion_parameters(
            self.model_alias, session_state.session_config, self.temperature
        )

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        """Handle signature validation and completion parameter validation."""
        # Common validation for all semantic functions
        self._validate_completion_parameters(plan, session_state)
        # Use mixin's implementation
        return super().to_column_field(plan, session_state)

    def _eq_specific(self, other: SemanticClassifyExpr) -> bool:
        return self.temperature == other.temperature and self.model_alias == other.model_alias and self.classes == other.classes and self.examples == other.examples


class AnalyzeSentimentExpr(ValidatedSignature, SemanticExpr):
    function_name = "semantic.analyze_sentiment"

    def __init__(
        self,
        expr: LogicalExpr,
        temperature: float,
        model_alias: Optional[ResolvedModelAlias] = None,
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

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        """Handle signature validation and completion parameter validation."""
        # Common validation for all semantic functions
        self._validate_completion_parameters(plan, session_state)
        # Use mixin's implementation
        return super().to_column_field(plan, session_state)

    def __str__(self):
        return f"semantic.analyze_sentiment({self.expr})"

    def _validate_completion_parameters(self, plan: LogicalPlan, session_state: BaseSessionState):
        """Validate completion parameters (no max_tokens for analyze_sentiment)."""
        validate_completion_parameters(
            self.model_alias, session_state.session_config, self.temperature
        )

    def _eq_specific(self, other: AnalyzeSentimentExpr) -> bool:
        return self.temperature == other.temperature and self.model_alias == other.model_alias

class EmbeddingsExpr(ValidatedDynamicSignature, SemanticExpr):
    """Expression for generating embeddings for a string column.

    This expression creates a new column of embeddings for each value in the input string column.
    The embeddings are a list of floats generated using RM
    """

    function_name = "semantic.embed"

    def __init__(self, expr: LogicalExpr, model_alias: Optional[ResolvedModelAlias] = None):
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

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        """Handle signature validation and completion parameter validation."""
        # Common validation for all semantic functions
        self._validate_completion_parameters(plan, session_state)
        # Use mixin's implementation with dynamic return type
        return super().to_column_field(plan, session_state)

    def __str__(self) -> str:
        return f"semantic.embed({self.expr}, {self.model_alias})"

    def _infer_dynamic_return_type(
        self, arg_types: List[DataType], plan: LogicalPlan
    , session_state: BaseSessionState) -> DataType:
        """Return EmbeddingType with specific dimensions based on model."""
        return_type = self._get_embedding_type_from_config(plan, session_state)
        return return_type

    def _get_embedding_type_from_config(self, plan: LogicalPlan, session_state: BaseSessionState) -> EmbeddingType:
        """Validate model configuration and return the correct EmbeddingType."""
        semantic_config = session_state.session_config.semantic

        # Check if any embedding models are configured
        if not semantic_config.embedding_models:
            raise ValidationError(
                "No embedding models configured. This operation requires embedding models. "
                "Please add embedding_models to your SemanticConfig."
            )
        embedding_model_configs = semantic_config.embedding_models
        model_alias = self.model_alias or ResolvedModelAlias(
            name=embedding_model_configs.default_model
        )
        if model_alias.name not in embedding_model_configs.model_configs:
            available = ", ".join(embedding_model_configs.model_configs.keys())
            raise ValidationError(
                f"Embedding model alias '{model_alias}' not found in SessionConfig. "
                f"Available models: {available}"
            )

        model_config = embedding_model_configs.model_configs[model_alias.name]
        model_provider = model_config.model_provider
        model_name = model_config.model_name
        embedding_params = model_catalog.get_embedding_model_parameters(
            model_provider, model_name
        )
        if isinstance(model_config, ResolvedGoogleModelConfig):
            profile_name = model_alias.profile if model_alias.profile else model_config.default_profile
            if profile_name not in model_config.profiles:
                raise ValidationError(
                    f"Embedding model preset '{model_alias.profile}' not found in SessionConfig."
                    f" Available presets for {model_alias.name}: {', '.join(model_config.profiles)}"
                )
            profile = model_config.profiles[profile_name]

            self.dimensions = (
                profile.embedding_dimensionality
                if profile.embedding_dimensionality
                else embedding_params.default_dimensions
            )
        else:
            self.dimensions = embedding_params.default_dimensions

        return EmbeddingType(
            embedding_model=f"{model_provider.value}/{model_name}",
            dimensions=self.dimensions,
        )

    def _validate_completion_parameters(self, plan: LogicalPlan, session_state: BaseSessionState):
        """Embeddings don't use completion parameters."""
        pass

    def _eq_specific(self, other: EmbeddingsExpr) -> bool:
        return self.model_alias == other.model_alias


class SemanticSummarizeExpr(ValidatedSignature, SemanticExpr):
    function_name = "semantic.summarize"

    def __init__(
        self,
        expr: LogicalExpr,
        format: Union[KeyPoints, Paragraph],
        temperature: float,
        model_alias: Optional[ModelAlias] = None,
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

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        """Handle signature validation and completion parameter validation."""
        # Common validation for all semantic functions
        self._validate_completion_parameters(plan, session_state)
        # Use mixin's implementation
        return super().to_column_field(plan, session_state)

    def _validate_completion_parameters(self, plan: LogicalPlan, session_state: BaseSessionState):
        """Validate completion parameters."""
        validate_completion_parameters(
            self.model_alias, session_state.session_config, self.temperature
        )

    def __str__(self) -> str:
        return f"semantic.summarize({self.expr})"

    def _eq_specific(self, other: SemanticSummarizeExpr) -> bool:
        return self.temperature == other.temperature and self.model_alias == other.model_alias and self.format == other.format
