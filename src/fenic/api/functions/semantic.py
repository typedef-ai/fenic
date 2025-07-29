"""Semantic functions for Fenic DataFrames - LLM-based operations."""

from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, validate_call

from fenic.api.column import Column, ColumnOrName
from fenic.core._logical_plan.expressions import (
    AnalyzeSentimentExpr,
    EmbeddingsExpr,
    ResolvedClassDefinition,
    SemanticClassifyExpr,
    SemanticExtractExpr,
    SemanticMapExpr,
    SemanticPredExpr,
    SemanticReduceExpr,
    SemanticSummarizeExpr,
)
from fenic.core._utils.structured_outputs import (
    OutputFormatValidationError,
    validate_output_format,
)
from fenic.core.error import ValidationError
from fenic.core.types import (
    ClassDefinition,
    ClassifyExampleCollection,
    KeyPoints,
    MapExampleCollection,
    Paragraph,
    PredicateExampleCollection,
)


@validate_call(config=ConfigDict(arbitrary_types_allowed=True, strict=True))
def map(
        instruction: str,
        examples: Optional[MapExampleCollection] = None,
        schema: Optional[type[BaseModel]] = None,
        model_alias: Optional[str] = None,
        temperature: float = 0,
        max_output_tokens: int = 512,
) -> Column:
    """Applies a natural language instruction to one or more text columns, enabling rich summarization and generation tasks.

    Args:
        instruction: A string containing the semantic.map prompt.
            The instruction must include placeholders in curly braces that reference one or more column names.
            These placeholders will be replaced with actual column values during prompt construction during
            query execution.
        examples: Optional collection of examples to guide the semantic mapping operation.
            Each example should demonstrate the expected input and output for the mapping.
            The examples should be created using MapExampleCollection.create_example(),
            providing instruction variables and their expected answers.
        schema: Optional Pydantic model that defines the output structure with descriptions for each field.
        model_alias: Optional alias for the language model to use for the mapping. If None, will use the language model configured as the default.
        temperature: Optional temperature parameter for the language model. If None, will use the default temperature (0.0).
        max_output_tokens: Optional parameter to constrain the model to generate at most this many tokens. If None, fenic will calculate the expected max
            tokens, based on the model's context length and other operator-specific parameters.

    Returns:
        Column: A column expression representing the semantic mapping operation.

    Raises:
        ValueError: If the instruction is not a string.

    Example: Mapping without examples
        ```python
        semantic.map("Given the product name: {name} and its description: {details}, generate a compelling one-line description suitable for a product catalog.", examples)
        ```

    Example: Mapping with few-shot examples
        ```python
        examples = MapExampleCollection()
        examples.create_example(MapExample(
            input={"name": "GlowMate", "details": "A rechargeable bedside lamp with adjustable color temperatures, touch controls, and a sleek minimalist design."},
            output="The modern touch-controlled lamp for better sleep and style."
        ))
        examples.create_example(MapExample(
            input={"name": "AquaPure", "details": "A compact water filter that attaches to your faucet, removes over 99% of contaminants, and improves taste instantly."},
            output="Clean, great-tasting water straight from your tap."
        ))
        semantic.map("Given the product name: {name} and its description: {details}, generate a compelling one-line description suitable for a product catalog.", examples)
        ```
    """
    if schema:
        try:
            validate_output_format(schema)
        except OutputFormatValidationError as e:
            raise ValidationError("Invalid response schema") from e

    return Column._from_logical_expr(
        SemanticMapExpr(
            instruction,
            examples=examples,
            max_tokens=max_output_tokens,
            model_alias=model_alias,
            temperature=temperature,
            response_format=schema,
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def extract(
        column: ColumnOrName,
        schema: type[BaseModel],
        max_output_tokens: int = 1024,
        temperature: float = 0,
        model_alias: Optional[str] = None,
) -> Column:
    """Extracts structured information from unstructured text using a provided Pydantic model schema.

    This function applies an instruction-driven extraction process to text columns, returning
    structured data based on the fields and descriptions provided. Useful for pulling out key entities,
    facts, or labels from documents.

    The schema must be a valid Pydantic model type with supported field types. These include:

    - Primitive types: `str`, `int`, `float`, `bool`
    - Optional fields: `Optional[T]` where `T` is a supported type
    - Lists: `List[T]` where `T` is a supported type
    - Literals: `Literal[...`] (for enum-like constraints)
    - Nested Pydantic models (recursive schemas are supported, but must be JSON-serializable and acyclic)

    Unsupported types (e.g., unions, custom classes, runtime circular references, or complex generics) will raise errors at runtime.

    Args:
        column: Column containing text to extract from.
        schema: A Pydantic model type that defines the output structure with descriptions for each field.
        model_alias: Optional alias for the language model to use for the extraction. If None, will use the language model configured as the default.
        temperature: Optional temperature parameter for the language model. If None, will use the default temperature (0.0).
        max_output_tokens: Optional parameter to constrain the model to generate at most this many tokens. If None, fenic will calculate the expected max
            tokens, based on the model's context length and other operator-specific parameters.

    Returns:
        Column: A new column with structured values (a struct) based on the provided schema.

    Example: Extracting knowledge graph triples and named entities from text
        ```python
        class Triple(BaseModel):
            subject: str = Field(description="The subject of the triple")
            predicate: str = Field(description="The predicate or relation")
            object: str = Field(description="The object of the triple")

        class KGResult(BaseModel):
            triples: List[Triple] = Field(description="List of extracted knowledge graph triples")
            entities: list[str] = Field(description="Flat list of all detected named entities")

        df.select(semantic.extract("blurb", KGResult))
        ```
    """
    try:
        validate_output_format(schema)
    except OutputFormatValidationError as e:
        raise ValidationError("Invalid extraction schema") from e

    return Column._from_logical_expr(
        SemanticExtractExpr(
            Column._from_col_or_name(column)._logical_expr,
            max_tokens=max_output_tokens,
            temperature=temperature,
            model_alias=model_alias,
            schema=schema,
        )
    )


@validate_call(config=ConfigDict(arbitrary_types_allowed=True, strict=True))
def predicate(
        instruction: str,
        examples: Optional[PredicateExampleCollection] = None,
        model_alias: Optional[str] = None,
        temperature: float = 0,
) -> Column:
    """Applies a natural language predicate to one or more string columns, returning a boolean result.

    This is useful for filtering rows based on user-defined criteria expressed in natural language.

    Args:
        instruction: A string containing the semantic.predicate prompt.
            The instruction must include placeholders in curly braces that reference one or more column names.
            These placeholders will be replaced with actual column values during prompt construction during
            query execution.
        examples: Optional collection of examples to guide the semantic predicate operation.
            Each example should demonstrate the expected boolean output for different inputs.
            The examples should be created using PredicateExampleCollection.create_example(),
            providing instruction variables and their expected boolean answers.
        model_alias: Optional alias for the language model to use for the mapping. If None, will use the language model configured as the default.
        temperature: Optional temperature parameter for the language model. If None, will use the default temperature (0.0).

    Returns:
        Column: A column expression that returns a boolean value after applying the natural language predicate.

    Raises:
        ValueError: If the instruction is not a string.

    Example: Identifying product descriptions that mention wireless capability
        ```python
        semantic.predicate("Does the product description: {product_description} mention that the item is wireless?")
        ```

    Example: Filtering support tickets that describe a billing issue
        ```python
        semantic.predicate("Does this support message: {ticket_text} describe a billing issue?")
        ```

    Example: Filtering support tickets that describe a billing issue with examples
        ```python
        examples = PredicateExampleCollection()
        examples.create_example(PredicateExample(
            input={"ticket_text": "I was charged twice for my subscription and need help."},
            output=True))
        examples.create_example(PredicateExample(
            input={"ticket_text": "How do I reset my password?"},
            output=False))
        semantic.predicate("Does this support ticket describe a billing issue? {ticket_text}", examples)
        ```
    """
    return Column._from_logical_expr(
        SemanticPredExpr(
            instruction,
            examples=examples,
            model_alias=model_alias,
            temperature=temperature,
        )
    )


@validate_call(config=ConfigDict(strict=True))
def reduce(
        instruction: str,
        model_alias: Optional[str] = None,
        temperature: float = 0,
        max_output_tokens: int = 512,
) -> Column:
    """Aggregate function: reduces a set of strings across columns into a single string using a natural language instruction.

    Args:
        instruction: A string containing the semantic.reduce prompt.
            The instruction can include placeholders in curly braces that reference column names.
            These placeholders will be replaced with actual column values during prompt construction during
            query execution.
        model_alias: Optional alias for the language model to use for the mapping. If None, will use the language model configured as the default.
        temperature: Optional temperature parameter for the language model. If None, will use the default temperature (0.0).
        max_output_tokens: Optional parameter to constrain the model to generate at most this many tokens. If None, fenic will calculate the expected max
            tokens, based on the model's context length and other operator-specific parameters.

    Returns:
        Column: A column expression representing the semantic reduction operation.

    Raises:
        ValueError: If the instruction is not a string.

    Example: Summarizing documents using their titles and bodies
        ```python
        semantic.reduce("Summarize these documents using each document's title: {title} and body: {body}.")
        ```
    """
    return Column._from_logical_expr(
        SemanticReduceExpr(
            instruction,
            max_tokens=max_output_tokens,
            model_alias=model_alias,
            temperature=temperature,
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def classify(
        column: ColumnOrName,
        classes: Union[List[str], List[ClassDefinition]],
        examples: Optional[ClassifyExampleCollection] = None,
        model_alias: Optional[str] = None,
        temperature: float = 0,
) -> Column:
    """Classifies a string column into one of the provided classes.

    This is useful for tagging incoming documents with predefined categories.

    Args:
        column: Column or column name containing text to classify.
        classes: List of class labels or ClassDefinition objects defining the available classes. Use ClassDefinition objects to provide descriptions for the classes.
        examples: Optional collection of example classifications to guide the model.
            Examples should be created using ClassifyExampleCollection.create_example(),
            with instruction variables mapped to their expected classifications.
        model_alias: Optional alias for the language model to use for the mapping. If None, will use the language model configured as the default.
        temperature: Optional temperature parameter for the language model. If None, will use the default temperature (0.0).

    Returns:
        Column: Expression containing the classification results.

    Raises:
        ValueError: If column is invalid or classes is empty or has duplicate labels.

    Example: Categorizing incoming support requests
        ```python
        # Categorize incoming support requests
        semantic.classify("message", ["Account Access", "Billing Issue", "Technical Problem"])
        ```

    Example: Categorizing incoming support requests using ClassDefinition objects
        ```python
        # Categorize incoming support requests
        semantic.classify("message", [
            ClassDefinition(label="Account Access", description="General questions, feature requests, or non-technical assistance"),
            ClassDefinition(label="Billing Issue", description="Questions about charges, payments, subscriptions, or account billing"),
            ClassDefinition(label="Technical Problem", description="Problems with product functionality, bugs, or technical difficulties")
        ])
        ```

    Example: Categorizing incoming support requests with ClassDefinition objects and examples
        ```python
        examples = ClassifyExampleCollection()
        class_definitions = [
            ClassDefinition(label="Account Access", description="General questions, feature requests, or non-technical assistance"),
            ClassDefinition(label="Billing Issue", description="Questions about charges, payments, subscriptions, or account billing"),
            ClassDefinition(label="Technical Problem", description="Problems with product functionality, bugs, or technical difficulties")
        ]
        examples.create_example(ClassifyExample(
            input="I can't reset my password or access my account.",
            output="Account Access"))
        examples.create_example(ClassifyExample(
            input="You charged me twice for the same month.",
            output="Billing Issue"))
        semantic.classify("message", class_definitions, examples)
        ```
    """
    if len(classes) < 2:
        raise ValidationError(
            "The `classes` list must contain at least two ClassDefinition objects. "
            "You provided only one. Classification requires at least two possible labels."
        )

    # Validate unique labels
    if isinstance(classes[0], ClassDefinition):
        classes = [ResolvedClassDefinition(label=class_def.label, description=class_def.description) for class_def in classes]
    else:
        classes = [ResolvedClassDefinition(label=class_def, description=None) for class_def in classes]

    labels = [class_def.label for class_def in classes]
    duplicates = {label for label in labels if labels.count(label) > 1}
    if duplicates:
        raise ValidationError(
            f"Class labels must be unique. The following duplicate label(s) were found: {sorted(duplicates)}"
        )

    return Column._from_logical_expr(
        SemanticClassifyExpr(
            Column._from_col_or_name(column)._logical_expr,
            classes,
            examples=examples,
            model_alias=model_alias,
            temperature=temperature,
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def analyze_sentiment(
        column: ColumnOrName,
        model_alias: Optional[str] = None,
        temperature: float = 0,
) -> Column:
    """Analyzes the sentiment of a string column. Returns one of 'positive', 'negative', or 'neutral'.

    Args:
        column: Column or column name containing text for sentiment analysis.
        model_alias: Optional alias for the language model to use for the mapping. If None, will use the language model configured as the default.
        temperature: Optional temperature parameter for the language model. If None, will use the default temperature (0.0).

    Returns:
        Column: Expression containing sentiment results ('positive', 'negative', or 'neutral').

    Raises:
        ValueError: If column is invalid or cannot be resolved.

    Example: Analyzing the sentiment of a user comment
        ```python
        semantic.analyze_sentiment(col('user_comment'))
        ```
    """
    return Column._from_logical_expr(
        AnalyzeSentimentExpr(
            Column._from_col_or_name(column)._logical_expr,
            model_alias=model_alias,
            temperature=temperature,
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def embed(
    column: ColumnOrName,
    model_alias: Optional[str] = None,
) -> Column:
    """Generate embeddings for the specified string column.

    Args:
        column: Column or column name containing the values to generate embeddings for.
        model_alias: Optional alias for the embedding model to use for the mapping.
            If None, will use the embedding model configured as the default.


    Returns:
        A Column expression that represents the embeddings for each value in the input column

    Raises:
        TypeError: If the input column is not a string column.

    Example: Generate embeddings for a text column
        ```python
        df.select(semantic.embed(col("text_column")).alias("text_embeddings"))
        ```
    """
    return Column._from_logical_expr(
        EmbeddingsExpr(Column._from_col_or_name(column)._logical_expr, model_alias=model_alias)
    )

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def summarize(
    column: ColumnOrName,
    format: Union[KeyPoints, Paragraph, None] = None,
    temperature: float = 0,
    model_alias: Optional[str] = None
) -> Column:
    """Summarizes strings from a column.

    Args:
        column: Column or column name containing text for summarization
        format: Format of the summary to generate. Can be either KeyPoints or Paragraph. If None, will default to Paragraph with a maximum of 120 words.
        temperature: Optional temperature parameter for the language model. If None, will use the default temperature (0.0).
        model_alias: Optional alias for the language model to use for the summarization. If None, will use the language model configured as the default.

    Returns:
        Column: Expression containing the summarized string
    Raises:
        ValueError: If column is invalid or cannot be resolved.

    Example:
        >>> semantic.summarize(col('user_comment')).
    """
    if format is None:
        format = Paragraph()
    return Column._from_logical_expr(
        SemanticSummarizeExpr(Column._from_col_or_name(column)._logical_expr, format, temperature, model_alias=model_alias)
    )
