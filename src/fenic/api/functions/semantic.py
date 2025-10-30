"""Semantic functions for Fenic DataFrames - LLM-based operations."""

from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, validate_call

from fenic._inference.request_utils import TimeoutParam
from fenic.api.column import Column, ColumnOrName
from fenic.core._logical_plan.expressions import (
    AliasExpr,
    AnalyzeSentimentExpr,
    ColumnExpr,
    EmbeddingsExpr,
    ResolvedClassDefinition,
    SemanticClassifyExpr,
    SemanticExtractExpr,
    SemanticMapExpr,
    SemanticParsePDFExpr,
    SemanticPredExpr,
    SemanticReduceExpr,
    SemanticSummarizeExpr,
)
from fenic.core._logical_plan.resolved_types import ResolvedResponseFormat
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
from fenic.core.types.semantic import ModelAlias, _resolve_model_alias


@validate_call(config=ConfigDict(arbitrary_types_allowed=True, strict=True))
def map(
        prompt: str,
        /,
        *,
        strict: bool = True,
        examples: Optional[MapExampleCollection] = None,
        response_format: Optional[type[BaseModel]] = None,
        model_alias: Optional[Union[str, ModelAlias]] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        request_timeout: TimeoutParam = None,
        **columns: Column,
) -> Column:
    """Applies a generation prompt to one or more columns, enabling rich summarization and generation tasks.

    Args:
        prompt: A Jinja2 template for the generation prompt. References column
            values using {{ column_name }} syntax. Each placeholder is replaced with the
            corresponding value from the current row during execution.
        strict: If True, when any of the provided columns has a None value for a row,
                the entire row's output will be None (template is not rendered).
                If False, None values are handled using Jinja2's null rendering behavior.
                Default is True.
        examples: Optional few-shot examples to guide the model's output format and style.
        response_format: Optional Pydantic model to enforce structured output. Must include descriptions for each field.
        model_alias: Optional language model alias. If None, uses the default model.
        temperature: Language model temperature (default: 0.0).
        max_output_tokens: Maximum tokens to generate (default: 512).
        request_timeout: Optional timeout in seconds for a single LLM request. If None, uses the default timeout (120 seconds).
        **columns: Named column arguments that correspond to template variables.
            Keys must match the variable names used in the template.

    Returns:
        Column: A column expression representing the semantic mapping operation.


    Example: Mapping without examples
        ```python
        fc.semantic.map(
            "Write a compelling one-line description for {{ name }}: {{ details }}",
            name=fc.col("name"),
            details=fc.col("details")
        )
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
        fc.semantic.map(
            "Write a compelling one-line description for {{ name }}: {{ details }}",
            name=fc.col("name"),
            details=fc.col("details"),
            examples=examples
        )
        ```
    """
    if not prompt:
        raise ValidationError("The `prompt` argument to `semantic.map` cannot be empty.")

    if not columns:
        raise ValidationError("`semantic.map` requires at least one named column argument (e.g. `text=col('text')`).")

    if response_format:
        try:
            validate_output_format(response_format)
        except OutputFormatValidationError as e:
            raise ValidationError(f"Invalid response format: {str(e)}") from None

    exprs: List[Union[ColumnExpr, AliasExpr]] = []
    for var_name, column in columns.items():
        if isinstance(column._logical_expr, ColumnExpr) and column._logical_expr.name == var_name:
            exprs.append(column._logical_expr)
        else:
            exprs.append(column.alias(var_name)._logical_expr)

    resolved_model_alias = _resolve_model_alias(model_alias)
    if response_format:
        resolved_response_format = ResolvedResponseFormat.from_pydantic_model(response_format)
    else:
        resolved_response_format = None
    return Column._from_logical_expr(
        SemanticMapExpr(
            prompt,
            strict=strict,
            exprs=exprs,
            max_tokens=max_output_tokens,
            temperature=temperature,
            model_alias=resolved_model_alias,
            response_format=resolved_response_format,
            examples=examples,
            request_timeout=request_timeout,
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def extract(
        column: ColumnOrName,
        response_format: type[BaseModel],
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
        model_alias: Optional[Union[str, ModelAlias]] = None,
        request_timeout: TimeoutParam = None,
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
        response_format: A Pydantic model type that defines the output structure with descriptions for each field.
        model_alias: Optional alias for the language model to use for the extraction. If None, will use the language model configured as the default.
        temperature: Optional temperature parameter for the language model. If None, will use the default temperature (0.0).
        max_output_tokens: Optional parameter to constrain the model to generate at most this many tokens. If None, fenic will calculate the expected max
            tokens, based on the model's context length and other operator-specific parameters.
        request_timeout: Optional timeout in seconds for a single LLM request. If None, uses the default timeout (120 seconds).

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
        validate_output_format(response_format)
    except OutputFormatValidationError as e:
        raise ValidationError(f"Invalid response format: {str(e)}") from None

    resolved_model_alias = _resolve_model_alias(model_alias)
    resolved_response_format = ResolvedResponseFormat.from_pydantic_model(response_format)
    return Column._from_logical_expr(
        SemanticExtractExpr(
            Column._from_col_or_name(column)._logical_expr,
            max_tokens=max_output_tokens,
            temperature=temperature,
            response_format=resolved_response_format,
            model_alias=resolved_model_alias,
            request_timeout=request_timeout,
        )
    )


@validate_call(config=ConfigDict(arbitrary_types_allowed=True, strict=True))
def predicate(
        predicate: str,
        /,
        *,
        strict: bool = True,
        examples: Optional[PredicateExampleCollection] = None,
        model_alias: Optional[Union[str, ModelAlias]] = None,
        temperature: float = 0.0,
        request_timeout: TimeoutParam = None,
        **columns: Column,
) -> Column:
    r"""Applies a boolean predicate to one or more columns, typically used for filtering.

    Args:
        predicate: A Jinja2 template containing a yes/no question or boolean claim.
            Should reference column values using {{ column_name }} syntax. The model will
            evaluate this condition for each row and return True or False.
        strict: If True, when any of the provided columns has a None value for a row,
                the entire row's output will be None (template is not rendered).
                If False, None values are handled using Jinja2's null rendering behavior.
                Default is True.
        examples: Optional few-shot examples showing how to evaluate the predicate.
            Helps ensure consistent True/False decisions.
        model_alias: Optional language model alias. If None, uses the default model.
        temperature: Language model temperature (default: 0.0).
        request_timeout: Optional timeout in seconds for a single LLM request. If None, uses the default timeout (120 seconds).
        **columns: Named column arguments that correspond to template variables.
            Keys must match the variable names used in the template.

    Returns:
        Column: A boolean column expression.

    Example: Filtering product descriptions
        ```python
        wireless_products = df.filter(
            fc.semantic.predicate(
                dedent('''\
                    Product: {{ description }}
                    Is this product wireless or battery-powered?'''),
                description=fc.col("product_description")
            )
        )
        ```

    Example: Filtering support tickets
        ```python
        df = df.with_column(
            "is_urgent",
            fc.semantic.predicate(
                dedent('''\
                    Subject: {{ subject }}
                    Body: {{ body }}
                    This ticket indicates an urgent issue.'''),
                subject=fc.col("ticket_subject"),
                body=fc.col("ticket_body")
            )
        )
        ```

    Example: Filtering with examples
        ```python
        examples = PredicateExampleCollection()
        examples.create_example(PredicateExample(
            input={"ticket": "I was charged twice for my subscription and need help."},
            output=True
        ))
        examples.create_example(PredicateExample(
            input={"ticket": "How do I reset my password?"},
            output=False
        ))
        fc.semantic.predicate(
            dedent('''\
                Ticket: {{ ticket }}
                This ticket is about billing.'''),
            ticket=fc.col("ticket_text"),
            examples=examples
        )
        ```
    """
    if not predicate:
        raise ValidationError("The `predicate` argument to `semantic.predicate` cannot be empty.")

    if not columns:
        raise ValidationError("`semantic.predicate` requires at least one named column argument (e.g. `text=col('text')`).")

    exprs: List[Union[ColumnExpr, AliasExpr]] = []
    for var_name, column in columns.items():
        if isinstance(column._logical_expr, ColumnExpr) and column._logical_expr.name == var_name:
            exprs.append(column._logical_expr)
        else:
            exprs.append(column.alias(var_name)._logical_expr)

    resolved_model_alias = _resolve_model_alias(model_alias)
    return Column._from_logical_expr(
        SemanticPredExpr(
            predicate,
            strict=strict,
            exprs=exprs,
            temperature=temperature,
            model_alias=resolved_model_alias,
            examples=examples,
            request_timeout=request_timeout,
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def reduce(
    prompt: str,
    column: ColumnOrName,
    *,
    group_context: Optional[Dict[str, Column]] = None,
    order_by: List[ColumnOrName] = None,
    model_alias: Optional[Union[str, ModelAlias]] = None,
    temperature: float = 0,
    max_output_tokens: int = 512,
    request_timeout: TimeoutParam = None,
) -> Column:
    """Aggregate function: reduces a set of strings in a column to a single string using a natural language instruction.

    Args:
        prompt: A string containing the semantic.reduce prompt.
            The instruction can optionally include Jinja2 template variables (e.g., {{variable}}) that
            reference columns from the group_context parameter. These will be replaced with
            actual values from the first row of each group during execution.
        column: The column containing documents/strings to reduce.
        group_context: Optional dictionary mapping variable names to columns. These columns
            provide context for each group and can be referenced in the instruction template.
        order_by: Optional list of columns to sort grouped documents by before reduction. Documents are
            processed in ascending order by default if no sort function is provided. Use a sort function
            (e.g., col("date").desc()/fc.desc("date")) for descending order. The order_by columns help
            preserve the temporal/logical sequence of the documents (e.g chunks in a document, speaker turns in a meeting transcript)
            for more coherent summaries.
        model_alias: Optional alias for the language model to use. If None, uses the default model.
        temperature: Temperature parameter for the language model (default: 0.0).
        max_output_tokens: Maximum tokens the model can generate (default: 512).
        request_timeout: Optional timeout in seconds for a single LLM request. If None, uses the default timeout (120 seconds).

    Returns:
        Column: A column expression representing the semantic reduction operation.

    Example: Simple reduction
        ```python
        # Simple reduction
        df.group_by("category").agg(
            semantic.reduce("Summarize the documents", col("document_text"))
        )
        ```

    Example: With group context
        ```python
        df.group_by("department", "region").agg(
            semantic.reduce(
                "Summarize these {{department}} reports from {{region}}",
                col("document_text"),
                group_context={
                    "department": col("department"),
                    "region": col("region")
                }
            )
        )
        ```

    Example: With sorting
        ```python
        df.group_by("category").agg(
            semantic.reduce(
                "Summarize the documents",
                col("document_text"),
                order_by=col("date")
            )
        )
        ```
    """
    if not prompt:
        raise ValidationError("The `prompt` argument to `semantic.reduce` cannot be empty.")

    group_context_exprs: List[Union[ColumnExpr, AliasExpr]] = []
    if group_context:
        for var_name, col in group_context.items():
            if isinstance(col, ColumnExpr) and col.expr.name == var_name:
                group_context_exprs.append(col._logical_expr)
            else:
                group_context_exprs.append(col.alias(var_name)._logical_expr)
    order_by_exprs = []
    if order_by:
        for col in order_by:
            order_by_exprs.append(Column._from_col_or_name(col)._logical_expr)
    resolved_model_alias = _resolve_model_alias(model_alias)
    return Column._from_logical_expr(
        SemanticReduceExpr(
            prompt,
            input_expr=Column._from_col_or_name(column)._logical_expr,
            max_tokens=max_output_tokens,
            temperature=temperature,
            group_context_exprs=group_context_exprs,
            model_alias=resolved_model_alias,
            order_by_exprs=order_by_exprs,
            request_timeout=request_timeout,
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def classify(
    column: ColumnOrName,
    classes: Union[List[str], List[ClassDefinition]],
    examples: Optional[ClassifyExampleCollection] = None,
    model_alias: Optional[Union[str, ModelAlias]] = None,
    temperature: float = 0,
    request_timeout: TimeoutParam = None,
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
        request_timeout: Optional timeout in seconds for a single LLM request. If None, uses the default timeout (120 seconds).

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
        classes = [ResolvedClassDefinition(label=class_def.label, description=class_def.description) for class_def in
                   classes]
    else:
        classes = [ResolvedClassDefinition(label=class_def, description=None) for class_def in classes]

    labels = [class_def.label for class_def in classes]
    duplicates = {label for label in labels if labels.count(label) > 1}
    if duplicates:
        raise ValidationError(
            f"Class labels must be unique. The following duplicate label(s) were found: {sorted(duplicates)}"
        )

    resolved_model_alias = _resolve_model_alias(model_alias)
    return Column._from_logical_expr(
        SemanticClassifyExpr(
            Column._from_col_or_name(column)._logical_expr,
            classes,
            examples=examples,
            model_alias=resolved_model_alias,
            temperature=temperature,
            request_timeout=request_timeout,
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def analyze_sentiment(
    column: ColumnOrName,
    model_alias: Optional[Union[str, ModelAlias]] = None,
    temperature: float = 0,
    request_timeout: TimeoutParam = None,
) -> Column:
    """Analyzes the sentiment of a string column. Returns one of 'positive', 'negative', or 'neutral'.

    Args:
        column: Column or column name containing text for sentiment analysis.
        model_alias: Optional alias for the language model to use for the mapping. If None, will use the language model configured as the default.
        temperature: Optional temperature parameter for the language model. If None, will use the default temperature (0.0).
        request_timeout: Optional timeout in seconds for a single LLM request. If None, uses the default timeout (120 seconds).

    Returns:
        Column: Expression containing sentiment results ('positive', 'negative', or 'neutral').

    Raises:
        ValueError: If column is invalid or cannot be resolved.

    Example: Analyzing the sentiment of a user comment
        ```python
        semantic.analyze_sentiment(col('user_comment'))
        ```
    """
    resolved_model_alias = _resolve_model_alias(model_alias)
    return Column._from_logical_expr(
        AnalyzeSentimentExpr(
            Column._from_col_or_name(column)._logical_expr,
            model_alias=resolved_model_alias,
            temperature=temperature,
            request_timeout=request_timeout,
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def embed(
    column: ColumnOrName,
    model_alias: Optional[Union[str, ModelAlias]] = None,
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
    resolved_model_alias = _resolve_model_alias(model_alias)
    return Column._from_logical_expr(
        EmbeddingsExpr(Column._from_col_or_name(column)._logical_expr, model_alias=resolved_model_alias)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def summarize(
    column: ColumnOrName,
    format: Union[KeyPoints, Paragraph, None] = None,
    temperature: float = 0,
    model_alias: Optional[Union[str, ModelAlias]] = None,
    request_timeout: TimeoutParam = None,
) -> Column:
    """Summarizes strings from a column.

    Args:
        column: Column or column name containing text for summarization
        format: Format of the summary to generate. Can be either KeyPoints or Paragraph. If None, will default to Paragraph with a maximum of 120 words.
        temperature: Optional temperature parameter for the language model. If None, will use the default temperature (0.0).
        model_alias: Optional alias for the language model to use for the summarization. If None, will use the language model configured as the default.
        request_timeout: Optional timeout in seconds for a single LLM request. If None, uses the default timeout (120 seconds).

    Returns:
        Column: Expression containing the summarized string
    Raises:
        ValueError: If column is invalid or cannot be resolved.

    Example:
        >>> semantic.summarize(col('user_comment')).
    """
    if format is None:
        format = Paragraph()
    resolved_model_alias = _resolve_model_alias(model_alias)
    return Column._from_logical_expr(
        SemanticSummarizeExpr(Column._from_col_or_name(column)._logical_expr, format, temperature,
                              model_alias=resolved_model_alias, request_timeout=request_timeout)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def parse_pdf(
    column: ColumnOrName,
    model_alias: Optional[Union[str, ModelAlias]] = None,
    page_separator: Optional[str] = None,
	describe_images: bool = False,  # for images that aren't tables
	max_output_tokens: Optional[int] = None,
	request_timeout: TimeoutParam = None,
) -> Column:
    r"""Parses a column of PDF paths into markdown.

    Returns:
        Dataframe: a dataframe with markdown strings for each PDF file.

    Args:
        column: Column or column name containing the PDF to parse.
        model_alias: Optional alias for the language model to use for the parsing. If None, will use the language model configured as the default.
        page_separator: Optional page separator to use for the parsing.  If the separator includes the {page} placeholder, the model will replace it with the current page number.
        describe_images:  Flag to describe images in the PDF. If True, the prompt will ask the model to include a description of the image in the markdown output.  If False, the prompt asks the model to ignore images that aren't tables or charts.
        max_output_tokens: Optional maximum number of output tokens per ~3 pages of PDF (does not include reasoning tokens). If None, don't constrain the model's output.
        request_timeout: Optional timeout in seconds for a single LLM request. If None, uses the default timeout (120 seconds).

    Note:
        For Gemini models, this function uses the google file API, uploading PDF files to Google's file store and deleting them after each request.

    Raises:
        ExecutionError: If paths in the column are not valid PDF files.

    Example: Parse PDF paths in a column into markdown
        ```python
        semantic.parse_pdf(col("pdf_path")).show()
        ```

    Example: Parsing PDFs into markdown with a page separator and describing images larger than 50 pixels
        ```python
        pdf_markdown = semantic.parse_pdf(col("pdf_path"), page_separator="--- PAGE {page} ---", describe_images=True)
        pdf_markdown.select(col("markdown_content")).show()
        ```

    Example: Grabbing PDFs from a glob pattern using pdf_metadata and parsing the PDF content
        ```python
        pdf_metadata = local_session.read.pdf_metadata("data/docs/**/*.pdf")
        pdf_markdown = pdf_metadata.select(semantic.parse_pdf(col("file_path"), page_separator="--- PAGE BREAK ---")
        pdf_markdown.select(col("markdown_content")).show()
        ```
    """
    resolved_model_alias = _resolve_model_alias(model_alias)

    return Column._from_logical_expr(
        SemanticParsePDFExpr(
            Column._from_col_or_name(column)._logical_expr,
            model_alias=resolved_model_alias,
            page_separator=page_separator,
            describe_images=describe_images,
            max_output_tokens=max_output_tokens,
            request_timeout=request_timeout,
        )
    )