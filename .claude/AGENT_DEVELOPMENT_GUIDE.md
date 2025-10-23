Become an amazing member of the typedef team and help us implement new features in our open source library!
Refer to the following guide for details:

# Fenic Feature Development Guide for AI Agents

This comprehensive guide provides AI agents with all the necessary knowledge, patterns, and conventions required to independently develop features for Fenic, a PySpark-inspired DataFrame framework for AI and LLM applications.

## Table of Contents

1. [Introduction & Philosophy](#1-introduction--philosophy)
2. [Architecture Overview](#2-architecture-overview)
3. [Development Environment](#3-development-environment)
4. [Code Style Guidelines](#4-code-style-guidelines)
5. [Adding DataFrame Operations](#5-adding-dataframe-operations)
6. [Adding Logical Expressions](#6-adding-logical-expressions)
7. [Adding Logical Plans](#7-adding-logical-plans)
8. [Protobuf Serialization System](#8-protobuf-serialization-system)
9. [Type System & Inference](#9-type-system--inference)
10. [Testing Requirements](#10-testing-requirements)
11. [Common Patterns](#11-common-patterns)
12. [Complete Example Walkthrough](#12-complete-example-walkthrough)
13. [Troubleshooting & Debugging](#13-troubleshooting--debugging)
14. [PR Checklist](#14-pr-checklist)

---

## 1. Introduction & Philosophy

### What is Fenic?

Fenic is a PySpark-inspired DataFrame framework specifically designed for AI and LLM applications. It provides:

- **Semantic operators** for LLM-powered data transformations
- **Native support** for unstructured data (markdown, transcripts, JSON)
- **Efficient batch inference** across multiple model providers (OpenAI, Anthropic, Google)
- **Dual-backend architecture** supporting both local (Polars) and cloud (gRPC) execution

### Core Design Principles

1. **Lazy Evaluation**: Operations build logical plans without immediate execution
2. **Session-Centric**: All operations flow through `Session.get_or_create()`
3. **Strong Typing**: Schema inference and type validation throughout
4. **PySpark Compatibility**: Similar API surface for familiar developer experience
5. **Cloud-Native**: First-class support for distributed execution
6. **Type Safety**: Comprehensive type hints and runtime validation

---

## 2. Architecture Overview

### Directory Structure

```
src/fenic/
├── api/                      # Public API surface
│   ├── session.py           # Session management
│   ├── dataframe/          # DataFrame class and methods
│   │   ├── dataframe.py    # Main DataFrame implementation
│   │   └── semantic_extension.py  # Semantic operations
│   ├── column.py           # Column expressions
│   └── functions/          # Built-in functions
│       ├── core.py         # Core functions (lit, col, when)
│       ├── aggregate.py    # Aggregation functions
│       ├── string.py       # String functions
│       └── semantic.py     # Semantic/LLM functions
├── core/                    # Core internal logic
│   ├── _logical_plan/      # Logical plan nodes
│   │   ├── plans/         # Plan types (Source, Transform, Sink)
│   │   └── expressions/   # Expression types
│   ├── _serde/            # Protobuf serialization
│   │   └── proto/         # Serde implementation
│   ├── _utils/            # Utilities (type inference, validation)
│   └── types/             # Type system (DataType, Schema)
├── _backends/              # Execution backends
│   ├── local/             # Local backend (Polars)
│   │   ├── physical_plan/ # Physical execution nodes
│   │   └── transpiler/    # Logical→Physical conversion
│   └── cloud/             # Cloud backend (gRPC)
└── _inference/            # LLM provider integrations
```

### Key Architectural Components

#### 1. Three-Layer Architecture

**API Layer** (`src/fenic/api/`)

- User-facing DataFrame and Column classes
- Function registry
- Semantic extensions
- Input validation and error handling

**Core Layer** (`src/fenic/core/`)

- Logical plan representation (what to compute)
- Expression trees
- Type system
- Optimization rules
- Serialization/deserialization

**Backend Layer** (`src/fenic/_backends/`)

- Physical plan representation (how to compute)
- Local execution via Polars
- Cloud execution via gRPC
- Resource management

#### 2. Logical vs Physical Plans

**Logical Plans** define WHAT to compute:

```python
# User writes:
df.filter(col("age") > 18).select("name")

# Creates logical plan:
Projection(
    input=Filter(
        input=Source(...),
        predicate=GreaterThan(Column("age"), Literal(18))
    ),
    columns=["name"]
)
```

**Physical Plans** define HOW to compute:

```python
# Local backend converts to:
ProjectionExec(
    input=FilterExec(
        input=SourceExec(...),
        predicate=pl.col("age") > pl.lit(18)  # Polars expression
    ),
    columns=["name"]
)
```

#### 3. Expression System

Expressions are the building blocks of computations:

```python
# Expression hierarchy:
LogicalExpr (base)
├── ColumnExpr           # Column references
├── LiteralExpr          # Constant values
├── SeriesLiteralExpr    # Series data
├── BinaryExpr           # Left op Right
│   ├── ArithmeticExpr
│   ├── BooleanExpr
│   └── ComparisonExpr
├── AggregateExpr        # Aggregations (sum, avg, count)
└── SemanticExpr         # LLM operations
    ├── SemanticMapExpr
    ├── SemanticExtractExpr
    └── SemanticClassifyExpr
```

---

## 3. Development Environment

### Setup Commands

```bash
# Initial setup (installs dependencies + builds Rust extensions)
just setup

# Sync Python dependencies
just sync

# Build Rust extensions (after Rust code changes)
just sync-rust

# Sync with cloud extras
just sync-cloud
```

### Testing Commands

```bash
# Run local tests (excludes cloud tests)
just test
# or
uv run pytest -m "not cloud"

# Run cloud tests
just test-cloud
# or
uv run pytest -m cloud

# Run specific test file
uv run pytest tests/path/to/test_file.py

# Run specific test
uv run pytest tests/path/to/test_file.py::test_function_name -xvs

# Run tests matching pattern
uv run pytest -k "pattern" -xvs
```

### Linting & Formatting

```bash
# Run ruff checks
uv run ruff check .

# Auto-format code
uv run ruff format .

# Run both
just trunk check
just trunk fmt
```

### Proto Generation

```bash
# Regenerate Python protobuf types (after modifying .proto files)
just generate-protos-py

# Generate all proto types (Python, Go, Rust)
just generate-protos
```

---

## 4. Code Style Guidelines

### Python Style

#### Docstrings

**ALWAYS use Google-style docstrings**, not NumPy or Sphinx style:

````python
def my_function(param1: str, param2: int) -> DataFrame:
    """Short description of what the function does.

    Longer description providing more context about the function's
    behavior, edge cases, and important notes.

    Args:
        param1: Description of first parameter
        param2: Description of second parameter

    Returns:
        Description of return value

    Raises:
        ValueError: When param2 is negative
        TypeError: When param1 is not a string

    Example: Basic usage
        ```python
        result = my_function("hello", 42)
        result.show()
        # Output:
        # +---------+
        # |   result|
        # +---------+
        # |something|
        # +---------+
        ```

    Example: Advanced usage
        ```python
        # More complex example
        result = my_function("world", 100)
        ```
    """
    pass
````

#### Type Hints

**ALWAYS provide type hints** for all function parameters and return values:

```python
# Good
def process_dataframe(df: DataFrame, threshold: float = 0.5) -> DataFrame:
    pass

# Bad - missing type hints
def process_dataframe(df, threshold=0.5):
    pass
```

Use `Union`, `Optional`, `List`, `Dict` from typing:

```python
from typing import Dict, List, Optional, Union

def with_column(
    self,
    col_name: str,
    col: Union[Any, Column, pl.Series, pd.Series]
) -> DataFrame:
    pass
```

#### Imports

**Prefer absolute imports** over relative imports:

```python
# Good
from fenic.core._logical_plan.expressions import ColumnExpr
from fenic.core.types import StringType

# Avoid (use only when necessary to avoid circular imports)
from ..expressions import ColumnExpr
from ...types import StringType
```

Group imports in this order:

1. Standard library
2. Third-party packages
3. Local/Fenic modules

```python
# Standard library
from typing import List, Optional
from io import BytesIO

# Third-party
import polars as pl
import pandas as pd

# Local
from fenic.core._logical_plan.expressions import LogicalExpr
from fenic.core.types import DataType
```

#### Code Formatting

- **Indentation**: 4 spaces (not tabs)
- **Line length**: ~100 characters (soft limit)
- **Trailing commas**: Use in multi-line collections
- **String quotes**: Use double quotes for strings

```python
# Good
my_dict = {
    "key1": "value1",
    "key2": "value2",
    "key3": "value3",  # Trailing comma
}

# Multi-line function calls
result = my_function(
    parameter1="long_value",
    parameter2=42,
    parameter3=True,  # Trailing comma
)
```

### Naming Conventions

| Type             | Convention           | Example                                        |
| ---------------- | -------------------- | ---------------------------------------------- |
| Classes          | PascalCase           | `DataFrame`, `ColumnExpr`, `SeriesLiteralExpr` |
| Functions        | snake_case           | `with_column`, `create_dataframe`              |
| Variables        | snake_case           | `col_name`, `data_type`                        |
| Constants        | UPPER_SNAKE_CASE     | `DEFAULT_TIMEOUT`, `MAX_RETRIES`               |
| Private members  | \_leading_underscore | `_source`, `_logical_expr`                     |
| Internal modules | \_leading_underscore | `_backends`, `_serde`                          |

---

## 5. Adding DataFrame Operations

DataFrame operations are methods on the `DataFrame` class in `src/fenic/api/dataframe/dataframe.py`.

### Step-by-Step Process

#### Step 1: Add method to DataFrame class

````python
# In src/fenic/api/dataframe/dataframe.py

def my_operation(
    self,
    param1: str,
    param2: Optional[int] = None
) -> DataFrame:
    """Short description of the operation.

    Args:
        param1: Description of parameter 1
        param2: Optional description of parameter 2

    Returns:
        DataFrame: New DataFrame with operation applied

    Example: Basic usage
        ```python
        df = session.create_dataframe({"col": [1, 2, 3]})
        result = df.my_operation("value")
        result.show()
        ```
    """
    # Create logical plan node
    new_plan = MyOperationPlan(
        input=self._plan,
        param1=param1,
        param2=param2,
        session_state=self._session_state
    )

    # Return new DataFrame with new plan
    return DataFrame(new_plan, self._session_state)
````

#### Step 2: Create logical plan node

```python
# In src/fenic/core/_logical_plan/plans/transform.py

class MyOperationPlan(LogicalPlan):
    """Logical plan for my_operation.

    This plan represents...
    """

    def __init__(
        self,
        input: LogicalPlan,
        param1: str,
        param2: Optional[int],
        session_state: Optional[BaseSessionState] = None
    ):
        super().__init__(session_state)
        self.input = input
        self.param1 = param1
        self.param2 = param2 or 10  # Default value

    def children(self) -> List[LogicalPlan]:
        return [self.input]

    def _eq_specific(self, other: MyOperationPlan) -> bool:
        return (
            self.param1 == other.param1 and
            self.param2 == other.param2
        )

    @property
    def schema(self) -> Schema:
        # Define output schema
        return self.input.schema  # Or modify as needed
```

#### Step 3: Add physical plan implementation

```python
# In src/fenic/_backends/local/physical_plan/transform.py

class MyOperationExec(PhysicalPlan):
    """Physical execution for MyOperationPlan."""

    def __init__(
        self,
        input: PhysicalPlan,
        param1: str,
        param2: int
    ):
        self.input = input
        self.param1 = param1
        self.param2 = param2

    def execute(self) -> pl.DataFrame:
        # Get input data
        df = self.input.execute()

        # Perform operation using Polars
        result = df.with_columns(
            # Your Polars transformation here
        )

        return result
```

#### Step 4: Add transpiler mapping

```python
# In src/fenic/_backends/local/transpiler/plan_converter.py

@_convert_plan.register
def _convert_my_operation_plan(
    plan: MyOperationPlan,
    context: TranspileContext
) -> PhysicalPlan:
    """Convert MyOperationPlan to physical plan."""
    return MyOperationExec(
        input=context.convert_plan(plan.input),
        param1=plan.param1,
        param2=plan.param2
    )
```

#### Step 5: Add tests

```python
# In tests/_backends/local/dataframe/test_core.py

def test_my_operation(local_session):
    """Test my_operation with basic input."""
    df = local_session.create_dataframe({
        "col1": [1, 2, 3],
        "col2": ["a", "b", "c"]
    })

    result = df.my_operation("test_value", param2=5)
    result_df = result.to_polars()

    # Assertions
    assert "new_col" in result_df.columns
    assert result_df["new_col"].to_list() == [expected, values, here]


def test_my_operation_with_defaults(local_session):
    """Test my_operation with default parameters."""
    df = local_session.create_dataframe({"col": [1, 2]})
    result = df.my_operation("value")

    # Test default behavior
    assert result.count() == 2


def test_my_operation_error_handling(local_session):
    """Test my_operation error handling."""
    df = local_session.create_dataframe({"col": [1, 2]})

    with pytest.raises(ValueError, match="Invalid parameter"):
        df.my_operation("bad_value", param2=-1)
```

### DataFrame Operation Patterns

#### Pattern 1: Transformation (returns DataFrame)

```python
def transform_operation(self, ...) -> DataFrame:
    """Most DataFrame operations follow this pattern."""
    new_plan = TransformPlan(input=self._plan, ...)
    return DataFrame(new_plan, self._session_state)
```

#### Pattern 2: Action (returns value)

```python
def action_operation(self, ...) -> Any:
    """Actions trigger execution and return values."""
    # Actions execute the plan
    result = self._execute()
    return process_result(result)
```

#### Pattern 3: Lazy Property (returns metadata)

```python
@property
def metadata_property(self) -> SomeType:
    """Properties that don't execute the plan."""
    return self._plan.some_metadata
```

---

## 6. Adding Logical Expressions

Expressions represent computations on data (columns, literals, operations).

### Expression Type Hierarchy

```python
LogicalExpr (ABC in core/_logical_plan/expressions/base.py)
├── to_column_field(plan, session_state) -> ColumnField  # Required
├── children() -> List[LogicalExpr]                      # Required
├── _eq_specific(other) -> bool                          # Required
└── __str__() -> str                                     # Required
```

### Step-by-Step Process

#### Step 1: Define expression class

```python
# In src/fenic/core/_logical_plan/expressions/your_category.py

from typing import List
from fenic.core._logical_plan.expressions.base import LogicalExpr
from fenic.core.types import DataType

class MyCustomExpr(LogicalExpr):
    """Expression for custom operation.

    This expression...
    """

    def __init__(self, input_expr: LogicalExpr, parameter: str):
        """Initialize MyCustomExpr.

        Args:
            input_expr: Input expression to operate on
            parameter: Configuration parameter
        """
        self.input_expr = input_expr
        self.parameter = parameter

    def __str__(self) -> str:
        return f"my_custom({self.input_expr}, {self.parameter})"

    def to_column_field(
        self,
        plan: LogicalPlan,
        session_state: BaseSessionState
    ) -> ColumnField:
        # Determine output type
        input_field = self.input_expr.to_column_field(plan, session_state)

        # Return output column with inferred type
        return ColumnField(
            name=str(self),
            data_type=StringType  # Or infer from input
        )

    def children(self) -> List[LogicalExpr]:
        return [self.input_expr]

    def _eq_specific(self, other: MyCustomExpr) -> bool:
        return self.parameter == other.parameter
```

#### Step 2: Add function to create expression

````python
# In src/fenic/api/functions/your_category.py

from fenic.api.column import Column

def my_custom(col: Column, parameter: str) -> Column:
    """Apply custom operation to column.

    Args:
        col: Input column
        parameter: Operation parameter

    Returns:
        Column with custom operation applied

    Example:
        ```python
        df.select(my_custom(col("text"), "config"))
        ```
    """
    return Column._from_logical_expr(
        MyCustomExpr(col._logical_expr, parameter)
    )
````

#### Step 3: Export function

```python
# In src/fenic/api/functions/__init__.py

from fenic.api.functions.your_category import my_custom

__all__ = [
    # ... existing functions ...
    "my_custom",
]
```

#### Step 4: Add transpiler (local backend)

```python
# In src/fenic/_backends/local/transpiler/expr_converter.py

@_convert_expr.register
def _convert_my_custom_expr(
    logical: MyCustomExpr,
    context: ExpressionContext
) -> pl.Expr:
    """Convert MyCustomExpr to Polars expression."""
    input_expr = context.convert_expr(logical.input_expr)

    # Implement using Polars operations
    return input_expr.str.replace(
        # Your Polars implementation
    )
```

#### Step 5: Add tests

```python
# In tests/_backends/local/expressions/test_your_category.py

def test_my_custom_expr(local_session):
    """Test my_custom expression."""
    df = local_session.create_dataframe({
        "text": ["hello", "world"]
    })

    result = df.select(my_custom(col("text"), "param")).to_polars()

    assert result["text"].to_list() == ["expected", "values"]
```

### Expression Mixins

Use mixins for common patterns:

```python
# UnparameterizedExpr: For expressions without config beyond children
class MySimpleExpr(UnparameterizedExpr, LogicalExpr):
    def _eq_specific(self, other):
        return True  # Auto-implemented by mixin

# ValidatedSignature: For expressions with input validation
class MyValidatedExpr(ValidatedSignature, LogicalExpr):
    function_name = "my_function"

    def validate_signature(self, input_types: List[DataType]):
        if not is_string_type(input_types[0]):
            raise TypeError(f"{self.function_name} requires string input")
```

---

## 7. Adding Logical Plans

Plans represent operations on DataFrames (sources, transformations, sinks).

### Plan Type Hierarchy

```python
LogicalPlan (ABC in core/_logical_plan/plans/base.py)
├── SourcePlan        # Data sources (files, tables, memory)
├── TransformPlan     # Transformations (filter, select, join)
└── SinkPlan         # Data sinks (write operations)
```

### Key Methods

```python
class LogicalPlan(ABC):
    @property
    @abstractmethod
    def schema(self) -> Schema:
        """Output schema of this plan."""
        pass

    @abstractmethod
    def children(self) -> List[LogicalPlan]:
        """Child plans (inputs)."""
        pass

    @abstractmethod
    def _eq_specific(self, other) -> bool:
        """Plan-specific equality check."""
        pass
```

### Example: Adding a Custom Transform Plan

```python
# In src/fenic/core/_logical_plan/plans/transform.py

class CustomTransformPlan(LogicalPlan):
    """Plan for custom transformation."""

    def __init__(
        self,
        input: LogicalPlan,
        config: str,
        session_state: Optional[BaseSessionState] = None
    ):
        super().__init__(session_state)
        self.input = input
        self.config = config

    @property
    def schema(self) -> Schema:
        # Define how schema changes
        input_schema = self.input.schema

        # Option 1: Same schema as input
        return input_schema

        # Option 2: Modified schema
        # return Schema([
        #     *input_schema.fields,
        #     StructField("new_col", StringType)
        # ])

    def children(self) -> List[LogicalPlan]:
        return [self.input]

    def _eq_specific(self, other: CustomTransformPlan) -> bool:
        return self.config == other.config
```

---

## 8. Protobuf Serialization System

Protobuf serialization enables cloud backend support by serializing logical plans to bytes for transmission.

### Why Serialization Matters

- **Cloud Backend**: Plans must be serialized to send to remote execution
- **Caching**: Serialized plans can be cached
- **Debugging**: Serialized plans can be inspected
- **Versioning**: Proto schemas provide backward compatibility

### Serialization Architecture

```
Python Object → LogicalPlanProto → bytes → [network] → bytes → LogicalPlanProto → Python Object
```

### Key Components

1. **Protocol Buffer Schema** (`.proto` files in `protos/logical_plan/v1/`)
2. **Generated Python Types** (`src/fenic/_gen/protos/`)
3. **Serde Functions** (`src/fenic/core/_serde/proto/`)
4. **SerdeContext** (centralized serde operations)

### Adding Serialization for New Expressions

#### Step 1: Update Protocol Buffer Schema

```protobuf
// In protos/logical_plan/v1/expressions.proto

message LogicalExpr {
  oneof expr_type {
    // ... existing expressions ...
    MyCustomExpr my_custom = 999;  // Use next available number
  }
}

message MyCustomExpr {
  LogicalExpr input_expr = 1;
  string parameter = 2;
  int32 optional_config = 3;
}
```

#### Step 2: Regenerate Python Types

```bash
just generate-protos-py
```

This generates `MyCustomExprProto` in `src/fenic/_gen/protos/logical_plan/v1/expressions_pb2.py`.

#### Step 3: Export Proto Type

```python
# In src/fenic/core/_serde/proto/types.py

# Add import
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    MyCustomExpr as MyCustomExprProto,
)

# Add to __all__
__all__ = [
    # ... existing exports ...
    "MyCustomExprProto",
]
```

#### Step 4: Implement Serde Functions

```python
# In src/fenic/core/_serde/proto/expressions/your_category.py

from fenic.core._logical_plan.expressions import MyCustomExpr
from fenic.core._serde.proto.expression_serde import (
    _deserialize_logical_expr_helper,
    serialize_logical_expr,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import (
    LogicalExprProto,
    MyCustomExprProto,
)

# Serialization
@serialize_logical_expr.register
def _serialize_my_custom_expr(
    logical: MyCustomExpr,
    context: SerdeContext
) -> LogicalExprProto:
    """Serialize MyCustomExpr to protobuf."""
    return LogicalExprProto(
        my_custom=MyCustomExprProto(
            input_expr=context.serialize_logical_expr(
                SerdeContext.EXPR,
                logical.input_expr
            ),
            parameter=logical.parameter,
            optional_config=logical.optional_config if hasattr(logical, 'optional_config') else 0,
        )
    )

# Deserialization
@_deserialize_logical_expr_helper.register
def _deserialize_my_custom_expr(
    proto: MyCustomExprProto,
    context: SerdeContext
) -> MyCustomExpr:
    """Deserialize MyCustomExpr from protobuf."""
    return MyCustomExpr(
        input_expr=context.deserialize_logical_expr(
            SerdeContext.EXPR,
            proto.input_expr
        ),
        parameter=proto.parameter,
        # Handle optional fields
        optional_config=proto.optional_config if proto.HasField("optional_config") else None,
    )
```

#### Step 5: Register Module

```python
# In src/fenic/core/_serde/proto/expressions/__init__.py

from .your_category import (
    _serialize_my_custom_expr,
    _deserialize_my_custom_expr,
)
```

#### Step 6: Add Serde Tests

```python
# In tests/_logical_plan/serde/test_expression_serde.py

# Add to imports
from fenic.core._logical_plan.expressions import MyCustomExpr

# Add to expression_examples dictionary
expression_examples = {
    # ... existing examples ...
    MyCustomExpr: [
        MyCustomExpr(ColumnExpr("col1"), "param1"),
        MyCustomExpr(LiteralExpr("text", StringType), "param2"),
    ],
}
```

The test suite automatically tests all expressions in `expression_examples` for round-trip serialization.

### Serializing Complex Data (bytes pattern)

For runtime data like Series or DataFrames, use the bytes pattern:

```python
# Protobuf schema
message SeriesLiteralExpr {
  bytes series_data = 1;  // Serialized data as bytes
  DataType data_type = 2;
}

# Serialization
@serialize_logical_expr.register
def _serialize_series_literal_expr(
    logical: SeriesLiteralExpr,
    context: SerdeContext
) -> LogicalExprProto:
    """Serialize by converting to bytes via Arrow IPC."""
    from io import BytesIO

    # Convert to bytes using Arrow IPC format
    buffer = BytesIO()
    logical.series.to_frame().write_ipc(buffer)

    return LogicalExprProto(
        series_literal=SeriesLiteralExprProto(
            series_data=buffer.getvalue(),  # bytes
            data_type=context.serialize_data_type(
                SerdeContext.DATA_TYPE,
                logical.data_type
            ),
        )
    )

# Deserialization
@_deserialize_logical_expr_helper.register
def _deserialize_series_literal_expr(
    proto: SeriesLiteralExprProto,
    context: SerdeContext
) -> SeriesLiteralExpr:
    """Deserialize from bytes back to original form."""
    from io import BytesIO
    import polars as pl

    # Reconstruct from bytes
    buffer = BytesIO(proto.series_data)
    df = pl.read_ipc(buffer)
    series = df.to_series()

    return SeriesLiteralExpr(series=series)
```

### Unserializable Expressions

Some expressions cannot be safely serialized (e.g., UDFExpr with user code):

```python
# In src/fenic/core/_serde/proto/expressions/unserializable.py

from fenic.core._serde.proto.errors import UnsupportedTypeError

@serialize_logical_expr.register
def _serialize_udf_expr(
    logical: UDFExpr,
    context: SerdeContext
) -> LogicalExprProto:
    raise context.create_serde_error(
        UnsupportedTypeError,
        "UDFExpr cannot be serialized for cloud execution. "
        "User-defined functions contain arbitrary Python code that "
        "cannot be safely transmitted. Use this feature only with "
        "local backend.",
        UDFExpr
    )
```

---

## 9. Type System & Inference

Fenic has a comprehensive type system for data validation and optimization.

### DataType Hierarchy

```python
DataType (base class)
├── BooleanType
├── IntegerType
├── FloatType
├── DoubleType
├── StringType
├── DateType
├── TimestampType
├── BinaryType
├── JsonType
├── ArrayType(element_type: DataType)
├── StructType(fields: List[StructField])
└── MapType(key_type: DataType, value_type: DataType)
```

### Type Inference

#### From Python Values

```python
# In src/fenic/core/_utils/type_inference.py

def infer_dtype_from_python(value: Any) -> DataType:
    """Infer Fenic DataType from Python value."""
    if isinstance(value, bool):
        return BooleanType
    elif isinstance(value, int):
        return IntegerType
    elif isinstance(value, float):
        return DoubleType
    elif isinstance(value, str):
        return StringType
    # ... more types
```

#### From Polars Types

```python
def infer_dtype_from_polars(pl_dtype: pl.DataType) -> DataType:
    """Convert Polars dtype to Fenic DataType."""
    if pl_dtype == pl.Boolean:
        return BooleanType
    elif pl_dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
        return IntegerType
    elif pl_dtype == pl.Float64:
        return DoubleType
    elif pl_dtype == pl.Utf8:
        return StringType
    elif isinstance(pl_dtype, pl.List):
        element_type = infer_dtype_from_polars(pl_dtype.inner)
        return ArrayType(element_type)
    # ... more types
```

#### Type Validation in Expressions

```python
class MyTypedExpr(ValidatedSignature, LogicalExpr):
    """Expression with type validation."""

    function_name = "my_function"

    def validate_signature(self, input_types: List[DataType]):
        """Validate input types."""
        if len(input_types) != 1:
            raise TypeError(
                f"{self.function_name} expects 1 argument, "
                f"got {len(input_types)}"
            )

        if not isinstance(input_types[0], StringType):
            raise TypeError(
                f"{self.function_name} expects string input, "
                f"got {input_types[0]}"
            )
```

---

## 10. Testing Requirements

Comprehensive testing ensures feature quality and prevents regressions.

### Test Organization

```
tests/
├── _backends/
│   ├── local/               # Local backend tests
│   │   ├── dataframe/       # DataFrame operation tests
│   │   ├── expressions/     # Expression tests
│   │   └── physical_plan/   # Physical plan tests
│   └── cloud/              # Cloud backend tests (marked with @pytest.mark.cloud)
├── _logical_plan/
│   └── serde/              # Serialization tests
├── api/                    # API surface tests
└── core/                   # Core logic tests
```

### Test Types

#### 1. Unit Tests

Test individual components in isolation:

```python
def test_my_expression_basic(local_session):
    """Test MyCustomExpr with basic input."""
    expr = MyCustomExpr(ColumnExpr("col1"), "param")

    # Test string representation
    assert str(expr) == "my_custom(col1, param)"

    # Test children
    assert len(expr.children()) == 1
    assert isinstance(expr.children()[0], ColumnExpr)
```

#### 2. Integration Tests

Test full workflows:

```python
def test_my_operation_workflow(local_session):
    """Test complete workflow with my_operation."""
    # Create source data
    df = local_session.create_dataframe({
        "id": [1, 2, 3],
        "value": ["a", "b", "c"]
    })

    # Apply operation
    result = (
        df
        .my_operation("config")
        .filter(col("id") > 1)
        .select("value")
    )

    # Execute and verify
    result_df = result.to_polars()
    assert len(result_df) == 2
    assert result_df["value"].to_list() == ["b", "c"]
```

#### 3. Serde Round-Trip Tests

Test serialization/deserialization:

```python
# Add to tests/_logical_plan/serde/test_expression_serde.py

expression_examples = {
    MyCustomExpr: [
        MyCustomExpr(ColumnExpr("col1"), "param1"),
        MyCustomExpr(ColumnExpr("col2"), "param2"),
    ],
}

# Automatic round-trip test runs for all expressions
```

#### 4. Error Handling Tests

Test error cases:

```python
def test_my_operation_invalid_input(local_session):
    """Test error handling for invalid input."""
    df = local_session.create_dataframe({"col": [1, 2]})

    with pytest.raises(ValueError, match="Invalid parameter"):
        df.my_operation("invalid")

def test_my_expression_type_error(local_session):
    """Test type validation error."""
    df = local_session.create_dataframe({"col": [1, 2]})

    with pytest.raises(TypeError, match="expects string input"):
        df.select(my_custom(col("col"), "param"))
```

### Cloud Backend Tests

Mark cloud tests with `@pytest.mark.cloud`:

```python
@pytest.mark.cloud
def test_my_operation_cloud(cloud_session):
    """Test my_operation with cloud backend."""
    df = cloud_session.create_dataframe({"col": [1, 2, 3]})
    result = df.my_operation("config")

    # Cloud execution
    result_df = result.to_polars()
    assert len(result_df) == 3
```

### Running Tests

```bash
# Run local tests only
just test
uv run pytest -m "not cloud"

# Run cloud tests
just test-cloud
uv run pytest -m cloud

# Run specific test
uv run pytest tests/path/to/test.py::test_function -xvs

# Run with pattern
uv run pytest -k "my_operation" -xvs

# Run with coverage
uv run pytest --cov=src/fenic --cov-report=html
```

---

## 11. Common Patterns

### Pattern: Conditional Column Addition

```python
df = df.with_column(
    "category",
    when(col("value") > 100, "high")
    .when(col("value") > 50, "medium")
    .otherwise("low")
)
```

### Pattern: Multiple Transformations

```python
result = (
    df
    .filter(col("status") == "active")
    .with_column("double_value", col("value") * 2)
    .select("id", "double_value")
    .sort(col("double_value").desc())
    .limit(10)
)
```

### Pattern: Aggregation

```python
summary = (
    df
    .group_by("category")
    .agg(
        count(col("id")).alias("count"),
        avg(col("value")).alias("avg_value"),
        max(col("value")).alias("max_value")
    )
)
```

### Pattern: Semantic Operation

```python
df = df.semantic.extract(
    "text_column",
    schema=MyPydanticModel,
    model="gpt-4o-mini"
)
```

---

## 12. Complete Example Walkthrough

Let's walk through the complete implementation of `SeriesLiteralExpr` as a real-world example.

### Use Case

Enable users to add Polars or pandas Series directly to DataFrames:

```python
import polars as pl

df = session.create_dataframe({"name": ["Alice", "Bob"]})
bonus = pl.Series([100, 200])
df.with_column("bonus", bonus).show()
```

### Step 1: Create Expression Class

```python
# src/fenic/core/_logical_plan/expressions/basic.py

class SeriesLiteralExpr(LogicalExpr):
    """Expression representing a Polars or pandas Series as a literal column.

    This expression allows users to directly add Series data to a DataFrame using
    with_column or with_columns. The Series is stored as a Polars Series internally
    and serialized via Arrow IPC format for cloud backend compatibility.

    Works with both local and cloud backends.
    """

    def __init__(self, series: Union[pl.Series, pd.Series]):
        """Initialize a SeriesLiteralExpr.

        Args:
            series: A Polars or pandas Series to be used as a literal column
        """
        # Convert pandas Series to Polars for consistent handling
        if isinstance(series, pd.Series):
            self.series = pl.from_pandas(series)
        else:
            self.series = series

        # Infer the Fenic data type from the Polars Series dtype
        self.data_type = self._infer_fenic_type(self.series.dtype)

    def _infer_fenic_type(self, pl_dtype: pl.DataType) -> DataType:
        """Convert Polars dtype to Fenic DataType."""
        from fenic.core._utils.type_inference import infer_dtype_from_polars
        return infer_dtype_from_polars(pl_dtype)

    def __str__(self) -> str:
        return f"series(len={len(self.series)})"

    def to_column_field(
        self,
        plan: LogicalPlan,
        session_state: BaseSessionState
    ) -> ColumnField:
        return ColumnField(str(self), self.data_type)

    def children(self) -> List[LogicalExpr]:
        return []

    def _eq_specific(self, other: SeriesLiteralExpr) -> bool:
        try:
            return self.series.equals(other.series)
        except Exception:
            return False
```

### Step 2: Update Type Inference

```python
# src/fenic/core/_utils/type_inference.py

def infer_dtype_from_polars(pl_dtype: pl.DataType) -> DataType:
    """Convert a Polars data type to a Fenic DataType."""
    if pl_dtype == pl.Boolean:
        return BooleanType
    elif pl_dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
        return IntegerType
    elif pl_dtype == pl.Float64:
        return DoubleType
    elif pl_dtype == pl.Utf8:
        return StringType
    # ... more types
```

### Step 3: Update DataFrame Methods

```python
# src/fenic/api/dataframe/dataframe.py

def with_column(
    self,
    col_name: str,
    col: Union[Any, Column, pl.Series, pd.Series]
) -> DataFrame:
    """Add a new column or replace an existing column.

    Args:
        col_name: Name of the new column
        col: Column expression, Series, or value to assign to the column.
            - Column: A Column expression (e.g., col("age") + 1)
            - pl.Series or pd.Series: A Polars or pandas Series with data
            - Any other value: Treated as a literal value (broadcast to all rows)

    Returns:
        DataFrame: New DataFrame with added/replaced column
    """
    exprs = []

    # Handle different input types: Column, Series, or literal value
    if isinstance(col, (pl.Series, pd.Series)):
        # Wrap Series in SeriesLiteralExpr and then in Column
        col = Column._from_logical_expr(SeriesLiteralExpr(col))
    elif not isinstance(col, Column):
        # Wrap other values as literals
        col = lit(col)

    # Build projection with new column
    for field in self.columns:
        if field != col_name:
            exprs.append(Column._from_column_name(field)._logical_expr)

    exprs.append(col.alias(col_name)._logical_expr)

    # Create new plan
    return DataFrame(
        Projection(self._plan, exprs, self._session_state),
        self._session_state
    )
```

### Step 4: Add Local Backend Support

```python
# src/fenic/_backends/local/transpiler/expr_converter.py

@_convert_expr.register
def _convert_series_literal_expr(
    logical: SeriesLiteralExpr,
    context: ExpressionContext
) -> pl.Expr:
    """Convert SeriesLiteralExpr to Polars expression.

    This wraps the Series in pl.lit() which allows Polars to handle:
    - Length matching with the DataFrame
    - Broadcasting of single-value Series
    - Type checking and coercion
    """
    return pl.lit(logical.series)
```

### Step 5: Add Protobuf Schema

```protobuf
// protos/logical_plan/v1/expressions.proto

message LogicalExpr {
  oneof expr_type {
    // ... existing expressions ...
    SeriesLiteralExpr series_literal = 165;
  }
}

message SeriesLiteralExpr {
  bytes series_data = 1;  // Serialized Polars Series in Arrow IPC format
  DataType data_type = 2;
}
```

### Step 6: Implement Serde

```python
# src/fenic/core/_serde/proto/expressions/basic.py

@serialize_logical_expr.register
def _serialize_series_literal_expr(
    logical: SeriesLiteralExpr,
    context: SerdeContext
) -> LogicalExprProto:
    """Serialize SeriesLiteralExpr by converting Polars Series to bytes."""
    from io import BytesIO

    # Serialize the Series to Arrow IPC binary format
    buffer = BytesIO()
    logical.series.to_frame().write_ipc(buffer)

    return LogicalExprProto(
        series_literal=SeriesLiteralExprProto(
            series_data=buffer.getvalue(),
            data_type=context.serialize_data_type(
                SerdeContext.DATA_TYPE,
                logical.data_type
            ),
        )
    )

@_deserialize_logical_expr_helper.register
def _deserialize_series_literal_expr(
    logical_proto: SeriesLiteralExprProto,
    context: SerdeContext
) -> SeriesLiteralExpr:
    """Deserialize SeriesLiteralExpr from bytes back to Polars Series."""
    from io import BytesIO
    import polars as pl

    # Deserialize the bytes back to a DataFrame, then extract Series
    buffer = BytesIO(logical_proto.series_data)
    df = pl.read_ipc(buffer)
    series = df.to_series()

    return SeriesLiteralExpr(series=series)
```

### Step 7: Add Tests

```python
# tests/_backends/local/dataframe/test_core.py

def test_with_column_polars_series(local_session):
    """Test adding a Polars Series to a DataFrame."""
    import polars as pl

    data = {"name": ["Alice", "Bob"], "age": [25, 30]}
    df = local_session.create_dataframe(data)

    # Create a Polars Series
    bonus_series = pl.Series("bonus", [100, 200])

    result = df.with_column("bonus", bonus_series).to_polars()

    assert "bonus" in result.columns
    assert result["bonus"].to_list() == [100, 200]

# tests/_logical_plan/serde/test_expression_serde.py

expression_examples = {
    # ... existing examples ...
    SeriesLiteralExpr: [
        SeriesLiteralExpr(pl.Series("test", [1, 2, 3])),
        SeriesLiteralExpr(pl.Series([1.5, 2.5, 3.5])),
        SeriesLiteralExpr(pd.Series([100, 200])),
    ],
}
```

---

## 13. Troubleshooting & Debugging

### Common Issues

#### Issue: Import Error for Proto Types

**Error:**

```
ImportError: cannot import name 'MyExprProto' from 'fenic.core._serde.proto.types'
```

**Solution:**

1. Regenerate protos: `just generate-protos-py`
2. Add import to `src/fenic/core/_serde/proto/types.py`
3. Add to `__all__` in types.py

#### Issue: Type Inference Fails

**Error:**

```
TypeInferenceError: Unsupported Polars dtype: Categorical
```

**Solution:**
Add handling in `infer_dtype_from_polars`:

```python
elif isinstance(pl_dtype, pl.Categorical):
    return StringType  # Categorical maps to String
```

#### Issue: Serde Round-Trip Fails

**Error:**

```
AssertionError: Deserialized expression doesn't match original
```

**Debug Steps:**

1. Check proto schema matches Python class
2. Verify all fields are serialized/deserialized
3. Check optional field handling (`proto.HasField()`)
4. Ensure constructor parameters match exactly

### Debugging Tools

#### 1. Plan Visualization

```python
# Print logical plan
print(df._plan)

# Get plan string representation
plan_str = str(df._plan)
```

#### 2. Expression Inspection

```python
# Inspect expression
expr = col("age") + 1
print(expr._logical_expr)
print(type(expr._logical_expr))
print(expr._logical_expr.children())
```

#### 3. Schema Debugging

```python
# Check schema
print(df.schema)
print(df.columns)
print(df.dtypes)
```

#### 4. Serde Debugging

```python
from fenic.core._serde.proto.serde_context import create_serde_context

context = create_serde_context()

# Serialize
proto = context.serialize_logical_expr("test", my_expr)
print(proto)

# Deserialize
deserialized = context.deserialize_logical_expr("test", proto)
print(deserialized)
```

---

## 14. PR Checklist

Before submitting a PR, ensure:

### Code Quality

- [ ] All new code has comprehensive type hints
- [ ] All new functions/classes have Google-style docstrings with examples
- [ ] Code follows Fenic style guidelines (4 spaces, ~100 chars, absolute imports)
- [ ] No commented-out code or debug print statements
- [ ] Ruff checks pass: `uv run ruff check .`
- [ ] Code is formatted: `uv run ruff format .`

### Functionality

- [ ] Feature works with local backend
- [ ] Feature works with cloud backend (or documented as local-only)
- [ ] Error handling is comprehensive with clear error messages
- [ ] Edge cases are handled
- [ ] Performance is reasonable (no obvious inefficiencies)

### Testing

- [ ] Unit tests added for new functionality
- [ ] Integration tests added for workflows
- [ ] Serde round-trip tests added (if applicable)
- [ ] Error handling tests added
- [ ] All existing tests pass: `just test`
- [ ] Cloud tests pass: `just test-cloud` (if applicable)

### Documentation

- [ ] Docstrings include multiple examples
- [ ] Examples demonstrate basic and advanced usage
- [ ] Examples show expected output
- [ ] Breaking changes are documented
- [ ] Migration guide provided (if breaking changes)

### Serialization (if applicable)

- [ ] Proto schema updated
- [ ] Proto types regenerated: `just generate-protos-py`
- [ ] Serde functions implemented
- [ ] Proto types exported in types.py
- [ ] Serde module registered in `__init__.py`
- [ ] Serde tests added to test_expression_serde.py

### Git

- [ ] Commit messages are clear and descriptive
- [ ] Commits are logical units of work
- [ ] No merge commits (rebase instead)
- [ ] Branch is up to date with main

---

## Appendix: Quick Reference

### File Locations

| Component           | Location                                    |
| ------------------- | ------------------------------------------- |
| DataFrame API       | `src/fenic/api/dataframe/dataframe.py`      |
| Functions           | `src/fenic/api/functions/`                  |
| Logical Expressions | `src/fenic/core/_logical_plan/expressions/` |
| Logical Plans       | `src/fenic/core/_logical_plan/plans/`       |
| Local Backend       | `src/fenic/_backends/local/`                |
| Serde               | `src/fenic/core/_serde/proto/`              |
| Proto Schemas       | `protos/logical_plan/v1/`                   |
| Tests               | `tests/`                                    |

### Command Reference

```bash
# Setup
just setup                  # Initial setup
just sync                   # Sync dependencies
just sync-rust             # Build Rust extensions

# Testing
just test                  # Local tests
just test-cloud           # Cloud tests
uv run pytest path        # Specific tests

# Code Quality
uv run ruff check .       # Lint
uv run ruff format .      # Format

# Protobuf
just generate-protos-py   # Regenerate Python protos
```

### Type Mapping

| Python            | Polars   | Fenic         |
| ----------------- | -------- | ------------- |
| bool              | Boolean  | BooleanType   |
| int               | Int64    | IntegerType   |
| float             | Float64  | DoubleType    |
| str               | Utf8     | StringType    |
| datetime.date     | Date     | DateType      |
| datetime.datetime | Datetime | TimestampType |
| list              | List     | ArrayType     |
| dict              | Struct   | StructType    |

---

This guide should provide everything needed to develop features for Fenic independently. When in doubt, refer to existing code for patterns, and don't hesitate to ask questions!
