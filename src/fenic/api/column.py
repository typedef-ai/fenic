"""Column API for Fenic DataFrames - represents column expressions and operations."""

from __future__ import annotations

from typing import Any, List, Union

from pydantic import ConfigDict, validate_call

from fenic.core._logical_plan.expressions import (
    AliasExpr,
    CastExpr,
    ContainsAnyExpr,
    ContainsExpr,
    EndsWithExpr,
    ILikeExpr,
    IndexExpr,
    InExpr,
    IsNullExpr,
    LikeExpr,
    LiteralExpr,
    LogicalExpr,
    OtherwiseExpr,
    RLikeExpr,
    SortExpr,
    StartsWithExpr,
    WhenExpr,
)
from fenic.core._logical_plan.expressions.arithmetic import ArithmeticExpr
from fenic.core._logical_plan.expressions.base import Operator
from fenic.core._logical_plan.expressions.basic import ColumnExpr, NotExpr
from fenic.core._logical_plan.expressions.comparison import (
    BooleanExpr,
    EqualityComparisonExpr,
    NumericComparisonExpr,
)
from fenic.core._utils.type_inference import (
    TypeInferenceError,
    infer_dtype_from_pyobj,
)
from fenic.core.error import ValidationError
from fenic.core.types.datatypes import (
    DataType,
)


class Column:
    """A column expression in a DataFrame.

    This class represents a column expression that can be used in DataFrame operations.
    It provides methods for accessing, transforming, and combining column data.

    Example: Create a column reference
        ```python
        # Reference a column by name using col() function
        col("column_name")
        ```

    Example: Use column in operations
        ```python
        # Perform arithmetic operations
        df.select(col("price") * col("quantity"))
        ```

    Example: Chain column operations
        ```python
        # Chain multiple operations
        df.select(col("name").upper().contains("John"))
        ```
    """

    _logical_expr: LogicalExpr

    def __new__(cls):
        """Prevent direct Column construction."""
        if cls is Column:
            raise TypeError("Direct construction of Column is not allowed")
        return super().__new__(cls)

    def get_item(self, key: Union[str, int]) -> Column:
        """Access an item in a struct or array column.

        This method allows accessing elements in complex data types:

        - For array columns, the key should be an integer index
        - For struct columns, the key should be a field name

        Args:
            key (Union[str, int]): The index (for arrays) or field name (for structs) to access

        Returns:
            Column: A Column representing the accessed item

        Example: Access an array element
            ```python
            # Get the first element from an array column
            df.select(col("array_column").get_item(0))
            ```

        Example: Access a struct field
            ```python
            # Get a field from a struct column
            df.select(col("struct_column").get_item("field_name"))
            ```
        """
        return Column._from_logical_expr(IndexExpr(self._logical_expr, key))

    def __getitem__(self, key: Union[str, int]) -> Column:
        """Access an item in a struct or array column using [] syntax.

        This method provides a convenient syntax for accessing elements in complex data types:

        - For array columns: col("array_column")[0]
        - For struct columns: col("struct_column")["field_name"]

        Args:
            key (Union[str, int]): The index (for arrays) or field name (for structs) to access

        Returns:
            Column: A Column representing the accessed item

        Example: Access an array element
            ```python
            # Get the first element from an array column using [] syntax
            df.select(col("array_column")[0])
            ```

        Example: Access a struct field
            ```python
            # Get a field from a struct column using [] syntax
            df.select(col("struct_column")["field_name"])
            ```
        """
        return self.get_item(key)

    def __getattr__(self, name: str) -> Column:
        """Access an attribute of a struct column.

        This method allows accessing fields in struct columns using dot notation.

        Args:
            name (str): The name of the field to access

        Returns:
            Column: A Column representing the accessed field

        Example: Access a struct field using dot notation
            ```python
            # Get a field from a struct column using dot notation
            df.select(col("struct_column").field_name)
            ```
        """
        if not isinstance(name, str):
            raise TypeError(
                f"Column attribute access requires a string name, got {type(name).__name__}"
            )
        return self.get_item(name)

    def alias(self, name: str) -> Column:
        """Create an alias for this column.

        This method assigns a new name to the column expression, which is useful
        for renaming columns or providing names for complex expressions.

        Args:
            name (str): The alias name to assign

        Returns:
            Column: Column with the specified alias

        Example: Rename a column
            ```python
            # Rename a column to a new name
            df.select(col("original_name").alias("new_name"))
            ```

        Example: Name a complex expression
            ```python
            # Give a name to a calculated column
            df.select((col("price") * col("quantity")).alias("total_value"))
            ```
        """
        return Column._from_logical_expr(AliasExpr(self._logical_expr, name))

    def cast(self, data_type: DataType) -> Column:
        """Cast the column to a new data type.

        This method creates an expression that casts the column to a specified data type.
        The casting behavior depends on the source and target types:

        Primitive type casting:

        - Numeric types (IntegerType, FloatType, DoubleType) can be cast between each other
        - Numeric types can be cast to/from StringType
        - BooleanType can be cast to/from numeric types and StringType
        - StringType cannot be directly cast to BooleanType (will raise TypeError)

        Complex type casting:

        - ArrayType can only be cast to another ArrayType (with castable element types)
        - StructType can only be cast to another StructType (with matching/castable fields)
        - Primitive types cannot be cast to/from complex types

        Args:
            data_type (DataType): The target DataType to cast the column to

        Returns:
            Column: A Column representing the casted expression

        Example: Cast integer to string
            ```python
            # Convert an integer column to string type
            df.select(col("int_col").cast(StringType))
            ```

        Example: Cast array of integers to array of strings
            ```python
            # Convert an array of integers to an array of strings
            df.select(col("int_array").cast(ArrayType(element_type=StringType)))
            ```

        Example: Cast struct fields to different types
            ```python
            # Convert struct fields to different types
            new_type = StructType([
                StructField("id", StringType),
                StructField("value", FloatType)
            ])
            df.select(col("data_struct").cast(new_type))
            ```

        Raises:
            TypeError: If the requested cast operation is not supported
        """
        return Column._from_logical_expr(CastExpr(self._logical_expr, data_type))

    def desc(self) -> Column:
        """Apply descending order to this column during a dataframe sort or order_by.

        This method creates an expression that provides a column and sort order to the sort function.

        Returns:
            Column: A Column expression that provides a column and sort order to the sort function

        Example: Sort by age in descending order
            ```python
            # Sort a dataframe by age in descending order
            df.sort(col("age").desc()).show()
            ```

        Example: Sort using column reference
            ```python
            # Sort using column reference with descending order
            df.sort(col("age").desc()).show()
            ```
        """
        return Column._from_logical_expr(SortExpr(self._logical_expr, ascending=False))

    def desc_nulls_first(self) -> Column:
        """Apply descending order putting nulls first to this column during a dataframe sort or order_by.

        This method creates an expression that provides a column and sort order to the sort function

        Returns:
            Column: A Column expression that provides a column and sort order to the sort function

        Example: Sort by age in descending order with nulls first
            ```python
            df.sort(col("age").desc_nulls_first()).show()
            ```

        Example: Sort using column reference
            ```python
            df.sort(col("age").desc_nulls_first()).show()
            ```
        """
        return Column._from_logical_expr(
            SortExpr(self._logical_expr, ascending=False, nulls_last=False)
        )

    def desc_nulls_last(self) -> Column:
        """Apply descending order putting nulls last to this column during a dataframe sort or order_by.

        This method creates an expression that provides a column and sort order to the sort function.

        Returns:
            Column: A Column expression that provides a column and sort order to the sort function

        Example: Sort by age in descending order with nulls last
            ```python
            # Sort a dataframe by age in descending order, with nulls appearing last
            df.sort(col("age").desc_nulls_last()).show()
            ```

        Example: Sort using column reference
            ```python
            # Sort using column reference with descending order and nulls last
            df.sort(col("age").desc_nulls_last()).show()
            ```
        """
        return Column._from_logical_expr(
            SortExpr(self._logical_expr, ascending=False, nulls_last=True)
        )

    def asc(self) -> Column:
        """Apply ascending order to this column during a dataframe sort or order_by.

        This method creates an expression that provides a column and sort order to the sort function.

        Returns:
            Column: A Column expression that provides a column and sort order to the sort function

        Example: Sort by age in ascending order
            ```python
            # Sort a dataframe by age in ascending order
            df.sort(col("age").asc()).show()
            ```

        Example: Sort using column reference
            ```python
            # Sort using column reference with ascending order
            df.sort(col("age").asc()).show()
            ```
        """
        return Column._from_logical_expr(SortExpr(self._logical_expr, ascending=True))

    def asc_nulls_first(self) -> Column:
        """Apply ascending order putting nulls first to this column during a dataframe sort or order_by.

        This method creates an expression that provides a column and sort order to the sort function.

        Returns:
            Column: A Column expression that provides a column and sort order to the sort function

        Example: Sort by age in ascending order with nulls first
            ```python
            # Sort a dataframe by age in ascending order, with nulls appearing first
            df.sort(col("age").asc_nulls_first()).show()
            ```

        Example: Sort using column reference
            ```python
            # Sort using column reference with ascending order and nulls first
            df.sort(col("age").asc_nulls_first()).show()
            ```
        """
        return Column._from_logical_expr(
            SortExpr(self._logical_expr, ascending=True, nulls_last=False)
        )

    def asc_nulls_last(self) -> Column:
        """Apply ascending order putting nulls last to this column during a dataframe sort or order_by.

        This method creates an expression that provides a column and sort order to the sort function.

        Returns:
            Column: A Column expression that provides a column and sort order to the sort function

        Example: Sort by age in ascending order with nulls last
            ```python
            # Sort a dataframe by age in ascending order, with nulls appearing last
            df.sort(col("age").asc_nulls_last()).show()
            ```

        Example: Sort using column reference
            ```python
            # Sort using column reference with ascending order and nulls last
            df.sort(col("age").asc_nulls_last()).show()
            ```
        """
        return Column._from_logical_expr(
            SortExpr(self._logical_expr, ascending=True, nulls_last=True)
        )

    def contains(self, other: Union[str, Column]) -> Column:
        """Check if the column contains a substring.

        This method creates a boolean expression that checks if each value in the column
        contains the specified substring. The substring can be either a literal string
        or a column expression.

        Args:
            other (Union[str, Column]): The substring to search for (can be a string or column expression)

        Returns:
            Column: A boolean column indicating whether each value contains the substring

        Example: Find rows where name contains "john"
            ```python
            # Filter rows where the name column contains "john"
            df.filter(col("name").contains("john"))
            ```

        Example: Find rows where text contains a dynamic pattern
            ```python
            # Filter rows where text contains a value from another column
            df.filter(col("text").contains(col("pattern")))
            ```
        """
        if isinstance(other, str):
            return Column._from_logical_expr(ContainsExpr(self._logical_expr, other))
        else:
            return Column._from_logical_expr(
                ContainsExpr(self._logical_expr, other._logical_expr)
            )

    def contains_any(self, others: List[str], case_insensitive: bool = True) -> Column:
        """Check if the column contains any of the specified substrings.

        This method creates a boolean expression that checks if each value in the column
        contains any of the specified substrings. The matching can be case-sensitive or
        case-insensitive.

        Args:
            others (List[str]): List of substrings to search for
            case_insensitive (bool): Whether to perform case-insensitive matching (default: True)

        Returns:
            Column: A boolean column indicating whether each value contains any substring

        Example: Find rows where name contains "john" or "jane" (case-insensitive)
            ```python
            # Filter rows where name contains either "john" or "jane"
            df.filter(col("name").contains_any(["john", "jane"]))
            ```

        Example: Case-sensitive matching
            ```python
            # Filter rows with case-sensitive matching
            df.filter(col("name").contains_any(["John", "Jane"], case_insensitive=False))
            ```
        """
        return Column._from_logical_expr(
            ContainsAnyExpr(self._logical_expr, others, case_insensitive)
        )

    def starts_with(self, other: Union[str, Column]) -> Column:
        """Check if the column starts with a substring.

        This method creates a boolean expression that checks if each value in the column
        starts with the specified substring. The substring can be either a literal string
        or a column expression.

        Args:
            other (Union[str, Column]): The substring to check for at the start (can be a string or column expression)

        Returns:
            Column: A boolean column indicating whether each value starts with the substring

        Example: Find rows where name starts with "Mr"
            ```python
            # Filter rows where name starts with "Mr"
            df.filter(col("name").starts_with("Mr"))
            ```

        Example: Find rows where text starts with a dynamic pattern
            ```python
            # Filter rows where text starts with a value from another column
            df.filter(col("text").starts_with(col("prefix")))
            ```

        Raises:
            ValueError: If the substring starts with a regular expression anchor (^)
        """
        if isinstance(other, str):
            return Column._from_logical_expr(StartsWithExpr(self._logical_expr, other))
        else:
            return Column._from_logical_expr(
                StartsWithExpr(self._logical_expr, other._logical_expr)
            )

    def ends_with(self, other: Union[str, Column]) -> Column:
        """Check if the column ends with a substring.

        This method creates a boolean expression that checks if each value in the column
        ends with the specified substring. The substring can be either a literal string
        or a column expression.

        Args:
            other (Union[str, Column]): The substring to check for at the end (can be a string or column expression)

        Returns:
            Column: A boolean column indicating whether each value ends with the substring

        Example: Find rows where email ends with "@gmail.com"
            ```python
            df.filter(col("email").ends_with("@gmail.com"))
            ```

        Example: Find rows where text ends with a dynamic pattern
            ```python
            df.filter(col("text").ends_with(col("suffix")))
            ```

        Raises:
            ValueError: If the substring ends with a regular expression anchor ($)
        """
        if isinstance(other, str):
            return Column._from_logical_expr(EndsWithExpr(self._logical_expr, other))
        else:
            return Column._from_logical_expr(
                EndsWithExpr(self._logical_expr, other._logical_expr)
            )

    def rlike(self, other: str) -> Column:
        r"""Check if the column matches a regular expression pattern.

        This method creates a boolean expression that checks if each value in the column
        matches the specified regular expression pattern. The pattern must be a literal string
        and cannot be a column expression.

        Args:
            other (str): The regular expression pattern to match against

        Returns:
            Column: A boolean column indicating whether each value matches the pattern

        Example: Find rows where phone number matches pattern
            ```python
            # Filter rows where phone number matches a specific pattern
            df.filter(col("phone").rlike(r"^\d{3}-\d{3}-\d{4}$"))
            ```

        Example: Find rows where text contains word boundaries
            ```python
            # Filter rows where text contains a word with boundaries
            df.filter(col("text").rlike(r"\bhello\b"))
            ```
        """
        return Column._from_logical_expr(RLikeExpr(self._logical_expr, other))

    def like(self, other: str) -> Column:
        r"""Check if the column matches a SQL LIKE pattern.

        This method creates a boolean expression that checks if each value in the column
        matches the specified SQL LIKE pattern. The pattern must be a literal string
        and cannot be a column expression.

        SQL LIKE pattern syntax:

        - % matches any sequence of characters
        - _ matches any single character

        Args:
            other (str): The SQL LIKE pattern to match against

        Returns:
            Column: A boolean column indicating whether each value matches the pattern

        Example: Find rows where name starts with "J" and ends with "n"
            ```python
            # Filter rows where name matches the pattern "J%n"
            df.filter(col("name").like("J%n"))
            ```

        Example: Find rows where code matches specific pattern
            ```python
            # Filter rows where code matches the pattern "A_B%"
            df.filter(col("code").like("A_B%"))
            ```
        """
        return Column._from_logical_expr(LikeExpr(self._logical_expr, other))

    def ilike(self, other: str) -> Column:
        r"""Check if the column matches a SQL LIKE pattern (case-insensitive).

        This method creates a boolean expression that checks if each value in the column
        matches the specified SQL LIKE pattern, ignoring case. The pattern must be a literal string
        and cannot be a column expression.

        SQL LIKE pattern syntax:

        - % matches any sequence of characters
        - _ matches any single character

        Args:
            other (str): The SQL LIKE pattern to match against

        Returns:
            Column: A boolean column indicating whether each value matches the pattern

        Example: Find rows where name starts with "j" and ends with "n" (case-insensitive)
            ```python
            # Filter rows where name matches the pattern "j%n" (case-insensitive)
            df.filter(col("name").ilike("j%n"))
            ```

        Example: Find rows where code matches pattern (case-insensitive)
            ```python
            # Filter rows where code matches the pattern "a_b%" (case-insensitive)
            df.filter(col("code").ilike("a_b%"))
            ```
        """
        return Column._from_logical_expr(ILikeExpr(self._logical_expr, other))

    def is_null(self) -> Column:
        """Check if the column contains NULL values.

        This method creates an expression that evaluates to TRUE when the column value is NULL.

        Returns:
            Column: A Column representing a boolean expression that is TRUE when this column is NULL

        Example: Filter rows where a column is NULL
            ```python
            # Filter rows where some_column is NULL
            df.filter(col("some_column").is_null())
            ```

        Example: Use in a complex condition
            ```python
            # Filter rows where col1 is NULL or col2 is greater than 100
            df.filter(col("col1").is_null() | (col("col2") > 100))
            ```
        """
        return Column._from_logical_expr(IsNullExpr(self._logical_expr, True))

    def is_not_null(self) -> Column:
        """Check if the column contains non-NULL values.

        This method creates an expression that evaluates to TRUE when the column value is not NULL.

        Returns:
            Column: A Column representing a boolean expression that is TRUE when this column is not NULL

        Example: Filter rows where a column is not NULL
            ```python
            df.filter(col("some_column").is_not_null())
            ```

        Example: Use in a complex condition
            ```python
            df.filter(col("col1").is_not_null() & (col("col2") <= 50))
            ```
        """
        return Column._from_logical_expr(IsNullExpr(self._logical_expr, False))

    def when(self, condition: Column, value: Column) -> Column:
        """Evaluates a list of conditions and returns one of multiple possible result expressions.

        If Column.otherwise() is not invoked, None is returned for unmatched conditions.
        Otherwise() will return for rows with None inputs.

        Args:
            condition (Column): A boolean Column expression
            value (Column): A literal value or Column expression to return if the condition is true

        Returns:
            Column: A Column expression representing whether each element of Column matches the condition

        Raises:
            TypeError: If the condition is not a boolean Column expression

        Example: Use when/otherwise for conditional logic
            ```python
            # Create a DataFrame with age and name columns
            df = session.createDataFrame(
                {"age": [2, 5]}, {"name": ["Alice", "Bob"]}
            )

            # Use when/otherwise to create a case result column
            df.select(
                col("name"),
                when(col("age") > 3, 1).otherwise(0).alias("case_result")
            ).show()
            # Output:
            # +-----+-----------+
            # | name|case_result|
            # +-----+-----------+
            # |Alice|          0|
            # |  Bob|          1|
            # +-----+-----------+
            ```
        """
        return Column._from_logical_expr(WhenExpr(self._logical_expr, condition._logical_expr, value._logical_expr))

    def otherwise(self, value: Column) -> Column:
        """Evaluates a list of conditions and returns one of multiple possible result expressions.

        If Column.otherwise() is not invoked, None is returned for unmatched conditions.
        Otherwise() will return for rows with None inputs.

        Args:
            value (Column): A literal value or Column expression to return

        Returns:
            Column: A Column expression representing whether each element of Column is not matched by any previous conditions

        Example: Use when/otherwise for conditional logic
            ```python
            # Create a DataFrame with age and name columns
            df = session.createDataFrame(
                {"age": [2, 5]}, {"name": ["Alice", "Bob"]}
            )

            # Use when/otherwise to create a case result column
            df.select(
                col("name"),
                when(col("age") > 3, 1).otherwise(0).alias("case_result")
            ).show()
            # Output:
            # +-----+-----------+
            # | name|case_result|
            # +-----+-----------+
            # |Alice|          0|
            # |  Bob|          1|
            # +-----+-----------+
            ```
        """
        return Column._from_logical_expr(OtherwiseExpr(self._logical_expr, value._logical_expr))

    def is_in(self, other: Union[List[Any], ColumnOrName]) -> Column:
        """Check if the column is in a list of values or a column expression.

        Args:
            other (Union[List[Any], ColumnOrName]): A list of values or a Column expression

        Returns:
            Column: A Column expression representing whether each element of Column is in the list

        Example: Check if name is in a list of values
            ```python
            # Filter rows where name is in a list of values
            df.filter(col("name").is_in(["Alice", "Bob"]))
            ```

        Example: Check if value is in another column
            ```python
            # Filter rows where name is in another column
            df.filter(col("name").is_in(col("other_column")))
            ```
        """
        if isinstance(other, list):
            try:
                type_ = infer_dtype_from_pyobj(other)
                return Column._from_logical_expr(InExpr(self._logical_expr, LiteralExpr(other, type_)))
            except TypeInferenceError as e:
                raise ValidationError(f"Cannot apply IN on {other}. List argument to IN must be be a valid Python List literal.") from e
        else:
            return Column._from_logical_expr(InExpr(self._logical_expr, other._logical_expr))

    @classmethod
    def _from_logical_expr(cls, logical_expr: LogicalExpr) -> Column:
        if not isinstance(logical_expr, LogicalExpr):
            raise TypeError(f"Expected LogicalExpr, got {type(logical_expr)}")
        column = super().__new__(cls)
        column._logical_expr = logical_expr
        return column

    @classmethod
    def _from_column_name(cls, col_name: str):
        if not isinstance(col_name, str):
            raise TypeError(f"Expected str, got {type(col_name)}")
        column = super().__new__(cls)
        column._logical_expr = ColumnExpr(col_name)
        return column

    @classmethod
    def _from_col_or_name(cls, col_or_name: ColumnOrName) -> Column:
        if isinstance(col_or_name, Column):
            return col_or_name
        else:
            return cls._from_column_name(col_or_name)

    def _create_binary_expr(
        self, other: Any, op: Operator, expr_class, reverse=False
    ) -> Column:
        if isinstance(other, Column):
            right_expr = other._logical_expr
        else:
            right_expr = LiteralExpr(other, infer_dtype_from_pyobj(other))
        if reverse:
            return Column._from_logical_expr(
                expr_class(right_expr, self._logical_expr, op)
            )
        else:
            return Column._from_logical_expr(
                expr_class(self._logical_expr, right_expr, op)
            )

    def __invert__(self) -> Column:
        """Logical NOT operation."""
        return Column._from_logical_expr(NotExpr(self._logical_expr))

    def __gt__(self, other: Any) -> Column:
        """Greater than comparison."""
        return self._create_binary_expr(other, Operator.GT, NumericComparisonExpr)

    def __ge__(self, other: Any) -> Column:
        """Greater than or equal comparison."""
        return self._create_binary_expr(other, Operator.GTEQ, NumericComparisonExpr)

    def __lt__(self, other: Any) -> Column:
        """Less than comparison."""
        return self._create_binary_expr(other, Operator.LT, NumericComparisonExpr)

    def __le__(self, other: Any) -> Column:
        """Less than or equal comparison."""
        return self._create_binary_expr(other, Operator.LTEQ, NumericComparisonExpr)

    def __eq__(self, other: Any) -> Column:
        """Equality comparison."""
        return self._create_binary_expr(other, Operator.EQ, EqualityComparisonExpr)

    def __ne__(self, other: Any) -> Column:
        """Not equal comparison."""
        return self._create_binary_expr(other, Operator.NOT_EQ, EqualityComparisonExpr)

    def __and__(self, other: Column) -> Column:
        """Logical AND operation."""
        return self._create_binary_expr(other, Operator.AND, BooleanExpr)

    def __rand__(self, other: Column) -> Column:
        """Reverse logical AND operation."""
        return self & other

    def __or__(self, other: Column) -> Column:
        """Logical OR operation."""
        return self._create_binary_expr(other, Operator.OR, BooleanExpr)

    def __ror__(self, other: Column) -> Column:
        """Reverse logical OR operation."""
        return self | other

    def __add__(self, other: Any) -> Column:
        """Addition operation."""
        return self._create_binary_expr(other, Operator.PLUS, ArithmeticExpr)

    def __radd__(self, other: Any) -> Column:
        """Reverse addition operation."""
        return self._create_binary_expr(
            other, Operator.PLUS, ArithmeticExpr, reverse=True
        )

    def __sub__(self, other: Any) -> Column:
        """Subtraction operation."""
        return self._create_binary_expr(other, Operator.MINUS, ArithmeticExpr)

    def __rsub__(self, other: Any) -> Column:
        """Reverse subtraction operation."""
        return self._create_binary_expr(
            other, Operator.MINUS, ArithmeticExpr, reverse=True
        )

    def __mul__(self, other: Any) -> Column:
        """Multiplication operation."""
        return self._create_binary_expr(other, Operator.MULTIPLY, ArithmeticExpr)

    def __rmul__(self, other) -> Column:
        """Reverse multiplication operation."""
        return self * other

    def __truediv__(self, other: Any) -> Column:
        """Division operation."""
        return self._create_binary_expr(other, Operator.DIVIDE, ArithmeticExpr)

    def __rtruediv__(self, other: Any) -> Column:
        """Reverse division operation."""
        return self._create_binary_expr(
            other, Operator.DIVIDE, ArithmeticExpr, reverse=True
        )

    def __bool__(self):
        """Prevent boolean conversion of Column objects."""
        raise TypeError(
            "Cannot use Column in boolean context. Use '&', '|', or '~', not 'and', 'or', or 'not'."
        )

ColumnOrName = Union[Column, str]

Column.get_item = validate_call(config=ConfigDict(strict=True))(Column.get_item)
Column.getItem = Column.get_item
Column.alias = validate_call(config=ConfigDict(strict=True))(Column.alias)
Column.cast = validate_call(
    config=ConfigDict(strict=True, arbitrary_types_allowed=True)
)(Column.cast)
Column.contains = validate_call(
    config=ConfigDict(strict=True, arbitrary_types_allowed=True)
)(Column.contains)
Column.contains_any = validate_call(config=ConfigDict(strict=True))(Column.contains_any)
Column.starts_with = validate_call(
    config=ConfigDict(strict=True, arbitrary_types_allowed=True)
)(Column.starts_with)
Column.ends_with = validate_call(
    config=ConfigDict(strict=True, arbitrary_types_allowed=True)
)(Column.ends_with)
Column.rlike = validate_call(config=ConfigDict(strict=True))(Column.rlike)
Column.like = validate_call(config=ConfigDict(strict=True))(Column.like)
Column.ilike = validate_call(config=ConfigDict(strict=True))(Column.ilike)
Column.when = validate_call(
    config=ConfigDict(strict=True, arbitrary_types_allowed=True)
)(Column.when)
Column.otherwise = validate_call(
    config=ConfigDict(strict=True, arbitrary_types_allowed=True)
)(Column.otherwise)
Column.is_in = validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))(Column.is_in)
