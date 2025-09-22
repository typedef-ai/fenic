"""Date and time functions."""

from typing import Optional, Union

from pydantic import ConfigDict, validate_call

from fenic.api.column import Column, ColumnOrName
from fenic.core._logical_plan.expressions import (
    DateAddExpr,
    DateDiffExpr,
    DateFormatExpr,
    DateTruncExpr,
    DayExpr,
    FromUTCTimestampExpr,
    HourExpr,
    LiteralExpr,
    MilliSecondExpr,
    MinuteExpr,
    MonthExpr,
    NowExpr,
    SecondExpr,
    TimestampAddExpr,
    TimestampDiffExpr,
    ToDateExpr,
    ToTimestampExpr,
    ToUTCTimestampExpr,
    YearExpr,
)
from fenic.core.types.datatypes import IntegerType
from fenic.core.types.enums import (
    DateTimeUnit,
)


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def year(column: ColumnOrName) -> Column:
    """Extract the year from a date column.

    Args:
        column: The column to extract the year from.

    Returns:
        A Column object with the year extracted.

    Raises:
        TypeError: If column type is not a DateType or TimestampType.

    Example:
        ```python
        # dates: "2025-01-01", "2025-01-02", "2025-01-03"]
        df.select(dt.year(col("date"))).to_pydict()
        # Output: [{'year': 2025}, {'year': 2025}, {'year': 2025}]
        ```
    """
    return Column._from_logical_expr(YearExpr(Column._from_col_or_name(column)._logical_expr))

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def month(column: ColumnOrName) -> Column:
    """Extract the month from a month column.

    Args:
        column: The column to extract the month from.

    Returns:
        A Column object with the month extracted.

    Raises:
        TypeError: If column type is not a DateType or TimestampType.

    Example:
        ```python
        # dates: "2025-01-01", "2025-01-02", "2024-12-03"]
        df.select(dt.month(col("date"))).to_pydict()
        # Output: [{'month': 1}, {'month': 1}, {'month': 12}]
        ```
    """
    return Column._from_logical_expr(MonthExpr(Column._from_col_or_name(column)._logical_expr))

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def day(column: ColumnOrName) -> Column:
    """Extract the day from a day column.

    Args:
        column: The column to extract the day from.

    Returns:
        A Column object with the day extracted.

    Raises:
        TypeError: If column type is not a DateType or TimestampType.

    Example:
        ```python
        # dates: "2025-01-01", "2025-01-02", "2025-01-03"]
        df.select(dt.day(col("date"))).to_pydict()
        # Output: [{'day': 1}, {'day': 2}, {'day': 3}]
        ```
    """
    return Column._from_logical_expr(DayExpr(Column._from_col_or_name(column)._logical_expr))

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def hour(column: ColumnOrName) -> Column:
    """Extract the hour from a day column.

    Args:
        column: The column to extract the hour from.

    Returns:
        A Column object with the hour extracted.

    Raises:
        TypeError: If column type is not a DateType or TimestampType.

    Notes:
        This will return 0 for DateType columns.

    Example:
        ```python
        # ts: "2025-01-01 10:00:00", "2025-01-02 11:00:00", "2025-01-03 12:00:00"]
        df.select(dt.hour(col("ts"))).to_pydict()
        # Output: [{'hour': 10}, {'hour': 11}, {'hour': 12}]
        ```
    """
    return Column._from_logical_expr(HourExpr(Column._from_col_or_name(column)._logical_expr))

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def minute(column: ColumnOrName) -> Column:
    """Extract the minute from a day column.

    Args:
        column: The column to extract the minute from.

    Returns:
        A Column object with the minute extracted.

    Raises:
        TypeError: If column type is not a DateType or TimestampType.

    Notes:
        This will return 0 for DateType columns.

    Example:
        ```python
        # ts: "2025-01-01 10:10:00", "2025-01-02 11:11:00", "2025-01-03 12:12:00"]
        df.select(dt.minute(col("ts"))).to_pydict()
        # Output: [{'minute': 10}, {'minute': 11}, {'minute': 12}]
        ```
    """
    return Column._from_logical_expr(MinuteExpr(Column._from_col_or_name(column)._logical_expr))

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def second(column: ColumnOrName) -> Column:
    """Extract the hour from a second column.

    Args:
        column: The column to extract the second from.

    Returns:
        A Column object with the second extracted.

    Raises:
        TypeError: If column type is not a DateType or TimestampType.

    Notes:
        This will return 0 for DateType columns.

    Example:
        ```python
        # ts: "2025-01-01 10:10:01", "2025-01-02 11:11:02", "2025-01-03 12:12:03"]
        df.select(dt.second(col("ts"))).to_pydict()
        # Output: [{'second': 1}, {'second': 2}, {'second': 3}]
        ```
    """
    return Column._from_logical_expr(SecondExpr(Column._from_col_or_name(column)._logical_expr))

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def millisecond(column: ColumnOrName) -> Column:
    """Extract the hour from a millisecond column.

    Args:
        column: The column to extract the millisecond from.

    Returns:
        A Column object with the millisecond extracted.

    Raises:
        TypeError: If column type is not a DateType or TimestampType.

    Notes:
        This will return 0 for DateType columns.

    Example:
        ```python
        # ts: "2025-01-01 10:10:01.123", "2025-01-02 11:11:02.234", "2025-01-03 12:12:03.345"]
        df.select(dt.millisecond(col("ts"))).to_pydict()
        # Output: [{'millisecond': 123}, {'millisecond': 234}, {'millisecond': 345}]
        ```
    """
    return Column._from_logical_expr(MilliSecondExpr(Column._from_col_or_name(column)._logical_expr))

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def to_date(column: ColumnOrName, format: Optional[str] = None) -> Column:
    """Transform a string into a DateType.

    Args:
        column: The column to transform into a DateType.
        format: The format of the date string.

    Returns:
        A Column object with the DateType transformed.

    Raises:
        TypeError: If column type is not a StringType.

    Notes:
        - If format is not provided, the default format is "YYYY-MM-DD".
        - The accepted formats should follow this pattern:
          https://spark.apache.org/docs/latest/sql-ref-datetime-pattern.html

    Example:
        ```python
        # date_str: "11-01-2025", "12-02-2025", "01-03-2025"]
        df.select(to_date(col("date_str"), format="MM-dd-yyyy").alias("date")).to_pydict()
        # Output: {'date': [datetime.datetime(2025, 11, 1, 0, 0, tzinfo=zoneinfo.ZoneInfo(key='UTC')), datetime.datetime(2025, 12, 2, 0, 0, tzinfo=zoneinfo.ZoneInfo(key='UTC')), datetime.datetime(2025, 1, 3, 0, 0, tzinfo=zoneinfo.ZoneInfo(key='UTC'))]}
        ```
    """
    return Column._from_logical_expr(ToDateExpr(Column._from_col_or_name(column)._logical_expr, format))

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def to_timestamp(column: ColumnOrName, format: Optional[str] = None) -> Column:
    """Transform a string into a TimestampType.

    Args:
        column: The column to transform into a TimestampType.
        format: The format of the timestamp string.

    Returns:
        A Column object with the `TimestampType` type, with a UTC timezone. If the provided `format` contains a timezone specifier, the result timestamp value will be converted from the `format` timezone to UTC.

    Raises:
        TypeError: If column type is not a StringType.

    Notes:
        - If format is not provided, the default format is ISO 8601 with milliseconds.
        - The accepted formats should follow this pattern:
          https://spark.apache.org/docs/latest/sql-ref-datetime-pattern.html

    Example:
        ```python
        # date_str: ["11-01-2025 10:00:00", "12-02-2025 11:00:00", "01-03-2025 12:00:00"]
        df.select(dt.to_date(col("date_str"), format="MM-dd-yyyy HH:mm:ss").alias("timestamp")).to_pydict()
        # Output: {'timestamp': [datetime.datetime(2025, 11, 1, 10, 0, tzinfo=zoneinfo.ZoneInfo(key='UTC')), datetime.datetime(2025, 11, 1, 10, 0, tzinfo=zoneinfo.ZoneInfo(key='UTC')), datetime.datetime(2025, 11, 1, 10, 0, tzinfo=zoneinfo.ZoneInfo(key='UTC'))]}
        ```
    """
    return Column._from_logical_expr(ToTimestampExpr(Column._from_col_or_name(column)._logical_expr, format))


def now() -> Column:
    """Get the current date and time.
    
    Returns:
        A Column object with the current date and time.
        The type of the column is TimestampType.

    Example:
        ```python
        df.select(dt.now()).to_pydict()
        # Output: [{'date': '<current date and time>'}]
        ```
    """
    return Column._from_logical_expr(NowExpr())

def current_timestamp() -> Column:
    """Get the current date and time.

    Returns:
        A Column object with the current date and time.
        The type of the column is TimestampType in UTC timezone.

    Example:
        ```python
        df.select(dt.current_timestamp().alias("cur_ts")).to_pydict()
        # Output: {'cur_ts': [datetime.datetime(2025, 9, 26, 10, 0)]}
        ```
    """
    return Column._from_logical_expr(NowExpr())

def current_date() -> Column:
    """Get the current date.

    Returns:
        A Column object with the current date.
        The type of the column is DateType.

    Example:
        ```python
        df.select(dt.current_date().alias("cur_date")).to_pydict()
        # Output: {'cur_date': [datetime.date(2025, 9, 26)]}
        ```
    """
    return Column._from_logical_expr(NowExpr(as_date=True))

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def date_trunc(column: ColumnOrName, unit: DateTimeUnit) -> Column:
    """Truncate a date to a given unit.

    Args:
        column: The column to truncate.
        unit: The unit to truncate to.

    Returns:
        A Column object with the date truncated.

    Raises:
        TypeError: If column type is not a DateType or TimestampType.
        ValueError: If unit is not supported, must be one of the supported ones.

    Notes:
        The supported units are: "year", "month", "day", "hour", "minute", "second", "millisecond".

    Example:
        ```python
        # dates: "2025-01-01", "2025-02-01", "2025-03-01"]
        df.select(dt.date_trunc(col("date"), "year").alias("date_trunc")).to_pydict()
        # Output: {'date_trunc': ['2025-01-01', '2025-01-01', '2025-01-01']}
        ```
    """
    return Column._from_logical_expr(DateTruncExpr(Column._from_col_or_name(column)._logical_expr, unit))

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def date_add(column: ColumnOrName, days: Union[int, ColumnOrName]) -> Column:
    """Adds the number of days to the date/timestamp column.

    Args:
        column: The column to add the days to.
        days: The number of days to add to the date/timestamp column. If the days is negative, the days will be subtracted.

    Returns:
        A Column object with the date/timestamp column with the days added.

    Raises:
        TypeError: If column type is not a DateType or TimestampType, or if days is not an IntegerType.
    
    Example:
        ```python
        # dates: "2025-01-01", "2025-02-01", "2025-03-01"]
        df.select(dt.date_add(col("date"), 1).alias("date_add")).to_pydict()
        # Output: {'date_add': ['2025-01-02', '2025-02-02', '2025-03-02']}
        ```
    """
    if isinstance(days, int):
        days_expr = LiteralExpr(days, IntegerType)
    else:
        days_expr = Column._from_col_or_name(days)._logical_expr

    return Column._from_logical_expr(DateAddExpr(Column._from_col_or_name(column)._logical_expr, days_expr))

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def date_sub(column: ColumnOrName, days: Union[int, ColumnOrName]) -> Column:
    """Subtracts the number of days from the date/timestamp column.

    Args:
        column: The column to subtract the days from.
        days: The amount of days to subtract. If the days is negative, the days will be added.

    Returns:
        A Column object with the date/timestamp column with the days substracted.

    Raises:
        TypeError: If column type is not a DateType or TimestampType, or if days is not an IntegerType.

    Example:
        ```python
        # dates: "2025-01-01", "2025-02-01", "2025-03-01"]
        df.select(dt.date_sub(col("date"), 1).alias("date_sub")).to_pydict()
        # Output: {'date_sub': ['2024-12-31', '2025-01-31', '2025-02-28']}
        ```
    """
    if isinstance(days, int):
        days_expr = LiteralExpr(days, IntegerType)
    else:
        days_expr = Column._from_col_or_name(days)._logical_expr

    return Column._from_logical_expr(DateAddExpr(Column._from_col_or_name(column)._logical_expr, days_expr, sub=True))

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def timestamp_add(column: ColumnOrName, quantity: Union[int, ColumnOrName], unit: DateTimeUnit) -> Column:
    """Adds the quantity of the given unit to the timestamp column.

    Args:
        column: The column to add the quantity to.
        quantity: The quantity to add. If the quantity is negative, the quantity will be subtracted.
        unit: The unit of the quantity.

    Returns:
        A Column object with the timestamp column with the quantity added.

    Raises:
        TypeError: If column type is not a TimestampType, or if quantity is not an IntegerType.
        ValueError: If unit is not supported, must be one of the supported ones.

    Notes:
        The supported units are: "year", "month", "day", "hour", "minute", "second", "millisecond".

    Example:
        ```python
        # ts: "2025-01-01 10:00:00", "2025-02-01 11:00:00", "2025-03-01 12:00:00"]
        df.select(dt.timestamp_add(col("ts"), 1, "day").alias("ts_add")).to_pydict()
        # Output: {'ts_add': ['2025-01-02 10:00:00', '2025-02-02 11:00:00', '2025-03-02 12:00:00']}
        ```
    """
    if isinstance(quantity, int):
        quantity_expr = LiteralExpr(quantity, IntegerType)
    else:
        quantity_expr = Column._from_col_or_name(quantity)._logical_expr

    return Column._from_logical_expr(TimestampAddExpr(Column._from_col_or_name(column)._logical_expr, quantity_expr, unit))

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def date_format(column: ColumnOrName, format: str) -> Column:
    """Formats a date/timestamp column to a given format.

    Args:
        column: The column to format.
        format: The format to format the column to.

    Returns:
        A Column object with the date/timestamp column formatted into a string.

    Raises:
        TypeError: If column type is not a DateType or TimestampType.

    Notes:
        - The accepted formats should follow this pattern:
          https://spark.apache.org/docs/latest/sql-ref-datetime-pattern.html

    Example:
        ```python
        # ts: "2025-01-01 10:00:00", "2025-02-01 11:00:00", "2025-03-01 15:00:00"]
        df.select(dt.date_format(col("date"), "MM-dd-yyyy hh:mm:ss a").alias("date")).to_pydict()
        # Output: {'date': ['01-01-2025 10:00:00 AM', '02-01-2025 11:00:00 AM', '03-01-2025 03:00:00 PM']}
        ```
    """
    return Column._from_logical_expr(DateFormatExpr(Column._from_col_or_name(column)._logical_expr, format))

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def datediff(end: ColumnOrName, start: ColumnOrName) -> Column:
    """Calculates the number of days between two date/timestamp columns.

    Args:
        end: To date column to work on.
        start: From date column to work on.

    Returns:
        A Column object with the difference in days between the two date/timestamp columns.

    Example:
        ```python
        # end: "2025-01-01", "2025-02-02", "2025-03-06"]
        # start: "2025-01-02", "2025-02-01", "2025-03-02"]
        df.select(dt.datediff(col("end"), col("start")).alias("date_diff")).to_pydict()
        # Output: {'date_diff': [-1, 1, 4]}
        ```
    """
    return Column._from_logical_expr(
        DateDiffExpr(
            Column._from_col_or_name(end)._logical_expr,
            Column._from_col_or_name(start)._logical_expr))

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def timestamp_diff(start: ColumnOrName, end: ColumnOrName, unit: DateTimeUnit) -> Column:
    """Calculates the difference between two timestamp columns.

    Args:
        start: The first column to calculate the difference from.
        end: The second column to calculate the difference from.
        unit: The unit of the difference.

    Returns:
        A Column object with the difference in the given unit between the two timestamp columns.

    Raises:
        ValueError: If unit is not supported, must be one of the supported ones.

    Notes:
        The supported units are: "year", "month", "day", "hour", "minute", "second", "millisecond".

    Example:
        ```python
        # start: "2025-01-01 10:00:00", "2025-02-02 11:00:00", "2025-03-06 12:00:00"]
        # end: "2025-01-02 10:00:00", "2025-02-01 11:00:00", "2025-03-01 12:00:00"]
        df.select(dt.timestamp_diff(col("start"), col("end"), "day").alias("ts_diff")).to_pydict()
        # Output: {'ts_diff': [-1, 1, 5]}
        ```
    """
    return Column._from_logical_expr(
        TimestampDiffExpr(
            Column._from_col_or_name(start)._logical_expr,
            Column._from_col_or_name(end)._logical_expr,
            unit))


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def to_utc_timestamp(column: ColumnOrName, tz: str) -> Column:
    """Accepts a Column with [TimestampType] (UTC), interprets each value as wall-clock time in the specified timezone `tz`, and converts it to a timestamp in UTC.

    Args:
        column: The column containing the timestamp.  Will be treated as timezone-agnostic.
        tz: A timezone that the input should be converted to.

    Returns:
        A Column object with timestamp expressed in UTC.

    Raises:
        TypeError: If column type is not a TimestampType.

    Notes:
        - In fenic, the TimestampType data type is always in UTC, which is not timezone-agnostic.
        - Spark also represents all timestamps as not timezone-agnostic, except Spark uses a timestamp type with the local session timezone.
        - Similarly to Spark to_utc_timestamp function, this function treats the input timestamp as timezone-agnostic, sets it to the requested timezone (without changing the timestamp), then converts the timestamp to UTC.
        - nulls are preserved.
        - Use this when data contains local/wall‑clock timestamps from `tz` (ignoring the UTC timezone in the type), and timestamp values converted to UTC are required.
        - For the inverse operation (UTC → local wall‑clock, then re‑expressed in UTC), see from_utc_timestamp.
        - see Spark documentation for more details: https://spark.apache.org/docs/4.0.1/api/python/reference/pyspark.sql/api/pyspark.sql.functions.to_utc_timestamp.html

    Example:
        ```python
        df.select("timestamp", dt.to_timestamp(col("timestamp"), "yyyy-MM-dd HH:mm:ss").alias("la_time")).show()
        # Output:
        #┌─────────────────────┬─────────────────────────┐
        #│ timestamp           ┆ la_time                 │
        #╞═════════════════════╪═════════════════════════╡
        #│ 2025-01-15 10:30:00 ┆ 2025-01-15 10:30:00 UTC │
        #│ 2025-01-16 14:00:00 ┆ 2025-01-16 14:00:00 UTC │
        #│ 2025-01-17 18:45:00 ┆ 2025-01-17 18:45:00 UTC │
        #└─────────────────────┴─────────────────────────┘
        #
        df.select(dt.to_utc_timestamp(col("la_time"), "America/Los_Angeles").alias("la_time_to_utc")).show()
        # Output:
        #┌─────────────────────────┐
        #│ la_time_to_utc          │
        #╞═════════════════════════╡
        #│ 2025-01-15 18:30:00 UTC │
        #│ 2025-01-16 22:00:00 UTC │
        #│ 2025-01-18 02:45:00 UTC │
        #└─────────────────────────┘
        ```
    """
    return Column._from_logical_expr(
        ToUTCTimestampExpr(
            Column._from_col_or_name(column)._logical_expr,
            tz))

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def from_utc_timestamp(column: ColumnOrName, tz: str) -> Column:
    """Accepts a Column with [TimestampType] (UTC).  For each row, converts the timestamp value to the provided `tz` timezone, then renders that timestamp as UTC without changing the timestamp value.  In other words, this function shifts the timestamp by the timezone offset `out = t+offset(t+tz)`.

    Args:
        column: The column containing the timestamp.
        tz: A timezone that the input will be converted from.

    Returns:
        A Column object with timestamp expressed in UTC.

    Raises:
        TypeError: If column type is not a TimestampType.

    Notes:
        - In fenic, the TimestampType data type is always in UTC, which is not timezone-agnostic.
        - Spark also represents all timestamps as not timezone-agnostic, except Spark uses a timestamp type with the local session timezone.
        - Similarly to Spark from_utc_timestamp function, this function will take a UTC timestamp and convert it to the requested timezone, then represent it as a timestamp in the session (UTC) timezone.
        - nulls are preserved.
        - Use when applying `tz` to timestamps in UTC and the resulting wall‑clock timestamp is required (though the result is still stored with a UTC timezone).
        - Use to_utc_timestamp for the inverse (treats naive/local values in tz, then converts to UTC).
        - see Spark documentation for more details: https://spark.apache.org/docs/4.0.1/api/python/reference/pyspark.sql/api/pyspark.sql.functions.from_utc_timestamp.html#

    Example:
        ```python
        df.select("timestamp", dt.to_timestamp(col("timestamp"), "yyyy-MM-dd HH:mm:ss").alias("utc_time")).show()
        # Output:
        #┌─────────────────────┬─────────────────────────┐
        #│ timestamp           ┆ utc_time                │
        #╞═════════════════════╪═════════════════════════╡
        #│ 2025-01-15 10:30:00 ┆ 2025-01-15 10:30:00 UTC │
        #│ 2025-01-16 14:00:00 ┆ 2025-01-16 14:00:00 UTC │
        #│ 2025-01-17 18:45:00 ┆ 2025-01-17 18:45:00 UTC │
        #└─────────────────────┴─────────────────────────┘
        #
        df.select(dt.from_utc_timestamp(col("utc_time"), "America/Los_Angeles").alias("la_time_in_utc")).show()
        # Output:
        #┌─────────────────────────┐
        #│ la_time_in_utc          │
        #╞═════════════════════════╡
        #│ 2025-01-15 02:30:00 UTC │
        #│ 2025-01-16 06:00:00 UTC │
        #│ 2025-01-17 10:45:00 UTC │
        #└─────────────────────────┘
        ```
    """
    return Column._from_logical_expr(
        FromUTCTimestampExpr(
            Column._from_col_or_name(column)._logical_expr,
            tz))

