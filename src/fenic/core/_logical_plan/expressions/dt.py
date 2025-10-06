from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from fenic.core._logical_plan import LogicalPlan

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.expressions.base import LogicalExpr
from fenic.core.types import (
    ColumnField,
    DataType,
    DateType,
    IntegerType,
    StringType,
    TimestampType,
)
from fenic.core.types.enums import (
    DateTimeUnit,
)

POLARS_DURATION_UNITS_MAP = {
    "year": "y",
    "month": "mo",
    "day": "d",
    "hour": "h",
    "minute": "m",
    "second": "s",
    "millisecond": "ms",
}

# DATE PART RELATED EXPRESSIONS

class DatePartRelatedExpr(LogicalExpr):
    """Base expression for date part related operations."""
    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self.temporal_type = DateType

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        self.temporal_type = _resolve_temporal_type(self.expr, plan, session_state)
        return ColumnField(str(self), IntegerType)

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class YearExpr(DatePartRelatedExpr):
    """Expression for extracting the year from a date type column."""
    def __init__(self, expr: LogicalExpr):
        super().__init__(expr)

    def __str__(self):
        return f"dt.year({self.expr})"

    def _eq_specific(self, other: YearExpr) -> bool:
        return self.expr == other.expr

class MonthExpr(DatePartRelatedExpr):
    """Expression for extracting the month from a date type column."""

    def __init__(self, expr: LogicalExpr):
        super().__init__(expr)

    def __str__(self):
        return f"dt.month({self.expr})"

    def _eq_specific(self, other: MonthExpr) -> bool:
        return True


class DayExpr(DatePartRelatedExpr):
    """Expression for extracting the day from a date type column."""
    def __init__(self, expr: LogicalExpr):
        super().__init__(expr)

    def __str__(self):
        return f"dt.day({self.expr})"

    def _eq_specific(self, other: DayExpr) -> bool:
        return True


class HourExpr(DatePartRelatedExpr):
    """Expression for extracting the hour from a date type column."""
    def __init__(self, expr: LogicalExpr):
        super().__init__(expr)

    def __str__(self):
        return f"dt.hour({self.expr})"

    def _eq_specific(self, other: HourExpr) -> bool:
        return True


class MinuteExpr(DatePartRelatedExpr):
    """Expression for extracting the minute from a date type column."""
    def __init__(self, expr: LogicalExpr):
        super().__init__(expr)

    def __str__(self):
        return f"dt.minute({self.expr})"

    def _eq_specific(self, other: MinuteExpr) -> bool:
        return True


class SecondExpr(DatePartRelatedExpr):
    """Expression for extracting the minute from a date type column."""
    def __init__(self, expr: LogicalExpr):
        self.expr = expr

    def __str__(self):
        return f"dt.second({self.expr})"

    def _eq_specific(self, other: SecondExpr) -> bool:
        return True


class MilliSecondExpr(DatePartRelatedExpr):
    """Expression for extracting the millisecond from a date type column."""
    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self.is_date = False

    def __str__(self):
        return f"dt.millisecond({self.expr})"

    def _eq_specific(self, other: MilliSecondExpr) -> bool:
        return True


# DATE CONVERSION EXPRESSIONS


class DateConversionExpr(LogicalExpr):
    """Base expression for date conversion operations."""
    def __init__(self, expr: LogicalExpr, format: Optional[str]):
        self.expr = expr
        self.original_format = format
        self.format = _java_like_to_chrono(self.original_format) if self.original_format else self.get_default_format()

    def children(self) -> List[LogicalExpr]:
        return [self.expr]
    
    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        col_expr = self.expr.to_column_field(plan, session_state)
        if col_expr.data_type != StringType:
            raise TypeError(f"Expected StringType, got {col_expr.data_type}")
        return ColumnField(str(self), self.get_return_type())

    @classmethod
    @abstractmethod
    def get_default_format(cls) -> str:
        """Returns the default chrono format for the expression."""
        pass
    
    @classmethod
    @abstractmethod
    def get_return_type(cls) -> DataType:
        """Returns the return type for the expression."""
        pass


class ToDateExpr(DateConversionExpr):
    """Parses a string to a date."""
    def __init__(self, expr: LogicalExpr, format: Optional[str]):
        super().__init__(expr, format)

    def __str__(self):
        return f"dt.to_date({self.expr})"

    def _eq_specific(self, other: ToDateExpr) -> bool:
        return True
    
    @classmethod
    def get_default_format(cls) -> str:
        """Returns the default chronoformat for the expression."""
        return "%Y-%m-%d"

    @classmethod
    def get_return_type(cls) -> DataType:
        """Returns the return type for the expression."""
        return DateType


class ToTimestampExpr(DateConversionExpr):
    """Parses a string to a timestamp."""
    def __init__(self, expr: LogicalExpr, format: Optional[str]):
        super().__init__(expr, format)

    def __str__(self):
        return f"dt.to_timestamp({self.expr})"

    def _eq_specific(self, other: ToTimestampExpr) -> bool:
        return True

    @classmethod
    def get_default_format(cls) -> str:
        """Returns ISO 8601 with milliseconds as the default chronoformat."""
        return "%Y-%m-%dT%H:%M:%S.%3f"
    
    @classmethod
    def get_return_type(cls) -> DataType:
        """Returns the return type for the expression."""
        return TimestampType


# CURRENT DATE AND TIME EXPRESSIONS


class NowExpr(LogicalExpr):
    """Expression for getting the current date and time."""
    def __init__(self, as_date: bool = False):
        super().__init__()
        self.as_date = as_date

    def __str__(self):
        return f"dt.now(as_date={self.as_date})"
    
    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        if self.as_date:
            return ColumnField(str(self), DateType)
        return ColumnField(str(self), TimestampType)
    
    def children(self) -> List[LogicalExpr]:
        return []

    def _eq_specific(self, other: NowExpr) -> bool:
        return True


 # DATE & TIME MANIPULATION EXPRESSIONS


class DateTruncExpr(LogicalExpr):
    """Expression for truncating a date to a given unit."""
    def __init__(self, expr: LogicalExpr, unit: DateTimeUnit):
        super().__init__()
        self.expr = expr
        self.original_unit = unit
        self.unit = "1" + _get_polars_duration_unit(self.original_unit)

    def __str__(self):
        return f"dt.date_trunc(unit={self.unit})"

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        self.temporal_type = _resolve_temporal_type(self.expr, plan, session_state)
        return ColumnField(str(self), self.temporal_type)
    
    def children(self) -> List[LogicalExpr]:
        return [self.expr]
    
    def _eq_specific(self, other: DateTruncExpr) -> bool:
        return self.unit == other.unit

class DateAddExpr(LogicalExpr):
    """Expression for adding a given amount of days to a date/timestamp column."""
    def __init__(self, expr: LogicalExpr, days: LogicalExpr, sub: bool = False):
        self.expr = expr
        self.days = days
        self.sub = sub

    def __str__(self):
        return f"dt.date_add(days={self.days})"
    
    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        self.temporal_type = _resolve_temporal_type(self.expr, plan, session_state)

        days_col = self.days.to_column_field(plan, session_state)
        if days_col.data_type != IntegerType:
            raise TypeError(f"Expected IntegerType, got {days_col.data_type}")

        return ColumnField(str(self), self.temporal_type)
    
    def children(self) -> List[LogicalExpr]:
        return [self.expr, self.days]
    
    def _eq_specific(self, other: DateAddExpr) -> bool:
        return self.days == other.days


class TimestampAddExpr(LogicalExpr):
    """Expression for adding a given quantity of a unit to a timestamp column."""
    def __init__(self, expr: LogicalExpr, quantity: LogicalExpr, unit: DateTimeUnit):
        self.expr = expr
        self.quantity = quantity
        self.original_unit = unit
        self.unit = _get_polars_duration_unit(self.original_unit)

    def __str__(self):
        return f"dt.timestamp_add(quantity={self.quantity}, unit={self.unit})"
    
    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        self.temporal_type = _resolve_temporal_type(self.expr, plan, session_state)

        quantity_col = self.quantity.to_column_field(plan, session_state)
        if quantity_col.data_type != IntegerType:
            raise TypeError(f"Expected IntegerType, got {quantity_col.data_type}")

        return ColumnField(str(self), self.temporal_type)
    
    def children(self) -> List[LogicalExpr]:
        return [self.expr, self.quantity]
    
    def _eq_specific(self, other: TimestampAddExpr) -> bool:
        return self.quantity == other.quantity and self.unit == other.unit


class DateFormatExpr(LogicalExpr):
    """Expression for formatting a date/timestamp column to a given format."""
    def __init__(self, expr: LogicalExpr, format: str):
        super().__init__()
        self.expr = expr
        self.original_format = format
        # Convert the format to a Chrono (strftime) format.
        self.format = _java_like_to_chrono(self.original_format)

    def __str__(self):
        return f"dt.date_format(format={self.format})"
    
    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        # In the case of date_format, we should accept a timestamp with a timezone.
        self.temporal_type = _resolve_temporal_type(self.expr, plan, session_state)
        return ColumnField(str(self), StringType)
    
    def children(self) -> List[LogicalExpr]:
        return [self.expr]
    
    def _eq_specific(self, other: DateFormatExpr) -> bool:
        return self.format == other.format


class DateDiffExpr(LogicalExpr):
    """Expression for calculating the difference between two date/timestamp columns."""
    def __init__(self, end: LogicalExpr, start: LogicalExpr):
        self.end = end
        self.start = start

    def __str__(self):
        return f"dt.datediff(end={self.end}, start={self.start})"

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        _resolve_temporal_type(self.end, plan, session_state)
        _resolve_temporal_type(self.start, plan, session_state)
        return ColumnField(str(self), IntegerType)

    def children(self) -> List[LogicalExpr]:
        return [self.start, self.end]

    def _eq_specific(self, other: DateDiffExpr) -> bool:
        return self.end == other.end and self.start == other.start


class TimestampDiffExpr(LogicalExpr):
    """Expression for calculating the difference between two timestamp columns."""
    def __init__(self, start: LogicalExpr, end: LogicalExpr, unit: DateTimeUnit):
        self.start = start
        self.end = end
        self.unit = unit

    def __str__(self):
        return f"dt.timestamp_diff(start={self.start}, end={self.end}, unit={self.unit})"

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        _resolve_temporal_type(self.start, plan, session_state)
        _resolve_temporal_type(self.end, plan, session_state)
        return ColumnField(str(self), IntegerType)

    def children(self) -> List[LogicalExpr]:
        return [self.start, self.end]

    def _eq_specific(self, other: TimestampDiffExpr) -> bool:
        return self.start == other.start and self.end == other.end and self.unit == other.unit


class ToUTCTimestampExpr(LogicalExpr):
    """Expression for converting a timestamp to UTC."""
    def __init__(self, expr: LogicalExpr, timezone: str):
        self.expr = expr
        self.timezone = timezone

    def __str__(self):
        return f"dt.to_utc_timestamp({self.expr}, timezone={self.timezone})"

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        _resolve_temporal_type(self.expr, plan, session_state)
        # The resulting type is always UTC
        return ColumnField(str(self), TimestampType)

    def children(self) -> List[LogicalExpr]:
        return [self.expr]

    def _eq_specific(self, other: ToUTCTimestampExpr) -> bool:
        return self.expr == other.expr and self.timezone == other.timezone


class FromUTCTimestampExpr(LogicalExpr):
    """Expression for converting a timestamp from UTC to a given tz."""
    def __init__(self, expr: LogicalExpr, timezone: str):
        self.expr = expr
        self.timezone = timezone

    def __str__(self):
        return f"dt.from_utc_timestamp({self.expr}, timezone={self.timezone})"

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        _resolve_temporal_type(self.expr, plan, session_state, check_utc=False)
        return ColumnField(str(self), TimestampType)

    def children(self) -> List[LogicalExpr]:
        return [self.expr]

    def _eq_specific(self, other: FromUTCTimestampExpr) -> bool:
        return self.expr == other.expr and self.timezone == other.timezone


# UTILITY FUNCTIONS

def _resolve_temporal_type(expr: LogicalExpr, plan: LogicalPlan, session_state: BaseSessionState, check_utc: bool = False) -> bool:
    """Resolve the expression's type if it's DATE or TIMESTAMP, otherwise raise an error."""
    col_field = expr.to_column_field(plan, session_state)
    if col_field.data_type == DateType or col_field.data_type == TimestampType:
        return col_field.data_type
    raise TypeError(f"Expected one of DateType or TimestampType, got {col_field.data_type}")

def _repeat_map(ch: str, n: int) -> str:
    """Map a run of the same pattern letter (e.g., 'yyyy' or 'SSS') to Chrono's strftime.

    Defaults to reasonable fallbacks when an exact width distinction doesn't exist.
    """
    # Year
    if ch == 'y':
        return "%Y" if n >= 3 else "%y"

    # Month (1-12 / names)
    if ch == 'M':
        if n == 1:
            return "%-m"  # no leading zero (Chrono on Unix); fallback "%m" if you prefer fixed two-digit
        if n == 2:
            return "%m"
        if n == 3:
            return "%b"   # Jan
        return "%B"       # January

    # Day of month
    if ch == 'd':
        if n == 1:
            return "%-d"
        return "%d"

    # 24-hour / 12-hour
    if ch == 'H':
        if n == 1:
            return "%-H"
        return "%H"
    if ch == 'h':
        if n == 1:
            return "%-I"
        return "%I"

    # Minute
    if ch == 'm':
        if n == 1:
            return "%-M"
        return "%M"

    # Second
    if ch == 's':
        if n == 1:
            return "%-S"
        return "%S"

    # Fractional seconds (Java SDF uses 'S'; some folks use 'f'—support both)
    if ch in ('S', 'f'):
        # User explicitly wants %3f style for milliseconds length 3, etc.
        return f"%{n}f"

    # Day of week (E or e in some impls). We'll handle 'E' like SDF.
    if ch == 'E':
        return "%A" if n >= 4 else "%a"

    # AM/PM
    if ch == 'a':
        return "%p"

    # Time zone:
    #   Z (RFC 822) -> %z (e.g., -0800)
    #   X (ISO 8601) widths vary; map to %z/%:z best-effort
    if ch == 'Z':
        return "%z"
    if ch == 'X':
        # SDF: X, XX, XXX => -08, -0800, -08:00 respectively.
        # Chrono supports %z (-0800) and %:z (-08:00). There's no bare -08, so map:
        return "%:z" if n >= 3 else "%z"
    if ch == 'V':
        # zone-name
        # this should be a the full zone name, that is not available from Polars,
        # so we'll print the zone name.
        # e.g. "America/Los_Angeles" -> 'PST'
        return "%Z"
    if ch == 'z':
        # zone-name
        return "%Z"

    # Quarter (not standard in strftime; leave literal Q's/numbers)
    if ch == 'Q':
        return "Q" * n

    # Fallback: treat unknown pattern letters as literals
    return ch * n


def _java_like_to_chrono(pattern: str) -> str:
    """Translate a PySpark/Java SimpleDateFormat-like pattern into Chrono's strftime format.

    - Supports quoted literals with single quotes.
    - Coalesces runs of letters and maps via _repeat_map.
    """
    # Split into literal and pattern segments (single quotes denote literals; '' => single quote)
    # This regex captures either a quoted literal (including escaped '' inside) or a run of non-quoted text.
    tokens = []
    i = 0
    n = len(pattern)

    while i < n:
        if pattern[i] == "'":
            # Parse a quoted literal, handling doubled single quotes
            i += 1
            lit = []
            while i < n:
                if pattern[i] == "'":
                    if i + 1 < n and pattern[i + 1] == "'":  # escaped quote
                        lit.append("'")
                        i += 2
                    else:
                        i += 1
                        break
                else:
                    lit.append(pattern[i])
                    i += 1
            tokens.append(("LIT", "".join(lit)))
        else:
            # Accumulate until next quote
            j = i
            while j < n and pattern[j] != "'":
                j += 1
            tokens.append(("PAT", pattern[i:j]))
            i = j

    # Now process pattern segments, mapping runs of identical letters; leave non-letters as literals
    def _map_segment(seg: str) -> str:
        out = []
        k = 0
        L = len(seg)
        while k < L:
            ch = seg[k]
            if ch.isalpha():
                # count run
                r = k + 1
                while r < L and seg[r] == ch:
                    r += 1
                out.append(_repeat_map(ch, r - k))
                k = r
            else:
                # literal character (e.g., -, :, /, T, space)
                out.append(ch)
                k += 1
        return "".join(out)

    result_parts = []
    for kind, val in tokens:
        if kind == "LIT":
            # Escape % inside literals by doubling %% so Chrono treats it literal
            result_parts.append(val.replace("%", "%%"))
        else:
            result_parts.append(_map_segment(val))

    return "".join(result_parts)


def _get_polars_duration_unit(unit: DateTimeUnit) -> str:
    """Returns the Polars duration unit for a given unit.

    For more information checkout the Polars documentation:
    https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.dt.offset_by.html
    """
    if unit not in POLARS_DURATION_UNITS_MAP:
        raise ValueError(f"Unsupported unit: {unit}. Supported units are: {POLARS_DURATION_UNITS_MAP.keys()}")

    return POLARS_DURATION_UNITS_MAP.get(unit)
