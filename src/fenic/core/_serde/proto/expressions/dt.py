from fenic.core._logical_plan.expressions.dt import (
    DateAddExpr,
    DateDiffExpr,
    DateFormatExpr,
    DateTruncExpr,
    DayExpr,
    FromUTCTimestampExpr,
    HourExpr,
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
from fenic.core._serde.proto.expression_serde import (
    _deserialize_logical_expr_helper,
    serialize_logical_expr,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import (
    DateAddExprProto,
    DateDiffExprProto,
    DateFormatExprProto,
    DateTruncExprProto,
    DayExprProto,
    FromUTCTimestampExprProto,
    HourExprProto,
    LogicalExprProto,
    MilliSecondExprProto,
    MinuteExprProto,
    MonthExprProto,
    NowExprProto,
    SecondExprProto,
    TimestampAddExprProto,
    TimestampDiffExprProto,
    ToDateExprProto,
    ToTimestampExprProto,
    ToUTCTimestampExprProto,
    YearExprProto,
)

# =============================================================================
# Date part expressions
# =============================================================================

# Year
@serialize_logical_expr.register
def _serialize_year_expr(logical: YearExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(year=YearExprProto(expr=serialize_logical_expr(logical.expr, context)))

@_deserialize_logical_expr_helper.register
def _deserialize_year_expr(logical_proto: YearExprProto, context: SerdeContext) -> YearExpr:
    return YearExpr(context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr))

# Month
@serialize_logical_expr.register
def _serialize_month_expr(logical: MonthExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(month=MonthExprProto(expr=serialize_logical_expr(logical.expr, context)))

@_deserialize_logical_expr_helper.register
def _deserialize_month_expr(logical_proto: MonthExprProto, context: SerdeContext) -> MonthExpr:
    return MonthExpr(context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr))

# Day
@serialize_logical_expr.register
def _serialize_day_expr(logical: DayExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(day=DayExprProto(expr=serialize_logical_expr(logical.expr, context)))

@_deserialize_logical_expr_helper.register
def _deserialize_day_expr(logical_proto: DayExprProto, context: SerdeContext) -> DayExpr:
    return DayExpr(context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr))

# Hour
@serialize_logical_expr.register
def _serialize_hour_expr(logical: HourExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(hour=HourExprProto(expr=serialize_logical_expr(logical.expr, context)))

@_deserialize_logical_expr_helper.register
def _deserialize_hour_expr(logical_proto: HourExprProto, context: SerdeContext) -> HourExpr:
    return HourExpr(context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr))
# Minute
@serialize_logical_expr.register
def _serialize_minute_expr(logical: MinuteExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(minute=MinuteExprProto(expr=serialize_logical_expr(logical.expr, context)))

@_deserialize_logical_expr_helper.register
def _deserialize_minute_expr(logical_proto: MinuteExprProto, context: SerdeContext) -> MinuteExpr:
    return MinuteExpr(context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr))

# Second
@serialize_logical_expr.register
def _serialize_second_expr(logical: SecondExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(second=SecondExprProto(expr=serialize_logical_expr(logical.expr, context)))

@_deserialize_logical_expr_helper.register
def _deserialize_second_expr(logical_proto: SecondExprProto, context: SerdeContext) -> SecondExpr:
    return SecondExpr(context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr))

# MilliSecond
@serialize_logical_expr.register
def _serialize_milli_second_expr(logical: MilliSecondExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(millisecond=MilliSecondExprProto(expr=serialize_logical_expr(logical.expr, context)))

@_deserialize_logical_expr_helper.register
def _deserialize_milli_second_expr(logical_proto: MilliSecondExprProto, context: SerdeContext) -> MilliSecondExpr:
    return MilliSecondExpr(context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr))

# =============================================================================
# Date arithmetic expressions
# =============================================================================

# DateAddExpr
@serialize_logical_expr.register
def _serialize_date_add_expr(logical: DateAddExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(
        date_add=DateAddExprProto(
            expr=serialize_logical_expr(logical.expr, context),
            days=serialize_logical_expr(logical.days, context),
            sub=logical.sub))

@_deserialize_logical_expr_helper.register
def _deserialize_date_add_expr(logical_proto: DateAddExprProto, context: SerdeContext) -> DateAddExpr:
    return DateAddExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        days=context.deserialize_logical_expr("days", logical_proto.days),
        sub=logical_proto.sub)

# TimestampAddExpr
@serialize_logical_expr.register
def _serialize_timestamp_add_expr(logical: TimestampAddExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(
        timestamp_add=TimestampAddExprProto(
            expr=serialize_logical_expr(logical.expr, context),
            quantity=serialize_logical_expr(logical.quantity, context),
            unit=logical.original_unit))

@_deserialize_logical_expr_helper.register
def _deserialize_timestamp_add_expr(logical_proto: TimestampAddExprProto, context: SerdeContext) -> TimestampAddExpr:
    return TimestampAddExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        quantity=context.deserialize_logical_expr("quantity", logical_proto.quantity),
        unit=logical_proto.unit)

# DateTruncExpr
@serialize_logical_expr.register
def _serialize_date_trunc_expr(logical: DateTruncExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(
        date_trunc=DateTruncExprProto(
            expr=serialize_logical_expr(logical.expr, context),
            unit=logical.original_unit))

@_deserialize_logical_expr_helper.register
def _deserialize_date_trunc_expr(logical_proto: DateTruncExprProto, context: SerdeContext) -> DateTruncExpr:
    return DateTruncExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        unit=logical_proto.unit)

# DateDiffExpr
@serialize_logical_expr.register
def _serialize_date_diff_expr(logical: DateDiffExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(
        date_diff=DateDiffExprProto(
            end=serialize_logical_expr(logical.end, context),
            start=serialize_logical_expr(logical.start, context)))

@_deserialize_logical_expr_helper.register
def _deserialize_date_diff_expr(logical_proto: DateDiffExprProto, context: SerdeContext) -> DateDiffExpr:
    return DateDiffExpr(
        end=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.end),
        start=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.start))

# TimestampDiffExpr
@serialize_logical_expr.register
def _serialize_timestamp_diff_expr(logical: TimestampDiffExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(
        timestamp_diff=TimestampDiffExprProto(
            start=serialize_logical_expr(logical.start, context),
            end=serialize_logical_expr(logical.end, context),
            unit=logical.unit))

@_deserialize_logical_expr_helper.register
def _deserialize_timestamp_diff_expr(logical_proto: TimestampDiffExprProto, context: SerdeContext) -> TimestampDiffExpr:
    return TimestampDiffExpr(
        start=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.start),
        end=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.end),
        unit=logical_proto.unit)


# =============================================================================
# Current date/time expressions
# =============================================================================

# NowExpr
@serialize_logical_expr.register
def _serialize_now_expr(logical: NowExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(now=NowExprProto(as_date=logical.as_date))

@_deserialize_logical_expr_helper.register
def _deserialize_now_expr(logical_proto: NowExprProto, context: SerdeContext) -> NowExpr:
    return NowExpr(as_date=logical_proto.as_date)

# =============================================================================
# Date conversion expressions
# =============================================================================

# ToDateExpr
@serialize_logical_expr.register
def _serialize_to_date_expr(logical: ToDateExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(
        to_date=ToDateExprProto(
            expr=serialize_logical_expr(logical.expr, context),
            format=logical.original_format))

@_deserialize_logical_expr_helper.register
def _deserialize_to_date_expr(logical_proto: ToDateExprProto, context: SerdeContext) -> ToDateExpr:
    return ToDateExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        format=logical_proto.format)

# ToTimestampExpr
@serialize_logical_expr.register
def _serialize_to_timestamp_expr(logical: ToTimestampExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(
        to_timestamp=ToTimestampExprProto(
            expr=serialize_logical_expr(logical.expr, context),
            format=logical.original_format))

@_deserialize_logical_expr_helper.register
def _deserialize_to_timestamp_expr(logical_proto: ToTimestampExprProto, context: SerdeContext) -> ToTimestampExpr:
    return ToTimestampExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        format=logical_proto.format)

# DateFormatExpr
@serialize_logical_expr.register
def _serialize_date_format_expr(logical: DateFormatExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(
        date_format=DateFormatExprProto(
            expr=serialize_logical_expr(logical.expr, context),
            format=logical.original_format))

@_deserialize_logical_expr_helper.register
def _deserialize_date_format_expr(logical_proto: DateFormatExprProto, context: SerdeContext) -> DateFormatExpr:
    return DateFormatExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        format=logical_proto.format)

# ToUTCTimestampExpr
@serialize_logical_expr.register
def _serialize_to_utc_timestamp_expr(logical: ToUTCTimestampExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(
        to_utc_timestamp=ToUTCTimestampExprProto(
            expr=serialize_logical_expr(logical.expr, context),
            timezone=logical.timezone))

@_deserialize_logical_expr_helper.register
def _deserialize_to_utc_timestamp_expr(logical_proto: ToUTCTimestampExprProto, context: SerdeContext) -> ToUTCTimestampExpr:
    return ToUTCTimestampExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        timezone=logical_proto.timezone)

# FromUTCTimestampExpr
@serialize_logical_expr.register
def _serialize_from_utc_timestamp_expr(logical: FromUTCTimestampExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(
        from_utc_timestamp=FromUTCTimestampExprProto(
            expr=serialize_logical_expr(logical.expr, context),
            timezone=logical.timezone))

@_deserialize_logical_expr_helper.register
def _deserialize_from_utc_timestamp_expr(logical_proto: FromUTCTimestampExprProto, context: SerdeContext) -> FromUTCTimestampExpr:
    return FromUTCTimestampExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        timezone=logical_proto.timezone)

