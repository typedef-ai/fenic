import datetime
from typing import Any, Callable, List, Optional, Type

import polars as pl
import pytest

from fenic._backends.local.physical_plan.utils import apply_ingestion_coercions
from fenic.api.dataframe import DataFrame
from fenic.api.functions import col, lit
from fenic.api.functions.dt import (
    current_date,
    current_timestamp,
    date_add,
    date_format,
    date_sub,
    date_trunc,
    datediff,
    day,
    hour,
    millisecond,
    minute,
    month,
    now,
    second,
    timestamp_add,
    timestamp_diff,
    to_date,
    to_timestamp,
    year,
)
from fenic.core.types import (
    ColumnField,
    DateType,
    IntegerType,
    StringType,
    TimestampType,
)

# Full timestamp results.
TS_FULL_RESULTS = [
    datetime.datetime(2025, 1, 1, 1, 0),
    datetime.datetime(2025, 2, 2, 2, 20, 0, 200000),
    datetime.datetime(2025, 3, 3, 15, 33, 30, 300000)
]

TS_FULL_RESULTS_PLUS1 = [
    datetime.datetime(2025, 1, 2, 1, 0),
    datetime.datetime(2025, 2, 3, 2, 20, 0, 200000),
    datetime.datetime(2025, 3, 4, 15, 33, 30, 300000)
]

TS_FULL_RESULTS_MINUS1 = [
    datetime.datetime(2024, 12, 31, 1, 0),
    datetime.datetime(2025, 2, 1, 2, 20, 0, 200000),
    datetime.datetime(2025, 3, 2, 15, 33, 30, 300000)
]

TS_FULL_RESULTS_MINUS_100_MS = [
    datetime.datetime(2025, 1, 1, 0, 59, 59, 900000),
    datetime.datetime(2025, 2, 2, 2, 20, 0, 100000),
    datetime.datetime(2025, 3, 3, 15, 33, 30, 200000)
]

TS_FULL_RESULTS_PLUS1YEAR = [
    datetime.datetime(2026, 1, 1, 1, 0),
    datetime.datetime(2026, 2, 2, 2, 20, 0, 200000),
    datetime.datetime(2026, 3, 3, 15, 33, 30, 300000)
]

# Full date results.
DT_FULL_RESULTS = [
    datetime.date(2025, 1, 1),
    datetime.date(2025, 1, 2),
    datetime.date(2025, 1, 3)
]

DT_FULL_RESULTS_MINUS1 = [
    datetime.date(2024, 12, 31),
    datetime.date(2025, 1, 1),
    datetime.date(2025, 1, 2)
]

DT_FULL_RESULTS_PLUS1 = [
    datetime.date(2025, 1, 2),
    datetime.date(2025, 1, 3),
    datetime.date(2025, 1, 4)
]

DT_FULL_RESULTS_PLUS1YEAR = [
    datetime.date(2026, 1, 1),
    datetime.date(2026, 1, 2),
    datetime.date(2026, 1, 3)
]


@pytest.fixture
def df_with_date_types(local_session):
    pl_df = pl.DataFrame({
        "col1": [1, 2, 3],
        "date_str": [
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
        ],
        "date": [
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
        ],
        "date_plus": [
            "2025-01-10",
            "2025-01-20",
            "2025-01-30",
        ],
        "ts_str": [
            "2025-01-01T1:00:00.000",
            "2025-02-02T02:20:00.200",
            "2025-03-03T15:33:30.300",
        ]
    },
    schema={
        "col1": pl.Int64,
        "date_str": pl.String,
        "date": pl.Date,
        "date_plus": pl.Date,
        "ts_str": pl.String,
    })

    pl_df = pl_df.with_columns(
        pl.col("ts_str").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S.%3f").alias("ts")
    )

    pl_df = pl_df.with_columns(
        pl.col("ts_str").str.strptime(pl.Date, format="%Y-%m-%dT%H:%M:%S.%3f").alias("dt")
    )

    return local_session.create_dataframe(pl_df)

class TestDateFunctions:
    def test_year(self, df_with_date_types):
        """Test the year function."""
        default_year_results = [2025]*3

        df = df_with_date_types.select(year(col("ts")).alias("year"))
        self._verify_results_on_df(df, IntegerType, "year", default_year_results)

        self._verify_results(df_with_date_types, "date", year, IntegerType, "year", default_year_results)
        self._verify_error(df_with_date_types, "col1", year, TypeError)

    def test_month(self, df_with_date_types):
        """Test the month function."""

        df = df_with_date_types.select(month(col("ts")).alias("month"))
        self._verify_results_on_df(df, IntegerType, "month", [1, 2, 3])

        self._verify_results(df_with_date_types, "date", month, IntegerType, "month", [1, 1, 1])
        self._verify_error(df_with_date_types, "col1", month, TypeError)

    def test_day(self, df_with_date_types):
        """Test the day function."""

        df = df_with_date_types.select(day(col("ts")).alias("day"))
        self._verify_results_on_df(df, IntegerType, "day", [1, 2, 3])

        self._verify_results(df_with_date_types, "date", day, IntegerType, "day", [1, 2, 3])
        self._verify_error(df_with_date_types, "col1", day, TypeError)

    def test_hour(self, df_with_date_types):
        """Test the hour function."""

        df = df_with_date_types.select(hour(col("ts")).alias("hour"))
        self._verify_results_on_df(df, IntegerType, "hour", [1, 2, 15])

        self._verify_results(df_with_date_types, "date", hour, IntegerType, "hour", [0, 0, 0])
        self._verify_error(df_with_date_types, "col1", hour, TypeError)

    def test_minute(self, df_with_date_types):
        """Test the minute function."""

        df = df_with_date_types.select(minute(col("ts")).alias("minute"))
        self._verify_results_on_df(df, IntegerType, "minute", [0, 20, 33])

        self._verify_results(df_with_date_types, "date", minute, IntegerType, "minute", [0, 0, 0])
        self._verify_error(df_with_date_types, "col1", minute, TypeError)

    def test_second(self, df_with_date_types):
        """Test the second function."""

        df = df_with_date_types.select(second(col("ts")).alias("second"))
        self._verify_results_on_df(df, IntegerType, "second", [0, 0, 30])

        self._verify_results(df_with_date_types, "date", second, IntegerType, "second", [0, 0, 0])
        self._verify_error(df_with_date_types, "col1", second, TypeError)

    def test_millisecond(self, df_with_date_types):
        """Test the millisecond function."""

        df = df_with_date_types.select(millisecond(col("ts")).alias("millisecond"))
        self._verify_results_on_df(df, IntegerType, "millisecond", [0, 200, 300])

        self._verify_results(df_with_date_types, "date", millisecond, IntegerType, "millisecond", [0, 0, 0])
        self._verify_error(df_with_date_types, "col1", millisecond, TypeError)

    def test_to_timestamp(self, df_with_date_types):
        """Test the to_timestamp function."""
        df = df_with_date_types.select(to_timestamp(col("ts_str")).alias("ts2"))
        self._verify_results_on_df(df, TimestampType, "ts2", TS_FULL_RESULTS)

        self._verify_error(df_with_date_types, "col1", to_timestamp, TypeError)

    def test_to_date(self, df_with_date_types, local_session):
        """Test the to_date function."""
        df = local_session.create_dataframe(pl.DataFrame({
            "date_str": ["01-27-2025", "11-02-2025"],
        }))
        df = df.select(to_date(col("date_str"), format="MM-dd-yyyy").alias("dt2"))
        self._verify_results_on_df(df, DateType, "dt2", [
            datetime.date(2025, 1, 27),
            datetime.date(2025, 11, 2),
        ])
        self._verify_results(df_with_date_types, "date_str", to_date, DateType, "dt2", DT_FULL_RESULTS)
        self._verify_error(df_with_date_types, "col1", to_date, TypeError)

    def test_current_date_functions(self, df_with_date_types):
        """Test the current date and time functions."""
        now_year = datetime.datetime.now().year
        now_year_results = [now_year]*3

        # now
        df = df_with_date_types.with_column("now", now())
        df = df.select(col("now"))
        assert df.schema.column_fields == [ColumnField("now", TimestampType)]

        df = df.select(year(col("now")).alias("now_year"))
        self._verify_results_on_df(df, IntegerType, "now_year", now_year_results)

        # current_timestamp
        df = df_with_date_types.with_column("current_timestamp", current_timestamp())
        df = df.select(col("current_timestamp"))
        assert df.schema.column_fields == [ColumnField("current_timestamp", TimestampType)]

        df = df.select(year(col("current_timestamp")).alias("current_timestamp_year"))
        self._verify_results_on_df(df, IntegerType, "current_timestamp_year", now_year_results)

        # current_date
        df = df_with_date_types.with_column("current_date", current_date())
        df = df.select(col("current_date"))
        assert df.schema.column_fields == [ColumnField("current_date", DateType)]

        df = df.select(year(col("current_date")).alias("current_date_year"))
        self._verify_results_on_df(df, IntegerType, "current_date_year", now_year_results)

    def test_date_trunc(self, df_with_date_types):
        """Test the date_trunc function."""
        self._verify_error(df_with_date_types, "ts", date_trunc, ValueError, ["invalid"])
        self._verify_error(df_with_date_types, "col1", date_trunc, TypeError, ["year"])

        df = df_with_date_types.select(date_trunc(col("ts"), "year").alias("ts_trunc"))
        self._verify_results_on_df(df, TimestampType, "ts_trunc", [datetime.datetime(2025, 1, 1, 0, 0)] * 3)

        self._verify_results(df_with_date_types, "date", date_trunc, DateType, "dt_trunc", [datetime.date(2025, 1, 1)] * 3, ["year"])
        self._verify_results(df_with_date_types, "ts", date_trunc, TimestampType, "ts_trunc",
            [
                datetime.datetime(2025, 1, 1, 0, 0),
                datetime.datetime(2025, 2, 1, 0, 0),
                datetime.datetime(2025, 3, 1, 0, 0)
            ], ["month"])
        self._verify_results(df_with_date_types, "ts", date_trunc, TimestampType, "ts_trunc",
            [
                datetime.datetime(2025, 1, 1, 0, 0),
                datetime.datetime(2025, 2, 2, 0, 0),
                datetime.datetime(2025, 3, 3, 0, 0)
            ], ["day"])
        self._verify_results(df_with_date_types, "ts", date_trunc, TimestampType, "ts_trunc",
            [
                datetime.datetime(2025, 1, 1, 1, 0),
                datetime.datetime(2025, 2, 2, 2, 0),
                datetime.datetime(2025, 3, 3, 15, 0)
            ], ["hour"])
        self._verify_results(df_with_date_types, "date", date_trunc, DateType, "dt_trunc", [
            datetime.date(2025, 1, 1),
            datetime.date(2025, 1, 2),
            datetime.date(2025, 1, 3)
        ], ["hour"])

    def test_date_add(self, local_session, df_with_date_types):
        """Test the date_add function."""
        self._verify_error(df_with_date_types, "col1", date_add, TypeError, ["1"])
        self._verify_error(df_with_date_types, "ts", date_add, TypeError, [col("ts")])

        df = df_with_date_types.select(date_add(col("ts"), 1).alias("ts_add"))
        self._verify_results_on_df(df, TimestampType, "ts_add", TS_FULL_RESULTS_PLUS1)

        self._verify_results(df_with_date_types, "date", date_add, DateType, "dt_add", DT_FULL_RESULTS_MINUS1, [-1])
        self._verify_results(df_with_date_types, "date", date_sub, DateType, "dt_sub", DT_FULL_RESULTS_PLUS1, [-1])

        df = df_with_date_types.with_column("days", lit(-1))
        self._verify_results(df, "date", date_sub, DateType, "dt_sub", DT_FULL_RESULTS_PLUS1, [col("days")])

    def test_date_sub(self, df_with_date_types):
        """Test the date_sub function."""
        self._verify_error(df_with_date_types, "col1", date_sub, TypeError, ["1"])
        self._verify_error(df_with_date_types, "ts", date_sub, TypeError, [col("ts")])

        df = df_with_date_types.select(date_sub(col("ts"), 1).alias("ts_sub"))
        self._verify_results_on_df(df, TimestampType, "ts_sub", TS_FULL_RESULTS_MINUS1)

        self._verify_results(df_with_date_types, "date", date_sub, DateType, "dt_sub", DT_FULL_RESULTS_MINUS1, [1])
        self._verify_results(df_with_date_types, "date", date_sub, DateType, "dt_sub", DT_FULL_RESULTS_PLUS1, [-1])

    def test_timestamp_add(self, df_with_date_types):
        """Test the timestamp_add function."""
        self._verify_error(df_with_date_types, "col1", timestamp_add, TypeError, ["1", "year"])
        self._verify_error(df_with_date_types, "ts", timestamp_add, ValueError, ["1", "invalid"])

        df = df_with_date_types.select(timestamp_add(col("ts"), 1, "year").alias("ts_add"))
        self._verify_results_on_df(df, TimestampType, "ts_add", TS_FULL_RESULTS_PLUS1YEAR)

        self._verify_results(df_with_date_types, "ts", timestamp_add, TimestampType, "ts_add", TS_FULL_RESULTS_MINUS1, [-1, "day"])
        self._verify_results(df_with_date_types, "date", timestamp_add, DateType, "dt_add", DT_FULL_RESULTS_PLUS1YEAR, [1, "year"])
        self._verify_results(df_with_date_types, "date", timestamp_add, DateType, "dt_add", DT_FULL_RESULTS_PLUS1YEAR, [12, "month"])
        self._verify_results(df_with_date_types, "date", timestamp_add, DateType, "dt_add", DT_FULL_RESULTS_MINUS1, [-1, "day"])
        self._verify_results(df_with_date_types, "ts", timestamp_add, TimestampType, "ts_add", TS_FULL_RESULTS_MINUS1, [-24, "hour"])
        self._verify_results(df_with_date_types, "ts", timestamp_add, TimestampType, "ts_add", TS_FULL_RESULTS_MINUS1, [-1440, "minute"])
        self._verify_results(df_with_date_types, "ts", timestamp_add, TimestampType, "ts_add", TS_FULL_RESULTS_MINUS1, [-86400, "second"])
        self._verify_results(df_with_date_types, "ts", timestamp_add, TimestampType, "ts_add", TS_FULL_RESULTS_MINUS1, [-86400000, "millisecond"])
        self._verify_results(df_with_date_types, "ts", timestamp_add, TimestampType, "ts_add", TS_FULL_RESULTS_MINUS_100_MS, [-100, "millisecond"])

    def test_date_format(self, df_with_date_types):
        """Test the date_format function."""
        self._verify_error(df_with_date_types, "col1", date_format, TypeError, ["yyyy-MM-dd"])

        df = df_with_date_types.select(date_format(col("ts"), "MM-dd-yyyy").alias("ts_format"))
        self._verify_results_on_df(df, StringType, "ts_format", [
            "01-01-2025",
            "02-02-2025",
            "03-03-2025",
        ])

        self._verify_results(df_with_date_types, "ts", date_format, StringType, "ts_format",
            [
                '01-01-2025 01:00:00 AM',
                '02-02-2025 02:20:00 AM',
                '03-03-2025 03:33:30 PM'
            ],
            ["MM-dd-yyyy hh:mm:ss a"])
        self._verify_results(df_with_date_types, "date", date_format, StringType, "dt_format",
            [
                "01-01-2025",
                "02-01-2025",
                "03-01-2025",
            ],
            ["dd-MM-yyyy"])

    def test_date_diff(self, local_session, df_with_date_types):
        """Test the date_diff function."""
        self._verify_error(df_with_date_types, "col1", datediff, TypeError, [col("date")])
        self._verify_error(df_with_date_types, "ts", datediff, TypeError, [col("col1")])
        self._verify_results(df_with_date_types, "date_plus", datediff, IntegerType, "dt_diff", [9, 18, 27], [col("date")])
        self._verify_results(df_with_date_types, "date", datediff, IntegerType, "dt_diff", [-9, -18, -27], [col("date_plus")])

        df = df_with_date_types.select(
            col("ts"),
            date_add(col("ts"), 10).alias("ts_plus"))
        self._verify_results(df, "ts_plus", datediff, IntegerType, "ts_diff", [10, 10, 10], [col("ts")])

        df = local_session.create_dataframe({
            "end": [datetime.date(2015,4,8)],
            "start": [datetime.date(2015, 5, 10)]
        })
        self._verify_results(df, "end", datediff, IntegerType, "dt_diff", [-32], [col("start")])
        self._verify_results(df, "start", datediff, IntegerType, "dt_diff", [32], [col("end")])

    def test_timestamp_diff(self, local_session, df_with_date_types):
        """Test the timestamp_diff function."""
        self._verify_error(df_with_date_types, "col1", timestamp_diff, TypeError, [col("col1"), "day"])
        self._verify_error(df_with_date_types, "ts", timestamp_diff, ValueError, [col("ts"), "invalid"])

        df = df_with_date_types.select(
            col("ts"),
            date_add(col("ts"), 10).alias("ts_plus"))
        self._verify_results(df, "ts_plus", timestamp_diff, IntegerType, "ts_diff", [-10, -10, -10], [col("ts"), "day"])
        self._verify_results(df, "ts_plus", timestamp_diff, IntegerType, "ts_diff", [-864000000] * 3, [col("ts"), "millisecond"])

        df = df_with_date_types.select(
            col("date"),
            timestamp_add(col("date"), 10, "month").alias("date_plus"))
        self._verify_results(df, "date", timestamp_diff, IntegerType, "dt_diff", [10, 10, 10], [col("date_plus"), "month"])

        df = local_session.create_dataframe({
            "ts1": [datetime.datetime(2016, 3, 11, 9, 0, 7)],
            "ts2": [datetime.datetime(2024, 4, 2, 9, 0, 7)],
        })
        self._verify_results(df, "ts1", timestamp_diff, IntegerType, "ts_diff", [8], [col("ts2"), "year"])

        df = local_session.create_dataframe({
            "ts1": [datetime.datetime(2016, 3, 29, 9, 0, 7), datetime.datetime(2016, 2, 28, 9, 0, 7)],
            "ts2": [datetime.datetime(2016, 4, 10, 9, 0, 7), datetime.datetime(2016, 4, 30, 9, 0, 7)],
        })
        self._verify_results(df, "ts1", timestamp_diff, IntegerType, "ts_diff", [0, 2], [col("ts2"), "month"])

    def test_nested_datetime_expressions(self, local_session, df_with_date_types):
        """Test nested datetime expressions with multiple function combinations."""
        df = df_with_date_types.select(
            # Add 1 year, 2 months, and 15 days to a timestamp, then extract the month
            month(timestamp_add(timestamp_add(timestamp_add(col("ts"), 1, "year"), 2, "month"), 15, "day")).alias("complex_month"),

            # Calculate time difference in hours between two complex timestamp operations
            timestamp_diff(timestamp_add(col("ts"), 5, "hour"),timestamp_add(col("ts"), -3, "hour"),"hour").alias("hour_diff"),

            # Format a truncated and modified timestamp
            date_format(timestamp_add(date_trunc(col("ts"), "day"), 12, "hour"), "yyyy-MM-dd HH:mm:ss").alias("formatted_noon"),
        )

        # Verify original months [1,2,3] + 2 months = [3,4,5]
        df_to_verify = df.select(col("complex_month"))
        self._verify_results_on_df(df_to_verify, IntegerType, "complex_month", [3, 4, 5])

        # Verify hour_diff: 5 - (-3) = 8 hours difference
        df_to_verify = df.select(col("hour_diff"))
        self._verify_results_on_df(df_to_verify, IntegerType, "hour_diff", [-8, -8, -8])

        # Verify formatted_noon: each date at 12:00:00
        df_to_verify = df.select(col("formatted_noon"))
        self._verify_results_on_df(df_to_verify, StringType, "formatted_noon", [
            "2025-01-01 12:00:00",
            "2025-02-02 12:00:00",
            "2025-03-03 12:00:00"
        ])

        df2 = df_with_date_types.select(
            # Convert date to string, parse it back, add days, then get year
            year(date_add(to_date(date_format(col("date"), "yyyy-MM-dd")),365)).alias("next_year"),

            # Complex date difference calculation
            datediff(timestamp_add(date_trunc(current_date(), "month"),1, "month"),current_date()).alias("days_to_next_month"),
        )

        # Verify next_year: 2025 + 1 = 2026
        df_to_verify = df2.select(col("next_year"))
        self._verify_results_on_df(df_to_verify, IntegerType, "next_year", [2026, 2026, 2026])

        # Millisecond precision operations
        df3 = df_with_date_types.select(
            # Extract milliseconds from a timestamp with added milliseconds
            millisecond(timestamp_add(timestamp_add(col("ts"), 500, "millisecond"),250, "millisecond")).alias("total_ms"),

            # Calculate precise time differences in milliseconds
            timestamp_diff(timestamp_add(col("ts"), 1, "second"),timestamp_add(col("ts"), 500, "millisecond"),"millisecond").alias("ms_diff"),
        )

        # Verify total_ms: original ms [0,200,300] + 750 = [750,950,1050] but millisecond() returns mod 1000
        df_to_verify = df3.select(col("total_ms"))
        self._verify_results_on_df(df_to_verify, IntegerType, "total_ms", [750, 950, 50])

        # Verify ms_diff: 1000ms - 500ms = 500ms
        df_to_verify = df3.select(col("ms_diff"))
        self._verify_results_on_df(df_to_verify, IntegerType, "ms_diff", [-500, -500, -500])

        # Mixed date and timestamp operations
        df4 = df_with_date_types.select(
            # Compare dates using complex timestamp operations
            datediff(to_date(date_format(timestamp_add(col("ts"), 7, "day"),"yyyy-MM-dd")),col("date")).alias("week_diff"),

            # Extract time components from nested operations
            hour(timestamp_add(date_trunc(timestamp_add(col("ts"), 6, "hour"),"hour"),30, "minute")).alias("adjusted_hour"),
        )

        # Verify week_diff: 7 days difference
        df_to_verify = df4.select(col("week_diff"))
        self._verify_results_on_df(df_to_verify, IntegerType, "week_diff", [7, 38, 66])

        # Verify adjusted_hour: original hours [1,2,15] + 6 = [7,8,21], truncated then +30min still same hour
        df_to_verify = df4.select(col("adjusted_hour"))
        self._verify_results_on_df(df_to_verify, IntegerType, "adjusted_hour", [7, 8, 21])

        df5 = local_session.create_dataframe({
            "dummy": [1, 2, 3]
        })

        df5 = df5.select(
            # Complex expression using current date/time
            day(date_add(timestamp_add(date_trunc(current_timestamp(), "month"),15, "day"),-1)).alias("mid_month_day"),

            # Year difference between current date and a future date
            timestamp_diff(current_date(),timestamp_add(current_date(), 2, "year"),"year").alias("year_diff"),
        )

        # Verify mid_month_day
        df_to_verify = df5.select(col("mid_month_day"))
        self._verify_results_on_df(df_to_verify, IntegerType, "mid_month_day", [15])

        # Verify year_diff: current - (current + 2 years) = -2
        df_to_verify = df5.select(col("year_diff"))
        self._verify_results_on_df(df_to_verify, IntegerType, "year_diff", [2])

        # Date type nested expression
        df = df_with_date_types.select(
            col("ts"),
            now().alias("now"),
            day(date_add(now(), 10)).alias("ts_plus"),
            datediff(date_add(now(), 10), timestamp_add(now(), -1, "minute")).alias("ts_diff"),
            (year(date_add(now(), 365)) + 1).alias("year_plus"),
        )

        df_to_verify = df.select(col("ts_plus"))
        self._verify_results_on_df(df_to_verify, IntegerType, "ts_plus", [(datetime.date.today() + datetime.timedelta(days=10)).day] * 3)

        df_to_verify = df.select(col("ts_diff"))
        self._verify_results_on_df(df_to_verify, IntegerType, "ts_diff", [10] * 3)

        df_to_verify = df.select(col("year_plus"))
        self._verify_results_on_df(df_to_verify, IntegerType, "year_plus", [datetime.date.today().year + 2] * 3)

        df_to_verify = df.select(year(timestamp_add(timestamp_add(current_date(), 1, "year"), 1, "year")).alias("year_pp"))
        self._verify_results_on_df(df_to_verify, IntegerType, "year_pp", [datetime.date.today().year + 2])

        # somewhat deep nested expression
        dt_to_verify = df.select(
            datediff(
                timestamp_add(
                    to_timestamp(
                        date_format(
                            date_trunc(now(), "year"), # jan 1st of the current year
                            "yyyy-MM-dd HH:mm:ss"), # format the date to a string
                        "yyyy-MM-dd HH:mm:ss"), # convert the string to a timestamp
                    1, "month"), # feb 1st
                to_date(lit(f"{datetime.date.today().year}-01-01")).alias("start_date"), # jan 1st of the current year
            ).alias("dt_pp"))
        # difference in days from jan 1st to feb 1st.
        self._verify_results_on_df(dt_to_verify, IntegerType, "dt_pp", [31])

    def test_no_coercion_needed(self):
        """Date and datetime columns should not be coerced to string."""
        df = pl.DataFrame({
            "date_col": [datetime.date(2023, 12, 25), datetime.date(2024, 1, 1)],
            "datetime_col": [datetime.datetime(2023, 12, 25, 14, 30), datetime.datetime(2024, 1, 1, 9, 15)],
            "int_col": [1, 2]
        })

        result = apply_ingestion_coercions(df)

        # Check that date_col is now string type
        assert result.schema["date_col"] == pl.Date
        assert result.schema["datetime_col"] == pl.Datetime
        assert result.schema["int_col"] == pl.Int64  # unchanged

        # Check actual values are string representations
        date_values = result["date_col"].to_list()
        datetime_values = result["datetime_col"].to_list()
        assert all(isinstance(val, datetime.date) for val in date_values)
        assert all(isinstance(val, datetime.datetime) for val in datetime_values)

    @classmethod
    def _verify_results(
        cls,
        df: DataFrame,
        field_name: str,
        function: Callable,
        expected_type: Type,
        alias: str,
        expected_results: List[Any],
        extra_args: Optional[List[Any]] = None,
    ):
        if extra_args is None:
            extra_args = []
        df = df.select(function(col(field_name), *extra_args).alias(alias))
        cls._verify_results_on_df(df, expected_type, alias, expected_results)

    @classmethod
    def _verify_results_on_df(
        cls,
        df: DataFrame,
        expected_type: Type,
        alias: str,
        expected_results: List[Any],
    ):
        assert df.schema.column_fields == [
            ColumnField(alias, expected_type),
        ]
        assert df.to_polars()[alias].to_list() == expected_results

    @classmethod
    def _verify_error(
        cls,
        df: DataFrame,
        field_name: str,
        function: Callable,
        error_type: Type,
        extra_args: Optional[List[Any]] = None,
    ):
        if extra_args is None:
            extra_args = []
        with pytest.raises(error_type):
            df.select(function(col(field_name), *extra_args))
