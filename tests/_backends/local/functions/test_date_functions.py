import datetime
import zoneinfo

import polars as pl
import pytest

from fenic._backends.local.physical_plan.utils import apply_ingestion_coercions
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
    from_utc_timestamp,
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
    to_utc_timestamp,
    year,
)
from fenic.core.types import (
    ArrayType,
    ColumnField,
    DateType,
    StructField,
    StructType,
    TimestampType,
)

# Full timestamp results.
TS_FULL_RESULTS = [
    datetime.datetime(2025, 1, 1, 1, 0, tzinfo=datetime.timezone.utc),
    datetime.datetime(2025, 2, 2, 2, 20, 0, 200000, tzinfo=datetime.timezone.utc),
    datetime.datetime(2025, 3, 3, 15, 33, 30, 300000, tzinfo=datetime.timezone.utc)
]

TS_FULL_RESULTS_PLUS1 = [
    datetime.datetime(2025, 1, 2, 1, 0, tzinfo=datetime.timezone.utc),
    datetime.datetime(2025, 2, 3, 2, 20, 0, 200000, tzinfo=datetime.timezone.utc),
    datetime.datetime(2025, 3, 4, 15, 33, 30, 300000, tzinfo=datetime.timezone.utc)
]

TS_FULL_RESULTS_MINUS1 = [
    datetime.datetime(2024, 12, 31, 1, 0, tzinfo=datetime.timezone.utc),
    datetime.datetime(2025, 2, 1, 2, 20, 0, 200000, tzinfo=datetime.timezone.utc),
    datetime.datetime(2025, 3, 2, 15, 33, 30, 300000, tzinfo=datetime.timezone.utc)
]

TS_FULL_RESULTS_MINUS_100_MS = [
    datetime.datetime(2025, 1, 1, 0, 59, 59, 900000, tzinfo=datetime.timezone.utc),
    datetime.datetime(2025, 2, 2, 2, 20, 0, 100000, tzinfo=datetime.timezone.utc),
    datetime.datetime(2025, 3, 3, 15, 33, 30, 200000, tzinfo=datetime.timezone.utc)
]

TS_FULL_RESULTS_PLUS1YEAR = [
    datetime.datetime(2026, 1, 1, 1, 0, tzinfo=datetime.timezone.utc),
    datetime.datetime(2026, 2, 2, 2, 20, 0, 200000, tzinfo=datetime.timezone.utc),
    datetime.datetime(2026, 3, 3, 15, 33, 30, 300000, tzinfo=datetime.timezone.utc)
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

UTC_TZ = "UTC"
LA_TZ = "America/Los_Angeles"
NYC_TZ = "America/New_York"


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
        pl.col("ts_str").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S.%3f").dt.replace_time_zone("UTC").alias("ts")
    )
    pl_df = pl_df.with_columns(
        pl.col("ts_str").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S.%3f").alias("ts_naive") # test timestamp without timezone passed directly to fenic
    )

    pl_df = pl_df.with_columns(
        pl.col("ts_str").str.strptime(pl.Date, format="%Y-%m-%dT%H:%M:%S.%3f").alias("dt")
    )

    df = local_session.create_dataframe(pl_df)
    df = df.with_column(
                "ts_la",
                lit(datetime.datetime(2025, 1, 2, 1, 1, 1, tzinfo=zoneinfo.ZoneInfo(key=LA_TZ))))
    return df.with_column(
                "ts_utc",
                lit(datetime.datetime(2025, 1, 2, 1, 1, 1, tzinfo=zoneinfo.ZoneInfo(key=UTC_TZ))))

# Timestamp values for testing cast conversions.
@pytest.fixture
def df_utc_timestamp(local_session):
    df = local_session.create_dataframe({
            "timestamp": [
                "2025-01-15 10:30:00",
                "2025-01-16 14:00:00",
                "2025-01-17 18:45:00"
            ]
        })
    return df.with_column("ts", to_timestamp(col("timestamp"), "yyyy-MM-dd HH:mm:ss"))

TS_UTC_NAIVE_INPUT = [
    datetime.datetime(2025, 1, 15, 10, 30, 0, tzinfo=None),
    datetime.datetime(2025, 1, 16, 14, 0, 0, tzinfo=None),
    datetime.datetime(2025, 1, 17, 18, 45, 0, tzinfo=None)
]
TS_UTC_INPUT = [
    datetime.datetime(2025, 1, 15, 10, 30, 0, tzinfo=zoneinfo.ZoneInfo(key=UTC_TZ)),
    datetime.datetime(2025, 1, 16, 14, 0, 0, tzinfo=zoneinfo.ZoneInfo(key=UTC_TZ)),
    datetime.datetime(2025, 1, 17, 18, 45, 0, tzinfo=datetime.timezone.utc)
]
TS_LA_INPUT = [
    datetime.datetime(2025, 1, 15, 10, 30, 0, tzinfo=zoneinfo.ZoneInfo(key=LA_TZ)),
    datetime.datetime(2025, 1, 16, 14, 0, 0, tzinfo=zoneinfo.ZoneInfo(key=LA_TZ)),
    datetime.datetime(2025, 1, 17, 18, 45, 0, tzinfo=zoneinfo.ZoneInfo(key=LA_TZ))
]
TS_FROM_UTC_TO_LA_RESULTS = [
    datetime.datetime(2025, 1, 15, 2, 30, 0, tzinfo=zoneinfo.ZoneInfo(key=UTC_TZ) ),
    datetime.datetime(2025, 1, 16, 6, 0, 0, tzinfo=zoneinfo.ZoneInfo(key=UTC_TZ)),
    datetime.datetime(2025, 1, 17, 10, 45, 0, tzinfo=zoneinfo.ZoneInfo(key=UTC_TZ))
]
TS_TO_UTC_FROM_LA_RESULTS = [
    datetime.datetime(2025, 1, 15, 18, 30, 0, tzinfo=zoneinfo.ZoneInfo(key=UTC_TZ)),
    datetime.datetime(2025, 1, 16, 22, 0, 0, tzinfo=zoneinfo.ZoneInfo(key=UTC_TZ)),
    datetime.datetime(2025, 1, 18, 2, 45, 0, tzinfo=zoneinfo.ZoneInfo(key=UTC_TZ))
]


class TestDateFunctions:
    def test_year(self, df_with_date_types):
        """Test the year function."""
        default_year_results = [2025]*3

        df = df_with_date_types.select(year(col("ts")).alias("year"))
        assert df.to_polars()["year"].to_list() == default_year_results

        df = df_with_date_types.select(year(col("date")).alias("year"))
        assert df.to_polars()["year"].to_list() == default_year_results

        with pytest.raises(TypeError):
            df_with_date_types.select(year(col("col1")))

    def test_month(self, df_with_date_types):
        """Test the month function."""

        df = df_with_date_types.select(month(col("ts")).alias("month"))
        assert df.to_polars()["month"].to_list() == [1, 2, 3]

        df = df_with_date_types.select(month(col("date")).alias("month"))
        assert df.to_polars()["month"].to_list() == [1, 1, 1]

        with pytest.raises(TypeError):
            df_with_date_types.select(month(col("col1")))

    def test_day(self, df_with_date_types):
        """Test the day function."""

        df = df_with_date_types.select(day(col("ts")).alias("day"))
        assert df.to_polars()["day"].to_list() == [1, 2, 3]

        df = df_with_date_types.select(day(col("date")).alias("day"))
        assert df.to_polars()["day"].to_list() == [1, 2, 3]

        with pytest.raises(TypeError):
            df_with_date_types.select(day(col("col1")))

    def test_hour(self, df_with_date_types):
        """Test the hour function."""

        df = df_with_date_types.select(hour(col("ts")).alias("hour"))
        assert df.to_polars()["hour"].to_list() == [1, 2, 15]

        df = df_with_date_types.select(hour(col("date")).alias("hour"))
        assert df.to_polars()["hour"].to_list() == [0, 0, 0]

        with pytest.raises(TypeError):
            df_with_date_types.select(hour(col("col1")))

    def test_minute(self, df_with_date_types):
        """Test the minute function."""

        df = df_with_date_types.select(minute(col("ts")).alias("minute"))
        assert df.to_polars()["minute"].to_list() == [0, 20, 33]

        df = df_with_date_types.select(minute(col("date")).alias("minute"))
        assert df.to_polars()["minute"].to_list() == [0, 0, 0]

        with pytest.raises(TypeError):
            df_with_date_types.select(minute(col("col1")))

    def test_second(self, df_with_date_types):
        """Test the second function."""

        df = df_with_date_types.select(second(col("ts")).alias("second"))
        assert df.to_polars()["second"].to_list() == [0, 0, 30]

        df = df_with_date_types.select(second(col("date")).alias("second"))
        assert df.to_polars()["second"].to_list() == [0, 0, 0]

        with pytest.raises(TypeError):
            df_with_date_types.select(second(col("col1")))

    def test_millisecond(self, df_with_date_types):
        """Test the millisecond function."""

        df = df_with_date_types.select(millisecond(col("ts")).alias("millisecond"))
        assert df.to_polars()["millisecond"].to_list() == [0, 200, 300]

        df = df_with_date_types.select(millisecond(col("date")).alias("millisecond"))
        assert df.to_polars()["millisecond"].to_list() == [0, 0, 0]

        with pytest.raises(TypeError):
            df_with_date_types.select(millisecond(col("col1")))

    def test_to_timestamp(self, df_with_date_types):
        """Test the to_timestamp function."""
        df = df_with_date_types.select(to_timestamp(col("ts_str")).alias("ts2"))
        assert df.to_polars()["ts2"].to_list() == TS_FULL_RESULTS
        # We default all timestamps to UTC
        assert df.to_pydict()['ts2'][0].tzinfo == zoneinfo.ZoneInfo(key='UTC')

        with pytest.raises(TypeError):
            df_with_date_types.select(to_timestamp(col("col1")))

    def test_to_date_format(self, df_with_date_types):
        """Test the to_timestamp function with timezone."""
        ts_format = "yyyy-MM-dd HH:mm:ss XXX"
        # convert ts to LA timezone, then convert to date time
        df = df_with_date_types.select(date_format(from_utc_timestamp(col("ts"), LA_TZ), ts_format).alias("ts_la_str"))
        assert df.to_polars()["ts_la_str"].to_list() == [
            "2024-12-31 17:00:00 +00:00",
            "2025-02-01 18:20:00 +00:00",
            "2025-03-03 07:33:30 +00:00",
        ]

    def test_to_timestamp_different_tz(self, local_session):
        """Test the to_timestamp function with timezone."""
        ts_format = "yyyy-MM-dd HH:mm:ss XXX"
        # convert ts to LA timezone, then convert to date time
        df_str = local_session.create_dataframe(pl.DataFrame({
            "ts": [
                "2024-12-31 17:00:00 +08:00",
                "2025-02-01 18:20:00 +08:00",
                "2025-03-03 07:33:30 +08:00",
            ]
        }))
        df = df_str.select(to_timestamp(col("ts"), ts_format).alias("ts_utc"))
        assert df.to_polars()["ts_utc"].to_list() == [
            datetime.datetime(2024, 12, 31, 9, 0, 0, tzinfo=zoneinfo.ZoneInfo(key='UTC')),
            datetime.datetime(2025, 2, 1, 10, 20, 0, tzinfo=zoneinfo.ZoneInfo(key='UTC')),
            datetime.datetime(2025, 3, 2, 23, 33, 30, tzinfo=zoneinfo.ZoneInfo(key='UTC')),
        ]

    def test_to_date(self, df_with_date_types, local_session):
        """Test the to_date function."""
        df = local_session.create_dataframe(pl.DataFrame({
            "date_str": ["01-27-2025", "11-02-2025"],
        }))
        df = df.select(to_date(col("date_str"), format="MM-dd-yyyy").alias("dt2"))
        assert df.to_polars()["dt2"].to_list() == [
            datetime.date(2025, 1, 27),
            datetime.date(2025, 11, 2),
        ]

        df = df_with_date_types.select(to_date(col("date_str")).alias("dt2"))
        assert df.to_polars()["dt2"].to_list() == DT_FULL_RESULTS

        with pytest.raises(TypeError):
            df_with_date_types.select(to_date(col("col1")))

    def test_current_date_functions(self, df_with_date_types):
        """Test the current date and time functions."""
        now_year = datetime.datetime.now(datetime.timezone.utc).year
        now_year_results = [now_year]*3

        # now
        df = df_with_date_types.with_column("now", now())
        df = df.select(col("now"))
        assert df.schema.column_fields == [ColumnField("now", TimestampType)]

        df = df.select(year(col("now")).alias("now_year"))
        assert df.to_polars()["now_year"].to_list() == now_year_results

        # current_timestamp
        df = df_with_date_types.with_column("current_timestamp", current_timestamp())
        df = df.select(col("current_timestamp"))
        assert df.schema.column_fields == [ColumnField("current_timestamp", TimestampType)]
        # We default all timestamps to UTC
        assert df.to_pydict()['current_timestamp'][0].tzinfo == zoneinfo.ZoneInfo(key='UTC')

        df = df.select(year(col("current_timestamp")).alias("current_timestamp_year"))
        assert df.to_polars()["current_timestamp_year"].to_list() == now_year_results

        # current_date
        df = df_with_date_types.with_column("current_date", current_date())
        df = df.select(col("current_date"))
        assert df.schema.column_fields == [ColumnField("current_date", DateType)]

        df = df.select(year(col("current_date")).alias("current_date_year"))
        assert df.to_polars()["current_date_year"].to_list() == now_year_results

    def test_date_trunc(self, df_with_date_types):
        """Test the date_trunc function."""
        with pytest.raises(ValueError):
            df_with_date_types.select(date_trunc(col("ts"), "invalid"))

        with pytest.raises(TypeError):
            df_with_date_types.select(date_trunc(col("col1"), "year"))

        df = df_with_date_types.select(date_trunc(col("ts"), "year").alias("ts_trunc"))
        assert df.to_polars()["ts_trunc"].to_list() == [datetime.datetime(2025, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)] * 3

        df = df_with_date_types.select(date_trunc(col("date"), "year").alias("dt_trunc"))
        assert df.to_polars()["dt_trunc"].to_list() == [datetime.date(2025, 1, 1)] * 3

        df = df_with_date_types.select(date_trunc(col("ts"), "month").alias("ts_trunc"))
        assert df.to_polars()["ts_trunc"].to_list() == [
            datetime.datetime(2025, 1, 1, 0, 0, tzinfo=datetime.timezone.utc),
            datetime.datetime(2025, 2, 1, 0, 0, tzinfo=datetime.timezone.utc),
            datetime.datetime(2025, 3, 1, 0, 0, tzinfo=datetime.timezone.utc)
        ]

        df = df_with_date_types.select(date_trunc(col("ts"), "day").alias("ts_trunc"))
        assert df.to_polars()["ts_trunc"].to_list() == [
            datetime.datetime(2025, 1, 1, 0, 0, tzinfo=datetime.timezone.utc),
            datetime.datetime(2025, 2, 2, 0, 0, tzinfo=datetime.timezone.utc),
            datetime.datetime(2025, 3, 3, 0, 0, tzinfo=datetime.timezone.utc)
        ]

        df = df_with_date_types.select(date_trunc(col("ts"), "hour").alias("ts_trunc"))
        assert df.to_polars()["ts_trunc"].to_list() == [
            datetime.datetime(2025, 1, 1, 1, 0, tzinfo=datetime.timezone.utc),
            datetime.datetime(2025, 2, 2, 2, 0, tzinfo=datetime.timezone.utc),
            datetime.datetime(2025, 3, 3, 15, 0, tzinfo=datetime.timezone.utc)
        ]

        df = df_with_date_types.select(date_trunc(col("date"), "hour").alias("dt_trunc"))
        assert df.to_polars()["dt_trunc"].to_list() == [
            datetime.date(2025, 1, 1),
            datetime.date(2025, 1, 2),
            datetime.date(2025, 1, 3)
        ]

    def test_date_add(self, local_session, df_with_date_types):
        """Test the date_add function."""
        with pytest.raises(TypeError):
            df_with_date_types.select(date_add(col("col1"), "1"))

        with pytest.raises(TypeError):
            df_with_date_types.select(date_add(col("ts"), col("ts")))

        df = df_with_date_types.select(date_add(col("ts"), 1).alias("ts_add"))
        assert df.to_polars()["ts_add"].to_list() == TS_FULL_RESULTS_PLUS1

        df = df_with_date_types.select(date_add(col("date"), -1).alias("dt_add"))
        assert df.to_polars()["dt_add"].to_list() == DT_FULL_RESULTS_MINUS1

        df = df_with_date_types.select(date_sub(col("date"), -1).alias("dt_sub"))
        assert df.to_polars()["dt_sub"].to_list() == DT_FULL_RESULTS_PLUS1

        df = df_with_date_types.with_column("days", lit(-1))
        df = df.select(date_sub(col("date"), col("days")).alias("dt_sub"))
        assert df.to_polars()["dt_sub"].to_list() == DT_FULL_RESULTS_PLUS1

    def test_date_sub(self, df_with_date_types):
        """Test the date_sub function."""
        with pytest.raises(TypeError):
            df_with_date_types.select(date_sub(col("col1"), "1"))

        with pytest.raises(TypeError):
            df_with_date_types.select(date_sub(col("ts"), col("ts")))

        df = df_with_date_types.select(date_sub(col("ts"), 1).alias("ts_sub"))
        assert df.to_polars()["ts_sub"].to_list() == TS_FULL_RESULTS_MINUS1

        df = df_with_date_types.select(date_sub(col("date"), 1).alias("dt_sub"))
        assert df.to_polars()["dt_sub"].to_list() == DT_FULL_RESULTS_MINUS1

        df = df_with_date_types.select(date_sub(col("date"), -1).alias("dt_sub"))
        assert df.to_polars()["dt_sub"].to_list() == DT_FULL_RESULTS_PLUS1

    def test_timestamp_add(self, df_with_date_types):
        """Test the timestamp_add function."""
        with pytest.raises(TypeError):
            df_with_date_types.select(timestamp_add(col("col1"), "1", "year"))

        with pytest.raises(ValueError):
            df_with_date_types.select(timestamp_add(col("ts"), "1", "invalid"))

        df = df_with_date_types.select(timestamp_add(col("ts"), 1, "year").alias("ts_add"))
        assert df.to_polars()["ts_add"].to_list() == TS_FULL_RESULTS_PLUS1YEAR

        df = df_with_date_types.select(timestamp_add(col("ts"), -1, "day").alias("ts_add"))
        assert df.to_polars()["ts_add"].to_list() == TS_FULL_RESULTS_MINUS1

        df = df_with_date_types.select(timestamp_add(col("date"), 1, "year").alias("dt_add"))
        assert df.to_polars()["dt_add"].to_list() == DT_FULL_RESULTS_PLUS1YEAR

        df = df_with_date_types.select(timestamp_add(col("date"), 12, "month").alias("dt_add"))
        assert df.to_polars()["dt_add"].to_list() == DT_FULL_RESULTS_PLUS1YEAR

        df = df_with_date_types.select(timestamp_add(col("date"), -1, "day").alias("dt_add"))
        assert df.to_polars()["dt_add"].to_list() == DT_FULL_RESULTS_MINUS1

        df = df_with_date_types.select(timestamp_add(col("ts"), -24, "hour").alias("ts_add"))
        assert df.to_polars()["ts_add"].to_list() == TS_FULL_RESULTS_MINUS1

        df = df_with_date_types.select(timestamp_add(col("ts"), -1440, "minute").alias("ts_add"))
        assert df.to_polars()["ts_add"].to_list() == TS_FULL_RESULTS_MINUS1

        df = df_with_date_types.select(timestamp_add(col("ts"), -86400, "second").alias("ts_add"))
        assert df.to_polars()["ts_add"].to_list() == TS_FULL_RESULTS_MINUS1

        df = df_with_date_types.select(timestamp_add(col("ts"), -86400000, "millisecond").alias("ts_add"))
        assert df.to_polars()["ts_add"].to_list() == TS_FULL_RESULTS_MINUS1

        df = df_with_date_types.select(timestamp_add(col("ts"), -100, "millisecond").alias("ts_add"))
        assert df.to_polars()["ts_add"].to_list() == TS_FULL_RESULTS_MINUS_100_MS

    def test_date_format(self, df_with_date_types):
        """Test the date_format function."""
        with pytest.raises(TypeError):
            df_with_date_types.select(date_format(col("col1"), "yyyy-MM-dd"))

        df = df_with_date_types.select(date_format(col("ts"), "MM-dd-yyyy").alias("ts_format"))
        assert df.to_polars()["ts_format"].to_list() == [
            "01-01-2025",
            "02-02-2025",
            "03-03-2025",
        ]

        df = df_with_date_types.select(date_format(col("ts"), "MM-dd-yyyy hh:mm:ss a").alias("ts_format"))
        assert df.to_polars()["ts_format"].to_list() == [
            '01-01-2025 01:00:00 AM',
            '02-02-2025 02:20:00 AM',
            '03-03-2025 03:33:30 PM'
        ]

        df = df_with_date_types.select(date_format(col("date"), "dd-MM-yyyy").alias("dt_format"))
        assert df.to_polars()["dt_format"].to_list() == [
            "01-01-2025",
            "02-01-2025",
            "03-01-2025",
        ]

    def test_date_format_tz(self, df_utc_timestamp):
        """Test the date_format function with timezone."""
        df = df_utc_timestamp.select(date_format(col("ts"), "MM-dd-yyyy hh:mm:ss a XXX").alias("ts_format"))
        assert df.to_polars()["ts_format"].to_list() == [
            "01-15-2025 10:30:00 AM +00:00",
            "01-16-2025 02:00:00 PM +00:00",
            "01-17-2025 06:45:00 PM +00:00"
        ]

    def test_date_diff(self, local_session, df_with_date_types):
        """Test the date_diff function."""
        with pytest.raises(TypeError):
            df_with_date_types.select(datediff(col("col1"), col("date")))

        with pytest.raises(TypeError):
            df_with_date_types.select(datediff(col("ts"), col("col1")))

        df = df_with_date_types.select(datediff(col("date_plus"), col("date")).alias("dt_diff"))
        assert df.to_polars()["dt_diff"].to_list() == [9, 18, 27]

        df = df_with_date_types.select(datediff(col("date"), col("date_plus")).alias("dt_diff"))
        assert df.to_polars()["dt_diff"].to_list() == [-9, -18, -27]

        df = df_with_date_types.select(
            col("ts"),
            date_add(col("ts"), 10).alias("ts_plus"))
        df = df.select(datediff(col("ts_plus"), col("ts")).alias("ts_diff"))
        assert df.to_polars()["ts_diff"].to_list() == [10, 10, 10]

        df = local_session.create_dataframe({
            "end": [datetime.date(2015,4,8)],
            "start": [datetime.date(2015, 5, 10)]
        })
        df_result = df.select(datediff(col("end"), col("start")).alias("dt_diff"))
        assert df_result.to_polars()["dt_diff"].to_list() == [-32]

        df_result = df.select(datediff(col("start"), col("end")).alias("dt_diff"))
        assert df_result.to_polars()["dt_diff"].to_list() == [32]

    def test_timestamp_diff(self, local_session, df_with_date_types):
        """Test the timestamp_diff function."""
        with pytest.raises(TypeError):
            df_with_date_types.select(timestamp_diff(col("col1"), col("col1"), "day"))

        with pytest.raises(ValueError):
            df_with_date_types.select(timestamp_diff(col("ts"), col("ts"), "invalid"))

        df = df_with_date_types.select(
            col("ts"),
            date_add(col("ts"), 10).alias("ts_plus"))
        df_result = df.select(timestamp_diff(col("ts_plus"), col("ts"), "day").alias("ts_diff"))
        assert df_result.to_polars()["ts_diff"].to_list() == [-10, -10, -10]

        df_result = df.select(timestamp_diff(col("ts_plus"), col("ts"), "millisecond").alias("ts_diff"))
        assert df_result.to_polars()["ts_diff"].to_list() == [-864000000] * 3

        df = df_with_date_types.select(
            col("date"),
            timestamp_add(col("date"), 10, "month").alias("date_plus"))
        df_result = df.select(timestamp_diff(col("date"), col("date_plus"), "month").alias("dt_diff"))
        assert df_result.to_polars()["dt_diff"].to_list() == [10, 10, 10]

        df = local_session.create_dataframe({
            "ts1": [datetime.datetime(2016, 3, 11, 9, 0, 7)],
            "ts2": [datetime.datetime(2024, 4, 2, 9, 0, 7)],
        })
        df_result = df.select(timestamp_diff(col("ts1"), col("ts2"), "year").alias("ts_diff"))
        assert df_result.to_polars()["ts_diff"].to_list() == [8]

        df = local_session.create_dataframe({
            "ts1": [datetime.datetime(2016, 3, 29, 9, 0, 7), datetime.datetime(2016, 2, 28, 9, 0, 7)],
            "ts2": [datetime.datetime(2016, 4, 10, 9, 0, 7), datetime.datetime(2016, 4, 30, 9, 0, 7)],
        })
        df_result = df.select(timestamp_diff(col("ts1"), col("ts2"), "month").alias("ts_diff"))
        assert df_result.to_polars()["ts_diff"].to_list() == [0, 2]

        # You can do math between a timestamp and a date.
        df = df_with_date_types.select(timestamp_diff(col("ts"), col("date"), "day").alias("ts_diff"))
        assert df.to_polars()["ts_diff"].to_list() == [0, -31, -59]

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
        assert df_to_verify.to_polars()["complex_month"].to_list() == [3, 4, 5]

        # Verify hour_diff: 5 - (-3) = 8 hours difference
        df_to_verify = df.select(col("hour_diff"))
        assert df_to_verify.to_polars()["hour_diff"].to_list() == [-8, -8, -8]

        # Verify formatted_noon: each date at 12:00:00
        df_to_verify = df.select(col("formatted_noon"))
        assert df_to_verify.to_polars()["formatted_noon"].to_list() == [
            "2025-01-01 12:00:00",
            "2025-02-02 12:00:00",
            "2025-03-03 12:00:00"
        ]

        df2 = df_with_date_types.select(
            # Convert date to string, parse it back, add days, then get year
            year(date_add(to_date(date_format(col("date"), "yyyy-MM-dd")),365)).alias("next_year"),

            # Complex date difference calculation
            datediff(timestamp_add(date_trunc(current_date(), "month"),1, "month"),current_date()).alias("days_to_next_month"),
        )

        # Verify next_year: 2025 + 1 = 2026
        df_to_verify = df2.select(col("next_year"))
        assert df_to_verify.to_polars()["next_year"].to_list() == [2026, 2026, 2026]

        # Millisecond precision operations
        df3 = df_with_date_types.select(
            # Extract milliseconds from a timestamp with added milliseconds
            millisecond(timestamp_add(timestamp_add(col("ts"), 500, "millisecond"),250, "millisecond")).alias("total_ms"),

            # Calculate precise time differences in milliseconds
            timestamp_diff(timestamp_add(col("ts"), 1, "second"),timestamp_add(col("ts"), 500, "millisecond"),"millisecond").alias("ms_diff"),
        )

        # Verify total_ms: original ms [0,200,300] + 750 = [750,950,1050] but millisecond() returns mod 1000
        df_to_verify = df3.select(col("total_ms"))
        assert df_to_verify.to_polars()["total_ms"].to_list() == [750, 950, 50]

        # Verify ms_diff: 1000ms - 500ms = 500ms
        df_to_verify = df3.select(col("ms_diff"))
        assert df_to_verify.to_polars()["ms_diff"].to_list() == [-500, -500, -500]

        # Mixed date and timestamp operations
        df4 = df_with_date_types.select(
            # Compare dates using complex timestamp operations
            datediff(to_date(date_format(timestamp_add(col("ts"), 7, "day"),"yyyy-MM-dd")),col("date")).alias("week_diff"),

            # Extract time components from nested operations
            hour(timestamp_add(date_trunc(timestamp_add(col("ts"), 6, "hour"),"hour"),30, "minute")).alias("adjusted_hour"),
        )

        # Verify week_diff: 7 days difference
        df_to_verify = df4.select(col("week_diff"))
        assert df_to_verify.to_polars()["week_diff"].to_list() == [7, 38, 66]

        # Verify adjusted_hour: original hours [1,2,15] + 6 = [7,8,21], truncated then +30min still same hour
        df_to_verify = df4.select(col("adjusted_hour"))
        assert df_to_verify.to_polars()["adjusted_hour"].to_list() == [7, 8, 21]

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
        assert df_to_verify.to_polars()["mid_month_day"].to_list() == [15]

        # Verify year_diff: current - (current + 2 years) = -2
        df_to_verify = df5.select(col("year_diff"))
        assert df_to_verify.to_polars()["year_diff"].to_list() == [2]

        # Date type nested expression
        df = df_with_date_types.select(
            col("ts"),
            now().alias("now"),
            day(date_add(now(), 10)).alias("ts_plus"),
            datediff(date_add(now(), 10), timestamp_add(now(), -1, "minute")).alias("ts_diff"),
            (year(date_add(now(), 365)) + 1).alias("year_plus"),
        )

        df_to_verify = df.select(col("ts_plus"))
        assert df_to_verify.to_polars()["ts_plus"].to_list() == [(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=10)).day] * 3

        df_to_verify = df.select(col("ts_diff"))
        assert df_to_verify.to_polars()["ts_diff"].to_list() == [10] * 3

        df_to_verify = df.select(col("year_plus"))
        assert df_to_verify.to_polars()["year_plus"].to_list() == [datetime.datetime.now(datetime.timezone.utc).year + 2] * 3

        df_to_verify = df.select(year(timestamp_add(timestamp_add(current_date(), 1, "year"), 1, "year")).alias("year_pp"))
        assert df_to_verify.to_polars()["year_pp"].to_list() == [datetime.datetime.now(datetime.timezone.utc).year + 2]

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
                to_date(lit(f"{datetime.datetime.now(datetime.timezone.utc).year}-01-01")).alias("start_date"), # jan 1st of the current year
            ).alias("dt_pp"))
        # difference in days from jan 1st to feb 1st.
        assert dt_to_verify.to_polars()["dt_pp"].to_list() == [31]

    def test_no_coercion_needed(self):
        """Date and datetime columns should not be coerced to string."""
        df = pl.DataFrame({
            "date_col": [datetime.date(2023, 12, 25), datetime.date(2024, 1, 1)],
            "datetime_col": [datetime.datetime(2023, 12, 25, 14, 30, tzinfo=datetime.timezone.utc), datetime.datetime(2024, 1, 1, 9, 15, tzinfo=datetime.timezone.utc)],
            "int_col": [1, 2]
        })

        result = apply_ingestion_coercions(df, coerce_array=False)

        # Check that date_col is now string type
        assert result.schema["date_col"] == pl.Date
        assert result.schema["datetime_col"] == pl.Datetime
        assert result.schema["int_col"] == pl.Int64  # unchanged

        # Check actual values are string representations
        date_values = result["date_col"].to_list()
        datetime_values = result["datetime_col"].to_list()
        assert all(isinstance(val, datetime.date) for val in date_values)
        assert all(isinstance(val, datetime.datetime) for val in datetime_values)

    def test_timezone(self, local_session):
        """Test the timezone parameter of the timestamp type."""
        df_tz_naive = pl.DataFrame({
            "timestamp": [
                "2025-01-15 10:30:00",
                "2025-01-16 14:00:00",
                "2025-01-17 18:45:00"
            ]
        }).with_columns(
            pl.col("timestamp").str.to_datetime(time_unit="us") # Convert string to naive datetime
        )

        df_fenic_default_tz = local_session.create_dataframe(df_tz_naive).select(col("timestamp"))
        df_fenic_default_cached_tz = local_session.create_dataframe(df_tz_naive).select(col("timestamp")).cache()
        assert df_fenic_default_tz.schema.column_fields == [
            ColumnField("timestamp", TimestampType),
        ]
        assert df_fenic_default_cached_tz.schema.column_fields == [
            ColumnField("timestamp", TimestampType),
        ]

        # Convert the naive datetime column to timezone-aware
        df_tz_aware_converted = df_tz_naive.with_columns(
            pl.col("timestamp").dt.replace_time_zone("Europe/London").alias("ts_london")
        )
        df = local_session.create_dataframe(df_tz_aware_converted)
        df = df.select(col("ts_london"))
        assert df.schema.column_fields == [
            ColumnField("ts_london", TimestampType),
        ]

    def test_to_utc_timestamp(self, df_utc_timestamp):
        """Test the to_utc_timestamp function."""
        df = df_utc_timestamp.select(to_utc_timestamp(col("ts"), LA_TZ).alias("ts_to_utc"))
        df_cached = df_utc_timestamp.select(to_utc_timestamp(col("ts"), LA_TZ).alias("ts_to_utc")).cache()
        assert df.to_polars()["ts_to_utc"].to_list() == TS_TO_UTC_FROM_LA_RESULTS
        assert df_cached.to_polars()["ts_to_utc"].to_list() == TS_TO_UTC_FROM_LA_RESULTS

    def test_from_utc_timestamp(self, df_utc_timestamp):
        """Test the from_utc_timestamp function."""
        df = df_utc_timestamp.select(from_utc_timestamp(col("ts"), LA_TZ).alias("ts_to_la"))
        df_cached = df_utc_timestamp.select(from_utc_timestamp(col("ts"), LA_TZ).alias("ts_to_la")).cache()
        assert df.to_polars()["ts_to_la"].to_list() == TS_FROM_UTC_TO_LA_RESULTS
        assert df_cached.to_polars()["ts_to_la"].to_list() == TS_FROM_UTC_TO_LA_RESULTS

    def test_format_timestamps_multiple_timezones(self, local_session):
        """Test formatting timestamps with multiple timezones."""
        event_times = [
            datetime.datetime(2025, 1, 15, 18, 30, 0, tzinfo=zoneinfo.ZoneInfo("UTC")),
            datetime.datetime(2025, 1, 16, 14, 45, 0, tzinfo=zoneinfo.ZoneInfo("UTC"))
        ]
        df = local_session.create_dataframe({"event_time": event_times})
        df = df.select(
            col("event_time"),
            date_format(
                from_utc_timestamp(col("event_time"), "America/Los_Angeles"),
                "yyyy-MM-dd HH:mm:ss"
            ).alias("time_pst"),
            date_format(
                from_utc_timestamp(col("event_time"), "America/New_York"),
                "yyyy-MM-dd HH:mm:ss"
            ).alias("time_est"),
            date_format(
                from_utc_timestamp(col("event_time"), "Europe/London"),
                "yyyy-MM-dd HH:mm:ss"
            ).alias("time_gmt")
        )
        result = df.to_polars()
        assert result["time_pst"][0] == "2025-01-15 10:30:00"
        assert result["time_est"][0] == "2025-01-15 13:30:00"
        assert result["time_gmt"][0] == "2025-01-15 18:30:00"
        assert result["time_pst"][1] == "2025-01-16 06:45:00"
        assert result["time_est"][1] == "2025-01-16 09:45:00"
        assert result["time_gmt"][1] == "2025-01-16 14:45:00"


    def test_timestamp_literal_with_tz(self, df_with_date_types):
        """Test the timestamp literal with timezone."""
        df = df_with_date_types.select(
            year(col("ts_la")).alias("year_ts_la"),
            date_trunc(col("ts_la"), "year").alias("ts_la_trunc"),
            timestamp_add(col("ts_la"), 1, "year").alias("ts_la_plus")
        )
        df_cached = df_with_date_types.select(
            year(col("ts_la")).alias("year_ts_la"),
            date_trunc(col("ts_la"), "year").alias("ts_la_trunc"),
            timestamp_add(col("ts_la"), 1, "year").alias("ts_la_plus")
        ).cache()
        assert df.to_polars()["year_ts_la"].to_list() == [2025]*3
        assert df.to_polars()["ts_la_trunc"].to_list() == [datetime.datetime(2025, 1, 1, 0, 0, tzinfo=zoneinfo.ZoneInfo(key='UTC'))]*3
        assert df.to_polars()["ts_la_plus"].to_list() == [datetime.datetime(2026, 1, 2, 9, 1, 1, tzinfo=zoneinfo.ZoneInfo(key='UTC'))]*3

        assert df_cached.to_polars()["year_ts_la"].to_list() == [2025]*3
        assert df_cached.to_polars()["ts_la_trunc"].to_list() == [datetime.datetime(2025, 1, 1, 0, 0, tzinfo=zoneinfo.ZoneInfo(key='UTC'))]*3
        assert df_cached.to_polars()["ts_la_plus"].to_list() == [datetime.datetime(2026, 1, 2, 9, 1, 1, tzinfo=zoneinfo.ZoneInfo(key='UTC'))]*3

    def test_timestamp_diff_with_tz(self, df_with_date_types):
        """Verify we can do math between """
        df = df_with_date_types.select(timestamp_diff(col("ts_la"), col("ts"), "hour").alias("ts_diff"))
        assert df.to_polars()["ts_diff"].to_list() == [-32, 737, 1446]

        df = df_with_date_types.select(timestamp_diff(col("ts_utc"), col("ts"), "hour").alias("ts_diff"))
        assert df.to_polars()["ts_diff"].to_list() == [-24, 745, 1454]

        df = df_with_date_types.select(timestamp_diff(col("ts_la"), col("date"), "hour").alias("ts_diff"))
        assert df.to_polars()["ts_diff"].to_list() == [-33, -9, 14]

    def test_datediff_with_tz(self, df_with_date_types):
        """Verify we can do math between """
        df = df_with_date_types.select(datediff(col("ts_la"), col("ts")).alias("dt_diff"))
        assert df.to_polars()["dt_diff"].to_list() == [1, -30, -60]

        df = df_with_date_types.select(datediff(col("ts_utc"), col("dt")).alias("dt_diff"))
        assert df.to_polars()["dt_diff"].to_list() == [1, -30, -59]

    def test_dt_with_tz_in_nested_struct_with_lists(self, local_session):
        """Test a struct with a temporal types including datetime with timezone."""
        df = local_session.create_dataframe({
            "struct_col": [{"ts_la_tz": TS_LA_INPUT[0],
                            "ts_naive": TS_UTC_NAIVE_INPUT[0],
                            "ts_utc": TS_UTC_INPUT[0],
                            "ts_naive_list": TS_UTC_NAIVE_INPUT,
                            "ts_la_list": TS_LA_INPUT,
                            "nested_ts_struct": [{
                                "ts_la_tz": TS_LA_INPUT[0],
                                "ts_naive": TS_UTC_NAIVE_INPUT[0],
                                "ts_naive_list": TS_UTC_NAIVE_INPUT,
                                "ts_la_list": TS_LA_INPUT,
                            }]
                        }],
        })
        df = df.select('*')
        df_cached = df.cache()
        expected_schema = [
            ColumnField("struct_col", StructType([
                StructField("ts_la_tz", TimestampType),
                StructField("ts_naive", TimestampType),
                StructField("ts_utc", TimestampType),
                StructField("ts_naive_list", ArrayType(TimestampType)),
                StructField("ts_la_list", ArrayType(TimestampType)),
                StructField("nested_ts_struct", ArrayType(StructType([
                    StructField("ts_la_tz", TimestampType),
                    StructField("ts_naive", TimestampType),
                    StructField("ts_naive_list", ArrayType(TimestampType)),
                    StructField("ts_la_list", ArrayType(TimestampType)),
                ]))),
            ]))]
        assert df.schema.column_fields == expected_schema
        assert df_cached.schema.column_fields == expected_schema

        assert df.to_polars()["struct_col"][0]["ts_la_tz"] == TS_TO_UTC_FROM_LA_RESULTS[0]
        assert df.to_polars()["struct_col"][0]["ts_naive"] == TS_UTC_INPUT[0]
        assert df.to_polars()["struct_col"][0]["ts_utc"] == TS_UTC_INPUT[0]
        assert df.to_polars()["struct_col"][0]["ts_naive_list"] == TS_UTC_INPUT
        assert df.to_polars()["struct_col"][0]["ts_la_list"] == TS_TO_UTC_FROM_LA_RESULTS
        assert df.to_polars()["struct_col"][0]["nested_ts_struct"][0]["ts_la_tz"] == TS_TO_UTC_FROM_LA_RESULTS[0]
        assert df.to_polars()["struct_col"][0]["nested_ts_struct"][0]["ts_naive"] == TS_UTC_INPUT[0]
        assert df.to_polars()["struct_col"][0]["nested_ts_struct"][0]["ts_naive_list"] == TS_UTC_INPUT
        assert df.to_polars()["struct_col"][0]["nested_ts_struct"][0]["ts_la_list"] == TS_TO_UTC_FROM_LA_RESULTS

        assert df_cached.to_polars()["struct_col"][0]["ts_la_tz"] == TS_TO_UTC_FROM_LA_RESULTS[0]
        assert df_cached.to_polars()["struct_col"][0]["ts_naive"] == TS_UTC_INPUT[0]
        assert df_cached.to_polars()["struct_col"][0]["ts_utc"] == TS_UTC_INPUT[0]
        assert df_cached.to_polars()["struct_col"][0]["ts_naive_list"] == TS_UTC_INPUT
        assert df_cached.to_polars()["struct_col"][0]["ts_la_list"] == TS_TO_UTC_FROM_LA_RESULTS
        assert df_cached.to_polars()["struct_col"][0]["nested_ts_struct"][0]["ts_la_tz"] == TS_TO_UTC_FROM_LA_RESULTS[0]
        assert df_cached.to_polars()["struct_col"][0]["nested_ts_struct"][0]["ts_naive"] == TS_UTC_INPUT[0]
        assert df_cached.to_polars()["struct_col"][0]["nested_ts_struct"][0]["ts_naive_list"] == TS_UTC_INPUT
        assert df_cached.to_polars()["struct_col"][0]["nested_ts_struct"][0]["ts_la_list"] == TS_TO_UTC_FROM_LA_RESULTS
