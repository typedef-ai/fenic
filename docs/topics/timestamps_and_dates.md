# Timestamps and Dates

This page describes how Fenic handles temporal data types (`TimestampType` and `DateType`), including timezone behavior, storage precision, and common operations.

## Overview

Fenic provides two temporal data types along with various temporal manipulation functions in the fenic.dt module.

- **`TimestampType`**: Represents a point in time with microsecond precision, always stored in UTC
- **`DateType`**: Represents a calendar date without time information

Both types and related functionality follow similar patterns to PySpark, except that fenic converts all timestamps to UTC timezone instead of the session local timezone.

## Key Features

- **Microsecond precision**: Timestamps support up to 1/1,000,000 second precision
- **UTC-first design**: All timestamps are normalized to UTC during ingestion
- **Automatic timezone conversion**: Timezone-aware inputs are converted to UTC automatically
- **Consistent behavior**: Same behavior across all data sources (Parquet, CSV, in-memory DataFrames, tables)
- **Rich date/time functions**: Comprehensive set of functions for temporal operations
- **Timezone conversion utilities**: Functions to work with local timezones while maintaining UTC storage

## TimestampType and DateType

### TimestampType

Represents a point in time stored as microseconds since the Unix epoch (1970-01-01 00:00:00 UTC).

**Characteristics:**

- Precision: Microseconds (μs)
- Timezone: Always UTC
- Storage: Int64 (microseconds from epoch)
- Python type: `datetime.datetime` with `tzinfo=UTC`

### DateType

Represents a calendar date without time-of-day information.

**Characteristics:**

- Precision: Day
- Timezone: Not applicable
- Storage: Int32 (days from epoch)
- Python type: `datetime.date`

## PySpark-like behavior

Fenic's timestamp handling and functionality is very similar to PySpark, with the following differences:

- **Default timezone**: Fenic always converts timestamps to **UTC** whereas PySpark converts to a configurable timezone that defaults to the session's global timezone.
- **Timezone consistency**: Fenic guarantees all timestamps are UTC, eliminating ambiguity when session configuration changes

## Conversion and Casting Behavior Reference

The following table shows how different input types are handled when loading or casting to `TimestampType` and `DateType`:

| **Input Type**                | **Operation**                                  | **Input Example**                                                      | **Result**                | **Notes**                                 |
| ----------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------- | ------------------------- | ----------------------------------------- |
| **Naive Python datetime**     | Load into TimestampType                        | `datetime.datetime(2025, 1, 1, 10, 0)`                                 | `2025-01-01 10:00:00 UTC` | Interpreted as UTC                        |
| **UTC Python datetime**       | Load into TimestampType                        | `datetime.datetime(2025, 1, 1, 10, 0, tzinfo=UTC)`                     | `2025-01-01 10:00:00 UTC` | Already UTC, no conversion                |
| **LA Python datetime**        | Load into TimestampType                        | `datetime.datetime(2025, 1, 1, 10, 0, tzinfo=LA)`                      | `2025-01-01 18:00:00 UTC` | Converted: +8 hour offset                 |
| **Naive Polars Datetime**     | Load into TimestampType                        | `pl.Datetime` (no timezone)                                            | Interpreted as UTC        | Applied during ingestion                  |
| **Polars Datetime (UTC)**     | Load into TimestampType                        | `pl.Datetime(time_zone="UTC")`                                         | Kept as UTC               | No conversion needed                      |
| **Polars Datetime (LA)**      | Load into TimestampType                        | `pl.Datetime(time_zone="America/Los_Angeles")`                         | Converted to UTC          | Timezone conversion applied               |
| **String (ISO 8601 with TZ)** | `cast(TimestampType)`                          | `"2025-01-01T10:00:00+08:00"`                                          | `2025-01-01 02:00:00 UTC` | Parses timezone, converts to UTC          |
| **String (ISO 8601 no TZ)**   | `cast(TimestampType)`                          | `"2025-01-01T10:00:00.000"`                                            | `None`                    | Currently returns null (limitation)       |
| **String with format**        | `to_timestamp(col, format)`                    | `"01-15-2025 10:30:00"` with format `"MM-dd-yyyy HH:mm:ss"`            | `2025-01-15 10:30:00 UTC` | Parsed as UTC if no TZ in format          |
| **String with format + TZ**   | `to_timestamp(col, format)`                    | `"01-15-2025 10:30:00 +08:00"` with format `"MM-dd-yyyy HH:mm:ss XXX"` | `2025-01-15 02:30:00 UTC` | Timezone parsed and converted             |
| **Integer (microseconds)**    | `cast(TimestampType)`                          | `1735729800000000`                                                     | `2025-01-01 10:30:00 UTC` | Interpreted as μs from epoch              |
| **TimestampType**             | `cast(DateType)`                               | `2025-01-01 10:30:00 UTC`                                              | `2025-01-01`              | Truncates time, keeps date                |
| **DateType**                  | `cast(TimestampType)`                          | `2025-01-01`                                                           | `2025-01-01 00:00:00 UTC` | Midnight UTC                              |
| **String (date format)**      | `cast(DateType)`                               | `"2025-01-01"`                                                         | `2025-01-01`              | Parses as date                            |
| **String with format**        | `to_date(col, format)`                         | `"01-27-2025"` with format `"MM-dd-yyyy"`                              | `2025-01-27`              | Custom format parsing                     |
| **Integer (days)**            | `cast(DateType)`                               | `20000`                                                                | `2024-10-04`              | Days since epoch                          |
| **SQL CAST to DATE**          | `session.sql("SELECT CAST(col AS DATE)")`      | `"2025-01-01"`                                                         | `2025-01-01`              | DuckDB parses date, returned as DateType  |
| **SQL CAST to TIMESTAMP**     | `session.sql("SELECT CAST(col AS TIMESTAMP)")` | `"2025-01-01 10:30:00"`                                                | `2025-01-01 10:30:00 UTC` | DuckDB parses timestamp, converted to UTC |
| **SQL DATE_TRUNC**            | `session.sql("SELECT DATE_TRUNC('day', ts)")`  | Timestamp column                                                       | `2025-01-01 00:00:00 UTC` | DuckDB truncates, result converted to UTC |
| **TimestampType**             | `date_format(col, format)`                     | `2025-01-15 10:30:45 UTC` with format `"yyyy-MM-dd HH:mm:ss"`          | `"2025-01-15 10:30:45"`   | Formats timestamp as string               |
| **DateType**                  | `date_format(col, format)`                     | `2025-01-15` with format `"MM/dd/yyyy"`                                | `"01/15/2025"`            | Formats date as string                    |

### Timezone Conversion Function Reference

| **Function**                                     | **Input**                 | **Result**                | **Explanation**                                         |
| ------------------------------------------------ | ------------------------- | ------------------------- | ------------------------------------------------------- |
| `to_utc_timestamp(col, "America/Los_Angeles")`   | `2025-01-15 10:30:00 UTC` | `2025-01-15 18:30:00 UTC` | Treats input as LA wall-clock time, converts to UTC     |
| `from_utc_timestamp(col, "America/Los_Angeles")` | `2025-01-15 10:30:00 UTC` | `2025-01-15 02:30:00 UTC` | Converts input to LA wall-clock, then represents as UTC |

## Generating Timestamps and Dates

### Creating Timestamps

You can create timestamps by loading Python `datetime` objects or Polars DataFrames with datetime columns, or by using Fenic's native functions like `current_timestamp()` and `now()`.

```python
import datetime
import zoneinfo
from fenic.api.functions.dt import current_timestamp, now

# Create a timestamp from Python datetime
df = session.create_dataframe({
    "event_time": [datetime.datetime(2025, 1, 15, 10, 30, 0, tzinfo=zoneinfo.ZoneInfo("UTC"))]
})
df.show()
# Output: 2025-01-15 10:30:00 UTC

# Get current timestamp
df = session.create_dataframe({"id": [1, 2, 3]})
df = df.select(current_timestamp().alias("ts"))
df.show()
# Output: 2025-01-15 14:23:45 UTC (current time)

# Alternative: use now()
df = df.select(now().alias("ts"))
df.show()
# Output: 2025-01-15 14:23:45 UTC (current time)
```

#### Creating Dates

You can create dates by loading Python `date` objects or Polars DataFrames with date columns, or by using Fenic's native `current_date()` function.

```python
import datetime
import zoneinfo
from fenic.api.functions.dt import current_timestamp, now

# Create a timestamp from Python datetime
df = session.create_dataframe({
    "event_time": [datetime.datetime(2025, 1, 15, 10, 30, 0, tzinfo=zoneinfo.ZoneInfo("UTC"))]
})
df.show()
# Output: 2025-01-15 10:30:00 UTC

# Get current timestamp using Fenic functions
df = session.create_dataframe({"id": [1, 2, 3]})
df = df.select(current_timestamp().alias("ts"))
df.show()
# Output: 2025-01-15 14:23:45 UTC (current time)

# Alternative: use now()
df = df.select(now().alias("ts"))
df.show()
# Output: 2025-01-15 14:23:45 UTC (current time)
```

## Ingestion and Persistence

### Reading and Writing Files

When you write timestamp or date data to files or tables, Fenic writes them in your **local session timezone** (the timezone of your computer). When you read them back, Fenic automatically converts timestamps back to **UTC**.

```python
import datetime
import zoneinfo

# Create a DataFrame with a date
df = session.create_dataframe({
    "event_date": [datetime.date(2024, 1, 4)]
})

# Write to CSV (saved in local session timezone, e.g., "2024-01-04" in PST)
df.write.csv("events.csv")

# Read back from CSV (automatically converted to UTC)
df_read = session.read.csv("events.csv")
df_read.show()
# Output: 2024-01-04 (dates are timezone-agnostic)
```

For timestamps:

```python
import datetime
import zoneinfo

# Create a DataFrame with a timestamp
df = session.create_dataframe({
    "created_at": [datetime.datetime(2024, 1, 4, 7, 10, 13, tzinfo=zoneinfo.ZoneInfo("UTC"))]
})

# Write to CSV (saved with timezone info, e.g., "2024-01-04 07:10:13.000000+00:00")
df.write.csv("events.csv")

# Read back from CSV (automatically normalized to UTC)
df_read = session.read.csv("events.csv")
df_read.show()
# Output: 2024-01-04 07:10:13 UTC
```

The same behavior applies to:

- **Parquet files**: Timestamps stored with timezone metadata, automatically converted to UTC on read
- **Fenic tables** (DuckDB): DuckDB session timezone is always UTC, ensuring consistency
- **In-memory DataFrames**: Polars DataFrames with timezone-aware or naive timestamps are normalized to UTC

### SQL Queries

When using `session.sql()`, the underlying SQL engine (DuckDB) converts timestamps to the local session timezone during query execution. However, when Fenic loads the results back from SQL queries (including views and tables), all timestamps are automatically converted back to UTC.

```python
import datetime
import zoneinfo

# Create DataFrame with timestamps in different timezones
ts_la = datetime.datetime(2025, 1, 2, 1, 1, 1, tzinfo=zoneinfo.ZoneInfo("America/Los_Angeles"))
ts_utc = datetime.datetime(2025, 1, 2, 1, 1, 1, tzinfo=zoneinfo.ZoneInfo("UTC"))

df = session.create_dataframe({
    "ts_la": [ts_la],
    "ts_utc": [ts_utc]
})

# Execute SQL query - DuckDB internally uses local session timezone
result = session.sql("SELECT * FROM {df1}", df1=df)

# Results are automatically converted back to UTC
result.show()
# Output:
#┌─────────────────────────┬─────────────────────────┐
#│ ts_la                   ┆ ts_utc                  │
#╞═════════════════════════╪═════════════════════════╡
#│ 2025-01-02 09:01:01 UTC ┆ 2025-01-02 01:01:01 UTC │  # Both in UTC
#└─────────────────────────┴─────────────────────────┘

# You can use SQL date/time functions
result = session.sql(
    "SELECT DATE_TRUNC('day', ts_utc) as day_start FROM {df1}",
    df1=df
)
result.show()
# Output: 2025-01-02 00:00:00 UTC (converted back to UTC)
```

This behavior ensures:

- Consistent UTC timestamps regardless of SQL operations
- Predictable behavior when working with views and tables created via SQL
- No timezone confusion when chaining SQL and DataFrame operations

### Timezone Conversion During Ingestion

Fenic automatically handles different timezone inputs:

```python
import datetime
import zoneinfo

# Naive timestamp → interpreted as UTC
naive = datetime.datetime(2025, 1, 15, 10, 30, 0)

# UTC timestamp → no conversion needed
utc = datetime.datetime(2025, 1, 15, 10, 30, 0, tzinfo=zoneinfo.ZoneInfo("UTC"))

# LA timestamp → converted to UTC
la = datetime.datetime(2025, 1, 15, 10, 30, 0, tzinfo=zoneinfo.ZoneInfo("America/Los_Angeles"))

df = session.create_dataframe({
    "naive_ts": [naive],
    "utc_ts": [utc],
    "la_ts": [la]
})

df.show()
# Output:
#┌─────────────────────────┬─────────────────────────┬─────────────────────────┐
#│ naive_ts                ┆ utc_ts                  ┆ la_ts                   │
#╞═════════════════════════╪═════════════════════════╪═════════════════════════╡
#│ 2025-01-15 10:30:00 UTC ┆ 2025-01-15 10:30:00 UTC ┆ 2025-01-15 18:30:00 UTC │
#└─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

## Working with Timestamps

### Parsing Timestamps from Strings

Use `to_timestamp()` to parse timestamp strings with custom formats:

```python
from fenic.api.functions.dt import to_timestamp
from fenic.api.functions import col

df = session.create_dataframe({
    "timestamp_str": ["01-15-2025 10:30:00", "01-16-2025 14:00:00"]
})

df = df.select(
    to_timestamp(col("timestamp_str"), "MM-dd-yyyy HH:mm:ss").alias("ts")
)
df.show()
# Output:
#┌─────────────────────────┐
#│ ts                      │
#╞═════════════════════════╡
#│ 2025-01-15 10:30:00 UTC │
#│ 2025-01-16 14:00:00 UTC │
#└─────────────────────────┘
```

With timezone information:

```python
df = session.create_dataframe({
    "timestamp_str": ["01-15-2025 10:30:00 +08:00", "01-16-2025 14:00:00 +08:00"]
})

df = df.select(
    to_timestamp(col("timestamp_str"), "MM-dd-yyyy HH:mm:ss XXX").alias("ts")
)
df.show()
# Output (converted to UTC):
#┌─────────────────────────┐
#│ ts                      │
#╞═════════════════════════╡
#│ 2025-01-15 02:30:00 UTC │
#│ 2025-01-16 06:00:00 UTC │
#└─────────────────────────┘
```

```python
from fenic import TimestampType, DateType

# Integer to Timestamp (microseconds from epoch)
df = session.create_dataframe({"ts_int": [1735729800000000]})
df = df.select(col("ts_int").cast(TimestampType).alias("ts"))
df.show()
# Output: 2025-01-01 10:30:00 UTC

# Timestamp to Date (truncate time)
df = df.select(col("ts").cast(DateType).alias("date"))
df.show()
# Output: 2025-01-01

# Date to Timestamp (midnight UTC)
df = df.select(col("date").cast(TimestampType).alias("ts"))
df.show()
# Output: 2025-01-01 00:00:00 UTC
```

### Extracting Date Components

```python
from fenic.api.functions.dt import year, month, day, hour, minute, second

df = session.create_dataframe({
    "ts": [datetime.datetime(2025, 1, 15, 10, 30, 45, tzinfo=zoneinfo.ZoneInfo("UTC"))]
})

df = df.select(
    year(col("ts")).alias("year"),
    month(col("ts")).alias("month"),
    day(col("ts")).alias("day"),
    hour(col("ts")).alias("hour"),
    minute(col("ts")).alias("minute"),
    second(col("ts")).alias("second")
)
df.show()
# Output:
#┌──────┬───────┬─────┬──────┬────────┬────────┐
#│ year ┆ month ┆ day ┆ hour ┆ minute ┆ second │
#╞══════╪═══════╪═════╪══════╪════════╪════════╡
#│ 2025 ┆ 1     ┆ 15  ┆ 10   ┆ 30     ┆ 45     │
#└──────┴───────┴─────┴──────┴────────┴────────┘
```

## Working with Timezones

### Timezone Conversion Functions

Fenic provides two functions for working with timestamps across timezones:

#### to_utc_timestamp

Interprets a timestamp (stored in UTC) as wall-clock time in a specific timezone, then converts it to UTC.

**Use case**: Your data contains timestamps representing local time but stored with UTC metadata. You want to convert them to actual UTC.

```python
from fenic.api.functions.dt import to_timestamp, to_utc_timestamp

# Timestamps representing LA wall-clock time
df = session.create_dataframe({
    "la_wall_clock": ["2025-01-15 10:30:00", "2025-01-16 14:00:00"]
})

df = df.select(to_timestamp(col("la_wall_clock"), "yyyy-MM-dd HH:mm:ss").alias("ts"))
df = df.select(to_utc_timestamp(col("ts"), "America/Los_Angeles").alias("utc_ts"))
df.show()
# Output:
#┌─────────────────────────┐
#│ utc_ts                  │
#╞═════════════════════════╡
#│ 2025-01-15 18:30:00 UTC │  # 10:30 AM LA → 6:30 PM UTC
#│ 2025-01-16 22:00:00 UTC │  # 2:00 PM LA → 10:00 PM UTC
#└─────────────────────────┘
```

#### from_utc_timestamp

Converts a UTC timestamp to wall-clock time in a specific timezone, then represents it as UTC.

**Use case**: Display or work with timestamps in a local timezone while keeping them stored as UTC.

```python
from fenic.api.functions.dt import to_timestamp, from_utc_timestamp, date_format

# UTC timestamps
df = session.create_dataframe({
    "utc_time": ["2025-01-15 10:30:00", "2025-01-16 14:00:00"]
})

df = df.select(to_timestamp(col("utc_time"), "yyyy-MM-dd HH:mm:ss").alias("ts"))
df = df.select(
    from_utc_timestamp(col("ts"), "America/Los_Angeles").alias("la_ts"),
    date_format(
        from_utc_timestamp(col("ts"), "America/Los_Angeles"),
        "yyyy-MM-dd HH:mm:ss"
    ).alias("la_formatted")
)
df.show()
# Output:
#┌─────────────────────────┬─────────────────────┐
#│ la_ts                   ┆ la_formatted        │
#╞═════════════════════════╪═════════════════════╡
#│ 2025-01-15 02:30:00 UTC ┆ 2025-01-15 02:30:00 │  # 10:30 AM UTC → 2:30 AM LA
#│ 2025-01-16 06:00:00 UTC ┆ 2025-01-16 06:00:00 │  # 2:00 PM UTC → 6:00 AM LA
#└─────────────────────────┴─────────────────────┘
```

## Formatting Timestamps and Dates

Use `date_format()` to create formatted strings from timestamps or dates with whatever format you want:

```python
from fenic.api.functions.dt import date_format, from_utc_timestamp

df = session.create_dataframe({
    "ts": [datetime.datetime(2025, 1, 15, 10, 30, 0, tzinfo=zoneinfo.ZoneInfo("UTC"))],
    "date": [datetime.date(2025, 1, 15)]
})

# Format timestamps and dates in various formats
df = df.select(
    col("ts"),
    col("date"),
    date_format(col("ts"), "yyyy-MM-dd HH:mm:ss").alias("ts_simple"),
    date_format(col("ts"), "MM/dd/yyyy hh:mm a").alias("ts_12hr"),
    date_format(
        from_utc_timestamp(col("ts"), "America/Los_Angeles"),
        "MM-dd-yyyy hh:mm:ss a XXX"
    ).alias("ts_la_with_tz"),
    date_format(col("date"), "MMMM dd, yyyy").alias("date_long"),
    date_format(col("date"), "MM/dd/yy").alias("date_short")
)
df.show()
# Output:
#┌─────────────────────────┬────────────┬─────────────────────┬──────────────────┬──────────────────────────┬──────────────────┬────────────┐
#│ ts                      ┆ date       ┆ ts_simple           ┆ ts_12hr          ┆ ts_la_with_tz            ┆ date_long        ┆ date_short │
#╞═════════════════════════╪════════════╪═════════════════════╪══════════════════╪══════════════════════════╪══════════════════╪════════════╡
#│ 2025-01-15 10:30:00 UTC ┆ 2025-01-15 ┆ 2025-01-15 10:30:00 ┆ 01/15/2025 10:30 ┆ 01-15-2025 02:30:00 AM   ┆ January 15, 2025 ┆ 01/15/25   │
#│                         ┆            ┆                     ┆ AM               ┆ +00:00                   ┆                  ┆            │
#└─────────────────────────┴────────────┴─────────────────────┴──────────────────┴──────────────────────────┴──────────────────┴────────────┘
```

## Date and Time Arithmetic

### Adding/Subtracting Days

```python
from fenic.api.functions.dt import date_add, date_sub

df = session.create_dataframe({
    "date": [datetime.date(2025, 1, 15)]
})

df = df.select(
    date_add(col("date"), 7).alias("next_week"),
    date_sub(col("date"), 7).alias("last_week")
)
df.show()
# Output:
#┌────────────┬────────────┐
#│ next_week  ┆ last_week  │
#╞════════════╪════════════╡
#│ 2025-01-22 ┆ 2025-01-08 │
#└────────────┴────────────┘
```

### Adding/Subtracting Time Units

```python
from fenic.api.functions.dt import timestamp_add

df = session.create_dataframe({
    "ts": [datetime.datetime(2025, 1, 15, 10, 30, 0, tzinfo=zoneinfo.ZoneInfo("UTC"))]
})

df = df.select(
    timestamp_add(col("ts"), 2, "hour").alias("in_2_hours"),
    timestamp_add(col("ts"), -30, "minute").alias("30_min_ago")
)
df.show()
# Output:
#┌─────────────────────────┬─────────────────────────┐
#│ in_2_hours              ┆ 30_min_ago              │
#╞═════════════════════════╪═════════════════════════╡
#│ 2025-01-15 12:30:00 UTC ┆ 2025-01-15 10:00:00 UTC │
#└─────────────────────────┴─────────────────────────┘
```

### Calculating Differences

```python
from fenic.api.functions.dt import datediff, timestamp_diff

df = session.create_dataframe({
    "start": [datetime.date(2025, 1, 1)],
    "end": [datetime.date(2025, 1, 15)]
})

df = df.select(datediff(col("end"), col("start")).alias("days_diff"))
df.show()
# Output: 14

# For timestamps
df = session.create_dataframe({
    "start_ts": [datetime.datetime(2025, 1, 1, 10, 0, 0, tzinfo=zoneinfo.ZoneInfo("UTC"))],
    "end_ts": [datetime.datetime(2025, 1, 15, 14, 30, 0, tzinfo=zoneinfo.ZoneInfo("UTC"))]
})

df = df.select(
    timestamp_diff(col("start_ts"), col("end_ts"), "day").alias("days"),
    timestamp_diff(col("start_ts"), col("end_ts"), "hour").alias("hours")
)
df.show()
# Output:
#┌──────┬───────┐
#│ days ┆ hours │
#╞══════╪═══════╡
#│ 14   ┆ 340   │
#└──────┴───────┘
```

## Common Patterns

### Current Date/Time

```python
from fenic.api.functions.dt import current_date, current_timestamp

df = session.create_dataframe({"id": [1, 2, 3]})

df = df.select(
    col("id"),
    current_date().alias("today"),
    current_timestamp().alias("now")
)
df.show()
# Output:
#┌────┬────────────┬─────────────────────────┐
#│ id ┆ today      ┆ now                     │
#╞════╪════════════╪═════════════════════════╡
#│ 1  ┆ 2025-01-15 ┆ 2025-01-15 10:30:45 UTC │
#│ 2  ┆ 2025-01-15 ┆ 2025-01-15 10:30:45 UTC │
#│ 3  ┆ 2025-01-15 ┆ 2025-01-15 10:30:45 UTC │
#└────┴────────────┴─────────────────────────┘
```

### Formatting Timestamps with Multiple Timezones

When displaying timestamps to users in different regions, format them to their local timezones without showing the timezone suffix:

```python
from fenic.api.functions.dt import from_utc_timestamp, date_format

df = session.create_dataframe({
    "event_time": [
        datetime.datetime(2025, 1, 15, 18, 30, 0, tzinfo=zoneinfo.ZoneInfo("UTC")),
        datetime.datetime(2025, 1, 16, 14, 45, 0, tzinfo=zoneinfo.ZoneInfo("UTC"))
    ]
})

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
df.show()
# Output:
#┌─────────────────────────┬─────────────────────┬─────────────────────┬─────────────────────┐
#│ event_time              ┆ time_pst            ┆ time_est            ┆ time_gmt            │
#╞═════════════════════════╪═════════════════════╪═════════════════════╪═════════════════════╡
#│ 2025-01-15 18:30:00 UTC ┆ 2025-01-15 10:30:00 ┆ 2025-01-15 13:30:00 ┆ 2025-01-15 18:30:00 │
#│ 2025-01-16 14:45:00 UTC ┆ 2025-01-16 06:45:00 ┆ 2025-01-16 09:45:00 ┆ 2025-01-16 14:45:00 │
#└─────────────────────────┴─────────────────────┴─────────────────────┴─────────────────────┘
```

### Truncating Timestamps

```python
from fenic.api.functions.dt import date_trunc

df = session.create_dataframe({
    "ts": [datetime.datetime(2025, 1, 15, 10, 30, 45, tzinfo=zoneinfo.ZoneInfo("UTC"))]
})

df = df.select(
    date_trunc(col("ts"), "year").alias("year_start"),
    date_trunc(col("ts"), "month").alias("month_start"),
    date_trunc(col("ts"), "day").alias("day_start"),
    date_trunc(col("ts"), "hour").alias("hour_start")
)
df.show()
# Output:
#┌─────────────────────────┬─────────────────────────┬─────────────────────────┬─────────────────────────┐
#│ year_start              ┆ month_start             ┆ day_start               ┆ hour_start              │
#╞═════════════════════════╪═════════════════════════╪═════════════════════════╪═════════════════════════╡
#│ 2025-01-01 00:00:00 UTC ┆ 2025-01-01 00:00:00 UTC ┆ 2025-01-15 00:00:00 UTC ┆ 2025-01-15 10:00:00 UTC │
#└─────────────────────────┴─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

## Limitations

- **String casting without timezone**: Currently, casting ISO 8601 timestamp strings without timezone information to `TimestampType` returns `None`. Use `to_timestamp()` instead:

```python
# This returns None:
df.select(col("ts_str").cast(TimestampType))  # "2025-01-01T10:00:00.000"

# Use this instead:
df.select(to_timestamp(col("ts_str")))  # Default ISO 8601 parsing
```

## References

- Date/Time Functions: `src/fenic/api/functions/dt.py`
- Cast Implementation: `src/fenic/api/column.py:197-228`
- Ingestion Coercions: `src/fenic/_backends/local/physical_plan/utils.py:6-54`
- Type Definitions: `src/fenic/core/types/datatypes.py`
- SQL Tests: `tests/_backends/local/test_sql.py`
