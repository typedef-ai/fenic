"""
Possible semantic.reduce API:
Goals:
- optionally pass group context to jinja template for improved per group context
- optionally allow user to sort within group (e.g. for time series data, where order does matter)

def reduce(
    instruction: str,
    column: ColumnOrName, # NEW
    *,
    group_context: Optional[Dict[str, Column]] = None, # NEW
    order_by: Optional[ColumnOrName] = None, # NEW
    model_alias: Optional[Union[str, ModelAlias]] = None,
    temperature: float = 0,
) -> Column:

# simple example (no group context, no order_by)
df = df.group_by(["department", "fiscal_year"]).agg(
    fc.semantic.reduce(
        instruction="Summarize the milestone documents",
        model_alias="oai-small",
        column=fc.col("data"),
    ).alias("summary")
)

# complex example (group context, order_by)
df = df.group_by(["department", "fiscal_year"]).agg(
    fc.semantic.reduce(
        instruction="Summarize the milestones for the {{d}} department that occurred in the {{fy}}",
        column=fc.col("data"),
        group_context={
            "d": fc.col("department"),
            "fy": fc.col("fiscal_year"),
        },
        order_by=fc.asc(fc.col("sort_key")),
    ).alias("summary")
)
"""
import polars as pl

# sketch to lower to polars
df = pl.DataFrame({
    "department": ["accounting", "accounting", "accounting", "accounting",
                   "engineering", "engineering", "engineering", "engineering"],
    "fiscal_year": ["2025", "2024", "2025", "2024", "2025", "2024", "2025", "2024"],
    "sort_key": [7, 6, 5, 4, 3, 2, 1, 0],
    "data": ["Q1 revenue exceeded projections by 15%",
             "Annual audit completed successfully",
             "New budget allocation approved for team",
             "Tax filing deadline met ahead of schedule",
             "Deployed new microservices architecture",
             "Code review process streamlined by 40%",
             "Machine learning model accuracy improved",
             "Infrastructure migration completed on time"],
    "other_data": [1, 2, 3, 4, 5, 6, 7, 8]
})

def agg_func(outer_series: pl.Series) -> pl.Series:
    res = []
    # for real, this loop would run in a threadpool
    for inner_series in outer_series:
        first_row = inner_series.first()
        group_keys = (first_row["d"], first_row["fy"])
        inner_sort_indices = inner_series.struct.field("sort_key").arg_sort(descending=False)
        sorted_data = ", ".join(inner_series.struct.field("data").gather(inner_sort_indices).to_list())
        # next step: render jinja template with group context
        # next step: call hierarchical reduction function
        res.append(f"group: {group_keys}, data: {sorted_data}")
    return pl.Series(values=res, dtype=pl.String)

df = df.group_by(["department", "fiscal_year"]).agg(
   pl.struct([
       pl.col("department").alias("d"),
       pl.col("fiscal_year").alias("fy"),
       pl.col("sort_key"),
       pl.col("data")
   ])
   .alias("struct")
   .map_batches(
       agg_func,
       agg_list=True,
       returns_scalar=True,
       return_dtype=pl.String
   )
)

print(df['struct'].to_list())

import polars as pl

print("Polars version:", pl.__version__)

# Group by struct column (should work)
df_struct = pl.DataFrame({
    "key": [{"a": 1, "b": 2}, {"a": 1, "b": 2}, {"a": 2, "b": 3}],
    "val": [10, 20, 30]
})

print("\nGrouping by Struct column (should work):")
try:
    out = df_struct.group_by("key").agg(pl.col("val").sum())
    print(out)
except Exception as e:
    print("Struct groupby error:", e)

# Group by variable-length list (should fail)
df_list = pl.DataFrame({
    "key": [[1, 2, 3], [1, 2], [2, 1]],
    "val": [10, 20, 30]
})

print("\nGrouping by List column (should fail):")
try:
    out = df_list.group_by("key").agg(pl.col("val").sum())
    print(out)
except Exception as e:
    print("List groupby error:", e)

# Group by fixed-size list (should fail)
# Manually construct a FixedSizeList column
df_fixed_size = pl.DataFrame({
    "key": [[1, 2], [1, 2], [2, 1]],
    "val": [10, 20, 30]
})
df_fixed_size = df_fixed_size.with_columns(pl.col("key").cast(pl.Array(pl.Int64, 2)))
print("\nGrouping by FixedSizeList column (should fail):")
try:
    out = df_fixed_size.group_by("key").agg(pl.col("val").sum())
    print(out)
except Exception as e:
    print("FixedSizeList groupby error:", e)
