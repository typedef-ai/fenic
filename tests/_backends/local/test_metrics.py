import polars as pl
import pytest

from fenic import avg, col, count, semantic, sum


@pytest.fixture
def sales_data():
    return {
        "sale_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "product_id": [101, 102, 103, 101, 102, 103, 104, 105],
        "customer_id": [1001, 1002, 1003, 1004, 1001, 1002, 1005, 1003],
        "quantity": [2, 1, 3, 1, 2, 2, 5, 1],
        "amount": [200.50, 150.75, 300.25, 200.50, 150.75, 300.25, 500.00, 75.25],
        "sale_date": [
            "2023-01-15",
            "2023-01-16",
            "2023-01-17",
            "2023-01-18",
            "2023-01-19",
            "2023-01-20",
            "2023-01-21",
            "2023-01-22",
        ],
    }

@pytest.fixture
def product_data():
    return {
        "product_id": [101, 102, 103, 104, 105, 106],
        "product_name": ["Laptop", "Phone", "Tablet", "Monitor", "Keyboard", "Mouse"],
        "category": [
            "Electronics",
            "Electronics",
            "Electronics",
            "Computer",
            "Computer",
            "Computer",
        ],
        "price": [1000.00, 800.00, 500.00, 300.00, 50.00, 25.00],
    }

@pytest.fixture
def customer_data():
    return {
        "customer_id": [1001, 1002, 1003, 1004, 1005],
        "customer_name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "city": ["New York", "San Francisco", "Chicago", "Boston", "Seattle"],
        "segment": ["Premium", "Standard", "Premium", "Standard", "Premium"],
    }

@pytest.fixture(params=["simple_dataframe", "semantic_dataframe"])
def dataframe(request, local_session, sales_data, product_data, customer_data):
    if request.param == "simple_dataframe":
        sales_df = local_session.create_dataframe(pl.DataFrame(sales_data))
        product_df = local_session.create_dataframe(pl.DataFrame(product_data))
        customer_df = local_session.create_dataframe(pl.DataFrame(customer_data))

        # First query - premium electronics sales
        premium_electronics = (
            sales_df.join(product_df, "product_id")
            .join(customer_df, "customer_id")
            .filter((col("segment") == "Premium") & (col("category") == "Electronics"))
            .select("product_name", "customer_name", "amount", "quantity")
            .group_by("product_name")
            .agg(
                sum("amount").alias("total_sales"),
                avg("amount").alias("avg_sale"),
                count("customer_name").alias("num_transactions"),
            )
        ).cache()
        # Union the queries and limit results
        final_result = premium_electronics.union(premium_electronics).limit(5)
        return final_result
    elif request.param == "semantic_dataframe":
        df = local_session.create_dataframe(pl.DataFrame({"name": ["Alice", "Bob"]}))
        df = df.select(
            semantic.map("What is a longer name for {{name}}?", name=col("name")).alias("longer_name"),
            semantic.embed(col("name")).alias("embedding"),
        )
        df = df.filter(
            semantic.predicate(
                "This name: '{{longer_name}}' is used as a placeholder in discussions about cryptographic systems.",
                longer_name=col("longer_name"),
            )
        )
        return df


@pytest.mark.parametrize("dataframe", ["simple_dataframe"], indirect=True)
def test_simple_metrics(dataframe):
    # Execute and collect metrics
    result = dataframe.collect("polars")
    metrics = result.metrics

    # Verify basic metrics
    assert metrics.execution_time_ms > 0
    assert metrics.num_output_rows == 5
    assert metrics.total_lm_metrics is not None
    assert metrics.total_lm_metrics.num_uncached_input_tokens == 0
    assert metrics.total_lm_metrics.num_output_tokens == 0
    assert metrics.total_lm_metrics.cost == 0

    # Verify operator metrics structure
    assert len(metrics._operator_metrics) == 11

    # Find operators by type
    limit_ops = [
        op for op_id, op in metrics._operator_metrics.items() if "LimitExec" in op_id
    ]
    union_ops = [
        op for op_id, op in metrics._operator_metrics.items() if "UnionExec" in op_id
    ]
    agg_ops = [
        op
        for op_id, op in metrics._operator_metrics.items()
        if "AggregateExec" in op_id
    ]
    filter_ops = [
        op for op_id, op in metrics._operator_metrics.items() if "FilterExec" in op_id
    ]
    join_ops = [
        op for op_id, op in metrics._operator_metrics.items() if "JoinExec" in op_id
    ]
    source_ops = [
        op for op_id, op in metrics._operator_metrics.items() if "SourceExec" in op_id
    ]

    # Verify we have the expected operator types
    assert len(limit_ops) == 1, "Should have exactly one limit operator"
    assert len(union_ops) == 1, "Should have exactly one union operator"
    assert len(agg_ops) == 2, "Should have exactly two aggregate operators"
    assert len(filter_ops) == 1, "Should have exactly one filter operator"
    assert len(join_ops) == 2, "Should have exactly two join operators"
    assert len(source_ops) == 3, "Should have exactly three source operators"

    # Verify operator metrics content
    limit_op = limit_ops[0]
    assert limit_op.num_output_rows == metrics.num_output_rows
    assert limit_op.execution_time_ms > 0

@pytest.mark.parametrize("dataframe", ["semantic_dataframe"], indirect=True)
def test_semantic_metrics(dataframe):
    """Test that the semantic API works at all without using the fixture, given that the fixture sets lm."""
    result = dataframe.collect("polars")
    metrics = result.metrics
    assert metrics.total_lm_metrics.num_uncached_input_tokens > 0
    assert metrics.total_lm_metrics.num_output_tokens > 0
    assert metrics.total_lm_metrics.cost > 0
    assert metrics.total_rm_metrics.num_input_tokens > 0
    assert metrics.total_rm_metrics.cost > 0
    for operator_metrics in metrics._operator_metrics.values():
        if "SourceExec" in operator_metrics.operator_id:
            continue
        if "ProjectionExec" in operator_metrics.operator_id:
            assert operator_metrics.lm_metrics.num_uncached_input_tokens > 0
            assert operator_metrics.lm_metrics.num_output_tokens > 0
            assert operator_metrics.lm_metrics.cost > 0
            assert operator_metrics.rm_metrics.num_input_tokens > 0
            assert operator_metrics.rm_metrics.cost > 0
        if "FilterExec" in operator_metrics.operator_id:
            assert operator_metrics.lm_metrics.num_uncached_input_tokens > 0
            assert operator_metrics.lm_metrics.num_output_tokens > 0
            assert operator_metrics.lm_metrics.cost > 0
            assert operator_metrics.rm_metrics.num_input_tokens == 0
            assert operator_metrics.rm_metrics.cost == 0


@pytest.mark.parametrize("dataframe", ["simple_dataframe", "semantic_dataframe"], indirect=True)
def test_metrics_trace_callback(dataframe):
    # Choose dataframe to test
    test_plan = dataframe._logical_plan

    # Define our callback
    cur_callback_count = 0
    final_callback_metrics = None

    # Make sure the callback is called each step of the way
    def trace_callback(query_metrics):
        nonlocal cur_callback_count, final_callback_metrics
        cur_callback_count += 1
        final_callback_metrics = query_metrics

    # Execute and collect metrics
    _, metrics = dataframe._session_state.execution.collect(test_plan, trace_callback=trace_callback)
    assert cur_callback_count == len(metrics._operator_metrics)

    # Verify the callback received the correct metrics
    assert metrics == final_callback_metrics

    # Verify callback metrics for each type of execution
    cur_callback_count = 0
    _, metrics = dataframe._session_state.execution.show(test_plan, trace_callback=trace_callback)
    assert cur_callback_count == len(metrics._operator_metrics)
    assert metrics == final_callback_metrics

    cur_callback_count = 0
    _, metrics = dataframe._session_state.execution.count(test_plan, trace_callback=trace_callback)
    assert cur_callback_count == len(metrics._operator_metrics)
    assert metrics == final_callback_metrics

    cur_callback_count = 0
    metrics = dataframe._session_state.execution.save_as_table(test_plan, "test_metrics_table", mode="overwrite", trace_callback=trace_callback)
    assert cur_callback_count == len(metrics._operator_metrics)
    assert metrics == final_callback_metrics

    cur_callback_count = 0
    metrics = dataframe._session_state.execution.save_to_file(test_plan, "test_metrics_file", mode="overwrite", trace_callback=trace_callback)
    assert cur_callback_count == len(metrics._operator_metrics)
    assert metrics == final_callback_metrics


def test_metrics_from_view(local_session, sales_data, product_data, customer_data):
    sales_df = local_session.create_dataframe(pl.DataFrame(sales_data))
    product_df = local_session.create_dataframe(pl.DataFrame(product_data))
    customer_df = local_session.create_dataframe(pl.DataFrame(customer_data))

    # First query - premium electronics sales
    premium_electronics = (
        sales_df.join(product_df, "product_id")
        .join(customer_df, "customer_id")
        .filter((col("segment") == "Premium") & (col("category") == "Electronics"))
        .select("product_name", "customer_name", "amount", "quantity")
        .group_by("product_name")
        .agg(
            sum("amount").alias("total_sales"),
            avg("amount").alias("avg_sale"),
            count("customer_name").alias("num_transactions"),
        )
    )

    final_result = premium_electronics.union(premium_electronics).limit(5)
    final_result.write.save_as_view("premium_electronics")

    # retrieve view
    df = local_session.view("premium_electronics")
    result = df.collect()
    metrics = result.metrics

    # Verify basic metrics
    assert metrics.execution_time_ms > 0
    assert metrics.num_output_rows == 5
    assert metrics.total_lm_metrics is not None
    assert metrics.total_lm_metrics.num_uncached_input_tokens == 0
    assert metrics.total_lm_metrics.num_output_tokens == 0
    assert metrics.total_lm_metrics.cost == 0

    # Verify operator metrics structure
    assert len(metrics._operator_metrics) == 18
    # Find operators by type
    limit_ops = [
        op for op_id, op in metrics._operator_metrics.items() if "LimitExec" in op_id
    ]
    union_ops = [
        op for op_id, op in metrics._operator_metrics.items() if "UnionExec" in op_id
    ]
    agg_ops = [
        op
        for op_id, op in metrics._operator_metrics.items()
        if "AggregateExec" in op_id
    ]
    filter_ops = [
        op for op_id, op in metrics._operator_metrics.items() if "FilterExec" in op_id
    ]
    join_ops = [
        op for op_id, op in metrics._operator_metrics.items() if "JoinExec" in op_id
    ]
    source_ops = [
        op for op_id, op in metrics._operator_metrics.items() if "SourceExec" in op_id
    ]

    # Verify we have the expected operator types
    assert len(limit_ops) == 1, "Should have exactly one limit operator"
    assert len(union_ops) == 1, "Should have exactly one union operator"
    assert len(agg_ops) == 2, "Should have exactly two aggregate operators"
    assert len(filter_ops) == 2, "Should have exactly two filter operator"
    assert len(join_ops) == 4, "Should have exactly four join operators"
    assert len(source_ops) == 6, "Should have exactly six source operators"

    # Verify operator metrics content
    limit_op = limit_ops[0]
    assert limit_op.num_output_rows == metrics.num_output_rows
    assert limit_op.execution_time_ms > 0

def test_metrics_from_view_with_cache(local_session, sales_data, product_data, customer_data):
    sales_df = local_session.create_dataframe(pl.DataFrame(sales_data))
    product_df = local_session.create_dataframe(pl.DataFrame(product_data))
    customer_df = local_session.create_dataframe(pl.DataFrame(customer_data))

    # First query - premium electronics sales
    premium_electronics = (
        sales_df.join(product_df, "product_id")
        .join(customer_df, "customer_id")
        .filter((col("segment") == "Premium") & (col("category") == "Electronics"))
        .select("product_name", "customer_name", "amount", "quantity")
        .group_by("product_name")
        .agg(
            sum("amount").alias("total_sales"),
            avg("amount").alias("avg_sale"),
            count("customer_name").alias("num_transactions"),
        )
    ).cache()

    final_result = premium_electronics.union(premium_electronics).limit(5)
    final_result.write.save_as_view("premium_electronics")

    # retrieve view
    df = local_session.view("premium_electronics")
    result = df.collect()
    metrics = result.metrics

    # Verify basic metrics
    assert metrics.execution_time_ms > 0
    assert metrics.num_output_rows == 5
    assert metrics.total_lm_metrics is not None
    assert metrics.total_lm_metrics.num_uncached_input_tokens == 0
    assert metrics.total_lm_metrics.num_output_tokens == 0
    assert metrics.total_lm_metrics.cost == 0

    # Verify operator metrics structure
    assert len(metrics._operator_metrics) == 11

    # Find operators by type
    limit_ops = [
        op for op_id, op in metrics._operator_metrics.items() if "LimitExec" in op_id
    ]
    union_ops = [
        op for op_id, op in metrics._operator_metrics.items() if "UnionExec" in op_id
    ]
    agg_ops = [
        op
        for op_id, op in metrics._operator_metrics.items()
        if "AggregateExec" in op_id
    ]
    filter_ops = [
        op for op_id, op in metrics._operator_metrics.items() if "FilterExec" in op_id
    ]
    join_ops = [
        op for op_id, op in metrics._operator_metrics.items() if "JoinExec" in op_id
    ]
    source_ops = [
        op for op_id, op in metrics._operator_metrics.items() if "SourceExec" in op_id
    ]

    # Verify we have the expected operator types
    assert len(limit_ops) == 1, "Should have exactly one limit operator"
    assert len(union_ops) == 1, "Should have exactly one union operator"
    assert len(agg_ops) == 2, "Should have exactly two aggregate operators"
    assert len(filter_ops) == 1, "Should have exactly one filter operator"
    assert len(join_ops) == 2, "Should have exactly two join operators"
    assert len(source_ops) == 3, "Should have exactly three source operators"

    # Verify operator metrics content
    limit_op = limit_ops[0]
    assert limit_op.num_output_rows == metrics.num_output_rows
    assert limit_op.execution_time_ms > 0
