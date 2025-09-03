import logging

from fenic_cloud.hasura_client.generated_graphql_client import Client
from fenic_cloud.hasura_client.generated_graphql_client.list_query_execution_metric_by_query_execution_id import (
    ListQueryExecutionMetricByQueryExecutionId,
)

from fenic.core.metrics import (
    LMMetrics,
    PhysicalPlanRepr,
    QueryMetrics,
    RMMetrics,
)

logger = logging.getLogger(__name__)

async def get_query_execution_metrics(client: Client, execution_id: str) -> QueryMetrics:
    """Get query execution metrics from the cloud catalog."""
    query_metrics = QueryMetrics(
        execution_id=execution_id,
        session_id=None,
        execution_time_ms=0,
        num_output_rows=0,
        total_lm_metrics=LMMetrics(),
        total_rm_metrics=RMMetrics(),
        _plan_repr=PhysicalPlanRepr(operator_id="empty"),
    )

    cloud_query_metrics: ListQueryExecutionMetricByQueryExecutionId = await client.list_query_execution_metric_by_query_execution_id(execution_id)
    if cloud_query_metrics.typedef_query_execution_by_pk.query_execution_metrics:
        for metric in cloud_query_metrics.typedef_query_execution_by_pk.query_execution_metrics:
            query_metrics.end_ts = metric.metric_timestamp
            if metric.metric_name == "execution_time_ms":
                query_metrics.execution_time_ms = metric.metric_value
            elif metric.metric_name == "num_output_rows":
                query_metrics.num_output_rows = metric.metric_value
            elif metric.metric_name == "lm_num_uncached_input_tokens":
                query_metrics.total_lm_metrics.num_uncached_input_tokens = metric.metric_value
            elif metric.metric_name == "lm_num_cached_input_tokens":
                query_metrics.total_lm_metrics.num_cached_input_tokens = metric.metric_value
            elif metric.metric_name == "lm_num_output_tokens":
                query_metrics.total_lm_metrics.num_output_tokens = metric.metric_value
            elif metric.metric_name == "lm_num_requests":
                query_metrics.total_lm_metrics.num_requests = metric.metric_value
            elif metric.metric_name == "lm_cost":
                query_metrics.total_lm_metrics.cost = metric.metric_value
            elif metric.metric_name == "rm_num_input_tokens":
                query_metrics.total_rm_metrics.num_input_tokens = metric.metric_value
            elif metric.metric_name == "rm_num_requests":
                query_metrics.total_rm_metrics.num_requests = metric.metric_value
            elif metric.metric_name == "rm_cost":
                query_metrics.total_rm_metrics.cost = metric.metric_value
            else:
                logger.error(f"unknown metric: {metric.metric_name} for execution_id: {execution_id}")
    else:
        logger.error(f"couldn't find metrics for execution_id: {execution_id}")

    return query_metrics
