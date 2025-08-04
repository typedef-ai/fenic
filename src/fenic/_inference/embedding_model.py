from typing import Optional

import pyarrow as pa

from fenic._inference.model_client import ModelClient
from fenic._inference.types import FenicEmbeddingsRequest
from fenic.core._inference.model_catalog import model_catalog
from fenic.core._logical_plan.resolved_types import (
    ResolvedModelAlias,
)
from fenic.core.metrics import RMMetrics


class EmbeddingModel:
    def __init__(self, client: ModelClient[FenicEmbeddingsRequest, list[float]]):
        self.client = client
        self.model = client.model
        self.model_provider = client.model_provider
        self.model_parameters = model_catalog.get_embedding_model_parameters(self.model_provider, self.model)

    def get_embeddings(
        self,
        docs: list[str],
        model_alias: Optional[ResolvedModelAlias] = None,
    ) -> pa.ListArray:
        model_profile = model_alias.profile if model_alias else None
        requests = []
        for doc in docs:
            if doc:
                requests.append(FenicEmbeddingsRequest(doc, model_profile))
            else:
                requests.append(None)
        results = self.client.make_batch_requests(requests, operation_name="semantic.embed")
        output_dimensions = self.model_parameters.default_dimensions
        if results:
            output_dimensions = len(results[0])
        return pa.array(results, type=pa.list_(pa.float32(), output_dimensions))


    def reset_metrics(self):
        self.client.reset_metrics()

    def get_metrics(self) -> RMMetrics:
        return self.client.get_metrics()
