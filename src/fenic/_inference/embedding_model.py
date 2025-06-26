import pyarrow as pa

from fenic._inference.model_client import ModelClient
from fenic.core._inference.model_catalog import model_catalog
from fenic.core.metrics import RMMetrics


class EmbeddingModel:
    def __init__(self, client: ModelClient[str, list[float]]):
        self.client = client
        self.model = client.model
        self.model_provider = client.model_provider
        self.output_dimensions = model_catalog.get_embedding_model_parameters(self.model_provider, self.model).output_dimensions

    def get_embeddings(self, docs: list[str | None]) -> pa.ListArray:
        results = self.client.make_batch_requests(docs, operation_name="semantic.embed")
        return pa.array(results, type=pa.list_(pa.float32(), self.output_dimensions))

    def reset_metrics(self):
        self.client.reset_metrics()

    def get_metrics(self) -> RMMetrics:
        return self.client.get_metrics()
