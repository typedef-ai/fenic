"""The quiet join adapter must retain the batch client's failure lifetime."""

import threading

import pytest

from fenic._backends.local.semantic_operators.join import _QuietJoinBatchClient
from fenic.core.error import ExecutionError
from tests._inference.test_model_client_thread_exception import (
    FlakyCompletionClient,
    _request,
)


def test_quiet_join_adapter_registers_and_cleans_each_batch(monkeypatch):
    client = FlakyCompletionClient()
    adapter = _QuietJoinBatchClient(client)
    original = client._submit_batch_requests
    batch_keys = []

    def observe(requests, batch_id, *args, **kwargs):
        key = (threading.get_ident(), batch_id)
        batch_keys.append(key)
        assert key in client.active_batches
        assert kwargs["show_progress"] is False
        return original(requests, batch_id, *args, **kwargs)

    monkeypatch.setattr(client, "_submit_batch_requests", observe)
    try:
        with pytest.raises(ExecutionError, match="first batch failure"):
            adapter.make_batch_requests([_request("first")], "join")
        assert client.active_batches == set()
        assert client.thread_exceptions == {}
        responses = adapter.make_batch_requests([_request("second")], "join")
        assert responses[0].completion == "response-for-second"
        assert len(batch_keys) == len(set(batch_keys)) == 2
        assert client.active_batches == set()
        assert client.thread_exceptions == {}
    finally:
        client.shutdown()
