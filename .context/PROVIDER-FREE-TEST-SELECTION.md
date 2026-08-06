# Provider-free test selection rule

This experiment has had three placeholder-key 401 incidents (P1c, TD-3384, and
TD-3385). A synthetic `OPENAI_API_KEY` may satisfy client construction in a
fixture; it is never evidence that a test is provider-free.

For every local-only node, use an **allowlist of exact pytest node IDs** whose
bodies were inspected and are known to use fake injection, custom local vectors,
or direct Polars-only logic. Do not use a whole-file path or `-k` selector for a
file that mixes local and provider tests. Before running a new node ID, inspect
its body and the fixtures it invokes for `semantic.embed`, `semantic.map`,
`semantic.extract`, `semantic.predicate`, `semantic.join`, provider validation,
or a default configured model client. If any appears, exclude it unless the
test replaces that boundary with an explicit fake.

Always exclude these known provider-capable targets from local-only runs:

- `tests/_backends/local/test_metrics.py::test_semantic_metrics`;
- `tests/_backends/local/functions/test_embed.py::test_embeddings`,
  `test_embedding_very_long_string`, and `test_embedding_without_models`;
- `tests/_backends/local/io/test_reader.py::test_read_embeddings_table`;
- `tests/_backends/local/dataframe/test_semantic_sim_join.py` test bodies that
  call `semantic.embed` (never select that mixed file wholesale); and
- `tests/_backends/local/dataframe/test_semantic_join.py` test bodies that use
  the default language-model path or provider validation (never select that
  mixed file wholesale).

The safe pattern is explicit nodes proved local for the task, such as TD-3385's
normalization/similarity/embedding-average node list or P1a's custom-vector
sim-join nodes. If an otherwise useful test needs provider behavior, it belongs
to an explicitly authorized, spend-recorded provider probe—not a local gate.

Record any newly discovered provider-capable node in this file before the next
local-only test command. On a 401 or any unexpected network attempt: stop the
command, preserve the fact in the lane record, charge no inferred cost, and do
not retry with another selector.
