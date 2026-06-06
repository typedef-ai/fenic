<!-- markdownlint-disable MD041 MD033 -->
<div align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="docs/images/typedef-fenic-logo-dark.png">
        <img src="docs/images/typedef-fenic-logo-github-yellow.png" alt="fenic, by typedef" width="90%">
    </picture>
</div>

# fenic: semantic DataFrames for humans and agents

[![PyPI version](https://img.shields.io/pypi/v/fenic.svg)](https://pypi.org/project/fenic/)
[![Python versions](https://img.shields.io/pypi/pyversions/fenic.svg)](https://pypi.org/project/fenic/)
[![License](https://img.shields.io/github/license/typedef-ai/fenic.svg)](https://github.com/typedef-ai/fenic/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1381706122322513952?label=Discord&logo=discord)](https://discord.gg/GdqF3J7huR)

**fenic turns AI-assisted exploration of structured and unstructured data into reusable, inspectable DataFrame pipelines.**

It's a DataFrame query engine for semantic data processing, with AI operators — `extract`, `classify`, `summarize`, `embed`, semantic `join`, and more — built into the query model. Use it to turn documents, transcripts, logs, eval traces, tickets, tables, and APIs into typed rows and repeatable workflows.

The point is a shift in what your data work _produces_. Humans and agents work on the same pipelines — both can author, inspect, and reuse them. The result isn't a one-off prompt or a brittle regex script that has to be reverse-engineered later — it's a durable artifact: typed, inspectable, rerunnable, and callable.

> **From exploration to artifact.**

```bash
pip install fenic
```

Optional feature extras install heavier dependencies only when you need them:

```bash
pip install "fenic[pdf]"       # semantic.parse_pdf and PDF metadata loading
pip install "fenic[cluster]"   # DataFrame.semantic.with_cluster_labels
pip install "fenic[sim-join]"  # semantic.sim_join
```

Extras can be combined with model provider extras, for example:

```bash
pip install "fenic[google,pdf,cluster,sim-join]"
```

> **Writing fenic with an AI coding agent?** Run `fenic skill install` so Claude Code / Cursor / Codex write it correctly, and `fenic check` to lint it — [details below](#writing-fenic-with-an-ai-coding-agent).

---

## What is fenic?

fenic is a **semantic DataFrame engine**. You write the PySpark/SQL-style operations you already know — `select`, `filter`, `join`, `group_by`, `agg` — alongside _semantic operators_ that call language models as a first-class part of the query. You configure models once on a `Session`, build a pipeline lazily, and fenic compiles and runs it on a query engine built for inference: automatic batching, rate limiting, retries, token/cost accounting, and response caching.

Two ideas make it different from gluing an LLM onto pandas:

- **Inference lives inside the query model.** Extraction, classification, summarization, and embeddings are operators with schemas and types — not side calls you orchestrate by hand.
- **The pipeline is the artifact.** Because the work is expressed as typed operators, it's already inspectable (row-level lineage, `explain`, per-query metrics), rerunnable (lazy plans + caching), and promotable into a named table, view, or **MCP tool** an agent can call.

---

## 60 seconds: messy text → typed rows

Replace brittle parsing and one-off prompts with a typed, schema-bound operator. Define the shape you want as a Pydantic model; fenic returns structured columns you can query.

```python
import fenic as fc
from pydantic import BaseModel, Field

class Ticket(BaseModel):
    product: str = Field(description="The product the user is asking about")
    sentiment: str = Field(description="positive, neutral, or negative")
    issue: str = Field(description="One-line summary of the user's problem")

session = fc.Session.get_or_create(
    fc.SessionConfig(
        app_name="quickstart",
        semantic=fc.SemanticConfig(
            language_models={
                "mini": fc.OpenAILanguageModel(model_name="gpt-4o-mini", rpm=500, tpm=200_000)
            },
        ),
    )
)

df = session.create_dataframe([
    {"id": 1, "text": "The CSV export in Reports keeps timing out since the last update."},
    {"id": 2, "text": "Love the new dashboard, but SSO login is broken on mobile."},
])

# Free text -> typed, queryable rows
tickets = (
    df.select("id", fc.semantic.extract("text", Ticket).alias("t"))
      .unnest("t")
)
tickets.show()
# id | product | sentiment | issue
# 1  | Reports | negative  | CSV export times out
# 2  | Auth    | negative  | SSO login broken on mobile
```

Set the API key for whichever provider you use:

```bash
export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY / GOOGLE_API_KEY / COHERE_API_KEY / OPENROUTER_API_KEY
```

---

## Why fenic?

**Unstructured data is everywhere, and working with it is brittle.** Teams reach for regex, one-off scripts, notebooks, and prompt chains to pull meaning out of documents, logs, tickets, transcripts, and traces. The results are hard to reproduce and hard to inspect.

**Agents made exploration easy and introduced a new problem.** An agent can dig through messy data and find something useful — but unless that discovery becomes code, data, or a pipeline, it dies as a chat transcript. The next person has to reverse-engineer what happened.

**fenic gives semantic data work a DataFrame abstraction.** Express the exploration as fenic operators and it's _already_ the artifact:

|                       | Without fenic                                 | With fenic                                                |
| --------------------- | --------------------------------------------- | --------------------------------------------------------- |
| **Extraction**        | regex + one-off prompts, re-derived each time | `extract(Schema)` → typed columns, validated at plan time |
| **Reproducibility**   | "what did the agent do?"                      | a lazy plan you can `explain()` and rerun                 |
| **Inspection**        | scroll the transcript                         | row-level `lineage()`, typed rows, per-query cost/tokens  |
| **Reuse**             | copy/paste the script                         | promote to a table, view, or MCP tool                     |
| **Humans vs. agents** | separate, incompatible workflows              | one shared pipeline both can read and run                 |

The model is still probabilistic — but the _pipeline_ around it is typed, bounded, cached, replayable, and inspectable.

---

## Featured workflow: from eval exploration to durable eval intelligence

Eval analysis is the perfect case: the data is messy and semi-structured (traces, tool calls, outputs, judge notes), and the useful patterns usually get discovered once and lost. With fenic, an agent (or you) explores the traces, and the useful operations become a pipeline that reruns on every new batch.

```python
import fenic as fc
from typing import Literal
from pydantic import BaseModel, Field

class FailureMode(BaseModel):
    failed: bool = Field(description="Whether the agent failed the task")
    category: Literal["tool_error", "instruction_following", "retrieval", "reasoning", "none"] = Field(
        description="Primary failure category, or 'none' if the run succeeded"
    )
    evidence: str = Field(description="Short quote or summary justifying the classification")

session = fc.Session.get_or_create(
    fc.SessionConfig(
        app_name="eval_triage",
        semantic=fc.SemanticConfig(
            language_models={
                "mini": fc.OpenAILanguageModel(model_name="gpt-4o-mini", rpm=500, tpm=200_000)
            },
        ),
    )
)

# Eval traces are almost always JSON — one file per run here (a JSONL file or a
# warehouse table work too). `content` is loaded as JSON.
traces = session.read.docs("eval_runs/**/*.json", content_type="json", recursive=True)

# Optional: project just the fields the model should see to control token cost, e.g.
#   traces = traces.with_column("convo", fc.json.jq(fc.col("content"), ".messages"))

failures = (
    traces
    .with_column("analysis", fc.semantic.extract(fc.col("content").cast(fc.StringType), FailureMode))
    .unnest("analysis")
    .filter(fc.col("failed"))
)

# One inspectable, root-caused summary per failure category
failure_modes = failures.group_by("category").agg(
    fc.count("*").alias("n"),
    fc.semantic.reduce(
        "Summarize the common root cause across these failures",
        column=fc.col("evidence"),
    ).alias("pattern"),
)

failure_modes.write.save_as_table("failure_modes", mode="overwrite")
```

The exploration is now **cumulative**: rerun it on the next model version to detect regressions, inspect any row back to its source trace with `failures.lineage()`, and — see below — hand the result to the agent as a tool. Each new analysis becomes another reusable transform in a growing library for understanding model behavior.

---

## Query meaning and metadata together

The interesting questions usually need both structured and unstructured data — customer metadata _and_ support tickets, eval scores _and_ trajectories, CRM fields _and_ call transcripts. fenic does relational and semantic work in the same pipeline, including joins on _meaning_ rather than exact keys.

```python
# Match on meaning, not exact values
matches = candidates.semantic.join(
    roles,
    predicate=(
        "Candidate background: {{ left_on }}\n"
        "Role requirements: {{ right_on }}\n"
        "The candidate is a strong fit for the role."
    ),
    left_on=fc.col("resume"),
    right_on=fc.col("job_description"),
)

# ...then group, aggregate, and rank with ordinary DataFrame ops
ranked = (
    matches.group_by("role_id")
    .agg(fc.count("*").alias("n_candidates"))
    .order_by(fc.desc("n_candidates"))
)
```

---

## Make it an artifact your agents reuse

A fenic table or view becomes a governed tool surface over **MCP** (Model Context Protocol) — so the same pipeline a human inspects is what an agent calls, with typed parameters and bounded result sizes.

Auto-generate a suite of system tools (schema, profile, read, search, analyze) over your tables:

```python
from fenic import SystemToolConfig

session.catalog.set_table_description(
    "failure_modes", "Recurring agent failure modes with counts and root-cause summaries"
)

server = fc.create_mcp_server(
    session,
    "Eval Intelligence",
    system_tools=SystemToolConfig(
        table_names=["failure_modes"],
        tool_namespace="evals",
        max_result_rows=100,
    ),
)

fc.run_mcp_server_sync(server, transport="http", port=8000)
```

Or define a precise, parameterized tool — like a typed SQL macro:

```python
recent = session.table("failure_modes").filter(
    fc.col("category") == fc.tool_param("category", fc.StringType)
)

session.catalog.create_tool(
    "failures_by_category",
    "Look up recurring failures for a given category",
    recent,
    tool_params=[fc.ToolParam(name="category", description="Failure category to filter by")],
)
```

Serve every registered tool straight from the catalog with the CLI:

```bash
fenic-serve --app-name eval_triage --port 8000
```

---

## Writing fenic with an AI coding agent

fenic is new, so a coding agent won't always know its exact API out of the box. Two tools make any agent reliable at writing fenic:

**Lint it.** `fenic check` statically resolves a pipeline's `fc.*` symbols against the installed fenic and flags namespace/import mistakes (`fenic.functions`, `fc.array` vs `fc.arr`, `fc.explode`, …) — it does **not** execute your script. Have your agent run it after writing fenic and fix what it reports:

```bash
fenic check pipeline.py
# → {"ok": true, "findings": []}   or   a namespace/symbol finding with a fix suggestion
```

**Teach your agent the API.** `fenic skill install` copies the `fenic-mechanics` skill — fenic's namespace rules, semantic-operator calling conventions, and the silent gotchas — into the skill directories your agents read. It detects which agents you have and asks where to install:

```bash
fenic skill install
# Detected agents: claude, codex, cursor, gemini
# → global (~/.claude/skills, ~/.agents/skills) or just this project
```

Works with Claude Code, OpenAI Codex, Cursor, Gemini CLI, and Copilot. The skill stays dormant until you're actually writing fenic, then keeps the agent on the real API.

---

## Semantic operators

Column operators (via `fc.semantic.*`), used inside `select` / `with_column` / `filter` / `agg`:

| Operator                    | What it does                                                                 |
| --------------------------- | ---------------------------------------------------------------------------- |
| `extract(col, Schema)`      | Unstructured text → a typed struct from a Pydantic schema                    |
| `classify(col, classes)`    | Label text into predefined classes (optionally with descriptions + examples) |
| `predicate(prompt, **cols)` | Natural-language boolean — use it to `filter` rows                           |
| `map(prompt, **cols)`       | Apply a templated generation prompt per row (optionally typed output)        |
| `reduce(prompt, column)`    | Aggregate many rows in a group into one result (great after `group_by`)      |
| `analyze_sentiment(col)`    | positive / negative / neutral                                                |
| `summarize(col)`            | Paragraph or key-points summary                                              |
| `embed(col)`                | Embeddings for similarity, clustering, and search                            |
| `parse_pdf(col)`            | PDF paths → Markdown                                                         |

DataFrame operators (via `df.semantic.*`):

| Operator                                    | What it does                                        |
| ------------------------------------------- | --------------------------------------------------- |
| `join(other, predicate, left_on, right_on)` | Join two DataFrames on a natural-language predicate |
| `sim_join(other, left_on, right_on, k)`     | Top-k embedding-similarity join                     |
| `with_cluster_labels(by, num_clusters)`     | K-means clustering over an embedding column         |

All of it composes with ordinary DataFrame operations — `select`, `filter`, `with_columns`, `join`, `group_by`/`agg`, `order_by`, `limit`, `unnest`, `explode` — and full **SQL** via `session.sql("... {df} ...", df=df)`. Few-shot examples, per-model aliases/profiles, and structured output are built in.

---

## Native support for unstructured data

First-class logical types mean text-heavy data is typed, not just strings:

- **Markdown** — parse, generate a table of contents, extract header-based chunks, convert to JSON
- **Transcript** — SRT / WebVTT / generic, with speaker and timestamp awareness
- **JSON** — query and reshape nested data with `jq` expressions
- **HTML**, **Document paths (PDF)**, and **Embeddings** (fixed-dimension vectors)
- Rust-accelerated text processing: recursive chunking with overlap, regex, fuzzy matching

Read from local files, **S3**, and **Hugging Face** datasets:

```python
session.read.csv("data/*.csv")
session.read.parquet("s3://bucket/data/*.parquet")
session.read.docs("docs/**/*.md", content_type="markdown", recursive=True)
session.read.pdf_metadata("data/**/*.pdf", recursive=True)
```

---

## Inspect and operate

Because everything is a typed plan, you can see exactly what happened:

- `df.explain()` — the logical/physical plan for a pipeline
- `df.lineage()` — trace specific rows **forwards and backwards** through every operation
- Per-query metrics — tokens and cost for each run
- Caching — an LLM response cache plus `.cache()` to materialize expensive intermediates

```python
result = df.lineage()
result.backwards(["<row-id>"])   # which source rows produced this output?
```

Install optional feature extras when you use heavier operators:

| Extra      | Enables                                       |
| ---------- | --------------------------------------------- |
| `pdf`      | `semantic.parse_pdf` and PDF metadata loading |
| `cluster`  | `DataFrame.semantic.with_cluster_labels`      |
| `sim-join` | `DataFrame.semantic.sim_join`                 |

```bash
pip install "fenic[pdf,cluster,sim-join]"
# or
uv add "fenic[pdf,cluster,sim-join]"
```

---

## Providers

| Provider   | Type             | Notes                                                  |
| ---------- | ---------------- | ------------------------------------------------------ |
| OpenAI     | LLM + embeddings | GPT, o-series, GPT-5 family; `text-embedding-3-*`      |
| Anthropic  | LLM              | Claude (Haiku / Sonnet / Opus), with thinking budgets  |
| Google     | LLM + embeddings | Gemini (AI Studio _and_ Vertex)                        |
| Cohere     | Embeddings       | `embed-v4.0`                                           |
| OpenRouter | LLM (aggregator) | provider routing, fallbacks, price/throughput controls |

Reasoning/thinking effort is configurable per model via profiles, and you can register multiple models and pick per operator with `model_alias`.

---

## Examples

End-to-end agent projects live in [**fenic-examples**](https://github.com/typedef-ai/fenic-examples):

- A deep research agent for Hacker News
- Log triage with LangGraph
- AI feature engineering for recommender systems

<details>
<summary><b>In-repo notebooks (with Colab links)</b></summary>

| Example                                                                 | Description                                                       |
| ----------------------------------------------------------------------- | ----------------------------------------------------------------- |
| [Hello World!](examples/hello_world)                                    | Semantic extraction and classification through error-log analysis |
| [Enrichment](examples/enrichment)                                       | Multi-stage pipelines: template extraction, joins, LLM transforms |
| [Meeting Transcript Processing](examples/meeting_transcript_processing) | Native transcript parsing + Pydantic schemas + aggregations       |
| [News Analysis](examples/news_analysis)                                 | Bias detection with classify/extract/reduce                       |
| [Podcast Summarization](examples/podcast_summarization)                 | Speaker-aware, multi-level summarization                          |
| [Semantic Join](examples/semantic_joins)                                | Match records across tables by meaning                            |
| [Named Entity Recognition](examples/named_entity_recognition)           | Multi-stage entity extraction + classification                    |
| [Markdown Processing](examples/markdown_processing)                     | Structure extraction from Markdown                                |
| [JSON Processing](examples/json_processing)                             | Nested JSON with `jq`                                             |
| [Feedback Clustering](examples/feedback_clustering)                     | Embeddings + clustering + summarization                           |
| [Document Extraction](examples/document_extraction)                     | Structured metadata from diverse documents                        |

</details>

---

## Community

Join us on [Discord](https://discord.gg/GdqF3J7huR) to ask questions and share what you're building. If fenic is useful to you, a ⭐ on the repo helps others find it.

## Contributing

Contributions of all kinds are welcome — code, docs, tests, and ideas. For code changes, please open an issue to discuss your approach before sending a PR. See our [contribution guidelines](CONTRIBUTING.md) for development setup and workflow.
