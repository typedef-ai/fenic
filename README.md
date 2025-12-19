<!-- markdownlint-disable MD041 MD033 -->
<div align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="docs/images/typedef-fenic-logo-dark.png">
        <img src="docs/images/typedef-fenic-logo-github-yellow.png" alt="fenic, by typedef" width="90%">
    </picture>
</div>

# fenic: Declarative context engineering for agents (works with any agent framework)

[![PyPI version](https://img.shields.io/pypi/v/fenic.svg)](https://pypi.org/project/fenic/)
[![Python versions](https://img.shields.io/pypi/pyversions/fenic.svg)](https://pypi.org/project/fenic/)
[![License](https://img.shields.io/github/license/typedef-ai/fenic.svg)](https://github.com/typedef-ai/fenic/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1381706122322513952?label=Discord&logo=discord)](https://discord.gg/GdqF3J7huR)

**Apply context ops in Python and SQL (extract, chunk, retrieve, store, compact, summarize) to produce typed, tool-bounded outputs your agents can use.**

Keep your runtime. Add fenic. Get sophisticated context construction with inference offloading with no framework lock-in and no rewrites.

fenic is a **context construction layer** that works with any agent framework. You declare what your agent should see, build it with deterministic + semantic transforms, and expose it as **typed**, **bounded tools**, while **offloading inference** so context work happens outside your agent's prompt/context window. This reduces context bloat so agents stay focused on reasoning.

<p align="center">
  <b>Quick links:</b>
  <a href="#what-you-can-build">Use Cases</a> •
  <a href="#quick-introduction">Intro</a> •
  <a href="#quickstart-pick-your-path">Quickstart</a> •
  <a href="#core-concepts">Concepts</a> •
  <a href="#key-capabilities">Capabilities</a>
</p>

## Quick Install

```bash
pip install fenic
```

---

## What You Can Build

### Memory & Personalization

- **Curated memory packs** — extract/dedupe/redact facts; serve read-only for recall
- **Blocks & episodes** — persistent profile + recent event timeline; scoped snapshots
- **Decaying resolution memory** — window functions for temporal compression (daily → weekly → monthly)
- **Cross-agent shared memory** — typed tables accessible by multiple agents in your framework

### Retrieval & Knowledge

- **Policy / KB Q&A** — parse PDFs → `extract(Schema)` → `embed` → neighbors with citations
- **Chunked retrieval** — chunk/overlap you control, hybrid filter, optional re-rank

### Context Operations (Inference Offloaded)

- **Summarization** — deterministic or LLM-powered, reducing context bloat so agents stay focused on reasoning
- **Invariant management** — store facts that should persist; re-inject at decision points
- **Token-budget-aware truncation** — shape tool responses to fit budgets
- ...and more - fenic's API allows you to define any context operation you might need

### Structured Context from Data

- **Entity matching** — resolve duplicates / link records
- **Theme extraction** — cluster + label patterns
- **Semantic linking** — connect records across systems by meaning
- …and more — fenic's declarative API supports any data transformation your agents need

---

## Quick Introduction

Context engineering is the practice of managing everything that goes into an LLM's context window, retrieval, memory, conversation history, tool responses, prompts. It's all tokens in, tokens out. And it's both an **information problem** (what information, in what structure) and an **optimization problem** (how much, when to compress, what to forget).

fenic's declarative approach fits naturally here. Instead of writing imperative code for each context operation, you describe _what_ your context should look like—and iterate quickly as you learn what works. Combine deterministic transforms (filters, aggregations, windows) with semantic ones (extract, embed, summarize) in a single composable flow.

Critically, fenic **offloads inference**: summarization, extraction, and embedding happen outside your agent's context window. Your runtime gets the results with less context bloat, so agents stay focused on reasoning.

> **Note:** LLM calls still cost tokens/$, but fenic keeps that work out of your agent's prompt/context window.

### The fenic Approach

| Without fenic                                             | With fenic                                               |
| --------------------------------------------------------- | -------------------------------------------------------- |
| Agent summarizes conversation → tokens consumed           | fenic summarizes → agent gets result; less context bloat |
| Agent extracts facts → tokens consumed                    | fenic extracts → agent gets structured data              |
| Agent searches, filters, aggregates → multiple tool calls | fenic pre-computes → agent gets precise rows             |
| Context ops compete with reasoning                        | Less context bloat → agents stay focused on reasoning    |

### Example: PDF → Typed Q&A → Bounded Tools

Build the exact context your agent is allowed to use with typed Q/A facts distilled from policy PDFs, then expose it as a narrow tool surface.

```python
import fenic as fc
from fenic import SemanticConfig, OpenAILanguageModel, OpenAIEmbeddingModel
from fenic.api.mcp._tool_generation_utils import auto_generate_system_tools_from_tables
from fenic.core.mcp._server import FenicMCPServer
from pydantic import BaseModel, Field

# 1) Typed schema for extraction
class FAQSchema(BaseModel):
    question: str = Field(description="A question extracted from the document")
    answer: str = Field(description="The answer to the question")

session = fc.Session.get_or_create(fc.SessionConfig(
    app_name="faq_app",
    semantic=SemanticConfig(
        language_models={
            "gpt": OpenAILanguageModel(model_name="gpt-5-nano", rpm=100, tpm=100_000)
        },
        embedding_models={
            "embed": OpenAIEmbeddingModel(model_name="text-embedding-3-small", rpm=100, tpm=100_000)
        },
        default_embedding_model="embed"
    )
))

# 2) Hydrate → shape with deterministic + semantic transforms
rows = [
    {"id": 1, "pdf_path": "data/policies/eu_refund_policy.pdf"},
    {"id": 2, "pdf_path": "data/policies/password_reset_guide.pdf"},
]
df = session.create_dataframe(rows)

ctx = (
    df.select(
        fc.col("id"),
        fc.col("pdf_path"),
        fc.semantic.parse_pdf(fc.col("pdf_path")).alias("text")  # parse to markdown
    )
    .filter(fc.col("text").cast(fc.StringType).contains("policy"))  # deterministic filter
    .select(
        fc.col("id"),
        fc.semantic.extract(fc.col("text").cast(fc.StringType), FAQSchema).alias("facts"),  # typed extraction
        fc.semantic.embed(fc.col("text").cast(fc.StringType)).alias("vec")                  # embedding
    )
    .unnest("facts")  # -> id, question, answer, vec
)

# 3) Serve as bounded tools (MCP or direct functions)
ctx.write.save_as_table("faq_context", mode="overwrite")

generated_tools = auto_generate_system_tools_from_tables(
    ["faq_context"], session, tool_namespace="faq", max_result_limit=100
)

server = FenicMCPServer(
    session._session_state,
    user_defined_tools=[],
    system_tools=generated_tools,
    server_name="FAQ Server",
)
```

**The result:** Agents fetch small, precise rows—not entire PDFs. Runs are faster, cheaper, reproducible, and more accurate.

**What happened here:**

- PDF parsing, extraction, embedding → **inference offloaded to fenic**
- Agent context → **only receives small, shaped results**
- Context construction → **less context bloat** (agents stay focused on reasoning)
- Framework dependency → **none—works with any runtime**

**Works with any agentic framework** (LangGraph, PydanticAI, CrewAI, ...) — fenic exposes MCP tools or Python functions.

---

## Quickstart (Pick Your Path)

Four core context construction patterns, each works with any agent framework:

| Pattern                           | What It Shows                        |
| --------------------------------- | ------------------------------------ |
| **Memory — facts**                | Extract → embed → semantic recall    |
| **Memory — blocks & episodes**    | Persistent profile + recent timeline |
| **Retrieval — schema-first rows** | Typed extraction with citations      |
| **Retrieval — semantic spans**    | Chunked docs with neighbors          |

Each builds a typed table and exposes tools via MCP or direct Python functions.

---

### 1) Memory — Facts _(extract → embed → recall)_

Add structure to free-form chats by extracting them into typed facts and recall them semantically so agents don't carry giant histories.

**Tools exposed:** `mem_recall` (semantic query), plus system tools over `preferences`

<details>
<summary><b>Show code</b></summary>

```python
import fenic as fc
from fenic import SemanticConfig, OpenAILanguageModel, OpenAIEmbeddingModel
from fenic.api.mcp._tool_generation_utils import auto_generate_system_tools_from_tables
from fenic.core.mcp._server import FenicMCPServer
from fenic.core.mcp.types import SystemTool
from pydantic import BaseModel, Field

class Preference(BaseModel):
    category: str = Field(description="The category of the preference")
    value: str = Field(description="The value of the preference")

session = fc.Session.get_or_create(fc.SessionConfig(
    app_name="mem_facts",
    semantic=SemanticConfig(
        language_models={"gpt": OpenAILanguageModel(model_name="gpt-4.1-nano", rpm=100, tpm=100_000)},
        embedding_models={"embed": OpenAIEmbeddingModel(model_name="text-embedding-3-small", rpm=100, tpm=100_000)},
        default_embedding_model="embed"
    )
))

msgs = session.create_dataframe([
    {"user_id": "user123", "message": "I'm vegetarian and allergic to nuts."},
    {"user_id": "user123", "message": "I prefer morning meetings."},
])

prefs = (
    msgs.select(
        fc.col("user_id"),
        fc.semantic.extract(fc.col("message"), Preference).alias("pref"),
        fc.semantic.embed(fc.col("message")).alias("vec"),
    )
    .unnest("pref")
)
prefs.write.save_as_table("preferences", mode="overwrite")

async def mem_recall(user_id: str, query: str, k: int = 3):
    user_prefs = session.table("preferences").filter(fc.col("user_id") == fc.lit(user_id))
    q = session.create_dataframe([{"q": query}])
    res = q.semantic.sim_join(
        user_prefs,
        left_on=fc.semantic.embed(fc.col("q")),
        right_on=fc.col("vec"),
        k=k,
        similarity_score_column="relevance",
    ).select("category", "value", "relevance")
    return res._plan

generated_system_tools = auto_generate_system_tools_from_tables(
    ["preferences"], session, tool_namespace="mem", max_result_limit=100
)

server = FenicMCPServer(
    session._session_state,
    user_defined_tools=[],
    system_tools=[
        SystemTool(
            name="mem_recall",
            description="Semantic memory recall",
            max_result_limit=100,
            func=mem_recall,
        ),
        *generated_system_tools,
    ],
    server_name="Memory (Facts)",
)
```

</details>

---

### 2) Memory — Blocks & Episodes _(profile + timeline)_

Maintain a profile block alongside a recent event timeline; return scoped snapshots.

**Tools exposed:** `get_user_context` (profile + last N events), plus system tools

<details>
<summary><b>Show code</b></summary>

```python
from datetime import datetime
from typing import Optional

import fenic as fc
from fenic import SemanticConfig, OpenAILanguageModel
from fenic.api.mcp._tool_generation_utils import auto_generate_system_tools_from_tables
from fenic.core.mcp._server import FenicMCPServer
from fenic.core.mcp.types import SystemTool
from pydantic import BaseModel, Field

class MemoryBlock(BaseModel):
    block_name: str = Field(description="Name of the memory block")
    content: str = Field(description="Content stored in the memory block")
    last_updated: str = Field(description="Timestamp of last update")

class AccountEvent(BaseModel):
    event_type: str = Field(description="Type of account event")
    amount: Optional[float] = Field(default=None, description="Amount involved in the event")
    status: Optional[str] = Field(default=None, description="Status of the event")
    description: Optional[str] = Field(default=None, description="Description of the event")

session = fc.Session.get_or_create(fc.SessionConfig(
    app_name="mem_blocks",
    semantic=SemanticConfig(
        language_models={"gpt": OpenAILanguageModel(model_name="gpt-4.1-nano", rpm=100, tpm=100_000)}
    )
))

blocks = session.create_dataframe([
    {"user_id": "user123", "block_name": "profile", "content": "Name: Taylor; Dept: Finance",
     "last_updated": datetime.now().isoformat()}
])
blocks.write.save_as_table("memory_blocks", mode="overwrite")

ev = session.create_dataframe([
    {"user_id": "user123", "event": "Failed transaction of $99.99", "timestamp": "2025-01-01"},
    {"user_id": "user123", "event": "Card expired",                   "timestamp": "2025-01-05"},
    {"user_id": "user123", "event": "Account suspended",              "timestamp": "2025-01-06"},
])
timeline = (
    ev.select(
        fc.col("user_id"),
        fc.col("timestamp"),
        fc.semantic.extract(fc.col("event"), AccountEvent).alias("data"),
    )
    .unnest("data")
)
timeline.write.save_as_table("account_timeline", mode="overwrite")

async def get_user_context(user_id: str, last_n: int = 3):
    profile = (
        session.table("memory_blocks")
        .filter((fc.col("user_id") == fc.lit(user_id)) & (fc.col("block_name") == fc.lit("profile")))
        .select("block_name", "content", "last_updated")
    )
    recent = (
        session.table("account_timeline")
        .filter(fc.col("user_id") == fc.lit(user_id))
        .sort(fc.col("timestamp").desc())
        .limit(last_n)
        .select("timestamp", "event_type", "status", "amount", "description")
    )
    return {"profile": profile._plan, "recent_events": recent._plan}

generated_system_tools = auto_generate_system_tools_from_tables(
    ["memory_blocks", "account_timeline"],
    session,
    tool_namespace="memctx",
    max_result_limit=100,
)

server = FenicMCPServer(
    session._session_state,
    user_defined_tools=[],
    system_tools=[
        SystemTool(
            name="get_user_context",
            description="Profile + recent events",
            max_result_limit=100,
            func=get_user_context,
        ),
        *generated_system_tools,
    ],
    server_name="Memory (Blocks & Episodes)",
)
```

</details>

---

### 3) Retrieval — From unstructured to structured data

Turn unstructured sources into typed rows (Q&A, policies, products), pre-embed, and retrieve with citations.

**Tools exposed:** `qa_neighbors(query, k)` with citations, plus system tools

<details>
<summary><b>Show code</b></summary>

```python
import fenic as fc
from fenic import SemanticConfig, OpenAILanguageModel, OpenAIEmbeddingModel
from fenic.api.mcp._tool_generation_utils import auto_generate_system_tools_from_tables
from fenic.core.mcp._server import FenicMCPServer
from fenic.core.mcp.types import SystemTool
from pydantic import BaseModel, Field

class QAPair(BaseModel):
    question: str = Field(description="A question extracted from the document")
    answer: str = Field(description="The answer to the question")

session = fc.Session.get_or_create(fc.SessionConfig(
    app_name="policy_qa",
    semantic=SemanticConfig(
        language_models={"gpt": OpenAILanguageModel(model_name="gpt-5-nano", rpm=100, tpm=100_000)},
        embedding_models={"embed": OpenAIEmbeddingModel(model_name="text-embedding-3-small", rpm=100, tpm=100_000)},
        default_embedding_model="embed"
    )
))

qa_pairs = (
    session.read.pdf_metadata("policies/*.pdf")
    .select(fc.col("file_path").alias("source"),
            fc.semantic.parse_pdf(fc.col("file_path")).alias("content"))
    .select(fc.col("source"),
            fc.semantic.extract(fc.col("content").cast(fc.StringType), QAPair).alias("qa"))
    .unnest("qa")
    .select("source",
            "question",
            "answer",
            fc.semantic.embed(fc.col("question")).alias("embedding"))
)
qa_pairs.write.save_as_table("policy_qa", mode="overwrite")

async def qa_neighbors(query: str, k: int = 3):
    q = session.create_dataframe([{"q": query}])
    res = q.semantic.sim_join(
        session.table("policy_qa"),
        left_on=fc.semantic.embed(fc.col("q")),
        right_on=fc.col("embedding"),
        k=k,
        similarity_score_column="relevance",
    ).select("question", "answer", "source", "relevance")
    return res._plan

generated_system_tools = auto_generate_system_tools_from_tables(
    ["policy_qa"], session, tool_namespace="qa", max_result_limit=50
)

server = FenicMCPServer(
    session._session_state,
    user_defined_tools=[],
    system_tools=[
        SystemTool(
            name="qa_neighbors",
            description="Semantic Q/A retrieval",
            max_result_limit=50,
            func=qa_neighbors,
        ),
        *generated_system_tools,
    ],
    server_name="Policy QA",
)
```

</details>

---

### 4) Retrieval — Semantic Spans

Break long documents into overlapping spans, embed once, serve semantic top-K.

**Tools exposed:** `docs_neighbors(query, k)`, plus system tools

<details>
<summary><b>Show code</b></summary>

```python
import fenic as fc
from fenic import SemanticConfig, OpenAILanguageModel, OpenAIEmbeddingModel
from fenic.api.mcp._tool_generation_utils import auto_generate_system_tools_from_tables
from fenic.core.mcp._server import FenicMCPServer
from fenic.core.mcp.types import SystemTool

session = fc.Session.get_or_create(fc.SessionConfig(
    app_name="docs",
    semantic=SemanticConfig(
        language_models={"gpt": OpenAILanguageModel(model_name="gpt-5-nano", rpm=100, tpm=100_000)},
        embedding_models={"embed": OpenAIEmbeddingModel(model_name="text-embedding-3-small", rpm=100, tpm=100_000)},
        default_embedding_model="embed"
    )
))

exploded = (
    session.read.pdf_metadata("docs/**/*.pdf")
    .select(fc.col("file_path").alias("source"),
            fc.semantic.parse_pdf(fc.col("file_path")).alias("content"))
    .select(fc.col("source"),
            fc.text.recursive_word_chunk(
                fc.col("content").cast(fc.StringType),
                chunk_size=500,
                chunk_overlap_percentage=10,
            ).alias("chunks"))
    .explode("chunks")
)

# Use SQL to generate unique chunk IDs with ROW_NUMBER()
chunks = (
    session.sql("SELECT ROW_NUMBER() OVER () as chunk_id, * FROM {df}", df=exploded)
    .select("chunk_id",
            "source",
            fc.col("chunks").alias("text"),
            fc.semantic.embed(fc.col("chunks")).alias("embedding"))
)
chunks.write.save_as_table("chunks", mode="overwrite")

async def docs_neighbors(query: str, k: int = 3):
    q = session.create_dataframe([{"q": query}])
    res = q.semantic.sim_join(
        session.table("chunks"),
        left_on=fc.semantic.embed(fc.col("q")),
        right_on=fc.col("embedding"),
        k=k,
        similarity_score_column="relevance",
    ).select("chunk_id", "source", "text", "relevance")
    return res._plan

generated_system_tools = auto_generate_system_tools_from_tables(
    ["chunks"], session, tool_namespace="docs", max_result_limit=100
)

server = FenicMCPServer(
    session._session_state,
    user_defined_tools=[],
    system_tools=[
        SystemTool(
            name="docs_neighbors",
            description="Semantic chunk retrieval",
            max_result_limit=100,
            func=docs_neighbors,
        ),
        *generated_system_tools,
    ],
    server_name="Docs",
)
```

</details>

---

## Core Concepts

### Lifecycle: Hydrate → Shape → Serve → Operate

1. **Hydrate** — Load sources (PDF/MD/CSV/DB)
2. **Shape** — Transform with deterministic ops (select/filter/join/window) + semantic ops (extract/embed/summarize)
3. **Serve** — Expose as bounded tools (MCP or Python functions) with result caps
4. **Operate** — Version with snapshots/tags; rollback instantly

### Design Principles

| Principle                   | What It Means                                                                |
| --------------------------- | ---------------------------------------------------------------------------- |
| **Framework-agnostic**      | Works with any runtime that can call tools or functions                      |
| **Inference offloading**    | Context operations happen in fenic, not your agent's context window          |
| **Context as typed tables** | Model context relationally; query it precisely                               |
| **Declarative transforms**  | Focus on _what_ context to build, not _how_—iterate fast on context strategy |
| **Bounded tool surfaces**   | Minimal, auditable interfaces with result caps                               |
| **Immutable snapshots**     | Version context                                                              |
| **Runtime enablement**      | Provide primitives; let your framework orchestrate                           |

---

## Key Capabilities

<details>
<summary><b>Inference Offloading</b> — LLM ops happen in fenic, not your agent's context</summary>

```python
# This summarization happens in fenic's inference—
# your agent receives only the result, with less context bloat
summary = (
    session.table("conversations")
    .select(
        fc.col("user_id"),
        fc.semantic.map(
            "Summarize this conversation in 2 sentences: {{ messages }}",
            messages=fc.col("messages")
        ).alias("summary")
    )
)
```

**Traditional approach:** Agent performs the summarization or it delegates to a sub-agent → tokens consumed from agent budget or complexity is increased by having to manage the context of multiple agents
**fenic approach:** fenic handles summarization → agent receives the result → less context bloat, so agents stay focused on reasoning

</details>

<details>
<summary><b>Token Budget Awareness</b> — Track and manage token budgets</summary>

Track and manage token budgets across your context operations:

```python
# Token statistics available throughout the pipeline
metrics = df.write.save_as_table("context", mode="overwrite")

print(f"Total cost: ${metrics.total_lm_metrics.cost:.4f}")
print(f"Input tokens: {metrics.total_lm_metrics.num_uncached_input_tokens}")
print(f"Output tokens: {metrics.total_lm_metrics.num_output_tokens}")

# Tool responses shaped to fit budgets
from fenic.api.mcp._tool_generation_utils import auto_generate_system_tools_from_tables
from fenic.core.mcp._server import FenicMCPServer

generated_tools = auto_generate_system_tools_from_tables(
    ["context"], session, tool_namespace="ctx", max_result_limit=50  # Cap response size
)

server = FenicMCPServer(
    session._session_state,
    user_defined_tools=[],
    system_tools=generated_tools,
    server_name="Budget-Aware Tools",
)
```

</details>

<details>
<summary><b>State Management & Rollback</b> — Relational model with versioned tables</summary>

```python
# Version your context with dated table names
ctx.write.save_as_table("policy_qa_2025_11_06", mode="overwrite")
ctx.write.save_as_table("policy_qa_2025_11_12", mode="overwrite")

# Point prod to a version by copying
ctx.write.save_as_table("policy_qa_prod", mode="overwrite")

# Rollback instantly by repointing prod to an older version
old_ctx = session.table("policy_qa_2025_11_06")
old_ctx.write.save_as_table("policy_qa_prod", mode="overwrite")
```

</details>

<details>
<summary><b>Temporal Memory with Window Functions</b> — Decaying resolution patterns</summary>

```python
from datetime import date, timedelta

# Define temporal boundaries
today = date.today()
week_start = today - timedelta(days=today.weekday())

# Window functions for temporal processing
daily_summary = (
    session.table("events")
    .filter(fc.col("timestamp") >= fc.lit(today))
    .select(
        fc.col("user_id"),
        fc.semantic.reduce(
            "Summarize today's events",
            fc.col("event_text")
        ).alias("daily_summary")
    )
)

# Weekly rollup from daily summaries
weekly_summary = (
    session.table("daily_summaries")
    .filter(fc.col("date") >= fc.lit(week_start))
    .group_by("user_id")
    .agg(
        fc.semantic.reduce(
            "Summarize this week's key events",
            fc.col("daily_summary")
        ).alias("weekly_summary")
    )
)
```

</details>

<details>
<summary><b>Tool Response Shaping</b> — Truncation strategies, less context bloat</summary>

```python
# Deterministic: pagination and filtering
async def search_with_pagination(query: str, page: int = 0, page_size: int = 10):
    filtered = (
        session.table("documents")
        .filter(fc.col("content").contains(query))
    )
    # Use SQL for OFFSET support
    return session.sql(
        f"SELECT * FROM {{df}} ORDER BY relevance DESC LIMIT {page_size} OFFSET {page * page_size}",
        df=filtered
    )._plan

# Semantic: summarize large results (inference offloaded)
async def search_with_summary(query: str, k: int = 20):
    results = session.table("documents").limit(k)
    return (
        results.select(
            fc.col("id"),
            fc.semantic.map(
                f"Extract the part most relevant to: {query}\n\nContent: {{{{ content }}}}",
                content=fc.col("content")
            ).alias("relevant_excerpt")
        )
    )._plan
```

</details>

<details>
<summary><b>Tool-Level Observability</b> — See inside tools, not just inputs/outputs</summary>

```python
# fenic tracks operations inside tools
# Close the loop between runtime traces and tool internals

metrics = df.write.save_as_table("context", mode="overwrite")

# Aggregated LM metrics across all operators
lm = metrics.total_lm_metrics
print(f"Tokens: {lm.num_uncached_input_tokens + lm.num_output_tokens}, Cost: ${lm.cost:.4f}")

# Detailed execution plan with per-operator metrics
print(metrics.get_execution_plan_details())
```

</details>

---

## Declarative Shaping

### Deterministic Transforms

`select`, `filter`, `sort`, `limit`, `join`, `group_by`, `agg`, `window`, `unnest`, `explode`

### Semantic Transforms

`semantic.extract` (to Pydantic), `semantic.embed`, `semantic.classify`, `semantic.map` (template-driven),
`semantic.reduce` (multi-row), `semantic.predicate` (NL filters), `semantic.sim_join`, `semantic.parse_pdf`

```python
from pydantic import BaseModel

class PolicyInfo(BaseModel):
    category: str
    date: str
    summary: str

# Combine both—works with any framework
results = (
    session.read.pdf_metadata("docs/*.pdf")
    .select(
        fc.semantic.parse_pdf(fc.col("file_path")).alias("content")  # semantic
    )
    .filter(fc.col("content").contains("policy"))                     # deterministic
    .select(
        fc.semantic.extract(fc.col("content"), PolicyInfo).alias("data"),  # semantic
        fc.semantic.embed(fc.col("content")).alias("vec")                  # semantic
    )
    .filter(fc.col("data").category == fc.lit("refund"))              # deterministic
    .sort(fc.col("data").date.desc())                                 # deterministic
    .limit(10)                                                        # deterministic
)
```

---

## Production Operations

### Batching & Rate Limiting

```python
config = fc.SessionConfig(
    semantic=fc.SemanticConfig(
        language_models={
            "gpt": fc.OpenAILanguageModel(
                model_name="gpt-4o-mini",
                rpm=1000,  # requests per minute
                tpm=1_000_000  # tokens per minute
            )
        }
    )
)
```

<details>
<summary><b>Show more production features</b> — async UDFs, error handling...</summary>

### Async UDFs with Concurrency Control

```python
@fc.async_udf(
    return_type=fc.StringType,
    max_concurrency=50,
    timeout_seconds=10,
    num_retries=3
)
async def fetch_user_profile(user_id: str) -> str:
    async with aiohttp.ClientSession() as s:
        async with s.get(f"https://api.example.com/{user_id}") as resp:
            return await resp.text()
```

### Error Handling

- Automatic retries for transient failures
- Graceful degradation: failed rows return `None`
- Per-op timeouts
- Schema validation before execution
</details>

---

## Integrations

### AI Providers

| Provider      | Type             | Models                                      |
| ------------- | ---------------- | ------------------------------------------- |
| OpenAI        | LLM + Embeddings | GPT-4, GPT-5, o-series, text-embedding-3-\* |
| Anthropic     | LLM              | Claude (Haiku/Sonnet/Opus)                  |
| Google Gemini | LLM + Embeddings | Gemini 2.0/2.5 Flash                        |
| OpenRouter    | LLM (aggregator) | 200+ models                                 |
| Cohere        | LLM + Embeddings | embed-v4.0                                  |

### Agent Frameworks

Any agentic framework (LangGraph, PydanticAI, CrewAI, ...)

<details>
<summary><b>Show more integrations</b> — AI providers, data sources, outputs, agent frameworks</summary>

### Data Sources

Local files, S3, Hugging Face Datasets, in-memory (Polars/Pandas/PyArrow)

### Outputs

CSV/Parquet, fenic native storage, DataFrame exports, MCP servers, Python functions

</details>

---

## Install

```bash
pip install fenic
# or
uv add fenic
# Requires Python 3.10+
```

---

## Configuration

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
```

---

## Examples

### Agent Projects ([fenic-examples](https://github.com/typedef-ai/fenic-examples))

- A deep research agent for Hacker News
- A log triage with LangGraph
- How to do AI Feature Engineering for RecSys

<details>
<summary><b>Show more examples</b> — 11 notebooks with Colab links</summary>

| Example                                                                 | Description                                                                                                                         |                                                                                          Colab                                                                                          |
| ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [Hello World!](examples/hello_world)                                    | Introduction to semantic extraction and classification using fenic's core operators through error log analysis.                     |               [![Open in Colab](docs/images/colab-badge.svg)](https://colab.research.google.com/github/typedef-ai/fenic/blob/main/examples/hello_world/hello_world.ipynb)               |
| [Enrichment](examples/enrichment)                                       | Multi-stage DataFrames with template-based text extraction, joins, and LLM-powered transformations demonstrated via log enrichment. |                [![Open in Colab](docs/images/colab-badge.svg)](https://colab.research.google.com/github/typedef-ai/fenic/blob/main/examples/enrichment/enrichment.ipynb)                |
| [Meeting Transcript Processing](examples/meeting_transcript_processing) | Native transcript parsing, Pydantic schema integration, and complex aggregations shown through meeting analysis.                    | [![Open in Colab](docs/images/colab-badge.svg)](https://colab.research.google.com/github/typedef-ai/fenic/blob/main/examples/meeting_transcript_processing/transcript_processing.ipynb) |
| [News Analysis](examples/news_analysis)                                 | Analyze and extract insights from news articles using semantic operators and structured data processing.                            |             [![Open in Colab](docs/images/colab-badge.svg)](https://colab.research.google.com/github/typedef-ai/fenic/blob/main/examples/news_analysis/news_analysis.ipynb)             |
| [Podcast Summarization](examples/podcast_summarization)                 | Process and summarize podcast transcripts with speaker-aware analysis and key point extraction.                                     |     [![Open in Colab](docs/images/colab-badge.svg)](https://colab.research.google.com/github/typedef-ai/fenic/blob/main/examples/podcast_summarization/podcast_summarization.ipynb)     |
| [Semantic Join](examples/semantic_joins)                                | Instead of simple fuzzy matching, use fenic's powerful semantic join functionality to match data across tables.                     |            [![Open in Colab](docs/images/colab-badge.svg)](https://colab.research.google.com/github/typedef-ai/fenic/blob/main/examples/semantic_joins/semantic_joins.ipynb)            |
| [Named Entity Recognition](examples/named_entity_recognition)           | Extract and classify named entities from text using semantic extraction and classification.                                         |            [![Open in Colab](docs/images/colab-badge.svg)](https://colab.research.google.com/github/typedef-ai/fenic/blob/main/examples/named_entity_recognition/ner.ipynb)             |
| [Markdown Processing](examples/markdown_processing)                     | Process and transform markdown documents with structured data extraction and formatting.                                            |       [![Open in Colab](docs/images/colab-badge.svg)](https://colab.research.google.com/github/typedef-ai/fenic/blob/main/examples/markdown_processing/markdown_processing.ipynb)       |
| [JSON Processing](examples/json_processing)                             | Handle complex JSON data structures with semantic operations and schema validation.                                                 |           [![Open in Colab](docs/images/colab-badge.svg)](https://colab.research.google.com/github/typedef-ai/fenic/blob/main/examples/json_processing/json_processing.ipynb)           |
| [Feedback Clustering](examples/feedback_clustering)                     | Group and analyze feedback using semantic similarity and clustering operations.                                                     |       [![Open in Colab](docs/images/colab-badge.svg)](https://colab.research.google.com/github/typedef-ai/fenic/blob/main/examples/feedback_clustering/feedback_clustering.ipynb)       |
| [Document Extraction](examples/document_extraction)                     | Extract structured information from various document formats using semantic operators.                                              |       [![Open in Colab](docs/images/colab-badge.svg)](https://colab.research.google.com/github/typedef-ai/fenic/blob/main/examples/document_extraction/document_extraction.ipynb)       |

</details>

---

## Support

Join our community on [Discord](https://discord.gg/GdqF3J7huR) where you can connect with other users, ask questions, and get help with your fenic projects. Our community is always happy to welcome newcomers!

If you find fenic useful, consider giving us a ⭐ at the top of this repository. Your support helps us grow and improve the framework for everyone!

## Contributing

We welcome contributions of all kinds! Whether you're interested in writing code, improving documentation, testing features, or proposing new ideas, your help is valuable to us.

For developers planning to submit code changes, we encourage you to first open an issue to discuss your ideas before creating a Pull Request. This helps ensure alignment with the project's direction and prevents duplicate efforts.

Please refer to our [contribution guidelines](CONTRIBUTING.md) for detailed information about the development process and project setup.
