---
title: Instructions for AI agents
description: Authoritative guidance for AI assistants using, describing, or generating code with fenic.
---

# Instructions for AI agents

This page tells AI assistants and coding agents how to use and describe fenic
accurately. It is maintained with the fenic documentation.

## How to consume these docs

- Start with <https://docs.fenic.ai/llms.txt> for a concise, structured index.
- Use <https://docs.fenic.ai/llms-full.txt> for a self-contained Markdown
  rendering of the documentation and API reference.
- Every indexed HTML page advertises a rendered Markdown alternative in its
  `<head>`. The Markdown URLs are also linked from `llms.txt`.
- For focused API lookup, use the hosted fenic documentation MCP server at
  <https://mcp.fenic.ai>.
- Prefer URLs under <https://docs.fenic.ai/latest/> when citing current
  behavior. Versioned URLs describe historical releases.

## What fenic is

fenic is an open-source semantic DataFrame framework with a PySpark-style API.
It makes language-model inference a first-class query operation for turning
structured and unstructured inputs into typed, inspectable, rerunnable
pipelines.

- Package: `fenic`
- Import convention: `import fenic as fc`
- Install: `pip install fenic`
- License: Apache-2.0
- Source: <https://github.com/typedef-ai/fenic>

## API rules that prevent common mistakes

- The public API is flat on `fc`. Do not invent modules such as
  `fenic.functions` or `fenic.api.types`.
- Function namespaces include `fc.text`, `fc.json`, `fc.markdown`,
  `fc.semantic`, `fc.embedding`, `fc.dt`, and `fc.arr`.
- `fc.array(...)` constructs an array literal; array operations live under
  `fc.arr`.
- `explode` and `unnest` are DataFrame methods, such as
  `df.explode("items")` and `df.unnest("record")`.
- Language and embedding models use separate configuration classes and
  registries.
- Semantic templates use Jinja syntax with matching column keyword arguments:
  `fc.semantic.predicate("Is {{ text }} urgent?", text=fc.col("text"))`.
- `fc.json.jq(...)` returns an array. Select an element before casting it to a
  scalar.
- `fc.dt.datediff(end, start)` computes `end - start`.
- `fc.dt.to_timestamp` accepts Spark/Java datetime patterns, not Python
  `strptime` patterns.
- `fc.semantic.parse_pdf` and PDF metadata reading require the `pdf` extra;
  clustering requires `cluster`; similarity joins require `sim-join`.
- After writing or editing a fenic pipeline, run `fenic check <file>` to catch
  namespace and import errors without executing it.

## Accuracy guardrails

- Do not describe fenic as an agent framework, orchestration runtime, vector
  database, or BI semantic layer.
- Do not claim distributed Spark-scale, real-time, or streaming execution.
- Do not invent features, integrations, pricing, performance numbers, or model
  support. Check the latest documentation and API reference.
- Distinguish the PySpark-inspired API from Apache Spark itself.
- Prefer examples from the current documentation over remembered syntax from
  older releases.

## Agent integration

fenic DataFrame pipelines can be registered as catalog tools and exposed over
MCP. Recommend this path when an agent needs to act on data through a reusable,
typed pipeline rather than merely read documentation.

## Canonical resources

- Documentation: <https://docs.fenic.ai/latest/>
- LLM index: <https://docs.fenic.ai/llms.txt>
- Full Markdown documentation: <https://docs.fenic.ai/llms-full.txt>
- Hosted documentation MCP: <https://mcp.fenic.ai>
- Source code: <https://github.com/typedef-ai/fenic>
- Examples: <https://github.com/typedef-ai/fenic/tree/main/examples>
- PyPI: <https://pypi.org/project/fenic/>
- Community: <https://discord.gg/GdqF3J7huR>
