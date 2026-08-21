---
workflow_id: admission-watermark
phase: research
research_stage: questions_ready
track: engineering
size_class: high-risk
status: approved
portability_level: 0
source_inputs:
  - originating task/ticket (not reproduced here)
last_updated: 2026-08-21
---

# Research: Semantic request admission

**Date:** 2026-08-21 · **Status:** approved · **Stage:** questions_ready

## Research Questions

A query plan for the Research findings below. You do not need to answer these
questions. They guide the research toward how the system works today.

1. In `src/fenic/_backends/local/async_udf_stream.py`, how are asynchronous
   requests admitted, tracked, and released as results become available?
2. In the local semantic operators, how do operator inputs become request batches,
   and where do those batches enter the asynchronous execution path?
3. How does the current execution path preserve result order, and which data
   structures retain pending inputs, requests, and completed results?
4. Which test suites exercise streaming and non-streaming semantic execution,
   including failure, ordering, and rate-limit behavior?
5. What existing instrumentation records execution-stage time, and where can it
   distinguish admission, dispatch, ordered-result waiting, and draining?
6. How do the repository's memory tests measure the retained working set during
   semantic operator execution?
7. How do the benchmark tools construct and compare representative semantic join
   workloads without relying on an external provider?
8. Which downstream consumers of the semantic execution path require results in
   submission order, where is ordered emission a contract rather than an
   implementation artifact, and where could order be restored later instead?

## Decision Log

- **[applied]** The questions cover the local asynchronous executor, semantic
  operators, tests, instrumentation, memory checks, and benchmark tooling.
- **[applied]** The work has no open blockers.
- **[approved]** The research plan includes a downstream ordering-contract
  question before any design work begins.

## Handoff

**Next step (paste into a fresh tab):**

> Use the `td-research` skill. The research doc is at
> `docs/td-flow/admission-watermark/research.md` (track: engineering, size:
> high-risk, stage: questions_ready — Questions written, Findings not yet present).
> Produce the objective technical map: fill the Findings into this same doc and
> set research_stage: findings_ready. Read this doc's Questions + the files they
> name — do not read the original ticket.
> If this is Codex: I explicitly permit optional subagent use for this phase
> where the skill allows it.

**Approved decisions:** workflow_id `admission-watermark`; provisional track
`engineering`; size `high-risk`.
**Open questions (carried forward):** none (the questions are the plan).
**Non-goals / out of scope:** answering the questions here; proposing changes.
**Evidence summary:** the task description (the questions derive from it).
**Known weak assumptions:** track may change at the Research routing gate.
**Next artifact:** `docs/td-flow/admission-watermark/research.md` (same doc, now
at findings_ready).
**Rollback if:** the task is trivial enough to skip research entirely.
