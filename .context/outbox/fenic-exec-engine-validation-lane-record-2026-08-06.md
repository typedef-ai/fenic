# fenic-exec-engine-validation — lane record

**Status:** DESIGN FROZEN — no keyed/provider call has occurred at this point.  
**Branch:** `herd/fenic-exec-engine-validation`, stacked at B1 final `bd7d89bc8301387f36652a7be2ea7ad66f3edefd`.  
**Authorization:** Captain/Herd Command real-provider epilogue; hard total backstop `$50.00`, with a mandatory pre-run halt if the projected total exceeds `$40.00`.

## Security and spend boundary

The only credential source will be this worktree's `.env`, copied from the approved
development environment with mode `600`. Its contents have not been read or
printed. No synthetic key, shell expansion, or provider configuration file is an
alternative credential source.

The probe creates a new temporary Fenic session and database per arm, so a prior
arm cannot make a subsequent arm artificially cheap through response-cache reuse.
It configures `gpt-4.1-nano`, the least expensive suitable model already supported
by Fenic and used in the test configuration. Fenic's model catalogue prices it at
`$0.100 / 1M` uncached input tokens and `$0.400 / 1M` output tokens
(`src/fenic/core/_inference/model_catalog.py`, checked 2026-08-06). The completion
metrics emitted by Fenic will be the actual-cost receipt; estimates deliberately
do not assume prompt caching.

The configured governor is conservative (`250 RPM`, `50,000 TPM`) and is not a
request to provoke provider 429s. Any observed provider retry/429 is incidental:
record it and let the P0 lifecycle calculation exclude its identified rate-limited
portion from upstream-idle attribution. Do not rerun to manufacture one.

## Corpus and matrix

All inputs are synthetic, public, short category records; no customer or secret
data are sent.

### Fusion gain (primary)

Each row contains a unique `record_id` and a 20–30 word operational note with an
explicit category token. Map normalizes that note. Extract parses the normalized
text into the one-field structured schema `{category: str}`. The final output is
`record_id, signal` in both arms.

| Size | Why it is in the matrix | Unfused arm | Fused arm |
|---:|---|---|---|
| 64 | One B0-sized block | map projection exposes `normalized`, forcing the existing materialization breaker | strict B1 map→extract shape |
| 160 | Crosses the 100-pair/request block boundary | same | same |
| 320 | Four blocks; makes overlap measurable without a broad spend | same | same |

For every arm record: process wall time; Fenic query time; lifecycle events and
P0 `compute_idle_gap_metrics` (gross, **non-rate-limited**, queue, and
rate-limited time); LMMetrics input/output tokens, request count and dollar cost;
row count; and output equality by `record_id → category`. The unfused design
intentionally retains its normal intermediate materialization; that is the
baseline B1 is intended to improve, not an attempt to reimplement old code.

### Bounded semantic.join (secondary)

Use 16 left and 16 right synthetic rows whose only meaningful values are one of
four exact category tokens. The predicate is deliberately literal: retain a pair
only if the two supplied category tokens are exactly equal. The fake-client
expectation is the 64 Cartesian pairs with equal category. This is 256 predicate
calls, which crosses the ModelClient's 100-request execution block boundary
without needlessly pressuring the provider. Record wall time,
child-process peak RSS (`resource.getrusage(RUSAGE_SELF).ru_maxrss`, normalized
for macOS), result pair IDs, lifecycle and token/cost metrics. No pre-B0/P1c
implementation is rerun: the receipt is bounded execution plus parity with that
deterministic expectation.

An unexpected output set is a semantic divergence HOLD: preserve the evidence,
stop all further keyed calls, and report rather than debug/retry it.

## Conservative cost estimate and go/no-go

Per map or extract call reserve **1,200 input + 384 output tokens**, much higher
than the synthetic row and one-field response should require. Fusion matrix:
`(64 + 160 + 320) × 2 arms × 2 operations = 2,176 calls`, hence at most
`2,611,200` input and `835,584` output tokens, or
`$0.261120 + $0.334234 = $0.595354` at the stated price.

Join reserve: `16 × 16 = 256` predicate calls × `1,000 input + 128 output` =
`256,000` input and `32,768` output tokens, or `$0.038707`.

The direct combined estimate is **$0.634061**. A tenfold uncertainty reserve for
provider/tokenization overhead is **$6.340610**, still below the mandatory `$40`
go/no-go ceiling and the `$50` hard backstop. The probe is therefore authorized to
start. Each run's actual LMMetrics cost will be appended below; stop immediately
if the running actual plus remaining conservative reserve would cross `$40`.

## Forward charter acknowledgement

After this report-review gate, continue in order with local-only `$0` nodes from
the same Linear project: TD-3384 audit first (prove/measure eager no-op physical
operator work before behavior-preserving fast paths), then TD-3385
NumPy-versus-native-Polars embedding benchmark/recommendation (implement only
clear, parity-preserving wins). Herd Command owns the Linear status transitions;
these nodes retain branches-only, digest, commit/freeze/review discipline.

## Run receipts

### Run 1 — fusion baseline, 64 rows, unfused — HOLD

The first keyed call began only after the design and estimate above were written.
It used the fresh `unfused/64` session and completed its map and extract request
batches. The harness then detected that the 64 observed `record_id → category`
values were not an exact match for the literal fake-client expectation. The
observed row count was still 64, but that is not semantic parity. This is the
predeclared surprise-divergence condition.

**Action:** stopped immediately. Do not run the fused comparator, larger sizes,
join probe, a diagnostic retry, or an induced rate-limit experiment. This is a
HOLD/report outcome, not a debugging session.

The failing process raised before it could serialize its in-memory Fenic
`LMMetrics`; consequently the provider's exact usage receipt is unavailable from
this terminated arm. This is a measurement-harness limitation, **not** evidence
of zero spend. The conservative charged-cost upper bound for the completed 128
requests (64 map + 64 extract), using 1,200 input and the observed 512-token
per-request output reservation, is `$0.041574`. Exact actual: **unavailable due
to safety-stop-before-metric-serialization**. No subsequent provider spend was
incurred by this lane.

No provider 429 or retry was visible in the captured process output before the
HOLD. Since lifecycle events likewise were not serialized, this is not a positive
rate-limit finding and no idle-gap attribution is claimed.

The raw `record_id → category` comparison was also process-local and was not
serialized before the count-only parity assertion raised. The evidence
availability receipt is `.context/outbox/fenic-exec-engine-validation-divergence-evidence-2026-08-06.md`.
This is a second harness deficiency: it proves that a divergence occurred, but
cannot honestly show representative actual values. No retry was issued.

## Frozen report-review request — FROM HERD COMMAND

The report, lane record, divergence-evidence availability receipt, anchor, and
daily digest are frozen as documentation-only validation evidence. **Request
final report review FROM HERD COMMAND** for the validation HOLD; this is not a
code review and authorizes neither a retry nor TD-3384. Review scope is the
committed evidence delta rooted at B1 final `bd7d89b`, particularly
`.context/outbox/fenic-exec-engine-validation-report-2026-08-06.md` and
`.context/outbox/fenic-exec-engine-validation-divergence-evidence-2026-08-06.md`.

**HOLD:** await that disposition before starting any new node.

## Amendment A — authorized 2026-08-06

Herd Command accepted the original HOLD as honest and authorized exactly one
Amendment A rerun. The previous report-review request is superseded for this
limited purpose; the report is no longer final until this round either completes
or reaches its single permitted second HOLD.

Before any Amendment A keyed call, the harness now writes a durable synthetic
evidence JSON file under `.context/validation/amendment-a-evidence/` **before**
the parity assertion. Each arm's receipt includes expected and raw observed
values, case/whitespace-normalized values, value-by-value mismatches,
LMMetrics, lifecycle summary, and raw lifecycle events. The extract field
description and upstream map instruction both require exactly one of `ALPHA`,
`BETA`, `GAMMA`, or `DELTA`; parity compares normalized values. A residual
mismatch is listed by record ID in that evidence file and is a final
stop-and-report HOLD for this probe.

### Amendment A cumulative spend estimate and go/no-go

The prior stopped arm remains charged at its conservative upper bound of
**$0.041574**. Amendment A uses the observed 512-token output reservation for
every map/extract request (more conservative than the original 384-token
estimate): 2,176 fusion requests reserve `$0.705024`; the 256-call join reserves
`$0.038707`; new direct reserve is `$0.743731`. Cumulative direct reserve is
therefore **$0.785305**. The pre-run projected total is prior upper bound plus a
10× remaining-run uncertainty reserve: **$7.478886**, below the `$40` halt line
and `$50` hard backstop.

Before every arm, the harness recomputes `prior upper + completed actual
LMMetrics cost + 10× remaining reserve` and raises before issuing a request if
that projection exceeds `$40`. The matrix restarts at unfused/64 and otherwise
retains the approved order. No rate limit is manufactured.

### Charter ordering amendment

After the amended probe report is frozen and passes Herd Command report review,
**TD-3334 generalized N-op fusion** is next, ahead of TD-3384 and TD-3385. Its
design must first survey typedef's data-intelligence sync pipelines read-only
for actual semantic-operator chains, then optimize for that shape without
special-casing it at the expense of open-source Fenic. Free-for-everyone,
adaptive-rate-limiting-style gains are the model. TD-3334 remains
branches-only, local/fake-first, and spend-gated for any later live check.
