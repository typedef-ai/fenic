# Ask once, choose thresholds later

The [news-analysis example](../news_analysis/news_analysis.py) generates descriptions
of editorial language, then classifies them. This example reuses its **25 synthetic
articles from eight fictional sources** to show a different shape: five independent
questions over each article, followed by ordinary DataFrame decisions.

Use both models for what they return well:

- A completion model generates open-ended descriptions and quoted evidence.
- A decision model answers closed questions and preserves probability distributions.
- Your code chooses thresholds, review bands, rankings, and which rows need generation.

These are demonstrations of workflow mechanics, not real reporting or an accuracy
benchmark. The articles include invented facts and deliberately strong rhetoric.

## Run

From this repository, install the local development dependencies and export
`TYPESAFE_API_KEY` and `OPENAI_API_KEY` through your normal secret manager:

```bash
just sync-local
uv run python examples/typed_judgments/typed_judgments.py
```

The defaults are `jev-1.13.0` and `gpt-4.1-nano`. `main(config)` also accepts the
shared example-test configuration. The live smoke test skips when the TypeSafe
key is absent; it never sends `semantic.judge` to a completion provider. If the
supplied configuration has a completion model, the comparison uses it. Otherwise
it adds the OpenAI default, which requires `OPENAI_API_KEY`.

The standalone run uses a temporary database. It does not overwrite saved tables.
It sends only the headline and content to the models, never the fictional source
name. Source names are used later for aggregation.

## Two different query plans

The existing route makes one extraction and three classifications per article:

```text
article ── classify topic ──────────┐
        └─ extract language fields ┴─ classify political lean
                                  └─ classify journalistic style
```

Topic classification reads the original article. The other two classifiers read
the topic and extracted text. The three classification outputs are labels without
confidence; extraction also supplies useful free-form descriptions. The comparison
reproduces this article-level plan, not the original example's later per-source
`semantic.reduce` or table writes.

The judgment route makes one request per article with five independent questions:

| Question                  | Primitive | Output used by code                                                                          |
| ------------------------- | --------- | -------------------------------------------------------------------------------------------- |
| Political lean            | Choice    | `far_left`, `left_leaning`, `neutral`, `right_leaning`, `far_right`, plus all probabilities  |
| Main topic                | Choice    | Topic and distribution, including an `other` option                                          |
| Sensationalist style      | Noul      | Probability that the article uses sensationalist language                                    |
| Opinion presented as fact | Noul      | Probability that an evaluative opinion is stated as an established fact                      |
| Emotional intensity       | Score     | Position on four concrete levels, from detached reporting to sustained inflammatory rhetoric |

The model answers every question against the same article. No question consumes
another answer. These small inputs and five questions fit in a single native
judge request. Larger question sets may be partitioned by the operator's limits.

```python
judged = articles.with_column(
    "judgment",
    fc.semantic.judge(
        state="text", questions=questions(), model_alias="decisions"
    ),
).unnest("judgment").collect()
saved = session.create_dataframe(judged.data)
```

Materializing once makes later thresholds and views purely local. The five
questions, two additional dimensions, and topic definitions are not identical to
the completion route's tasks. Fewer requests does not establish equal accuracy.

## What probabilities buy

The script demonstrates four operations without repeating inference:

1. Change a sensationalism threshold from 0.50 to 0.80 or 0.95.
2. Abstain on `0.2 < sensationalist_p < 0.8`, instead of forcing a yes/no.
3. Rank articles by `sensationalist_p`.
4. Group by source and average probability columns, rather than count winning labels.

A Noul near **0.5 means uncertainty**, not medium intensity. Emotional intensity
uses a separate Score on a 0–3 scale. Its fractional value is a probability-weighted
level position, not a percentage of emotional words.

The source profile displays three of the five lean probabilities, alongside the
two Nouls. These are mean model probabilities, not measured frequencies of actual
bias. All five lean probabilities remain available in the materialized result.
Missing judgments are counted separately for review, not treated as neutral.

## A judge-gated extraction cascade

Closed judgments do not produce evidence quotes. For that, keep the completion model:

```python
saved = saved.with_column("advocacy_p", fc.lit(1.0) - fc.col("lean_p_neutral"))
gated = saved.filter(fc.col("advocacy_p") >= 0.8)
evidence = gated.select(
    "source", "headline", "advocacy_p",
    fc.semantic.extract(
        "text", BiasIndicators, max_output_tokens=256, model_alias="writer"
    ).alias("evidence"),
).unnest("evidence")
```

Here the gate means “likely political advocacy,” not “bad,” “unsafe,” or “false.”
The threshold was fixed before the run. The model selects candidates; the second
model generates quotes. A quote still needs checking against its source.

## Real output

Recorded on **2026-09-21**, using the models above and all 25 articles. Times are
fenic's `QueryMetrics.execution_time_ms`, divided by 1,000: query wall time,
excluding session setup and printing. Requests, token usage, and costs come from
`QueryMetrics.total_lm_metrics`. Dollar figures use fenic's model catalog rates,
not a provider invoice.

The first development run and the final display-validation run are both shown,
rather than presenting one timing as a stable performance result:

| Run   | Route              | Requests | Query seconds | Input tokens | Output tokens |   Cost USD |
| ----- | ------------------ | -------: | ------------: | -----------: | ------------: | ---------: |
| First | Extract + classify |       98 |        11.072 |       26,230 |         1,921 | 0.00339140 |
| First | Parallel judge     |       25 |         3.622 |       22,949 |         4,376 | 0.00096386 |
| First | Gated extract      |       18 |         1.633 |        7,219 |           649 | 0.00098150 |
| Final | Extract + classify |      100 |         4.129 |       26,389 |         1,865 | 0.00338490 |
| Final | Parallel judge     |       25 |         0.359 |       22,949 |         4,376 | 0.00096386 |
| Final | Gated extract      |       18 |         3.981 |        7,219 |           718 | 0.00100910 |

Provider-reported cached-input tokens were zero. The first route has a nominal
100 requests, but fenic deduplicates identical requests. In the first run, two
rows produced the same dependent-classifier input, saving one request in each
classifier. In the final run, those inputs were all distinct.

These are two sequential runs, not a controlled latency benchmark. Service warm-up,
batch concurrency, and model variability can affect timings. In the final run,
judge plus gated extraction took **4.340 seconds**, slightly longer than the
4.129-second completion route. The cascade reduced generation calls, not necessarily
elapsed time, and the outputs serve different purposes.

Selected printed output from the final run:

```text
Nominal baseline: 100 requests; distinct inputs to each dependent classifier: 25/25 (fenic deduplicates identical requests).
Thresholds changed AFTER inference (P(sensationalist)):
  p >= 0.50: 16 articles
  p >= 0.80: 13 articles
  p >= 0.95: 4 articles
Unavailable judgments (send to review): 0
Cascade: extracted 18/25 articles; avoided 7 generation calls versus extracting all.
Total measured model cost: $0.00535786
```

The abstain view contained three articles:

| Fictional source   | Headline                                                   | P(sensationalist) |
| ------------------ | ---------------------------------------------------------- | ----------------: |
| Free Market Weekly | Medical Innovation Delivers Hope for Alzheimer's Families  |              0.77 |
| Free Market Weekly | Amazon's Success Proves American Capitalism Delivers Value |              0.50 |
| Balanced Tribune   | American Innovation Leadership Drives AI Breakthrough      |              0.73 |

Selected source-profile rows, with the same printed numeric precision:

| Fictional source    | Mean left-leaning | Mean neutral | Mean right-leaning | Mean sensationalist |
| ------------------- | ----------------: | -----------: | -----------------: | ------------------: |
| Balanced Tribune    |            0.4825 |          0.0 |                0.5 |                0.84 |
| Global Wire Service |          0.033333 |     0.966667 |                0.0 |            0.046667 |
| Independent Monitor |          0.376667 |         0.29 |           0.333333 |                0.66 |

The first gated extraction printed these quotes:

```json
{
  "source": "Progressive Voice",
  "headline": "Fed's Rate Hike Threatens Working Families as Corporate Profits Soar",
  "advocacy_p": 1.0,
  "quotes": [
    "burden working families with higher borrowing costs",
    "Wall Street celebrates record profits",
    "This regressive monetary policy prioritizes the wealthy elite"
  ]
}
```

There were also failures worth retaining: the third gated article, “Climate
Summit's Weak Language Betrays Future Generations,” returned an empty quote list
despite advocacy probability 1.0. In the first run, the first article also returned
no quotes. A high gate probability neither guarantees successful extraction nor
verifies the generated evidence.

**Limits:** these synthetic examples do not establish calibration, political-bias
accuracy, or safe thresholds for another dataset. The PR's held-out calibration
caveat still applies. Evaluate error rates, abstentions, and missed cases on
held-out data from your own workload before choosing production thresholds.

## Design references

The independent-question design follows TypeSafe's
[Choice](https://docs.typesafe.ai/primitives/choice),
[Noul](https://docs.typesafe.ai/primitives/noul), and
[Score](https://docs.typesafe.ai/primitives/score) guidance. The composition patterns
come from [composite scoring](https://docs.typesafe.ai/patterns/composite-scoring)
and [speculative fan-out](https://docs.typesafe.ai/patterns/fan-out).
The [extraction cascade](https://docs.typesafe.ai/cookbooks/sde_cascade) illustrates
gating expensive work with narrow signals. Its extract-then-verify sequence differs
from this example's decide-then-extract sequence; its published accuracy claims
are not results for this example.
