# Typed judgments

`fc.semantic.judge` evaluates several closed questions about the same string
column. It returns a struct containing probabilities rather than hiding a decision
threshold inside a text response. The initial provider is TypeSafe System One.

Install the optional dependency and supply `TYPESAFE_API_KEY` through your normal
environment configuration:

```bash
pip install 'fenic[typesafe]'
```

## Evaluate and inspect

```python
import fenic as fc

session = fc.Session.get_or_create(
    fc.SessionConfig(
        app_name="typed_judgments",
        semantic=fc.SemanticConfig(
            language_models={
                "decisions": fc.TypeSafeLanguageModel(
                    model_name="jev-1.13.0", rpm=60, tpm=64_000
                )
            }
        ),
    )
)

questions = [
    fc.JudgeQuestion.noul(
        name="billing",
        instructions="Does the text describe a billing problem?",
    ),
    fc.JudgeQuestion.choice(
        name="topic",
        instructions="Which topic best describes the text?",
        options={
            "invoice": "Charges, invoices, or payment",
            "account": "Account access or settings",
            "other": "Neither of the other topics",
        },
    ),
]

messages = session.create_dataframe({"text": ["My invoice includes a duplicate charge."]})
scored = messages.with_column(
    "judgment",
    fc.semantic.judge(state="text", questions=questions, model_alias="decisions"),
).unnest("judgment")
selected = scored.filter(fc.col("billing_p") >= 0.8)
selected.show()
session.stop()
```

The output includes `billing_p`, `topic`, `topic_confidence`, and
`topic_p_invoice`, `topic_p_account`, and `topic_p_other`. Thresholds remain normal
column expressions, so a saved judgment can support several downstream decisions.

## Question types

Questions are immutable. Their names must be identifiers, and all generated output
fields must be unique.

| Factory                | Criteria                                   | Output                                                                             |
| ---------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------- |
| `JudgeQuestion.noul`   | Optional `true` and `false` descriptions   | `<name>_p`, the affirmative probability                                            |
| `JudgeQuestion.choice` | Between 2 and 255 described string options | `<name>`, `<name>_confidence`, and `<name>_p_<option>`                             |
| `JudgeQuestion.score`  | Between 2 and 10 ordered text levels       | Numeric `<name>` on the zero-based level scale, confidence, and `<name>_p_<index>` |

Choice probability suffixes normalize punctuation to underscores and use lowercase
ASCII. Colliding suffixes are rejected before execution. Score values may be
fractional; they are not integer class indices.

A question may name another Noul question as `premise`. This adds
`<name>_premise_p` containing that question's probability. It is an annotation for
downstream reasoning, not conditional execution: both questions are evaluated and
the provider is not asked to enforce a logical implication. Premise references
must exist and cannot form cycles.

## Execution and limits

The operator uses the session's scheduler, rate limiting, request cache, and usage
accounting. Ordered question definitions and the provider endpoint participate in
cache identity. Identical requests can share an in-flight call. Invalid answers
are not cached; known billed usage is retained even when an answer is rejected.
The SDK's internal retry loop is disabled so the session scheduler owns retries.

Null input states produce null structs without a call. Empty strings are valid
states. Malformed or failed responses produce null structs. A failure in any
partition nulls that row's entire judgment rather than returning a partial struct.
Authentication and other fatal configuration failures can still fail a query.

Questions share one state per request. The provider documents a 32,000-token
state-plus-longest-question limit and a 64,000-token total request limit.
Fenic estimates these sizes with a general-purpose tokenizer because the
provider tokenizer is not public. Questions are partitioned when necessary;
the state is never truncated. These estimates do not guarantee provider admission.
Premise-connected questions remain together. A state or indivisible question group
that exceeds the estimated envelope yields a null struct without a call.

Probabilities must be finite and bounded; full distributions must sum to one.
These checks do not establish calibration or consistency across questions.
Validate thresholds and error rates on held-out examples from your own workload.

This initial operation requires a model advertising typed-judgment support.
Other providers are unchanged. Free-form `map`, `extract`, `summarize`, and `reduce`
remain unsupported by this decision provider. Model profiles and automatic
operator fusion are not part of this initial implementation.
