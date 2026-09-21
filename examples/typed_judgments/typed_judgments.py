"""Compare generated analysis with parallel judgments, then combine them."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel, Field

import fenic as fc

LEAN = {
    "far_left": "Explicitly advocates replacing capitalism or radical socialist change.",
    "left_leaning": "Favors economic redistribution, labor protections, or progressive policy.",
    "neutral": "No clear political advocacy, or presents competing positions evenhandedly.",
    "right_leaning": "Favors free markets, traditional institutions, or conservative policy.",
    "far_right": "Explicitly advocates authoritarian nationalism or exclusionary extremism.",
}
TOPICS = ["politics", "technology", "business", "climate", "healthcare"]
GATE = 0.8  # Illustrative policy, fixed before the run, not a calibrated threshold.


class ArticleAnalysis(BaseModel):
    """The same open-ended fields as the original news example."""

    bias_indicators: str = Field(
        description="Key words or phrases that indicate political bias"
    )
    emotional_language: str = Field(
        description="Emotionally charged words or neutral descriptive language"
    )
    opinion_markers: str = Field(
        description="Words or phrases that signal opinion vs. factual reporting"
    )


class BiasIndicators(BaseModel):
    """Quoted evidence remains a generation task, not a closed decision."""

    quotes: list[str] = Field(
        description="Up to three verbatim phrases showing political advocacy; empty if none"
    )


def article_data() -> list[dict[str, str]]:
    """Reuse the original example's data without copying its article text."""
    path = Path(__file__).resolve().parents[1] / "news_analysis" / "news_analysis.py"
    spec = importlib.util.spec_from_file_location("news_article_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_articles()


def questions() -> list[fc.JudgeQuestion]:
    """Ask five independent questions; none reads another question's answer."""
    return [
        fc.JudgeQuestion.choice(
            name="lean",
            instructions=(
                "Which political position does the article's own narrative advocate? "
                "Judge headline and article text, not a position merely quoted from someone."
            ),
            options=LEAN,
        ),
        fc.JudgeQuestion.choice(
            name="topic",
            instructions="What is the main subject of the headline and article?",
            options={
                "politics": "Government, elections, courts, or public policy.",
                "technology": "Computing, artificial intelligence, or technology products.",
                "business": "Companies, commerce, monetary policy, or financial markets.",
                "climate": "Climate change, energy transition, or climate negotiations.",
                "healthcare": "Medical treatment, clinical research, or healthcare access.",
                "other": "A subject outside these categories.",
            },
        ),
        fc.JudgeQuestion.noul(
            name="sensationalist",
            instructions="Does the article use sensationalist language to provoke a reaction?",
            criteria={
                "true": "Exaggerated alarm, sweeping accusations, or dramatic hype in its own voice.",
                "false": "Restrained descriptions; reporting alarming events alone does not count.",
            },
        ),
        fc.JudgeQuestion.noul(
            name="opinion_as_fact",
            instructions="Does the article present an evaluative opinion as an established fact?",
            criteria={
                "true": "Unattributed value judgments or speculative motives stated as settled facts.",
                "false": "Factual reporting, clearly attributed opinions, or explicitly qualified views.",
            },
        ),
        fc.JudgeQuestion.score(
            name="emotion",
            instructions="How emotionally charged is the article's own language?",
            levels=[
                "Detached reporting with factual descriptions and no emotional appeals.",
                "Occasional approving or disapproving words within mostly restrained reporting.",
                "Repeated emotive judgments or appeals to hope, fear, anger, or sympathy.",
                "Sustained inflammatory rhetoric, denunciations, or urgent emotional appeals.",
            ],
        ),
    ]


def completion_route(articles: fc.DataFrame) -> fc.DataFrame:
    """Reproduce one extraction plus three classifications per article."""
    enriched = (
        articles.with_column(
            "primary_topic",
            fc.semantic.classify("text", TOPICS, model_alias="writer"),
        )
        .with_column(
            "analysis",
            fc.semantic.extract(
                "text", ArticleAnalysis, max_output_tokens=512, model_alias="writer"
            ),
        )
        .unnest("analysis")
    )
    enriched = enriched.with_column(
        "combined_extracts",
        fc.text.jinja(
            "Primary Topic: {{topic}}\nPolitical Bias Indicators: {{bias}}\n"
            "Emotional Language Summary: {{emotion}}\nOpinion Markers: {{opinion}}",
            topic=fc.col("primary_topic"),
            bias=fc.col("bias_indicators"),
            emotion=fc.col("emotional_language"),
            opinion=fc.col("opinion_markers"),
        ),
    )
    return enriched.with_column(
        "content_bias",
        fc.semantic.classify("combined_extracts", list(LEAN), model_alias="writer"),
    ).with_column(
        "journalistic_style",
        fc.semantic.classify(
            "combined_extracts",
            ["sensationalist", "informational"],
            model_alias="writer",
        ),
    )


def measure(name: str, frame: fc.DataFrame, usage: list[dict]):
    """Materialize once and print fenic's query metrics, not nominal call counts."""
    result = frame.collect()
    metrics = result.metrics
    lm = metrics.total_lm_metrics
    usage.append(
        {
            "route": name,
            "requests": lm.num_requests,
            "seconds": round(metrics.execution_time_ms / 1000, 3),
            "input_tokens": lm.num_uncached_input_tokens,
            "cached_input_tokens": lm.num_cached_input_tokens,
            "output_tokens": lm.num_output_tokens,
            "cost_usd": round(lm.cost, 8),
        }
    )
    print(json.dumps(usage[-1], sort_keys=True))
    return result.data


def run(config: fc.SessionConfig) -> None:
    """Compare both providers over the same synthetic articles."""
    # The test harness may supply only TypeSafe. Standalone defaults use OpenAI.
    existing = config.semantic.language_models if config.semantic else {}
    writer = next(
        (
            model
            for model in (existing or {}).values()
            if not isinstance(model, fc.TypeSafeLanguageModel)
        ),
        fc.OpenAILanguageModel(model_name="gpt-4.1-nano", rpm=500, tpm=200_000),
    )
    config = config.model_copy(
        update={
            "semantic": fc.SemanticConfig(
                language_models={
                    "decisions": fc.TypeSafeLanguageModel(
                        model_name="jev-1.13.0", rpm=500, tpm=200_000
                    ),
                    "writer": writer,
                },
                default_language_model="writer",
            ),
        }
    )
    session = fc.Session.get_or_create(config)
    usage = []
    try:
        articles = session.create_dataframe(article_data()).with_column(
            "text", fc.text.concat(fc.col("headline"), fc.lit(" | "), fc.col("content"))
        )
        print("25 synthetic articles; source names are not sent to either model.")
        print(f"Completion model: {writer.model_name}; decision model: jev-1.13.0")
        print("Measured requests, query wall time, and provider-reported usage:")
        baseline = measure("extract_plus_classify", completion_route(articles), usage)
        print(
            "Nominal baseline: 100 requests; distinct inputs to each dependent "
            f"classifier: {baseline['combined_extracts'].n_unique()}/25 "
            "(fenic deduplicates identical requests)."
        )
        print("Completion labels (first five):")
        print(
            baseline.select(
                "source", "content_bias", "primary_topic", "journalistic_style"
            ).head(5)
        )
        judged = measure(
            "parallel_judge",
            articles.with_column(
                "judgment",
                fc.semantic.judge(
                    state="text", questions=questions(), model_alias="decisions"
                ),
            ).unnest("judgment"),
            usage,
        )
        # Re-enter as materialized data: these views cannot repeat model inference.
        saved = session.create_dataframe(judged).with_column(
            "advocacy_p", fc.lit(1.0) - fc.col("lean_p_neutral")
        )
        print("Thresholds changed AFTER inference (P(sensationalist)):")
        for threshold in (0.5, GATE, 0.95):
            print(
                f"  p >= {threshold:.2f}: {saved.filter(fc.col('sensationalist_p') >= threshold).count()} articles"
            )
        print("Abstain band: uncertain sensationalism, 0.2 < p < 0.8:")
        saved.filter(
            (fc.col("sensationalist_p") > 0.2) & (fc.col("sensationalist_p") < 0.8)
        ).select("source", "headline", "sensationalist_p").show()
        print("Most likely sensationalist, ranked by probability:")
        saved.order_by(fc.desc("sensationalist_p")).select(
            "source", "headline", "sensationalist_p", "opinion_as_fact_p", "emotion"
        ).limit(5).show()
        print("Source profiles: mean probabilities, not counts of winning labels:")
        saved.group_by("source").agg(
            fc.avg("lean_p_left_leaning").alias("mean_left"),
            fc.avg("lean_p_neutral").alias("mean_neutral"),
            fc.avg("lean_p_right_leaning").alias("mean_right"),
            fc.avg("sensationalist_p").alias("mean_sensationalist"),
            fc.avg("opinion_as_fact_p").alias("mean_opinion_as_fact"),
        ).order_by("source").show()
        missing = saved.filter(fc.col("advocacy_p").is_null()).count()
        print(f"Unavailable judgments (send to review): {missing}")
        gated = saved.filter(fc.col("advocacy_p") >= GATE)
        selected = gated.count()
        evidence = measure(
            "gated_extract",
            gated.select(
                "source",
                "headline",
                "advocacy_p",
                fc.semantic.extract(
                    "text", BiasIndicators, max_output_tokens=256, model_alias="writer"
                ).alias("evidence"),
            ).unnest("evidence"),
            usage,
        )
        print(
            f"Cascade: extracted {selected}/25 articles; avoided {25 - selected} generation calls versus extracting all."
        )
        print("Gated evidence (first three):")
        print(json.dumps(evidence.head(3).to_dicts(), indent=2))
        print(
            f"Total measured model cost: ${sum(row['cost_usd'] for row in usage):.8f}"
        )
        print("Noul near 0.5 means uncertainty, not medium emotional intensity.")
        print(
            "These thresholds are illustrations, not measured accuracy or calibration."
        )
    finally:
        session.stop(skip_usage_summary=True)


def main(config: fc.SessionConfig | None = None) -> None:
    """Accept the shared test config; keep standalone runs isolated and uncached."""
    if config is not None:
        run(config)
    else:
        with TemporaryDirectory(prefix="fenic-news-judgments-") as directory:
            run(fc.SessionConfig(app_name="news_judgments", db_path=Path(directory)))


if __name__ == "__main__":
    main()
