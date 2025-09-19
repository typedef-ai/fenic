from textwrap import dedent

import polars as pl
import pytest

from fenic import (
    BooleanType,
    ColumnField,
    IntegerType,
    PredicateExample,
    PredicateExampleCollection,
    StringType,
    col,
    semantic,
)
from fenic.api.session import (
    OpenAIEmbeddingModel,
    SemanticConfig,
    Session,
    SessionConfig,
)
from fenic.core.error import InvalidExampleCollectionError, ValidationError


def test_single_semantic_filter(local_session):
    claim = "Review: {{review}}. The review has positive sentiment about apache spark."
    source = local_session.create_dataframe(
        {
            "blurb": [
                "Apache Spark is the worst piece of software I've ever used. It's so slow and inefficient and I hate the JVM.",
                "Apache Spark is amazing. It's so fast and effortlessly scales to petabytes of data. Couldn't be happier.",
            ],
            "a_boolean_column": [
                True,
                False,
            ],
            "a_numeric_column": [
                1,
                -1,
            ],
        }
    )
    df = source.filter(
        semantic.predicate(claim, review=col("blurb"))
        & (col("a_boolean_column"))
        & (col("a_numeric_column") > 0)
    )
    assert df.schema.column_fields == [
        ColumnField(name="blurb", data_type=StringType),
        ColumnField(name="a_boolean_column", data_type=BooleanType),
        ColumnField(name="a_numeric_column", data_type=IntegerType),
    ]
    result = df.to_polars()
    assert result.schema == {
        "blurb": pl.String,
        "a_boolean_column": pl.Boolean,
        "a_numeric_column": pl.Int64,
    }

    df = source.select(semantic.predicate(claim, review=col("blurb")).alias("sentiment"))
    result = df.to_polars()
    assert result.schema == {
        "sentiment": pl.Boolean,
    }

def test_semantic_filter_with_nulls(local_session):
    source = local_session.create_dataframe(
        {
            "blurb": [
                "Apache Spark is the worst piece of software I've ever used. It's so slow and inefficient and I hate the JVM.",
                None,
            ],
        }
    )
    template = "Review: {{review}}. The review has positive sentiment about apache spark."
    df = source.select(semantic.predicate(template, review=col("blurb")).alias("sentiment"))
    result = df.to_polars()
    assert result.schema == {
        "sentiment": pl.Boolean,
    }
    result_list = result["sentiment"].to_list()
    assert len(result_list) == 2
    assert result_list[1] is None

    df = source.select(semantic.predicate(template, strict=False, review=col("blurb")).alias("sentiment"))
    result = df.to_polars()
    result_list = result["sentiment"].to_list()
    assert len(result_list) == 2
    assert result_list[1] is not None

def test_semantic_filter_with_examples(local_session):
    claim = (
        "Review: {{part1}}. {{part2}}. The review has positive sentiment about apache spark."
    )
    source = local_session.create_dataframe(
        {
            "blurb1": [
                "Apache Spark is the worst piece of software I've ever used. It's so slow and inefficient and I hate the JVM.",
                "Apache Spark is amazing. It's so fast and effortlessly scales to petabytes of data. Couldn't be happier.",
            ],
            "blurb2": [
                "Apache Spark is the best thing since sliced bread.",
                "Apache Spark is the worst thing since sliced bread.",
            ],
        }
    )
    sentiment_collection = PredicateExampleCollection().create_example(
        PredicateExample(
            input={
                "part1": "Apache Spark has an amazing community.",
                "part2": "Apache Spark has good fault tolerance.",
            },
            output=True,
        )
    )
    df = source.filter(semantic.predicate(claim, part1=col("blurb1"), part2=col("blurb2"), examples=sentiment_collection))
    result = df.to_polars()
    assert result.schema == {
        "blurb1": pl.String,
        "blurb2": pl.String,
    }


def test_many_semantic_filter_or(local_session):
    source = local_session.create_dataframe(
        {
            "review": [
                "Apache Spark runs incredibly fast on our cluster, processing terabytes in minutes.",
                "Apache Spark has never crashed in production, running stable for months.",
                "Apache Spark's documentation is confusing and hard to follow.",
            ]
        }
    )

    df = source.filter(
        semantic.predicate("Review: {{review}}. The review discusses performance or speed", review=col("review"))
        | semantic.predicate("Review: {{review}}. The review discusses reliability or stability", review=col("review"))
    )
    result = df.to_polars()

    # Should match first two reviews (performance and reliability) but not the third (documentation)
    assert result.schema == {
        "review": pl.String,
    }

def test_semantic_predicate_without_models(tmp_path):
    """Test that an error is raised if no language models are configured."""
    session_config = SessionConfig(
        app_name="semantic_predicate_without_models",
        db_path=tmp_path,
    )
    session = Session.get_or_create(session_config)
    with pytest.raises(ValidationError, match="No language models configured."):
        source = session.create_dataframe(
            {"name": ["Alice", "Bob"]}
        )
        predicate_prompt = "The name: {{name}} has 10 letters."
        source.select(semantic.predicate(predicate_prompt, name=col("name")).alias("predicate"))
    session.stop()

    session_config = SessionConfig(
        app_name="semantic_predicate_with_models",
        semantic=SemanticConfig(
            embedding_models={"oai-small": OpenAIEmbeddingModel(model_name="text-embedding-3-small", rpm=3000, tpm=1_000_000)},
        ),
        db_path=tmp_path,
    )
    session = Session.get_or_create(session_config)
    with pytest.raises(ValidationError, match="No language models configured."):
        source = session.create_dataframe(
            {"name": ["Alice", "Bob"]}
        )
        predicate_prompt = "The name: {{name}} has 10 letters."
        source.select(semantic.predicate(predicate_prompt, name=col("name")).alias("predicate"))
    session.stop()

def test_semantic_predicate_complex_jinja_template(local_session):
    source = local_session.create_dataframe({
        "job": [
            {
                "title": "Data Analyst",
                "requirements": ["Python", "SQL", "Data Analysis"]
            },
            {
                "title": "Software Engineer",
                "requirements": ["Java", "Scala", "Spark"]
            }
        ],
        "resume": [
            {
                "name": "John Doe",
                "age": 30,
                "experience": [
                    {
                        "company": "Google",
                        "title": "Software Engineer",
                        "description": "Developed and maintained web applications."
                    }
                ]
            },
            {
                "name": "Jane Smith",
                "age": 25,
                "experience": [
                    {
                        "company": "Amazon",
                        "title": "Data Analyst",
                        "description": "Analyzed data and provided insights."
                    }
                ]
            }
        ]
    })

    result = source.select(
        semantic.predicate(
            dedent("""\
                Given the following resume and job requirements, determine if the candidate is a good fit for the job.
                ### Required Qualifications:
                {% for req in job.requirements %}
                - {{ req }}
                {% endfor %}
                ### Resume:
                {{resume.name}} {{resume.age}}
                {% for exp in resume.experience %}
                {{ exp.company }}: {{ exp.title }} - {{ exp.description }}
                {% endfor %}
            """),
            job=col("job"),
            resume=col("resume"),
        ).alias("is_good_fit")
    )
    assert result.schema.column_fields == [
        ColumnField(name="is_good_fit", data_type=BooleanType),
    ]

def test_semantic_predicate_complex_jinja_template_with_examples(local_session):
    source = local_session.create_dataframe({
        "job": [
            {
                "title": "Data Analyst",
                "requirements": ["Python", "SQL", "Data Analysis"]
            },
            {
                "title": "Software Engineer",
                "requirements": ["Java", "Scala", "Spark"]
            }
        ],
        "resume": [
            {
                "name": "John Doe",
                "age": 30,
                "experience": [
                    {
                        "company": "Google",
                        "title": "Software Engineer",
                        "description": "Developed and maintained web applications."
                    }
                ]
            },
            {
                "name": "Jane Smith",
                "age": 25,
                "experience": [
                    {
                        "company": "Amazon",
                        "title": "Data Analyst",
                        "description": "Analyzed data and provided insights."
                    }
                ]
            }
        ]
    })
    examples = PredicateExampleCollection()

    # Example 1: Candidate matches job requirements well
    examples.create_example(PredicateExample(
        input={
            "job": {
                "title": "Data Scientist",
                "requirements": ["Python", "Machine Learning", "SQL"]
            },
            "resume": {
                "name": "Alice Johnson",
                "age": 28,
                "experience": [
                    {
                        "company": "TechCorp",
                        "title": "Machine Learning Engineer",
                        "description": "Built ML models using Python and SQL."
                    }
                ]
            }
        },
        output=True  # Qualified
    ))

    # Example 2: Candidate has unrelated experience
    examples.create_example(PredicateExample(
        input={
            "job": {
                "title": "Data Scientist",
                "requirements": ["Python", "Machine Learning", "SQL"]
            },
            "resume": {
                "name": "Bob Smith",
                "age": 32,
                "experience": [
                    {
                        "company": "RetailCo",
                        "title": "Store Manager",
                        "description": "Managed store operations and supervised staff."
                    }
                ]
            }
        },
        output=False  # Not qualified
    ))

    prompt = dedent("""\
        Given the following resume and job requirements, determine if the candidate is a good fit for the job.
        ### Required Qualifications:
        {% for req in job.requirements %}
        - {{ req }}
        {% endfor %}
        ### Resume:
        {{resume.name}} {{resume.age}}
        {% for exp in resume.experience %}
        {{ exp.company }}: {{ exp.title }} - {{ exp.description }}
        {% endfor %}
    """)


    result = source.select(
        semantic.predicate(
            prompt,
            job=col("job"),
            resume=col("resume"),
            examples=examples,
        ).alias("is_good_fit")
    )
    assert result.schema.column_fields == [
        ColumnField(name="is_good_fit", data_type=BooleanType),
    ]

    bad_examples = PredicateExampleCollection()
    bad_examples.create_example(PredicateExample(
        input={"job": {"not_requirements": ["Python", "SQL", "Data Analysis"]}, "resume": {"name": "Alice Johnson", "age": 28, "experience": [{"company": "TechCorp", "title": "Machine Learning Engineer", "description": "Built ML models using Python and SQL."}]}},
        output=True
    ))
    with pytest.raises(InvalidExampleCollectionError, match="Field 'job' type mismatch: operator expects"):
        source.select(semantic.predicate(prompt, job=col("job"), resume=col("resume"), examples=bad_examples).alias("is_good_fit"))

def test_semantic_predicate_invalid_jinja_template(local_session):
    source = local_session.create_dataframe({"name": ["GlowMate"], "details": ["A rechargeable bedside lamp"]})
    with pytest.raises(ValidationError, match="The `predicate` argument to `semantic.predicate` cannot be empty."):
        source.select(
            semantic.predicate("", name=col("name"), details=col("details")).alias("summary")
        )
    with pytest.raises(ValidationError, match="`semantic.predicate` prompt requires at least one template variable."):
        source.select(
            semantic.predicate("hello", name=col("name")).alias("summary")
        )

def test_semantic_predicate_missing_column_names(local_session):
    source = local_session.create_dataframe({"name": ["GlowMate"], "details": ["A rechargeable bedside lamp"]})
    with pytest.raises(ValidationError, match="`semantic.predicate` requires at least one named column argument"):
        source.select(
            semantic.predicate("{{name}}").alias("summary")
        )

def test_semantic_predicate_missing_jinja_variable(local_session):
    source = local_session.create_dataframe({"name": ["GlowMate"], "details": ["A rechargeable bedside lamp"]})
    with pytest.raises(ValidationError, match="Template variable 'details' is not defined."):
        source.select(
            semantic.predicate("{{name}}{{details}}", name=col("name")).alias("summary")
        )
