import pytest

from fenic import ColumnField, col, text
from fenic.core.error import TypeMismatchError, ValidationError
from fenic.core.types.datatypes import StringType


def test_simple_variable(local_session):
    """Test simple variable substitution."""
    data = {
        "first_name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35]
    }
    df = local_session.create_dataframe(data)

    # Test simple variable substitution
    result = df.select(
        text.jinja("Hello {{ 'there' }} {{ name }}!", name=text.upper(col("first_name"))).alias("greeting")
    )
    assert result.schema.column_fields == [
        ColumnField(name="greeting", data_type=StringType)
    ]

    expected = ["Hello there ALICE!", "Hello there BOB!", "Hello there CHARLIE!"]
    assert result.to_polars()["greeting"].to_list() == expected


def test_multiple_variables(local_session):
    """Test template with multiple variables."""
    data = {
        "name": ["Alice", "Bob"],
        "age": [25, 30],
        "city": ["New York", "London"]
    }
    df = local_session.create_dataframe(data)

    result = df.select(
        text.jinja(
            "{{ name }} is {{ age }} years old and lives in {{ city }}",
            name=col("name"),
            age=col("age"),
            city=col("city")
        ).alias("description")
    ).to_polars()

    expected = [
        "Alice is 25 years old and lives in New York",
        "Bob is 30 years old and lives in London"
    ]
    assert result["description"].to_list() == expected


def test_struct_access(local_session):
    """Test accessing struct fields in templates."""
    data = {
        "user": [{"name": "Alice", "age": 25, "address": {"city": "New York"}},
                 {"name": "Bob", "age": 30, "address": {"city": None}},
                 {"name": "Charlie", "age": 35, "address": None},
                 None]
    }
    df = local_session.create_dataframe(data)

    result = df.select(
        text.jinja("Hello {{ user.name }}, you are {{ user['age'] }} and live in {{ user.address.city }}!", user=col("user")).alias("greeting")
    ).to_polars()

    expected = ["Hello Alice, you are 25 and live in New York!", "Hello Bob, you are 30 and live in none!",
                "Hello Charlie, you are 35 and live in !", None]
    assert result["greeting"].to_list() == expected


def test_array_access(local_session):
    """Test array access in templates."""
    data = {
        "items": [["hello"], ["hi", "hello"], [], None]
    }
    df = local_session.create_dataframe(data)

    result = df.select(
        text.jinja("{{ items[0] }} {{items[10] }}", items=col("items")).alias("result")
    ).to_polars()

    expected = ["hello ", "hi ", " ", None]
    assert result["result"].to_list() == expected

def test_bool_conditional(local_session):
    """Test conditional rendering in templates."""
    data = {
        "name": ["Alice", "Bob", "Charlie", "David"],
        "premium": [False, True, False, None]
    }
    df = local_session.create_dataframe(data)

    template = "Hello {{ name }}{% if premium %} (Premium Member){% endif %}!"

    result = df.select(
        text.jinja(template, name=col("name"), premium=~col("premium")).alias("greeting")
    ).to_polars()

    expected = [
        "Hello Alice (Premium Member)!",
        "Hello Bob!",
        "Hello Charlie (Premium Member)!",
        None,
    ]
    assert result["greeting"].to_list() == expected

def test_for_loop(local_session):
    """Test for loop in templates."""
    data = {
        "items": [["hello"], ["hi", "hello"], [], None]
    }
    df = local_session.create_dataframe(data)

    result = df.select(
        text.jinja("{% for item in items %}{{loop.index}} {{ item }} {% endfor %}", items=col("items")).alias("result")
    ).to_polars()

    expected = ["1 hello ", "1 hi 2 hello ", "", None]
    assert result["result"].to_list() == expected

    data = {
        "item": [{"items":["hello"]}, {"items":["hi", "hello"]}, {"items":[]}, None]
    }
    df = local_session.create_dataframe(data)

    result = df.select(
        text.jinja("{% for item in item.items %}{{loop.index}} {{ item }} {% endfor %}", item=col("item")).alias("result")
    ).to_polars()

    expected = ["1 hello ", "1 hi 2 hello ", "", None]
    assert result["result"].to_list() == expected

def test_nested_loop(local_session):
    """Test nested loop in templates."""
    data = {
        "items": [[["a", "b"], ["c", "d"]], [[]], [None], None]
    }
    df = local_session.create_dataframe(data)

    result = df.select(
        text.jinja("{% for item in items %}{% for inner_item in item %}outer: {{item[0]}} inner:{{inner_item}} {% endfor %}{% endfor %}", items=col("items")).alias("result")
    ).to_polars()

    expected = ['outer: a inner:a outer: a inner:b outer: c inner:c outer: c inner:d ', '', '', None]
    assert result["result"].to_list() == expected


def test_jinja_else_null_handling(local_session):
    """Test how nulls are handled in templates."""
    data = {
        "name": ["Alice", None, "Charlie"],
        "age": [25, 30, None]
    }
    df = local_session.create_dataframe(data)
    template = "{% if name %}{{ name }}{% else %}Unknown{% endif %} is {% if age %}{{ age }}{% else %}N/A{% endif %} years old"

    result = df.select(
        text.jinja(
            template,
            strict=False,
            name=col("name"),
            age=col("age")
        ).alias("description")
    ).to_polars()

    expected = [
        "Alice is 25 years old",
        "Unknown is 30 years old",
        "Charlie is N/A years old"
    ]
    assert result["description"].to_list() == expected

    result = df.select(
        text.jinja(
            template,
            strict=True,
            name=col("name"),
            age=col("age")
        ).alias("description")
    ).to_polars()

    expected = [
        "Alice is 25 years old",
        None,
        None,
    ]
    assert result["description"].to_list() == expected

def test_jinja_shadowing_and_scoping(local_session):
    """Test how shadowing and scoping works in templates."""
    data = {
        "names": [["Alice", "Bob", "Charlie"]],
        "name": ["David"]
    }
    df = local_session.create_dataframe(data)

    result = df.select(
        text.jinja(
            "{% for name in names %}{{ name }}{% endfor %}{{ name }}",
            names=col("names"),
            name=col("name")
        ).alias("description")
    ).to_polars()

    expected = [
        "AliceBobCharlieDavid"
    ]
    assert result["description"].to_list() == expected

def test_multiple_chunks(local_session):
    """Test multiple chunks in Jinja."""
    data = {
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35]
    }
    df = local_session.create_dataframe(data).union(local_session.create_dataframe(data)).union(local_session.create_dataframe(data))
    result = df.select(text.jinja("{{ name }} {{ age }}", name=col("name"), age=col("age")).alias("result")).to_polars()
    expected = ["Alice 25", "Bob 30", "Charlie 35", "Alice 25", "Bob 30", "Charlie 35", "Alice 25", "Bob 30", "Charlie 35"]
    assert result["result"].to_list() == expected

def test_invalid_jinja_template(local_session):
    """Test invalid Jinja template."""
    data = {
        "names": [["Alice"], ["Bob"], ["Charlie"]]
    }
    df = local_session.create_dataframe(data)

    with pytest.raises(ValidationError):
        df.select(
            text.jinja("{{ names[0] }} {{ names['foo']}}", names=col("names")).alias("result")
        ).to_polars()

def test_invalid_jinja_type_checking(local_session):
    """Test invalid Jinja template."""
    data = {
        "names": ["Alice", "Bob", "Charlie"]
    }
    df = local_session.create_dataframe(data)

    with pytest.raises(TypeMismatchError):
        df.select(
            text.jinja("{{ names[0] }}", names=col("names")).alias("result")
        ).to_polars()
