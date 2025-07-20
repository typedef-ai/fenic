
from fenic import col, text


def test_jinja_simple_variable(local_session):
    """Test simple variable substitution."""
    data = {
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35]
    }
    df = local_session.create_dataframe(data)

    # Test simple variable substitution
    result = df.select(
        text.jinja("Hello {{ name }}!", name=col("name")).alias("greeting")
    ).to_polars()

    expected = ["Hello Alice!", "Hello Bob!", "Hello Charlie!"]
    assert result["greeting"].to_list() == expected


def test_jinja_multiple_variables(local_session):
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


def test_jinja_struct_access(local_session):
    """Test accessing struct fields in templates."""
    # This would require creating a struct column first
    # For now, let's test the basic functionality
    data = {
        "user": [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]
    }
    df = local_session.create_dataframe(data)

    result = df.select(
        text.jinja("Hello {{ user.name }}, you are {{ user.age }}!", user=col("user")).alias("greeting")
    ).to_polars()

    expected = ["Hello Alice, you are 25!", "Hello Bob, you are 30!"]
    assert result["greeting"].to_list() == expected


def test_jinja_conditional(local_session):
    """Test conditional rendering in templates."""
    data = {
        "name": ["Alice", "Bob", "Charlie"],
        "premium": [True, False, True]
    }
    df = local_session.create_dataframe(data)

    template = "Hello {{ name }}{% if premium %} (Premium Member){% endif %}!"

    result = df.select(
        text.jinja(template, name=col("name"), premium=col("premium")).alias("greeting")
    ).to_polars()
    print(result)

    expected = [
        "Hello Alice (Premium Member)!",
        "Hello Bob!",
        "Hello Charlie (Premium Member)!"
    ]
    assert result["greeting"].to_list() == expected


def test_jinja_null_handling(local_session):
    """Test how nulls are handled in templates."""
    data = {
        "name": ["Alice", None, "Charlie"],
        "age": [25, 30, None]
    }
    df = local_session.create_dataframe(data)

    result = df.select(
        text.jinja(
            "{% if name %}{{ name }}{% else %}Unknown{% endif %} is {% if age %}{{ age }}{% else %}N/A{% endif %} years old",
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
