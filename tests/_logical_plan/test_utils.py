import pytest

from fenic.core._logical_plan.utils import validate_and_parse_jinja_template
from fenic.core.error import ValidationError


def test_struct_access():
    template = "{{ user.name }}"
    assert validate_and_parse_jinja_template(template) == ["user"]

def test_static_index_access():
    template = "{{ items[0] }}"
    assert validate_and_parse_jinja_template(template) == ["items"]

def test_for_loop_with_attr_access():
    template = "{% for item in items %}{{ item.name }}{% endfor %}"
    assert validate_and_parse_jinja_template(template) == ["items"]

def test_complex_template_with_multiple_variables():
    template = """
    Hello {{ user.name }}!

    {% if user.premium %}
      Premium member since {{ user.metadata.start_date }}
    {% endif %}

    {% for order in orders %}
      Order ID: {{ order['id'] }}, Total: {{ order.total }} First item: {{ order.items[0] }}
    {% endfor %}

    You have {{ notifications[0] }} unread messages.
    """
    vars = validate_and_parse_jinja_template(template)
    assert sorted(vars) == ["notifications", "orders", "user"]

def test_disallowed_filter():
    with pytest.raises(ValidationError, match="Unsupported template feature"):
        validate_and_parse_jinja_template("{{ name|upper }}")

def test_function_call():
    with pytest.raises(ValidationError, match="Unsupported template feature"):
        validate_and_parse_jinja_template("{{ get_user() }}")

def test_set_statement():
    with pytest.raises(ValidationError, match="Unsupported template feature"):
        validate_and_parse_jinja_template("{% set x = 5 %}")

def test_using_loop_object():
    template = "{% for item in items %}{{ loop.index }}{% endfor %}"
    with pytest.raises(ValidationError, match="The 'loop' variable is not allowed"):
        validate_and_parse_jinja_template(template)

def test_dynamic_index():
    template = "{{ items[i] }}"
    with pytest.raises(ValidationError, match="Dynamic indices using variables"):
        validate_and_parse_jinja_template(template)

def test_index_not_str_or_int():
    template = "{{ items[true] }}"
    with pytest.raises(ValidationError, match="Index must be a number or text string"):
        validate_and_parse_jinja_template(template)

def test_literal_const_expression():
    template = "{{ 5 }}"
    with pytest.raises(ValidationError, match="Literal values are not allowed"):
        validate_and_parse_jinja_template(template)

def test_filter_inside_loop():
    template = "{% for item in items %}{{ item|upper }}{% endfor %}"
    with pytest.raises(ValidationError, match="Unsupported template feature"):
        validate_and_parse_jinja_template(template)
