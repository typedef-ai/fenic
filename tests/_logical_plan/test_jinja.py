import pytest

from fenic.core._logical_plan.jinja import (
    TypeRequirement,
    VariableNode,
    validate_and_parse_jinja_template,
)
from fenic.core.error import ValidationError


def assert_variable_node(
    node: VariableNode,
    expected_req: TypeRequirement | None,
    expected_children: dict[str, dict] | None = None,
):
    assert node.requirement == expected_req
    expected_children = expected_children or {}
    assert set(node.children.keys()) == set(expected_children.keys())
    for child_name, child_expectations in expected_children.items():
        assert_variable_node(node.children[child_name], **child_expectations)

def test_struct_access():
    template = "{{ user.name }}"
    tree = validate_and_parse_jinja_template(template)
    assert "user" in tree.variables
    assert_variable_node(
        tree.variables["user"],
        expected_req=TypeRequirement.STRUCT,
        expected_children={"name": {"expected_req": None, "expected_children": {}}},
    )

def test_static_index_access():
    template = "{{ items[0] }}"
    tree = validate_and_parse_jinja_template(template)
    assert "items" in tree.variables
    assert_variable_node(tree.variables["items"], expected_req=TypeRequirement.ARRAY, expected_children={"*": {"expected_req": None, "expected_children": {}}})

def test_for_loop_with_attr_access():
    template = "{% for item in items %}{{ item.name }}{% endfor %}"
    tree = validate_and_parse_jinja_template(template)
    assert "items" in tree.variables
    assert_variable_node(
        tree.variables["items"],
        expected_req=TypeRequirement.ARRAY,
        expected_children={"*": {"expected_req": TypeRequirement.STRUCT, "expected_children": {"name": {"expected_req": None, "expected_children": {}}}}},
    )

def test_multiple_variables():
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
    tree = validate_and_parse_jinja_template(template)

    # user should be STRUCT with children
    assert "user" in tree.variables
    assert_variable_node(
        tree.variables["user"],
        expected_req=TypeRequirement.STRUCT,
        expected_children={
            "name": {"expected_req": None, "expected_children": {}},
            "premium": {"expected_req": TypeRequirement.BOOLEAN, "expected_children": {}},
            "metadata": {
                "expected_req": TypeRequirement.STRUCT,
                "expected_children": {"start_date": {"expected_req": None, "expected_children": {}}},
            },
        },
    )

    # orders is ARRAY, children are STRUCT
    assert "orders" in tree.variables
    assert_variable_node(
        tree.variables["orders"],
        expected_req=TypeRequirement.ARRAY,
        expected_children={
            "*": {
                "expected_req": TypeRequirement.STRUCT,
                "expected_children": {
                    "id": {"expected_req": None, "expected_children": {}},
                    "total": {"expected_req": None, "expected_children": {}},
                    "items": {
                        "expected_req": TypeRequirement.ARRAY,
                        "expected_children": {"*": {"expected_req": None, "expected_children": {}}},
                    },
                },
            }
        },
    )

    # notifications is ARRAY leaf
    assert "notifications" in tree.variables
    assert_variable_node(tree.variables["notifications"], expected_req=TypeRequirement.ARRAY, expected_children={"*": {"expected_req": None, "expected_children": {}}})

def test_nested_loops_with_conditional():
    template = """
    {% for user in users %}
      {% for order in user.orders %}
        {% if order.paid %}
          Order ID: {{ order.id }}
        {% endif %}
      {% endfor %}
    {% endfor %}
    """
    tree = validate_and_parse_jinja_template(template)

    # users should be ARRAY with child '*'
    assert "users" in tree.variables
    assert_variable_node(
        tree.variables["users"],
        expected_req=TypeRequirement.ARRAY,
        expected_children={
            "*": {
                "expected_req": TypeRequirement.STRUCT,
                "expected_children": {
                    "orders": {
                        "expected_req": TypeRequirement.ARRAY,
                        "expected_children": {
                            "*": {
                                "expected_req": TypeRequirement.STRUCT,
                                "expected_children": {
                                    "paid": {"expected_req": TypeRequirement.BOOLEAN, "expected_children": {}},
                                    "id": {"expected_req": None, "expected_children": {}},
                                },
                            }
                        },
                    }
                },
            }
        },
    )

def test_mixed_nesting_object_access():
    """Mix of array access and struct access should work"""
    template = "{{ data[0].users[1].profile.name }}"
    tree = validate_and_parse_jinja_template(template)

    assert "data" in tree.variables
    assert_variable_node(
        tree.variables["data"],
        expected_req=TypeRequirement.ARRAY,
        expected_children={
            "*": {
                "expected_req": TypeRequirement.STRUCT,
                "expected_children": {
                    "users": {
                        "expected_req": TypeRequirement.ARRAY,
                        "expected_children": {
                            "*": {
                                "expected_req": TypeRequirement.STRUCT,
                                "expected_children": {
                                    "profile": {
                                        "expected_req": TypeRequirement.STRUCT,
                                        "expected_children": {
                                            "name": {"expected_req": None, "expected_children": {}}
                                        },
                                    }
                                },
                            }
                        },
                    }
                },
            }
        },
    )

def test_jinja_template_with_no_output():
    """Templates with no output should return empty schema"""

    # Complex template with loops, conditions, but no actual output
    template = """
    {% for user in users %}
        {% if user.active %}
            {% for order in user.orders %}
                {% if order.paid %}
                    {% for item in order.items %}
                        {% if item.available %}
                        {% endif %}
                    {% endfor %}
                {% endif %}
            {% endfor %}
        {% endif %}
    {% endfor %}

    {% if admin.logged_in %}
        <!-- Another condition with no output -->
        {% for notification in admin.notifications %}
            <!-- Nested loop, still no output -->
        {% endfor %}
    {% endif %}
    """

    tree = validate_and_parse_jinja_template(template)

    # Should be completely empty because there is no output
    assert tree.variables == {}

    # Also test simpler cases
    assert validate_and_parse_jinja_template("").variables == {}
    assert validate_and_parse_jinja_template("<!-- just comments -->").variables == {}
    assert validate_and_parse_jinja_template("<h1>Static HTML</h1>").variables == {}
    assert validate_and_parse_jinja_template("{% for x in y %}{% endfor %}").variables == {}

def test_loop_variable_shadowing():
    """Nested loops can use same variable name (inner shadows outer)"""
    template = """
    {% for item in outer_items %}
      {{ item.outer_field }}
      {% for item in inner_items %}
        {{ item.inner_field }}
      {% endfor %}
      {{ item.outer_field_2 }}
    {% endfor %}
    """
    tree = validate_and_parse_jinja_template(template)

    # Should have both outer_items and inner_items in schema
    assert "outer_items" in tree.variables
    assert "inner_items" in tree.variables

    # outer_items should have structure for outer fields
    assert_variable_node(
        tree.variables["outer_items"],
        expected_req=TypeRequirement.ARRAY,
        expected_children={
            "*": {
                "expected_req": TypeRequirement.STRUCT,
                "expected_children": {
                    "outer_field": {"expected_req": None, "expected_children": {}},
                    "outer_field_2": {"expected_req": None, "expected_children": {}},
                },
            }
        },
    )

    # inner_items should have structure for inner fields
    assert_variable_node(
        tree.variables["inner_items"],
        expected_req=TypeRequirement.ARRAY,
        expected_children={
            "*": {
                "expected_req": TypeRequirement.STRUCT,
                "expected_children": {
                    "inner_field": {"expected_req": None, "expected_children": {}},
                },
            }
        },
    )

    template = """
    {% for item in outer_items %}
      {{ item.name }}
      {% for item in inner_items %}
        {{ item.name }}
      {% endfor %}
    {% endfor %}
    """
    tree = validate_and_parse_jinja_template(template)

    # Both arrays should have items with 'name' field
    assert "outer_items" in tree.variables
    assert "inner_items" in tree.variables

    # Both should have name field
    assert "name" in tree.variables["outer_items"].children["*"].children
    assert "name" in tree.variables["inner_items"].children["*"].children


def test_disallowed_filter():
    with pytest.raises(ValidationError, match="Unsupported Jinja template syntax on line"):
        validate_and_parse_jinja_template("{{ name|upper }}")

def test_function_call():
    with pytest.raises(ValidationError, match="Unsupported Jinja template syntax on line"):
        validate_and_parse_jinja_template("{{ get_user() }}")

def test_set_statement():
    with pytest.raises(ValidationError, match="Unsupported Jinja template syntax on line"):
        validate_and_parse_jinja_template("{% set x = 5 %}")

def test_using_loop_object():
    template = "{% for item in items %}{{ loop.index }}{% endfor %}"
    with pytest.raises(ValidationError, match= "Unsupported Jinja template syntax on line"):
        validate_and_parse_jinja_template(template)

def test_dynamic_index():
    template = "{{ items[i] }}"
    with pytest.raises(ValidationError, match="Unsupported Jinja template syntax on line"):
        validate_and_parse_jinja_template(template)

def test_index_not_str_or_int():
    template = "{{ items[true] }}"
    with pytest.raises(ValidationError, match="Unsupported Jinja template syntax on line"):
        validate_and_parse_jinja_template(template)

def test_literal_const_expression():
    template = "{{ 5 }}"
    with pytest.raises(ValidationError, match="Unsupported Jinja template syntax on line"):
        validate_and_parse_jinja_template(template)

def test_filter_inside_loop():
    template = "{% for item in items %}{{ item|upper }}{% endfor %}"
    with pytest.raises(ValidationError, match="Unsupported Jinja template syntax on line"):
        validate_and_parse_jinja_template(template)

def test_conflicting_type_requirements():
    template = """
    {% for user in users %}
      {{ users.name }}
    {% endfor %}
    """
    with pytest.raises(ValidationError, match="Variable used inconsistently across the jinja template"):
        validate_and_parse_jinja_template(template)

    # Conflict between ARRAY (index 0) and STRUCT (string key)
    template2 = """
    {{ data[0] }}
    {{ data["foo"] }}
    """
    with pytest.raises(ValidationError, match="Variable used inconsistently across the jinja template"):
        validate_and_parse_jinja_template(template2)

def test_jinja_syntax_error():
    template = "{{ data[0] }"
    with pytest.raises(ValidationError, match="Jinja template syntax error on line 1"):
        validate_and_parse_jinja_template(template)
