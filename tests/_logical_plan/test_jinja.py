from typing import Optional

import pytest

from fenic.core._logical_plan.jinja import (
    TypeRequirement,
    VariableNode,
    validate_and_parse_jinja_template,
)
from fenic.core.error import ValidationError


def assert_variable_node(
    node: VariableNode,
    expected_req: Optional[TypeRequirement],
    expected_children: Optional[dict[str, dict]] = None,
):
    assert node.requirement == expected_req
    expected_children = expected_children or {}
    assert set(node.children.keys()) == set(expected_children.keys())
    for child_name, child_expectations in expected_children.items():
        assert_variable_node(node.children[child_name], **child_expectations)

def test_variable_access():
    template = "{{ user }}"
    tree = validate_and_parse_jinja_template(template)
    assert len(tree.variables) == 1
    assert "user" in tree.variables
    assert_variable_node(tree.variables["user"], expected_req=None, expected_children={})


def test_struct_access():
    template = "{{ user.name }} {{ user['name'] }}"
    tree = validate_and_parse_jinja_template(template)
    assert len(tree.variables) == 1
    assert "user" in tree.variables
    assert_variable_node(
        tree.variables["user"],
        expected_req=TypeRequirement.STRUCT,
        expected_children={"name": {"expected_req": None, "expected_children": {}}},
    )

def test_array_access():
    template = "{{ items[0] }} {{ items[1] }}"
    tree = validate_and_parse_jinja_template(template)
    assert len(tree.variables) == 1
    assert "items" in tree.variables
    assert_variable_node(tree.variables["items"], expected_req=TypeRequirement.ARRAY, expected_children={"*": {"expected_req": None, "expected_children": {}}})

def test_for_loop():
    template = "{% for item in items %}{{ item }} {% else %} {{ fallback }} {% endfor %}"
    tree = validate_and_parse_jinja_template(template)
    assert len(tree.variables) == 2
    assert "items" in tree.variables
    assert_variable_node(
        tree.variables["items"],
        expected_req=TypeRequirement.ARRAY,
        expected_children={"*": {"expected_req": None, "expected_children": {}}},
    )
    assert "fallback" in tree.variables
    assert_variable_node(tree.variables["fallback"], expected_req=None, expected_children={})

    template = """
    {% for user in users %}
      {% for order in user.orders %}
          Order ID: {{ order.id }}
      {% endfor %}
    {% endfor %}
    """
    tree = validate_and_parse_jinja_template(template)
    assert len(tree.variables) == 1
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
                                    "id": {
                                        "expected_req": None,
                                        "expected_children": {}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    )


def test_conditional():
    template = """
    {% if x %}
        {{ x }}
    {% elif y %}
        {{ y }}
    {% else %}
        {{ z }}
    {% endif %}
    """
    tree = validate_and_parse_jinja_template(template)
    assert len(tree.variables) == 3
    assert "x" in tree.variables
    assert_variable_node(tree.variables["x"], expected_req=TypeRequirement.BOOLEAN, expected_children={})
    assert "y" in tree.variables
    assert_variable_node(tree.variables["y"], expected_req=TypeRequirement.BOOLEAN, expected_children={})
    assert "z" in tree.variables
    assert_variable_node(tree.variables["z"], expected_req=None, expected_children={})

    template = """
    {% if x %}
        {{ x }}
    {% else %}
        {% if y %}
            {{ y }}
        {% else %}
            {{ z }}
        {% endif %}
    {% endif %}
    """
    tree = validate_and_parse_jinja_template(template)
    assert len(tree.variables) == 3
    assert "x" in tree.variables
    assert_variable_node(tree.variables["x"], expected_req=TypeRequirement.BOOLEAN, expected_children={})
    assert "y" in tree.variables
    assert_variable_node(tree.variables["y"], expected_req=TypeRequirement.BOOLEAN, expected_children={})
    assert "z" in tree.variables
    assert_variable_node(tree.variables["z"], expected_req=None, expected_children={})


def test_complex():
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
    assert len(tree.variables) == 3
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


def test_mixed_nesting_object_access():
    """Mix of array access and struct access should work"""
    template = "{{ data[0].users[1].profile.name }}"
    tree = validate_and_parse_jinja_template(template)
    assert len(tree.variables) == 1

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
    assert validate_and_parse_jinja_template("{% for x in y %}{% endfor %}").variables == {}
    assert validate_and_parse_jinja_template("{% if x %}{% endif %}").variables == {}

def test_loop_variable_shadowing():
    """Nested loops can use same variable name (inner shadows outer)"""
    template = """
    {% for item in outer_items %}
      {{ item.outer_field }}
      {{ item.name }}
      {% for item in inner_items %}
        {{ item.inner_field }}
        {{ item.name }}
      {% endfor %}
      {{ item.inner_field_2 }}
    {% endfor %}
    """
    tree = validate_and_parse_jinja_template(template)
    assert len(tree.variables) == 2

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
                    "name": {"expected_req": None, "expected_children": {}},
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
                    "inner_field_2": {"expected_req": None, "expected_children": {}},
                    "name": {"expected_req": None, "expected_children": {}},
                },
            }
        },
    )

    template = """
    {% for item in stores %}
      {{ item.name }}
      {% for manager in item.managers %}
        {{ manager.first_name }}
      {% endfor %}
    {% endfor %}
    {% for item in products %}
      {{ item.name }}
      {% for item in item.reviews %}
        {{ item.rating }}
      {% endfor %}
      {{ item.author }}
    {% endfor %}
    {{ item.date }}
    {{ manager.last_name }}
    """
    tree = validate_and_parse_jinja_template(template)
    assert len(tree.variables) == 2
    assert_variable_node(
        tree.variables["stores"],
        expected_req=TypeRequirement.ARRAY,
        expected_children={
            "*": {
                "expected_req": TypeRequirement.STRUCT,
                "expected_children": {
                    "name": {
                        "expected_req": None,
                        "expected_children": {}
                    },
                    "managers": {
                        "expected_req": TypeRequirement.ARRAY,
                        "expected_children": {
                            "*": {
                                "expected_req": TypeRequirement.STRUCT,
                                "expected_children": {
                                    "first_name": {
                                        "expected_req": None,
                                        "expected_children": {}
                                    },
                                    "last_name": {
                                        "expected_req": None,
                                        "expected_children": {}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    )
    assert_variable_node(
        tree.variables["products"],
        expected_req=TypeRequirement.ARRAY,
        expected_children={
            "*": {
                "expected_req": TypeRequirement.STRUCT,
                "expected_children": {
                    "name": {
                        "expected_req": None,
                        "expected_children": {}
                    },
                    "reviews": {
                        "expected_req": TypeRequirement.ARRAY,
                        "expected_children": {
                            "*": {
                                "expected_req": TypeRequirement.STRUCT,
                                "expected_children": {
                                    "rating": {
                                        "expected_req": None,
                                        "expected_children": {}
                                    },
                                    "author": {
                                        "expected_req": None,
                                        "expected_children": {}
                                    },
                                    "date": {
                                        "expected_req": None,
                                        "expected_children": {}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    )


def test_array_with_both_loop_variable_and_index_access():
    template = """
    {% for item in products %}
    Product: {{ item.name }}
    {% endfor %}
    {{ products[0] }}
    {{ item.name.first }}
    """

    tree = validate_and_parse_jinja_template(template)
    assert len(tree.variables) == 1
    assert_variable_node(
        tree.variables["products"],
        expected_req=TypeRequirement.ARRAY,
        expected_children={
            "*": {
                "expected_req": TypeRequirement.STRUCT,
                "expected_children": {
                    "name": {
                        "expected_req": TypeRequirement.STRUCT,
                        "expected_children": {
                            "first": {
                                "expected_req": None,
                                "expected_children": {}
                            }
                        }
                    }
                }
            }
        },
    )

@pytest.mark.parametrize("template,expected_error", [
    ("{{ name|upper }}", "Unsupported Jinja template syntax"),
    ("{{ get_user() }}", "Unsupported Jinja template syntax"),
    ("{% set x = 5 %}", "Unsupported Jinja template syntax"),
    ("{% for item in items %}{{ loop.index }}{% endfor %}", "Unsupported Jinja template syntax"),
    ("{{ items[i] }}", "Unsupported Jinja template syntax"),
    ("{{ items[true] }}", "Unsupported Jinja template syntax"),
    ("{{ 5 }}", "Unsupported Jinja template syntax"),
    ("{% for item in items %}{{ item|upper }}{% endfor %}", "Unsupported Jinja template syntax"),
])

def test_unsupported_syntax(template, expected_error):
    with pytest.raises(ValidationError, match=expected_error):
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
