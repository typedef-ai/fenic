import re
from typing import Optional

import pytest

from fenic.core._logical_plan.jinja import (
    TypeRequirement,
    VariableNode,
    VariableTree,
)
from fenic.core.error import TypeMismatchError, ValidationError
from fenic.core.types import (
    ArrayType,
    BooleanType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


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
    tree = VariableTree.from_jinja_template(template)
    assert len(tree.variables) == 1
    assert "user" in tree.variables
    assert_variable_node(tree.variables["user"], expected_req=None, expected_children={})


def test_struct_access():
    template = "{{ user.name }} {{ user['name'] }}"
    tree = VariableTree.from_jinja_template(template)
    assert len(tree.variables) == 1
    assert "user" in tree.variables
    assert_variable_node(
        tree.variables["user"],
        expected_req=TypeRequirement.STRUCT,
        expected_children={"name": {"expected_req": None, "expected_children": {}}},
    )

def test_array_access():
    template = "{{ items[0] }} {{ items[1] }}"
    tree = VariableTree.from_jinja_template(template)
    assert len(tree.variables) == 1
    assert "items" in tree.variables
    assert_variable_node(tree.variables["items"], expected_req=TypeRequirement.ARRAY, expected_children={"*": {"expected_req": None, "expected_children": {}}})

def test_for_loop():
    template = "{% for item in items %}{{ item }} {% else %} {{ fallback }} {% endfor %}"
    tree = VariableTree.from_jinja_template(template)
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
    tree = VariableTree.from_jinja_template(template)
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
    tree = VariableTree.from_jinja_template(template)
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
    tree = VariableTree.from_jinja_template(template)
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
    tree = VariableTree.from_jinja_template(template)
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
    tree = VariableTree.from_jinja_template(template)
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

def test_jinja_template_with_dead_code():
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
    {% if should_greet %}
        {{ hello }}
    {% endif %}
    """

    tree = VariableTree.from_jinja_template(template)

    # Should only have hello and should_greet in the schema
    assert len(tree.variables) == 2
    assert "hello" in tree.variables
    assert_variable_node(tree.variables["hello"], expected_req=None, expected_children={})
    assert "should_greet" in tree.variables
    assert_variable_node(tree.variables["should_greet"], expected_req=TypeRequirement.BOOLEAN, expected_children={})

    # Also test simpler cases
    assert VariableTree.from_jinja_template("").variables == {}
    assert VariableTree.from_jinja_template("<!-- just comments -->").variables == {}
    assert VariableTree.from_jinja_template("{% for x in y %}{% endfor %}").variables == {}
    assert VariableTree.from_jinja_template("{% if x %}{% endif %}").variables == {}

    # Make sure that scoped leaked variables are not treated as dead code even if they are not used in loop body
    template = """
    {% for item in items %}
    {% endfor %}
    {{ item }}
    """
    tree = VariableTree.from_jinja_template(template)
    assert len(tree.variables) == 1
    assert "items" in tree.variables
    assert_variable_node(tree.variables["items"], expected_req=TypeRequirement.ARRAY, expected_children={"*": {"expected_req": None, "expected_children": {}}})

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
    tree = VariableTree.from_jinja_template(template)
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
    tree = VariableTree.from_jinja_template(template)
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

    tree = VariableTree.from_jinja_template(template)
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
        VariableTree.from_jinja_template(template)

def test_conflicting_type_requirements():
    template = """
    {% for user in users %}
      {{ users.name }}
    {% endfor %}
    """
    with pytest.raises(ValidationError, match="Variable used inconsistently across the jinja template"):
        VariableTree.from_jinja_template(template)

    # Conflict between ARRAY (index 0) and STRUCT (string key)
    template2 = """
    {{ data[0] }}
    {{ data["foo"] }}
    """
    with pytest.raises(ValidationError, match="Variable used inconsistently across the jinja template"):
        VariableTree.from_jinja_template(template2)

def test_jinja_syntax_error():
    template = "{{ data[0] }"
    with pytest.raises(ValidationError, match="Jinja template syntax error on line 1"):
        VariableTree.from_jinja_template(template)


def test_array_indexing_requires_array_type():
    template = "{{ data[0] }}"
    tree = VariableTree.from_jinja_template(template)
    tree.validate_jinja_variable("data", ArrayType(element_type=StringType))

    with pytest.raises(TypeMismatchError, match="Column 'data' used in Jinja template must be an ArrayType, but found StringType. This variable is used in a for-loop and must be an array column."):
        tree.validate_jinja_variable("data", StringType)

def test_for_loop_iteration_requires_array_type():
    template = "{% for item in items %}{{ item }}{% endfor %}"
    tree = VariableTree.from_jinja_template(template)
    tree.validate_jinja_variable("items", ArrayType(element_type=StringType))

    with pytest.raises(TypeMismatchError, match="Column 'items' used in Jinja template must be an ArrayType, but found StringType. This variable is used in a for-loop and must be an array column."):
        tree.validate_jinja_variable("items", StringType)

def test_field_access_requires_struct_type_and_valid_field():
    template = "{{ data.name }}"
    tree = VariableTree.from_jinja_template(template)
    tree.validate_jinja_variable("data", StructType(struct_fields=[StructField(name="name", data_type=StringType)]))

    with pytest.raises(TypeMismatchError, match=re.escape("Column 'data' used in Jinja template must be a StructType, but found StringType. This variable is accessed using field notation (e.g., data.fieldname) and must be a struct column.")):
        tree.validate_jinja_variable("data", StringType)

    template = "{{ data.invalid_field }}"
    tree = VariableTree.from_jinja_template(template)
    with pytest.raises(ValidationError, match=re.escape("Field 'invalid_field' in Jinja template does not exist in StructType at 'data'. Available StructFields: name.")):
        tree.validate_jinja_variable("data", StructType(struct_fields=[StructField(name="name", data_type=StringType)]))


def test_conditional_expression_requires_boolean_type():
    template = "{%if x %}{{ x }}{% endif %}"
    tree = VariableTree.from_jinja_template(template)
    tree.validate_jinja_variable("x", BooleanType)

    with pytest.raises(TypeMismatchError, match="Column 'x' used in Jinja template must be a BooleanType, but found StringType. This variable is used in a conditional expression and must evaluate to a boolean."):
        tree.validate_jinja_variable("x", StringType)

def test_complex_expression_against_datatypes():
    template = """
    {% for item in items %}
        {% if item.has_name %}
            {{ foo.bar }}
            {{ bar.baz }}
        {% endif %}
    {% endfor %}
    """
    tree = VariableTree.from_jinja_template(template)

    # Valid type assertions (should not raise)
    tree.validate_jinja_variable(
        "items", ArrayType(element_type=StructType(struct_fields=[StructField(name="has_name", data_type=BooleanType)]))
    )
    tree.validate_jinja_variable("foo", StructType(struct_fields=[StructField(name="bar", data_type=StringType)]))
    tree.validate_jinja_variable(
        "bar", StructType(struct_fields=[StructField(name="baz", data_type=IntegerType)])
    )

    # Validate that the error message for nested array access is correct
    with pytest.raises(TypeMismatchError, match=re.escape("Column 'items[*].has_name' used in Jinja template must be a BooleanType, but found StringType. This variable is used in a conditional expression and must evaluate to a boolean.")):
        tree.validate_jinja_variable("items", ArrayType(element_type=StructType(struct_fields=[StructField(name="has_name", data_type=StringType)])))
