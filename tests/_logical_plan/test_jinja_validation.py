import re
from typing import Optional

import pytest

from fenic.core._logical_plan.jinja_validation import (
    TypeRequirement,
    VariableNode,
    VariableTree,
)
from fenic.core.error import TypeMismatchError, ValidationError
from fenic.core.types import (
    ArrayType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


def test_constants_allowed():
    """Test that constants are allowed in templates"""
    template = """
    {{ "Hello World" }}
    {{ 42 }}
    {{ true }}
    {% if "static string" %}
        Always true
    {% endif %}
    """
    # Should not raise any validation errors
    tree = VariableTree.from_jinja_template(template)
    assert len(tree.variables) == 0  # No variables needed

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


def test_conditionals():
    # Basic conditional
    template = """
    {% if message %}
        {{ message }}
    {% endif %}
    """
    tree = VariableTree.from_jinja_template(template)
    assert len(tree.variables) == 1
    assert_variable_node(tree.variables["message"], expected_req=None, expected_children={})

    # Array in condition (empty vs non-empty)
    template = """
    {% if items %}
        Items found: {{ items[0] }}
    {% endif %}
    """
    tree = VariableTree.from_jinja_template(template)
    assert len(tree.variables) == 1
    assert_variable_node(
        tree.variables["items"],
        expected_req=TypeRequirement.ARRAY,
        expected_children={"*": {"expected_req": None, "expected_children": {}}}
    )

    # Struct in condition
    template = """
    {% if user %}
        Welcome {{ user.name }}
    {% endif %}
    {% if user.address %}
        {{ user }}
    {% endif %}
    """
    tree = VariableTree.from_jinja_template(template)
    assert len(tree.variables) == 1
    assert_variable_node(
        tree.variables["user"],
        expected_req=TypeRequirement.STRUCT,
        expected_children={"name": {"expected_req": None, "expected_children": {}}, "address": {"expected_req": None, "expected_children": {}}}
    )

    # Nested if-else
    template = """
    {% if user.active %}
        {% if user.premium %}
            {{ premium_message }}
        {% else %}
            {{ standard_message }}
        {% endif %}
    {% else %}
        {{ inactive_message }}
    {% endif %}
    """
    tree = VariableTree.from_jinja_template(template)
    assert len(tree.variables) == 4
    assert_variable_node(
        tree.variables["user"],
        expected_req=TypeRequirement.STRUCT,
        expected_children={
            "active": {"expected_req": None, "expected_children": {}},
            "premium": {"expected_req": None, "expected_children": {}}
        }
    )
    assert_variable_node(tree.variables["premium_message"], expected_req=None, expected_children={})
    assert_variable_node(tree.variables["standard_message"], expected_req=None, expected_children={})
    assert_variable_node(tree.variables["inactive_message"], expected_req=None, expected_children={})

    # If-elif-else
    template = """
    {% if user.is_admin %}
        {{ admin_dashboard }}
    {% elif user.is_moderator %}
        {{ mod_dashboard }}
    {% else %}
        {{ basic_content }}
    {% endif %}
    """
    tree = VariableTree.from_jinja_template(template)
    assert len(tree.variables) == 4
    assert_variable_node(
        tree.variables["user"],
        expected_req=TypeRequirement.STRUCT,
        expected_children={
            "is_admin": {"expected_req": None, "expected_children": {}},
            "is_moderator": {"expected_req": None, "expected_children": {}},
        }
    )
    assert_variable_node(tree.variables["admin_dashboard"], expected_req=None, expected_children={})
    assert_variable_node(tree.variables["mod_dashboard"], expected_req=None, expected_children={})
    assert_variable_node(tree.variables["basic_content"], expected_req=None, expected_children={})


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
            "premium": {"expected_req": None, "expected_children": {}},
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
    # Complex nested structure with whitespace trimming - produces NO output
    # The {%- -%} syntax strips all whitespace, making these loops/conditions empty
    # Therefore, all users/admin related variables are dead code
    template = """
{%- for user in users -%}
{%- if user.active -%}
{%- for order in user.orders -%}
{%- if order.paid -%}
{%- for item in order.items -%}
{%- if item.available -%}
{%- endif -%}
{%- endfor -%}
{%- endif -%}
{%- endfor -%}
{%- endif -%}
{%- endfor -%}

{%- if admin.logged_in -%}
{%- for notification in admin.notifications -%}
{%- endfor -%}
{%- endif -%}
{% if should_greet %}
    {{ hello }}
{% endif %}
    """

    tree = VariableTree.from_jinja_template(template)

    # Only variables that affect output are included:
    # - should_greet: controls whether output appears
    # - hello: is actually output
    # All the users/admin variables are eliminated as dead code
    assert len(tree.variables) == 2
    assert "hello" in tree.variables
    assert_variable_node(tree.variables["hello"], expected_req=None, expected_children={})
    assert "should_greet" in tree.variables
    assert_variable_node(tree.variables["should_greet"], expected_req=None, expected_children={})

    # Edge case: completely empty templates have no variables
    assert VariableTree.from_jinja_template("").variables == {}

    # No variables in template.
    assert VariableTree.from_jinja_template("<!-- just comments -->").variables == {}

    # Empty for loop (no whitespace between tags) produces no output
    # Therefore 'y' is dead code and not included in schema
    assert VariableTree.from_jinja_template("{% for x in y %}{% endfor %}").variables == {}

    # Empty if block produces no output, so 'x' is dead code
    assert VariableTree.from_jinja_template("{% if x %}{% endif %}").variables == {}

    # Loop with trimmed whitespace followed by variable access
    # The loop produces no output (trimmed), but 'item' is used outside the loop
    # Note: 'items' is dead code (loop produces no output), but 'item' is live code
    template = """
    {%- for item in items -%}
    {%- endfor -%}
    {{ item }}
    """
    tree = VariableTree.from_jinja_template(template)
    assert len(tree.variables) == 1
    assert "item" in tree.variables  # 'item' is a regular variable here, not the loop variable
    assert_variable_node(tree.variables["item"], expected_req=None, expected_children={})

    # Control structures WITH whitespace inside produce output!
    # The newline between {% for %} and {% endfor %} is output for each iteration
    # The newlines and {{ 'bar' }} between {% if %} and {% endif %} is output when condition is true
    # Therefore both 'items' and 'foo' must be in the schema
    template = """
    {% for item in items %}
    {% endfor %}
    {% if foo %}
    {{ 'bar' }}
    {% endif %}
    """
    tree = VariableTree.from_jinja_template(template)
    assert len(tree.variables) == 2
    assert "foo" in tree.variables
    assert_variable_node(tree.variables["foo"], expected_req=None, expected_children={})
    assert "items" in tree.variables
    assert_variable_node(tree.variables["items"], expected_req=TypeRequirement.ARRAY, expected_children={"*": {"expected_req": None, "expected_children": {}}})

def test_iter_variable_shadowing():
    """Nested loops can use same variable name (inner shadows outer)"""
    template = """
    {% for item in outer_items %}
      {{ item.outer_field }}
      {{ item.name }}
      {% for item in inner_items %}
        {{ item.inner_field }}
        {{ item.name }}
      {% endfor %}
      {{ item.outer_field_2 }}
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
                    "outer_field_2": {"expected_req": None, "expected_children": {}},
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
      {{ item.category }}
    {% endfor %}
    {{ item.date }}
    {{ manager.last_name }}
    """
    tree = VariableTree.from_jinja_template(template)
    assert len(tree.variables) == 4
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
                    "category": {
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
                                }
                            }
                        }
                    }
                }
            }
        }
    )

    assert_variable_node(tree.variables["item"], expected_req=TypeRequirement.STRUCT, expected_children={
        "date": {"expected_req": None, "expected_children": {}},
    })
    assert_variable_node(tree.variables["manager"], expected_req=TypeRequirement.STRUCT, expected_children={
        "last_name": {"expected_req": None, "expected_children": {}},
    })


def test_array_with_both_iter_variable_and_index_access():
    template = """
    {% for item in products %}
    Product: {{ item.name }}
    {% endfor %}
    {{ products[0] }}
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
                        "expected_req": None,
                        "expected_children": {}
                    }
                }
            }
        },
    )

def test_loop_variable_outside_loop():
    """Test that 'loop' outside a for loop is treated as a regular variable"""
    template = """
    {{ loop }}
    {{ loop.counter }}
    {% for item in items %}
        {{ loop.index }}
    {% endfor %}
    {{ loop.value }}
    """
    tree = VariableTree.from_jinja_template(template)
    assert len(tree.variables) == 2

    # 'loop' should be in variables as a struct (used outside loop)
    assert "loop" in tree.variables
    assert_variable_node(
        tree.variables["loop"],
        expected_req=TypeRequirement.STRUCT,
        expected_children={
            "counter": {"expected_req": None, "expected_children": {}},
            "value": {"expected_req": None, "expected_children": {}}
        }
    )

    # 'items' from the for loop
    assert "items" in tree.variables
    assert_variable_node(
        tree.variables["items"],
        expected_req=TypeRequirement.ARRAY,
        expected_children={"*": {"expected_req": None, "expected_children": {}}}
    )


def test_invalid_loop_attribute():
    """Test that invalid loop attributes raise validation errors"""
    template = """
    {% for item in items %}
        {{ loop.foo }}
    {% endfor %}
    """
    with pytest.raises(ValidationError, match="Invalid loop attribute 'foo'"):
        VariableTree.from_jinja_template(template)


def test_nested_loops_with_loop_variables():
    """Test loop variables in nested loops"""
    template = """
    {% for category in categories %}
        Category {{ loop.index }}: {{ category.name }}
        {% for item in category.items %}
            Item {{ loop.index }} (previous item: {{ loop.previtem }})
        {% endfor %}
    {% endfor %}
    """
    tree = VariableTree.from_jinja_template(template)
    assert len(tree.variables) == 1
    assert_variable_node(
        tree.variables["categories"],
        expected_req=TypeRequirement.ARRAY,
        expected_children={
            "*": {
                "expected_req": TypeRequirement.STRUCT,
                "expected_children": {
                    "name": {"expected_req": None, "expected_children": {}},
                    "items": {
                        "expected_req": TypeRequirement.ARRAY,
                        "expected_children": {"*": {"expected_req": None, "expected_children": {}}}
                    }
                }
            }
        }
    )


@pytest.mark.parametrize("template,expected_error", [
    ("{{ name|upper }}", "Unsupported Jinja template syntax"),
    ("{{ get_user() }}", "Unsupported Jinja template syntax"),
    ("{% set x = 5 %}", "Unsupported Jinja template syntax"),
    ("{{ items[i] }}", "Unsupported Jinja template syntax"),
    ("{{ items[true] }}", "Unsupported Jinja template syntax"),
    ("{% for item in items %}{{ item|upper }}{% endfor %}", "Unsupported Jinja template syntax"),
    ('{% if status == "active" %}{{ message }}{% endif %}', "Unsupported Jinja template syntax"),
    ('{% if count > 5 %}{{ message }}{% endif %}', "Unsupported Jinja template syntax"),
    ('{% if price <= 100 %}{{ message }}{% endif %}', "Unsupported Jinja template syntax"),
    ('{% if name != "admin" %}{{ message }}{% endif %}', "Unsupported Jinja template syntax"),
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
    {{ data["0"] }}
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

def test_complex_expression_validation():
    template = """
    {% for product in products %}
        {% for review in product.reviews %}
            {% if review.name %}
                {{ foo.bar }}
                {{ bar.baz }}
                {{ review.name }}
            {% endif %}
        {% endfor %}
    {% endfor %}
    """
    tree = VariableTree.from_jinja_template(template)

    # Valid type assertions (should not raise)
    tree.validate_jinja_variable(
        "products",
        ArrayType(
            element_type=StructType(
                struct_fields=[
                    StructField(
                        name="reviews",
                        data_type=ArrayType(
                            element_type=StructType(
                                struct_fields=[
                                    StructField(name="name", data_type=StringType)
                                ]
                            )
                        )
                    )
                ]
            )
        )
    )
    tree.validate_jinja_variable(
        "foo",
        StructType(
            struct_fields=[StructField(name="bar", data_type=StringType)]
        )
    )
    tree.validate_jinja_variable(
        "bar",
        StructType(
            struct_fields=[StructField(name="baz", data_type=IntegerType)]
        )
    )

    # Validate that the error message for nested array access is correct
    with pytest.raises(
        ValidationError,
        match=re.escape(
            "Field 'name' in Jinja template does not exist in StructType at 'products[*].reviews[*]'. Available StructFields: first_name. Please check for typos or confirm the struct schema."
        )
    ):
        tree.validate_jinja_variable(
            "products",
            ArrayType(
                element_type=StructType(
                    struct_fields=[
                        StructField(
                            name="reviews",
                            data_type=ArrayType(
                                element_type=StructType(
                                    struct_fields=[
                                        StructField(name="first_name", data_type=StringType)
                                    ]
                                )
                            )
                        )
                    ]
                )
            )
        )
