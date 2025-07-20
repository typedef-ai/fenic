#!/usr/bin/env python3
"""
Simple Jinja2 Demo - Basic Usage
"""

from jinja2 import Template

# # Simple template with various Jinja2 features
# template_string = """
# Hello {{ name }}!

# Your Information:
# - Age: {{ age }}
# - Email: {{ email|default('Not provided') }}
# - Status: {{ 'Active' if is_active else 'Inactive' }}

# Your Hobbies:
# {% for hobby in hobbies %}
#   {{ loop.index }}. {{ hobby|upper }}
# {% endfor %}

# Friends List:
# {% for friend in friends %}
#   - {{ friend.name }} ({{ friend.age }} years old)
#   {% if friend.age >= 18 %}
#     * Can vote!
#   {% else %}
#     * Too young to vote
#   {% endif %}
# {% endfor %}

# Total friends: {{ friends|length }}

# Price List:
# {% for item, price in prices.items() %}
#   - {{ item }}: ${{ "%.2f"|format(price) }}
# {% endfor %}
# SQUIDWARD!!!!
# {{ item }} {{ price }}

# Fun Facts:
# {% if age > 30 %}
# You're over 30!
# {% elif age > 20 %}
# You're in your twenties!
# {% else %}
# You're quite young!
# {% endif %}

# {% set greeting = "Have a great day" %}
# {{ greeting }}, {{ name }}!
# """

# # Create template
# template = Template(template_string)

# # Data to pass to template
# data = {
#     'name': 'Alice',
#     'age': 25,
#     'email': 'alice@example.com',
#     'is_active': True,
#     'hobbies': ['reading', 'hiking', 'coding'],
#     'friends': [
#         {'name': 'Bob', 'age': 30},
#         {'name': 'Charlie', 'age': 17},
#         {'name': 'Diana', 'age': 22}
#     ],
#     'prices': {
#         'Coffee': 3.50,
#         'Sandwich': 8.99,
#         'Salad': 7.25
#     }
# }

# # Render template
# output = template.render(**data)
# print(output)

# # Example with missing data to show default filter
# print("\n" + "="*50 + "\n")
# print("Example with missing email:")
# minimal_data = {
#     'name': 'John',
#     'age': 15,
#     'is_active': False,
#     'hobbies': [],
#     'friends': [],
#     'prices': {}
# }
# print(template.render(**minimal_data))

template_string = """
    {% for item in outer_items %}
    {{ item.outer_field }}
    {{ loop }}
    {% endfor %}
    {{ loop }}

"""

template = Template(template_string)

data = {
    "outer_items": [
        {"outer_field": "foo1", "name": "bar1", "outer_field_2": "baz1"},
        {"outer_field": "foo2", "name": "bar2", "outer_field_2": "baz2"},
    ],
    "inner_items": [
        {"inner_field": "jim", "name": "jimmy", "outer_field_2": "fuck"},
    ],
    "loop": "yo baby",
}

print(template.render(**data))

"""
static analysis
1. no dynamic array access (this is probably ok actually because we cant bounds check constant array access
- we could determine whether the index is string or integer and then validate whether whats indexed is array or struct
2. no variables
3. loop variables ok
4. constants ok
5. truthy ness (no limit about the booleans!)
6. no dictionarty iteration
7. treat scope leakage as new variable
"""
