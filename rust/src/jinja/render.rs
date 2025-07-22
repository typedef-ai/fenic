use minijinja::Environment;
use polars::prelude::*;
use std::collections::BTreeMap;

use crate::arrow_scalar_extractor::ArrowScalarConverter;

/// Core implementation of Jinja template rendering
pub fn render(inputs: &[Series], template: &str) -> PolarsResult<Series> {
    // Create minijinja environment and compile template
    let env = Environment::new();
    let template = env.template_from_str(template).map_err(|e| {
        PolarsError::ComputeError(format!("Template compilation failed: {}", e).into())
    })?;

    // Extract the struct column from inputs
    let struct_series = &inputs[0];
    let struct_array = struct_series
        .struct_()
        .map_err(|e| PolarsError::ComputeError(format!("Expected struct input: {}", e).into()))?;

    // Get all field names from the struct
    let field_names: Vec<String> = struct_array
        .struct_fields()
        .iter()
        .map(|s| s.name().to_string())
        .collect();

    // Prepare output string builder
    let mut builder = StringChunkedBuilder::new("jinja".into(), struct_series.len());
    let converter = ArrowScalarConverter;

    // Process each row
    for row_idx in 0..struct_series.len() {
        // Build context for this row
        let mut ctx = BTreeMap::new();

        // Extract values for all fields
        for field_name in &field_names {
            match struct_array.field_by_name(field_name) {
                Ok(field_series) => {
                    // Handle multiple chunks if necessary
                    let chunk_idx = 0; // For now, assuming single chunk
                    match converter.to_jinja(field_series.chunks()[chunk_idx].as_ref(), row_idx) {
                        Ok(value) => {
                            ctx.insert(field_name.clone(), value);
                        }
                        Err(e) => {
                            return Err(PolarsError::ComputeError(
                                format!(
                                    "Failed to convert field {} to JinjaValue: {}",
                                    field_name, e
                                )
                                .into(),
                            ));
                        }
                    }
                }
                Err(e) => {
                    return Err(PolarsError::ComputeError(
                        format!("Field {} not found in struct: {}", field_name, e).into(),
                    ));
                }
            }
        }

        // Render template with context
        match template.render(&ctx) {
            Ok(rendered) => builder.append_value(&rendered),
            Err(e) => {
                return Err(PolarsError::ComputeError(
                    format!("Template rendering failed: {}", e).into(),
                ));
            }
        }
    }

    Ok(builder.finish().into_series())
}

#[cfg(test)]
mod tests {
    use super::*;
    use indoc::indoc;

    #[test]
    fn test_jinja_render_invalid_template() {
        let name_series = Series::new("name".into(), vec![Some("Alice"), Some("Bob"), None]);
        let age_series = Series::new("age".into(), vec![Some(25i32), Some(30i32), Some(35i32)]);

        let struct_series =
            StructChunked::from_series("test_struct".into(), 3, [name_series, age_series].iter())
                .unwrap()
                .into_series();
        let inputs = vec![struct_series];

        let template = "{{ unclosed";

        let result = render(&inputs, template);
        assert!(result.is_err()); // Should fail due to invalid template
    }

    #[test]
    fn test_jinja_render_simple() {
        let name_series = Series::new("name".into(), vec![Some("Alice"), Some("Bob"), None]);
        let age_series = Series::new("age".into(), vec![Some(25i32), Some(30i32), Some(35i32)]);

        let struct_series =
            StructChunked::from_series("test_struct".into(), 3, [name_series, age_series].iter())
                .unwrap()
                .into_series();
        let inputs = vec![struct_series];

        let template = "Hello {{ name }}!";

        let result = render(&inputs, template).unwrap();
        let result_values: Vec<Option<&str>> = result.str().unwrap().into_iter().collect();

        assert_eq!(result_values[0], Some("Hello Alice!"));
        assert_eq!(result_values[1], Some("Hello Bob!"));
        assert_eq!(result_values[2], Some("Hello !")); // Null name renders as empty
    }

    #[test]
    fn test_jinja_render_multiple_variables() {
        let name_series = Series::new("name".into(), vec![Some("Alice"), Some("Bob"), None]);
        let age_series = Series::new("age".into(), vec![Some(25i32), Some(30i32), Some(35i32)]);

        let struct_series =
            StructChunked::from_series("test_struct".into(), 3, [name_series, age_series].iter())
                .unwrap()
                .into_series();
        let inputs = vec![struct_series];

        let template = "{{ name }} is {{ age }} years old";

        let result = render(&inputs, template).unwrap();
        let result_values: Vec<Option<&str>> = result.str().unwrap().into_iter().collect();

        assert_eq!(result_values[0], Some("Alice is 25 years old"));
        assert_eq!(result_values[1], Some("Bob is 30 years old"));
        assert_eq!(result_values[2], Some(" is 35 years old")); // null name renders as empty
    }

    #[test]
    fn test_jinja_render_truthiness() {
        // Test cases:
        // - premium_str: empty string → falsey
        // - premium_null: null → falsey
        // - premium_zero: 0 → falsey
        // - premium_true_str: "yes" → truthy
        // - premium_bool: bool values (true/false)

        use polars::prelude::*;

        let name_series = Series::new(
            "name".into(),
            vec!["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"],
        );

        let premium_str = Series::new(
            "premium".into(),
            vec![Some("yes"), Some(""), None, Some("yes"), Some("no"), None],
        );
        let premium_bool = Series::new(
            "premium".into(),
            vec![
                Some(true),
                Some(false),
                Some(false),
                Some(true),
                Some(true),
                Some(false),
            ],
        );
        let premium_zero = Series::new(
            "premium".into(),
            vec![Some(1), Some(0), Some(0), Some(1), Some(2), Some(0)],
        );

        // Use premium_str to simulate string-based truthiness
        let struct_str = StructChunked::from_series(
            "test_struct".into(),
            6,
            [name_series.clone(), premium_str.clone()].iter(),
        )
        .unwrap()
        .into_series();

        // Use premium_bool to simulate bool-based truthiness
        let struct_bool = StructChunked::from_series(
            "test_struct".into(),
            6,
            [name_series.clone(), premium_bool.clone()].iter(),
        )
        .unwrap()
        .into_series();

        // Use premium_zero to simulate numeric truthiness
        let struct_zero = StructChunked::from_series(
            "test_struct".into(),
            6,
            [name_series.clone(), premium_zero.clone()].iter(),
        )
        .unwrap()
        .into_series();

        let template = "Hello {{ name }}{% if premium %} (Premium) {% endif %}!";

        // Check string-based truthiness
        let result_str = render(&[struct_str], template).unwrap();
        let values_str: Vec<_> = result_str.str().unwrap().into_iter().collect();

        assert_eq!(
            values_str,
            vec![
                Some("Hello Alice (Premium) !"), // "yes" → truthy
                Some("Hello Bob!"),              // "" → falsey
                Some("Hello Charlie!"),          // None → falsey
                Some("Hello Diana (Premium) !"), // "yes" → truthy
                Some("Hello Eve (Premium) !"),   // "no" is non-empty → truthy
                Some("Hello Frank!")             // None → falsey
            ]
        );

        // Check boolean-based truthiness
        let result_bool = render(&[struct_bool], template).unwrap();
        let values_bool: Vec<_> = result_bool.str().unwrap().into_iter().collect();

        assert_eq!(
            values_bool,
            vec![
                Some("Hello Alice (Premium) !"),
                Some("Hello Bob!"),
                Some("Hello Charlie!"),
                Some("Hello Diana (Premium) !"),
                Some("Hello Eve (Premium) !"),
                Some("Hello Frank!")
            ]
        );

        // Check numeric-based truthiness
        let result_zero = render(&[struct_zero], template).unwrap();
        let values_zero: Vec<_> = result_zero.str().unwrap().into_iter().collect();

        assert_eq!(
            values_zero,
            vec![
                Some("Hello Alice (Premium) !"), // 1 → truthy
                Some("Hello Bob!"),              // 0 → falsey
                Some("Hello Charlie!"),          // 0 → falsey
                Some("Hello Diana (Premium) !"), // 1 → truthy
                Some("Hello Eve (Premium) !"),   // 2 → truthy
                Some("Hello Frank!")             // 0 → falsey
            ]
        );
    }

    #[test]
    fn test_jinja_render_with_loops() {
        // Create a struct with a list field
        let names_series = Series::new(
            "names".into(),
            vec![
                Some(Series::new("names".into(), vec!["Alice", "Bob", "Charlie"])),
                Some(Series::new("names".into(), vec!["David", "Eve"])),
                Some(Series::new("names".into(), Vec::<&str>::new())), // Empty list
            ],
        );
        let title_series = Series::new(
            "title".into(),
            vec![Some("Team A"), Some("Team B"), Some("Team C")],
        );

        let struct_series = StructChunked::from_series(
            "test_struct".into(),
            3,
            [names_series, title_series].iter(),
        )
        .unwrap()
        .into_series();

        let inputs = vec![struct_series];

        let template =
            "{{ title }}:{% for name in names %} {{ loop.index }}: {{ name }}{% endfor %}";

        let result = render(&inputs, template).unwrap();
        let result_values: Vec<Option<&str>> = result.str().unwrap().into_iter().collect();

        assert_eq!(result_values[0], Some("Team A: 1: Alice 2: Bob 3: Charlie"));
        assert_eq!(result_values[1], Some("Team B: 1: David 2: Eve"));
        assert_eq!(result_values[2], Some("Team C:")); // Empty list
    }

    #[test]
    fn test_jinja_render_array_access() {
        // Create a struct with list fields
        let colors_series = Series::new(
            "colors".into(),
            vec![
                Some(Series::new("colors".into(), vec!["red", "green", "blue"])),
                Some(Series::new("colors".into(), vec!["yellow", "purple"])),
                Some(Series::new("colors".into(), vec!["orange"])),
                None,
            ],
        );
        let numbers_series = Series::new(
            "numbers".into(),
            vec![
                Some(Series::new("numbers".into(), vec![10, 20, 30])),
                Some(Series::new("numbers".into(), vec![40, 50])),
                Some(Series::new("numbers".into(), vec![60])),
                None,
            ],
        );

        let struct_series = StructChunked::from_series(
            "test_struct".into(),
            4,
            [colors_series, numbers_series].iter(),
        )
        .unwrap()
        .into_series();

        let inputs = vec![struct_series];

        // Test hardcoded array index access
        let template = "First color: {{ colors[0] }}, Second number: {{ numbers[1] }}, Fifth number: {{ numbers[5] }}";

        let result = render(&inputs, template).unwrap();
        let result_values: Vec<Option<&str>> = result.str().unwrap().into_iter().collect();

        assert_eq!(
            result_values[0],
            Some("First color: red, Second number: 20, Fifth number: ")
        );
        assert_eq!(
            result_values[1],
            Some("First color: yellow, Second number: 50, Fifth number: ")
        );
        assert_eq!(
            result_values[2],
            Some("First color: orange, Second number: , Fifth number: ")
        );
        assert_eq!(
            result_values[3],
            Some("First color: , Second number: , Fifth number: ")
        );
    }

    #[test]
    fn test_jinja_render_struct_field_access() {
        // Create a struct with some null fields
        let name_series = Series::new(
            "name".into(),
            vec![Some("Alice"), None, Some("Charlie"), None],
        );
        let age_series = Series::new("age".into(), vec![Some(25), Some(30), None, None]);

        let struct_series =
            StructChunked::from_series("user".into(), 4, [name_series, age_series].iter())
                .unwrap()
                .into_series();

        let inputs = vec![
            StructChunked::from_series("user".into(), 4, [struct_series].iter())
                .unwrap()
                .into_series(),
        ];

        let template = "Name: {{ user.name }}, Age: {{ user['age'] }}";

        let result = render(&inputs, template).unwrap();
        let result_values: Vec<Option<&str>> = result.str().unwrap().into_iter().collect();

        assert_eq!(result_values[0], Some("Name: Alice, Age: 25"));
        assert_eq!(result_values[1], Some("Name: , Age: 30"));
        assert_eq!(result_values[2], Some("Name: Charlie, Age: "));
        assert_eq!(result_values[3], Some("Name: , Age: "));
    }

    #[test]
    fn test_jinja_render_complex_template() {
        // Create a struct with mixed types
        let items_series = Series::new(
            "items".into(),
            vec![
                Some(Series::new(
                    "items".into(),
                    vec!["apple", "banana", "cherry"],
                )),
                Some(Series::new("items".into(), vec!["book", "pen"])),
                Some(Series::new("items".into(), Vec::<&str>::new())),
            ],
        );
        let count_series = Series::new("count".into(), vec![Some(3), Some(2), Some(0)]);
        let active_series = Series::new("active".into(), vec![Some(true), Some(true), Some(false)]);

        let struct_series = StructChunked::from_series(
            "test_struct".into(),
            3,
            [items_series, count_series, active_series].iter(),
        )
        .unwrap()
        .into_series();

        let inputs = vec![struct_series];

        // Complex template with conditionals and loops - no extra indentation
        let template = indoc! {r#"
                {%- if active -%}
                Status: Active
                Items ({{ count }}):
                {%- for item in items %}
                - {{ item }}
                {%- endfor %}
                {%- else -%}
                Status: Inactive
                {%- endif -%}
                "#};

        let result = render(&inputs, template).unwrap();
        let result_values: Vec<Option<&str>> = result.str().unwrap().into_iter().collect();

        assert_eq!(
            result_values[0],
            Some(indoc! {"
            Status: Active
            Items (3):
            - apple
            - banana
            - cherry"})
        );
        assert_eq!(
            result_values[1],
            Some(indoc! {"
            Status: Active
            Items (2):
            - book
            - pen"})
        );
        assert_eq!(
            result_values[2],
            Some(indoc! {"
            Status: Inactive"})
        );
    }

    #[test]
    fn test_jinja_render_safe_array_access() {
        // Test what happens with out of bounds array access
        let items_series = Series::new(
            "items".into(),
            vec![
                Some(Series::new("items".into(), vec!["first", "second"])),
                Some(Series::new("items".into(), vec!["only"])),
            ],
        );

        let struct_series =
            StructChunked::from_series("test_struct".into(), 2, [items_series].iter())
                .unwrap()
                .into_series();

        let inputs = vec![struct_series];

        // This template tries to access index that might not exist
        let template = "Item 0: {{ items[0] }}, Item 1: {% if items[1] %}{{ items[1] }}{% else %}N/A{% endif %}";

        let result = render(&inputs, template).unwrap();
        let result_values: Vec<Option<&str>> = result.str().unwrap().into_iter().collect();

        assert_eq!(result_values[0], Some("Item 0: first, Item 1: second"));
        assert_eq!(result_values[1], Some("Item 0: only, Item 1: N/A"));
    }

    #[test]
    fn test_jinja_render_direct_struct_and_list_printing() {
        // Create nested structs and lists
        let colors_series = Series::new(
            "colors".into(),
            vec![
                Some(Series::new("colors".into(), vec!["red", "green", "blue"])),
                Some(Series::new("colors".into(), vec!["yellow"])),
            ],
        );

        let city_series = Series::new("city".into(), vec![Some("New York"), Some("London")]);
        let zip_series = Series::new("zip".into(), vec![Some("10001"), None]);

        let address_struct =
            StructChunked::from_series("address".into(), 2, [city_series, zip_series].iter())
                .unwrap()
                .into_series();

        let person_struct =
            StructChunked::from_series("test".into(), 2, [colors_series, address_struct].iter())
                .unwrap()
                .into_series();

        let inputs = vec![person_struct];

        // Test rendering structs and lists directly without field access
        let template = indoc! {"
                Colors list: {{ colors }}
                Address struct: {{ address }}
            "};

        let result = render(&inputs, template).unwrap();
        let result_values: Vec<Option<&str>> = result.str().unwrap().into_iter().collect();

        assert_eq!(
            result_values[0],
            Some("Colors list: [\"red\", \"green\", \"blue\"]\nAddress struct: {\"city\": \"New York\", \"zip\": \"10001\"}")
        );

        assert_eq!(
            result_values[1],
            Some(
                "Colors list: [\"yellow\"]\nAddress struct: {\"city\": \"London\", \"zip\": \"\"}"
            )
        );
    }
}
