use crate::dtypes::cast::cast_series_to_fenic_dtype;
use crate::dtypes::types::{FenicDType, StructField};
use polars::chunked_array::builder::get_list_builder;
use polars::prelude::*;
use polars_arrow::array::ValueSize;

/// Cast between struct types, handling field additions, removals, and type changes.
pub fn cast_struct_to_struct(
    s: &Series,
    src_struct_fields: &[StructField],
    dest_struct_fields: &[StructField],
) -> PolarsResult<Series> {
    let struct_len = s.len();
    let input_fields = s.struct_()?.fields_as_series();

    let field_map: std::collections::HashMap<&str, &Series> = input_fields
        .iter()
        .map(|f| (f.name().as_str(), f))
        .collect();

    let src_type_map: std::collections::HashMap<&str, &FenicDType> = src_struct_fields
        .iter()
        .map(|f| (f.name.as_str(), f.data_type.as_ref()))
        .collect();

    let mut casted_fields = Vec::with_capacity(dest_struct_fields.len());

    for dest_field in dest_struct_fields {
        let casted = match field_map.get(dest_field.name.as_str()) {
            Some(series) => {
                // SAFETY: All field_map entries are built from src_struct_fields,
                // so src_type_map must contain the corresponding entry.
                let src_type = src_type_map
                    .get(dest_field.name.as_str())
                    .unwrap_or_else(|| {
                        unreachable!(
                            "Internal error: `src_type_map` is missing field '{}'. \
                         This indicates a bug, as `field_map` and `src_type_map` are both derived \
                         from the same source and should always be in sync.",
                            dest_field.name
                        )
                    });

                if **src_type == *dest_field.data_type.as_ref() {
                    (*series).clone()
                } else {
                    cast_series_to_fenic_dtype(series, src_type, &dest_field.data_type)?
                }
            }
            None => Series::full_null(
                dest_field.name.as_str().into(),
                struct_len,
                &dest_field.data_type.canonical_polars_type(),
            ),
        };

        casted_fields.push(casted);
    }

    let out_struct =
        StructChunked::from_series(s.name().clone(), struct_len, casted_fields.iter())?;
    Ok(out_struct.into_series())
}

/// Cast between array types by recursively casting element types.
pub fn cast_array_to_array(
    s: &Series,
    src_type: &FenicDType,
    dest_type: &FenicDType,
) -> PolarsResult<Series> {
    if *src_type == *dest_type {
        return Ok(s.clone());
    }

    let ca = s.list()?;
    let name = s.name();

    let mut builder = get_list_builder(
        &dest_type.canonical_polars_type(),
        ca.get_values_size(),
        ca.len(),
        name.clone(),
    );

    for opt_s in ca.into_iter() {
        match opt_s {
            Some(inner) => {
                let casted = cast_series_to_fenic_dtype(&inner, src_type, dest_type)?;
                builder.append_series(&casted)?;
            }
            None => builder.append_null(),
        }
    }

    Ok(builder.finish().into_series())
}

/// Cast an array (list) to a fixed-size embedding array.
pub fn cast_array_to_embedding(s: &Series, dimensions: usize) -> PolarsResult<Series> {
    match s.dtype() {
        DataType::List(inner_type) => match inner_type.as_ref() {
            DataType::Float32 | DataType::Float64 => {
                s.cast(&DataType::Array(Box::new(DataType::Float32), dimensions))
            }
            other => unreachable!("Expected List(Float32 or Float64), got {:?}", other),
        },
        other => unreachable!("Expected Array or List, got {:?}", other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_string_series(name: &str, values: Vec<Option<&str>>) -> Series {
        Series::new(name.into(), values)
    }

    fn create_int_series(name: &str, values: Vec<Option<i64>>) -> Series {
        Series::new(name.into(), values)
    }

    #[test]
    fn test_struct_to_struct_casting_same_fields() {
        let name_series = create_string_series("name", vec![Some("Alice"), Some("Bob"), None]);
        let age_series = create_int_series("age", vec![Some(30), None, Some(40)]);

        let struct_series =
            StructChunked::from_series("person".into(), 3, [name_series, age_series].iter())
                .unwrap()
                .into_series();

        let fields = vec![
            StructField {
                name: "name".to_string(),
                data_type: Box::new(FenicDType::StringType),
            },
            StructField {
                name: "age".to_string(),
                data_type: Box::new(FenicDType::IntegerType),
            },
        ];

        let result = cast_struct_to_struct(&struct_series, &fields, &fields).unwrap();

        // The series should remain unchanged when casting to the same schema
        assert_eq!(result.len(), 3);
        let result_struct = result.struct_().unwrap();
        let result_fields = result_struct.fields_as_series();

        assert_eq!(result_fields.len(), 2);
        assert_eq!(result_fields[0].str().unwrap().get(0), Some("Alice"));
        assert_eq!(result_fields[1].i64().unwrap().get(1), None);
    }

    #[test]
    fn test_struct_to_struct_casting_add_field() {
        let name_series = create_string_series("name", vec![Some("Alice"), Some("Bob")]);

        let struct_series = StructChunked::from_series("person".into(), 2, [name_series].iter())
            .unwrap()
            .into_series();

        let src_fields = vec![StructField {
            name: "name".to_string(),
            data_type: Box::new(FenicDType::StringType),
        }];

        let dest_fields = vec![
            StructField {
                name: "name".to_string(),
                data_type: Box::new(FenicDType::StringType),
            },
            StructField {
                name: "age".to_string(),
                data_type: Box::new(FenicDType::IntegerType),
            },
        ];

        let result = cast_struct_to_struct(&struct_series, &src_fields, &dest_fields).unwrap();

        let result_struct = result.struct_().unwrap();
        let result_fields = result_struct.fields_as_series();

        assert_eq!(result_fields.len(), 2);
        // Original field preserved
        assert_eq!(result_fields[0].str().unwrap().get(0), Some("Alice"));
        // New field filled with nulls
        assert_eq!(result_fields[1].i64().unwrap().get(0), None);
        assert_eq!(result_fields[1].i64().unwrap().get(1), None);
    }

    #[test]
    fn test_struct_to_struct_casting_remove_field() {
        let name_series = create_string_series("name", vec![Some("Alice"), Some("Bob")]);
        let age_series = create_int_series("age", vec![Some(30), Some(25)]);
        let city_series = create_string_series("city", vec![Some("NYC"), Some("LA")]);

        let struct_series = StructChunked::from_series(
            "person".into(),
            2,
            [name_series, age_series, city_series].iter(),
        )
        .unwrap()
        .into_series();

        let src_fields = vec![
            StructField {
                name: "name".to_string(),
                data_type: Box::new(FenicDType::StringType),
            },
            StructField {
                name: "age".to_string(),
                data_type: Box::new(FenicDType::IntegerType),
            },
            StructField {
                name: "city".to_string(),
                data_type: Box::new(FenicDType::StringType),
            },
        ];

        let dest_fields = vec![
            StructField {
                name: "name".to_string(),
                data_type: Box::new(FenicDType::StringType),
            },
            StructField {
                name: "age".to_string(),
                data_type: Box::new(FenicDType::IntegerType),
            },
        ];

        let result = cast_struct_to_struct(&struct_series, &src_fields, &dest_fields).unwrap();

        let result_struct = result.struct_().unwrap();
        let result_fields = result_struct.fields_as_series();

        assert_eq!(result_fields.len(), 2); // Only 2 fields now

        // Remaining fields should be preserved
        assert_eq!(result_fields[0].str().unwrap().get(0), Some("Alice"));
        assert_eq!(result_fields[1].i64().unwrap().get(0), Some(30));
    }

    #[test]
    fn test_cast_array_to_embedding_from_list() {
        let list_data = vec![
            Some(Series::new("".into(), vec![1.0f32, 2.0f32, 3.0f32])),
            Some(Series::new("".into(), vec![4.0f32, 5.0f32, 6.0f32])),
            None,
        ];
        let list_series = Series::new("embeddings".into(), list_data);

        let result = cast_array_to_embedding(&list_series, 3).unwrap();

        assert_eq!(
            result.dtype(),
            &DataType::Array(Box::new(DataType::Float32), 3)
        );
        assert_eq!(result.len(), 3);
        assert!(result.get(2).unwrap().is_null());
    }

    #[test]
    fn test_cast_array_string_to_integer() {
        // Create array of strings that should convert to integers
        let string_arrays = vec![
            Some(Series::new("".into(), vec!["1", "2", "3"])),
            Some(Series::new("".into(), vec!["42", "-10", "0"])),
            Some(Series::new(
                "".into(),
                vec!["invalid", "123", "not_a_number"],
            )),
            None,
        ];
        let array_series = Series::new("test".into(), string_arrays);

        let src_type = FenicDType::StringType;
        let dest_type = FenicDType::IntegerType;

        let result = cast_array_to_array(&array_series, &src_type, &dest_type).unwrap();
        assert_eq!(result.len(), 4);

        let list_ca = result.list().unwrap();

        // First array: all valid integers
        let first_array = list_ca.get_as_series(0).unwrap();
        assert_eq!(first_array.len(), 3);
        assert_eq!(first_array.i64().unwrap().get(0), Some(1));
        assert_eq!(first_array.i64().unwrap().get(1), Some(2));
        assert_eq!(first_array.i64().unwrap().get(2), Some(3));

        // Second array: valid integers including negative and zero
        let second_array = list_ca.get_as_series(1).unwrap();
        assert_eq!(second_array.len(), 3);
        assert_eq!(second_array.i64().unwrap().get(0), Some(42));
        assert_eq!(second_array.i64().unwrap().get(1), Some(-10));
        assert_eq!(second_array.i64().unwrap().get(2), Some(0));

        // Third array: mixed valid and invalid
        let third_array = list_ca.get_as_series(2).unwrap();
        assert_eq!(third_array.len(), 3);
        assert_eq!(third_array.i64().unwrap().get(0), None); // "invalid" -> null
        assert_eq!(third_array.i64().unwrap().get(1), Some(123));
        assert_eq!(third_array.i64().unwrap().get(2), None); // "not_a_number" -> null

        // Fourth array: null
        assert!(list_ca.get_as_series(3).is_none());
    }

    #[test]
    fn test_cast_array_integer_to_float() {
        let int_arrays = vec![
            Some(Series::new("".into(), vec![1i64, 2i64, 3i64])),
            Some(Series::new("".into(), vec![i64::MAX, i64::MIN, 0i64])),
            None,
        ];
        let array_series = Series::new("test".into(), int_arrays);

        let src_type = FenicDType::IntegerType;
        let dest_type = FenicDType::FloatType;

        let result = cast_array_to_array(&array_series, &src_type, &dest_type).unwrap();
        assert_eq!(result.len(), 3);

        let list_ca = result.list().unwrap();

        // First array: simple integers to floats
        let first_array = list_ca.get_as_series(0).unwrap();
        assert_eq!(first_array.len(), 3);
        assert_eq!(first_array.f32().unwrap().get(0), Some(1.0));
        assert_eq!(first_array.f32().unwrap().get(1), Some(2.0));
        assert_eq!(first_array.f32().unwrap().get(2), Some(3.0));

        // Second array: extreme values
        let second_array = list_ca.get_as_series(1).unwrap();
        assert_eq!(second_array.len(), 3);
        // Note: i64::MAX/MIN might lose precision when cast to f32, but should not be null
        assert!(second_array.f32().unwrap().get(0).is_some());
        assert!(second_array.f32().unwrap().get(1).is_some());
        assert_eq!(second_array.f32().unwrap().get(2), Some(0.0));

        // Third array: null
        assert!(list_ca.get_as_series(2).is_none());
    }

    #[test]
    fn test_cast_array_float_to_integer() {
        let float_arrays = vec![
            Some(Series::new("".into(), vec![1.0f32, 2.9f32, -3.7f32])),
            Some(Series::new(
                "".into(),
                vec![f32::NAN, f32::INFINITY, 42.0f32],
            )),
        ];
        let array_series = Series::new("test".into(), float_arrays);

        let src_type = FenicDType::FloatType;
        let dest_type = FenicDType::IntegerType;

        let result = cast_array_to_array(&array_series, &src_type, &dest_type).unwrap();
        assert_eq!(result.len(), 2);

        let list_ca = result.list().unwrap();

        // First array: floats truncated to integers
        let first_array = list_ca.get_as_series(0).unwrap();
        assert_eq!(first_array.len(), 3);
        assert_eq!(first_array.i64().unwrap().get(0), Some(1)); // 1.0 -> 1
        assert_eq!(first_array.i64().unwrap().get(1), Some(2)); // 2.9 -> 2 (truncated)
        assert_eq!(first_array.i64().unwrap().get(2), Some(-3)); // -3.7 -> -3

        // Second array: special float values
        let second_array = list_ca.get_as_series(1).unwrap();
        assert_eq!(second_array.len(), 3);
        assert_eq!(second_array.i64().unwrap().get(0), None); // NaN -> null
        assert_eq!(second_array.i64().unwrap().get(1), None); // INFINITY -> null
        assert_eq!(second_array.i64().unwrap().get(2), Some(42)); // 42.0 -> 42
    }

    #[test]
    fn test_cast_array_boolean_to_string() {
        let bool_arrays = vec![
            Some(Series::new("".into(), vec![true, false, true])),
            Some(Series::new("".into(), vec![false])),
            None,
        ];
        let array_series = Series::new("test".into(), bool_arrays);

        let src_type = FenicDType::BooleanType;
        let dest_type = FenicDType::StringType;

        let result = cast_array_to_array(&array_series, &src_type, &dest_type).unwrap();
        assert_eq!(result.len(), 3);

        let list_ca = result.list().unwrap();

        // First array: booleans to strings
        let first_array = list_ca.get_as_series(0).unwrap();
        assert_eq!(first_array.len(), 3);
        assert_eq!(first_array.str().unwrap().get(0), Some("true"));
        assert_eq!(first_array.str().unwrap().get(1), Some("false"));
        assert_eq!(first_array.str().unwrap().get(2), Some("true"));

        // Second array: single boolean
        let second_array = list_ca.get_as_series(1).unwrap();
        assert_eq!(second_array.len(), 1);
        assert_eq!(second_array.str().unwrap().get(0), Some("false"));

        // Third array: null
        assert!(list_ca.get_as_series(2).is_none());
    }

    #[test]
    fn test_cast_array_with_nulls() {
        // Create arrays containing null values
        let string_arrays = vec![
            Some(Series::new("".into(), vec![Some("1"), None, Some("3")])),
            None,
            Some(Series::new("".into(), vec![Some("42")])),
        ];
        let array_series = Series::new("test".into(), string_arrays);

        let src_type = FenicDType::StringType;
        let dest_type = FenicDType::IntegerType;

        let result = cast_array_to_array(&array_series, &src_type, &dest_type).unwrap();
        assert_eq!(result.len(), 3);

        let list_ca = result.list().unwrap();

        // First array: mixed nulls and valid values
        let first_array = list_ca.get_as_series(0).unwrap();
        assert_eq!(first_array.len(), 3);
        assert_eq!(first_array.i64().unwrap().get(0), Some(1));
        assert_eq!(first_array.i64().unwrap().get(1), None); // null stays null
        assert_eq!(first_array.i64().unwrap().get(2), Some(3));

        // Second array: all nulls
        let second_array = list_ca.get_as_series(1);
        assert!(second_array.is_none());

        // Third array: single valid value
        let third_array = list_ca.get_as_series(2).unwrap();
        assert_eq!(third_array.len(), 1);
        assert_eq!(third_array.i64().unwrap().get(0), Some(42));
    }

    #[test]
    fn test_cast_array_identity() {
        // Test casting array to same type (should be no-op)
        let int_arrays = vec![
            Some(Series::new("".into(), vec![1i64, 2i64, 3i64])),
            Some(Series::new("".into(), vec![42i64])),
            None,
        ];
        let array_series = Series::new("test".into(), int_arrays);

        let src_type = FenicDType::IntegerType;
        let dest_type = FenicDType::IntegerType;

        let result = cast_array_to_array(&array_series, &src_type, &dest_type).unwrap();
        assert_eq!(result.len(), 3);

        let list_ca = result.list().unwrap();

        // Should be identical to input
        let first_array = list_ca.get_as_series(0).unwrap();
        assert_eq!(first_array.len(), 3);
        assert_eq!(first_array.i64().unwrap().get(0), Some(1));
        assert_eq!(first_array.i64().unwrap().get(1), Some(2));
        assert_eq!(first_array.i64().unwrap().get(2), Some(3));

        let second_array = list_ca.get_as_series(1).unwrap();
        assert_eq!(second_array.len(), 1);
        assert_eq!(second_array.i64().unwrap().get(0), Some(42));

        assert!(list_ca.get_as_series(2).is_none());
    }

    #[test]
    fn test_cast_nested_struct_with_arrays() {
        // Test casting structs that contain arrays
        let tags_data = vec![
            Some(Series::new("".into(), vec!["rust", "polars"])),
            Some(Series::new("".into(), vec!["json", "test"])),
        ];
        let tags_series = Series::new("tags".into(), tags_data);

        let scores_data = vec![
            Some(Series::new("".into(), vec![95.5f32, 87.2f32])),
            Some(Series::new("".into(), vec![88.0f32, 91.5f32])),
        ];
        let scores_series = Series::new("scores".into(), scores_data);

        let name_series = create_string_series("name", vec![Some("Project A"), Some("Project B")]);

        let original_struct = StructChunked::from_series(
            "project".into(),
            2,
            [name_series, tags_series, scores_series].iter(),
        )
        .unwrap()
        .into_series();

        // Source struct type
        let src_struct_type = vec![
            StructField {
                name: "name".to_string(),
                data_type: Box::new(FenicDType::StringType),
            },
            StructField {
                name: "tags".to_string(),
                data_type: Box::new(FenicDType::ArrayType {
                    element_type: Box::new(FenicDType::StringType),
                }),
            },
            StructField {
                name: "scores".to_string(),
                data_type: Box::new(FenicDType::ArrayType {
                    element_type: Box::new(FenicDType::FloatType),
                }),
            },
        ];

        // Target struct type - change scores array from Float to Integer
        let dest_struct_type = vec![
            StructField {
                name: "name".to_string(),
                data_type: Box::new(FenicDType::StringType),
            },
            StructField {
                name: "tags".to_string(),
                data_type: Box::new(FenicDType::ArrayType {
                    element_type: Box::new(FenicDType::StringType),
                }),
            },
            StructField {
                name: "scores".to_string(),
                data_type: Box::new(FenicDType::ArrayType {
                    element_type: Box::new(FenicDType::IntegerType),
                }),
            },
        ];

        let result =
            cast_struct_to_struct(&original_struct, &src_struct_type, &dest_struct_type).unwrap();
        assert_eq!(result.len(), 2);

        let struct_ca = result.struct_().unwrap();
        let fields = struct_ca.fields_as_series();

        // Check that name field is preserved
        let name_field = &fields[0];
        assert_eq!(name_field.str().unwrap().get(0), Some("Project A"));
        assert_eq!(name_field.str().unwrap().get(1), Some("Project B"));

        // Check that tags array is preserved
        let tags_field = &fields[1];
        let first_tags = tags_field.list().unwrap().get_as_series(0).unwrap();
        assert_eq!(first_tags.str().unwrap().get(0), Some("rust"));
        assert_eq!(first_tags.str().unwrap().get(1), Some("polars"));

        // Check that scores array was converted from float to integer
        let scores_field = &fields[2];
        let first_scores = scores_field.list().unwrap().get_as_series(0).unwrap();
        assert_eq!(first_scores.i64().unwrap().get(0), Some(95)); // 95.5 -> 95 (truncated)
        assert_eq!(first_scores.i64().unwrap().get(1), Some(87)); // 87.2 -> 87 (truncated)
    }

    #[test]
    fn test_cast_array_of_structs() {
        // Test arrays containing struct elements - simulated with nested struct data
        let person1_name = create_string_series("name", vec![Some("Alice"), Some("Bob")]);
        let person1_age = create_int_series("age", vec![Some(30), Some(25)]);

        let person_struct1 =
            StructChunked::from_series("person".into(), 2, [person1_name, person1_age].iter())
                .unwrap()
                .into_series();

        let person2_name = create_string_series("name", vec![Some("Charlie"), Some("Diana")]);
        let person2_age = create_int_series("age", vec![Some(35), Some(28)]);

        let person_struct2 =
            StructChunked::from_series("person".into(), 2, [person2_name, person2_age].iter())
                .unwrap()
                .into_series();

        // Create an array of these structs (conceptually)
        let struct_arrays = vec![Some(person_struct1), Some(person_struct2)];

        // Source and destination struct types for the elements
        let struct_fields_src = vec![
            StructField {
                name: "name".to_string(),
                data_type: Box::new(FenicDType::StringType),
            },
            StructField {
                name: "age".to_string(),
                data_type: Box::new(FenicDType::IntegerType),
            },
        ];

        let struct_fields_dest = vec![
            StructField {
                name: "name".to_string(),
                data_type: Box::new(FenicDType::StringType),
            },
            StructField {
                name: "age".to_string(),
                data_type: Box::new(FenicDType::StringType), // Convert age to string
            },
        ];

        // Test casting each struct individually (simulating array of structs)
        for struct_elem in struct_arrays {
            if let Some(struct_series) = struct_elem {
                let result =
                    cast_struct_to_struct(&struct_series, &struct_fields_src, &struct_fields_dest)
                        .unwrap();

                let struct_ca = result.struct_().unwrap();
                let fields = struct_ca.fields_as_series();

                // Check that age was converted from integer to string
                let age_field = &fields[1];
                assert!(age_field.str().is_ok()); // Should now be string type
            }
        }
    }

    #[test]
    fn test_cast_deeply_nested_structs() {
        // Test struct containing struct containing array
        let inner_tags = vec![Some(Series::new("".into(), vec!["tag1", "tag2"])), None];
        let inner_tags_series = Series::new("tags".into(), inner_tags);

        let inner_value_series =
            create_string_series("value", vec![Some("inner_val"), Some("inner_val2")]);

        // Create inner struct
        let inner_struct = StructChunked::from_series(
            "metadata".into(),
            2,
            [inner_value_series, inner_tags_series].iter(),
        )
        .unwrap()
        .into_series();

        let outer_name_series = create_string_series("name", vec![Some("outer1"), Some("outer2")]);

        // Create outer struct containing inner struct
        let outer_struct = StructChunked::from_series(
            "record".into(),
            2,
            [outer_name_series, inner_struct].iter(),
        )
        .unwrap()
        .into_series();

        // Source type: outer struct -> inner struct -> {value: string, tags: array<string>}
        let src_fields = vec![
            StructField {
                name: "name".to_string(),
                data_type: Box::new(FenicDType::StringType),
            },
            StructField {
                name: "metadata".to_string(),
                data_type: Box::new(FenicDType::StructType {
                    struct_fields: vec![
                        StructField {
                            name: "value".to_string(),
                            data_type: Box::new(FenicDType::StringType),
                        },
                        StructField {
                            name: "tags".to_string(),
                            data_type: Box::new(FenicDType::ArrayType {
                                element_type: Box::new(FenicDType::StringType),
                            }),
                        },
                    ],
                }),
            },
        ];

        // Destination type: change inner struct's value field from string to integer
        let dest_fields = vec![
            StructField {
                name: "name".to_string(),
                data_type: Box::new(FenicDType::StringType),
            },
            StructField {
                name: "metadata".to_string(),
                data_type: Box::new(FenicDType::StructType {
                    struct_fields: vec![
                        StructField {
                            name: "value".to_string(),
                            data_type: Box::new(FenicDType::IntegerType), // String -> Integer
                        },
                        StructField {
                            name: "tags".to_string(),
                            data_type: Box::new(FenicDType::ArrayType {
                                element_type: Box::new(FenicDType::StringType),
                            }),
                        },
                    ],
                }),
            },
        ];

        let result = cast_struct_to_struct(&outer_struct, &src_fields, &dest_fields).unwrap();
        assert_eq!(result.len(), 2);

        let struct_ca = result.struct_().unwrap();
        let outer_fields = struct_ca.fields_as_series();

        // Check outer name field
        let name_field = &outer_fields[0];
        assert_eq!(name_field.str().unwrap().get(0), Some("outer1"));

        // Check nested struct field
        let metadata_field = &outer_fields[1];
        let metadata_struct = metadata_field.struct_().unwrap();
        let inner_fields = metadata_struct.fields_as_series();

        // Check that inner value was converted from string to integer (should be null since "inner_val" is not a number)
        let inner_value_field = &inner_fields[0];
        assert_eq!(inner_value_field.i64().unwrap().get(0), None); // "inner_val" -> null

        // Check that inner tags array is preserved
        let inner_tags_field = &inner_fields[1];
        let first_tags = inner_tags_field.list().unwrap().get_as_series(0).unwrap();
        assert_eq!(first_tags.str().unwrap().get(0), Some("tag1"));
        assert_eq!(first_tags.str().unwrap().get(1), Some("tag2"));

        // Check null handling in nested structure
        assert!(inner_tags_field.list().unwrap().get_as_series(1).is_none());
    }

    #[test]
    fn test_cast_mixed_nested_types() {
        // Test a complex structure with multiple levels of nesting
        let config_values = vec![
            Some(Series::new("".into(), vec!["true", "false", "1"])),
            Some(Series::new("".into(), vec!["100", "200"])),
        ];
        let config_series = Series::new("config_values".into(), config_values);

        let env_series = create_string_series("environment", vec![Some("prod"), Some("dev")]);

        let settings_struct =
            StructChunked::from_series("settings".into(), 2, [env_series, config_series].iter())
                .unwrap()
                .into_series();

        let app_name_series = create_string_series("app_name", vec![Some("app1"), Some("app2")]);
        let version_series = create_string_series("version", vec![Some("1.0"), Some("2.0")]);

        let final_struct = StructChunked::from_series(
            "application".into(),
            2,
            [app_name_series, version_series, settings_struct].iter(),
        )
        .unwrap()
        .into_series();

        // Define nested type structure
        let src_fields = vec![
            StructField {
                name: "app_name".to_string(),
                data_type: Box::new(FenicDType::StringType),
            },
            StructField {
                name: "version".to_string(),
                data_type: Box::new(FenicDType::StringType),
            },
            StructField {
                name: "settings".to_string(),
                data_type: Box::new(FenicDType::StructType {
                    struct_fields: vec![
                        StructField {
                            name: "environment".to_string(),
                            data_type: Box::new(FenicDType::StringType),
                        },
                        StructField {
                            name: "config_values".to_string(),
                            data_type: Box::new(FenicDType::ArrayType {
                                element_type: Box::new(FenicDType::StringType),
                            }),
                        },
                    ],
                }),
            },
        ];

        // Change nested array element type from string to integer
        let dest_fields = vec![
            StructField {
                name: "app_name".to_string(),
                data_type: Box::new(FenicDType::StringType),
            },
            StructField {
                name: "version".to_string(),
                data_type: Box::new(FenicDType::FloatType), // String -> Float
            },
            StructField {
                name: "settings".to_string(),
                data_type: Box::new(FenicDType::StructType {
                    struct_fields: vec![
                        StructField {
                            name: "environment".to_string(),
                            data_type: Box::new(FenicDType::StringType),
                        },
                        StructField {
                            name: "config_values".to_string(),
                            data_type: Box::new(FenicDType::ArrayType {
                                element_type: Box::new(FenicDType::IntegerType), // String -> Integer
                            }),
                        },
                    ],
                }),
            },
        ];

        let result = cast_struct_to_struct(&final_struct, &src_fields, &dest_fields).unwrap();
        assert_eq!(result.len(), 2);

        let struct_ca = result.struct_().unwrap();
        let fields = struct_ca.fields_as_series();

        // Check app_name (unchanged)
        assert_eq!(fields[0].str().unwrap().get(0), Some("app1"));

        // Check version (string -> float)
        assert_eq!(fields[1].f32().unwrap().get(0), Some(1.0));
        assert_eq!(fields[1].f32().unwrap().get(1), Some(2.0));

        // Check nested settings struct
        let settings_field = &fields[2];
        let settings_struct = settings_field.struct_().unwrap();
        let settings_fields = settings_struct.fields_as_series();

        // Check environment (unchanged)
        assert_eq!(settings_fields[0].str().unwrap().get(0), Some("prod"));

        // Check config_values array (string -> integer)
        let config_field = &settings_fields[1];
        let first_config = config_field.list().unwrap().get_as_series(0).unwrap();
        assert_eq!(first_config.i64().unwrap().get(0), None); // "true" -> null (invalid integer)
        assert_eq!(first_config.i64().unwrap().get(1), None); // "false" -> null (invalid integer)
        assert_eq!(first_config.i64().unwrap().get(2), Some(1)); // "1" -> 1

        let second_config = config_field.list().unwrap().get_as_series(1).unwrap();
        assert_eq!(second_config.i64().unwrap().get(0), Some(100)); // "100" -> 100
        assert_eq!(second_config.i64().unwrap().get(1), Some(200)); // "200" -> 200
    }
}
