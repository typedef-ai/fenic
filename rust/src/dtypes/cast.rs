use crate::dtypes::collections::{
    cast_array_to_array, cast_array_to_embedding, cast_struct_to_struct,
};
use crate::dtypes::json::{
    cast_fenic_dtype_to_json, cast_json_to_fenic_dtype, cast_string_to_json,
};
use crate::dtypes::primitives::cast_primitive_to_primitive;
use crate::dtypes::types::FenicDType;
use polars::prelude::*;

/// Casts a Polars `Series` from one logical type to another.
///
/// This function is the main entry point for all type casting operations.
/// It dispatches to specialized casting functions based on the source and
/// destination types.
///
/// # Arguments
///
/// * `s` - The input `Series` to cast.
/// * `src_type` - The source logical type.
/// * `dest_type` - The target logical type to cast to.
///
/// # Returns
///
/// Returns a new `Series` cast to the target logical type, or an error if casting fails.
///
/// # Notes
///
/// * This function assumes the cast from source to destination type is valid.
///   Castability is checked by the `can_cast` function in the cast expression
///   during Python planning phase.
/// * If source and destination types are equal, returns a clone of the input.
/// * Struct fields missing in the input series are filled with nulls.
/// * Embedding types are expected to be represented as fixed-size arrays or lists of floats.
/// * For JSON types, strings are validated as valid JSON and nulls inserted for invalid values.
///
/// # Errors
///
/// Returns a `PolarsError` if the cast is not supported or fails.
pub fn cast_series_to_fenic_dtype(
    s: &Series,
    src_type: &FenicDType,
    dest_type: &FenicDType,
) -> PolarsResult<Series> {
    if src_type == dest_type {
        return Ok(s.clone());
    }

    use FenicDType::*;
    match (src_type, dest_type) {
        // MarkdownType treated as StringType for casting purposes
        (_, MarkdownType) => {
            match src_type {
                StringType => Ok(s.clone()),
                _ => cast_series_to_fenic_dtype(s, src_type, &StringType),
            }
        }
        (MarkdownType, _) => {
            match dest_type {
                StringType => Ok(s.clone()),
                _ => cast_series_to_fenic_dtype(s, &StringType, dest_type),
            }
        }

        // JSON casting
        (StringType, JsonType) => cast_string_to_json(s), // Validate strings as JSON
        (_, JsonType) => cast_fenic_dtype_to_json(s),    // Convert values to JSON representation
        (JsonType, _) => cast_json_to_fenic_dtype(s, dest_type),

        // Struct casting
        (
            StructType {
                struct_fields: src_struct_fields,
            },
            StructType {
                struct_fields: dest_struct_fields,
            },
        ) => cast_struct_to_struct(s, src_struct_fields, dest_struct_fields),
        (StructType { .. }, other) => unreachable!(
            "Attempted invalid cast from StructType to {:?}. This should have been filtered earlier.",
            other
        ),

        // Array casting
        (
            ArrayType {
                element_type: src_element_type,
            },
            ArrayType {
                element_type: dest_element_type,
            },
        ) => cast_array_to_array(s, src_element_type, dest_element_type),
        (ArrayType { .. }, EmbeddingType { dimensions, .. }) => {
            cast_array_to_embedding(s, *dimensions)
        }
        (
            EmbeddingType {
                dimensions: _,
                embedding_model: _,
            },
            ArrayType { element_type: _ },
        ) => {
            s.cast(&DataType::List(Box::new(DataType::Float32)))
        }
        (ArrayType { .. }, other) => unreachable!(
            "Attempted invalid cast from ArrayType to {:?}. This should have been filtered earlier.",
            other
        ),

        // Logical types (not yet implemented)
        (_, HtmlType) => todo!("Casting to HtmlType not yet implemented"),
        (_, TranscriptType { .. }) => todo!("Casting to TranscriptType not yet implemented"),
        (_, DocumentPathType { .. }) => todo!("Casting to DocumentPathType not yet implemented"),
        (HtmlType, _) => todo!("Casting from HtmlType not yet implemented"),
        (TranscriptType { .. }, _) => todo!("Casting from TranscriptType not yet implemented"),
        (DocumentPathType { .. }, _) => todo!("Casting from DocumentPathType not yet implemented"),
        (EmbeddingType { .. }, _) => todo!("Casting from EmbeddingType not yet implemented"),

        // Primitive casting
        _ => cast_primitive_to_primitive(s, src_type, dest_type),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtypes::types::StructField;
    use serde_json::json;

    fn create_string_series(name: &str, values: Vec<Option<&str>>) -> Series {
        Series::new(name.into(), values)
    }

    fn create_int_series(name: &str, values: Vec<Option<i64>>) -> Series {
        Series::new(name.into(), values)
    }

    #[test]
    fn test_identity_cast() {
        let series = Series::new("test".into(), vec![1, 2, 3]);
        let result =
            cast_series_to_fenic_dtype(&series, &FenicDType::IntegerType, &FenicDType::IntegerType)
                .unwrap();

        assert_eq!(series.len(), result.len());
        assert_eq!(series.name(), result.name());
    }

    #[test]
    fn test_cast_series_same_type() {
        let s = create_string_series("test", vec![Some("hello"), Some("world"), None]);
        let src_type = FenicDType::StringType;
        let dest_type = FenicDType::StringType;

        let result = cast_series_to_fenic_dtype(&s, &src_type, &dest_type).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result.str().unwrap().get(0), Some("hello"));
        assert_eq!(result.str().unwrap().get(1), Some("world"));
        assert_eq!(result.str().unwrap().get(2), None);
    }

    #[test]
    fn test_cast_json_to_embedding_valid() {
        let json_values = vec![
            json!([1.0, 2.0, 3.0]),    // Valid 3D embedding
            json!([1, 2, 3]),          // Integers, should convert to floats
            json!([0.1, -0.5, 999.9]), // Various float values
        ];

        let json_strings: Vec<String> = json_values
            .iter()
            .map(|v| serde_json::to_string(v).unwrap())
            .collect();

        let s = Series::new("test".into(), json_strings);

        let embedding_type = FenicDType::EmbeddingType {
            dimensions: 3,
            embedding_model: "oai-small".to_string(),
        };

        let result =
            cast_series_to_fenic_dtype(&s, &FenicDType::JsonType, &embedding_type).unwrap();
        assert_eq!(result.len(), 3);

        // Check that we got the right type
        if let DataType::Array(inner_type, size) = result.dtype() {
            assert_eq!(**inner_type, DataType::Float32);
            assert_eq!(*size, 3);
        }

        // Test that all values are non-null by getting them
        assert!(result.get(0).is_ok());
        assert!(result.get(1).is_ok());
        assert!(result.get(2).is_ok());

        let null_count = result.null_count();
        assert_eq!(null_count, 0);
    }

    #[test]
    fn test_cast_json_to_embedding_invalid() {
        let json_values = vec![
            json!([1.0, 2.0]),   // Wrong dimensions (2 instead of 3)
            json!([1, 2, 3, 4]), // Wrong dimensions (4 instead of 3)
            json!("not an array"),
            json!(null),
        ];

        let json_strings: Vec<String> = json_values
            .iter()
            .map(|v| serde_json::to_string(v).unwrap())
            .collect();

        let s = Series::new("test".into(), json_strings);

        let embedding_type = FenicDType::EmbeddingType {
            dimensions: 3,
            embedding_model: "oai-small".to_string(),
        };

        let result =
            cast_series_to_fenic_dtype(&s, &FenicDType::JsonType, &embedding_type).unwrap();
        assert_eq!(result.len(), 4);

        // All should be null due to wrong dimensions or invalid data
        let null_count = result.null_count();
        assert_eq!(null_count, 4);
    }

    #[test]
    fn test_json_struct_roundtrip() {
        // Create original struct data
        let name_series = create_string_series("name", vec![Some("Alice"), Some("Bob"), None]);
        let age_series = create_int_series("age", vec![Some(30), None, Some(40)]);
        let active_series = Series::new("active".into(), vec![Some(true), Some(false), None]);

        let original_struct = StructChunked::from_series(
            "person".into(),
            3,
            [name_series, age_series, active_series].iter(),
        )
        .unwrap()
        .into_series();

        let struct_type = FenicDType::StructType {
            struct_fields: vec![
                StructField {
                    name: "name".to_string(),
                    data_type: Box::new(FenicDType::StringType),
                },
                StructField {
                    name: "age".to_string(),
                    data_type: Box::new(FenicDType::IntegerType),
                },
                StructField {
                    name: "active".to_string(),
                    data_type: Box::new(FenicDType::BooleanType),
                },
            ],
        };

        // Roundtrip: Struct -> JSON -> Struct
        let json_series =
            cast_series_to_fenic_dtype(&original_struct, &struct_type, &FenicDType::JsonType)
                .unwrap();

        let roundtrip_struct =
            cast_series_to_fenic_dtype(&json_series, &FenicDType::JsonType, &struct_type).unwrap();

        // Verify the roundtrip preserved the data
        assert_eq!(original_struct.len(), roundtrip_struct.len());

        let original_fields = original_struct.struct_().unwrap().fields_as_series();
        let roundtrip_fields = roundtrip_struct.struct_().unwrap().fields_as_series();

        assert_eq!(original_fields.len(), roundtrip_fields.len());

        // Check name field
        let orig_names = &original_fields[0];
        let round_names = &roundtrip_fields[0];
        for i in 0..orig_names.len() {
            assert_eq!(
                orig_names.str().unwrap().get(i),
                round_names.str().unwrap().get(i)
            );
        }

        // Check age field
        let orig_ages = &original_fields[1];
        let round_ages = &roundtrip_fields[1];
        for i in 0..orig_ages.len() {
            assert_eq!(
                orig_ages.i64().unwrap().get(i),
                round_ages.i64().unwrap().get(i)
            );
        }

        // Check active field
        let orig_active = &original_fields[2];
        let round_active = &roundtrip_fields[2];
        for i in 0..orig_active.len() {
            assert_eq!(
                orig_active.bool().unwrap().get(i),
                round_active.bool().unwrap().get(i)
            );
        }
    }

    #[test]
    fn test_json_struct_roundtrip_with_arrays() {
        // Test roundtrip with nested arrays
        let tags_data = vec![
            Some(Series::new("".into(), vec!["rust", "polars"])),
            Some(Series::new("".into(), vec!["json", "struct", "test"])),
            None,
        ];
        let tags_series = Series::new("tags".into(), tags_data);

        let scores_data = vec![
            Some(Series::new("".into(), vec![95.5f32, 87.2f32, 92.1f32])),
            Some(Series::new("".into(), vec![88.0f32, 91.5f32])),
            None,
        ];
        let scores_series = Series::new("scores".into(), scores_data);

        let original_struct =
            StructChunked::from_series("record".into(), 3, [tags_series, scores_series].iter())
                .unwrap()
                .into_series();

        let struct_type = FenicDType::StructType {
            struct_fields: vec![
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
            ],
        };

        // Roundtrip: Struct -> JSON -> Struct
        let json_series =
            cast_series_to_fenic_dtype(&original_struct, &struct_type, &FenicDType::JsonType)
                .unwrap();

        let roundtrip_struct =
            cast_series_to_fenic_dtype(&json_series, &FenicDType::JsonType, &struct_type).unwrap();

        // Verify structure is preserved
        assert_eq!(original_struct.len(), roundtrip_struct.len());

        let original_fields = original_struct.struct_().unwrap().fields_as_series();
        let roundtrip_fields = roundtrip_struct.struct_().unwrap().fields_as_series();

        // Check tags array (first field)
        let orig_tags = &original_fields[0];
        let round_tags = &roundtrip_fields[0];

        // First record: should have 2 tags
        let orig_first_tags = orig_tags.list().unwrap().get_as_series(0).unwrap();
        let round_first_tags = round_tags.list().unwrap().get_as_series(0).unwrap();
        assert_eq!(orig_first_tags.len(), round_first_tags.len());
        assert_eq!(orig_first_tags.str().unwrap().get(0), Some("rust"));
        assert_eq!(round_first_tags.str().unwrap().get(0), Some("rust"));

        // Check null handling
        assert!(orig_tags.list().unwrap().get_as_series(2).is_none());
        assert!(round_tags.list().unwrap().get_as_series(2).is_none());
    }

    #[test]
    fn test_json_struct_roundtrip_edge_cases() {
        // Test with empty struct and various edge cases
        let empty_struct =
            StructChunked::from_series("empty".into(), 0, std::iter::empty::<&Series>())
                .unwrap()
                .into_series();

        let empty_struct_type = FenicDType::StructType {
            struct_fields: vec![],
        };

        // Empty struct roundtrip
        let json_series =
            cast_series_to_fenic_dtype(&empty_struct, &empty_struct_type, &FenicDType::JsonType)
                .unwrap();

        let roundtrip_struct =
            cast_series_to_fenic_dtype(&json_series, &FenicDType::JsonType, &empty_struct_type)
                .unwrap();

        assert_eq!(empty_struct.len(), roundtrip_struct.len());
        assert_eq!(empty_struct.len(), 0);

        // Test with all null struct
        let null_name_series = create_string_series("name", vec![None, None]);
        let null_age_series = create_int_series("age", vec![None, None]);

        let all_null_struct = StructChunked::from_series(
            "nulls".into(),
            2,
            [null_name_series, null_age_series].iter(),
        )
        .unwrap()
        .into_series();

        let simple_struct_type = FenicDType::StructType {
            struct_fields: vec![
                StructField {
                    name: "name".to_string(),
                    data_type: Box::new(FenicDType::StringType),
                },
                StructField {
                    name: "age".to_string(),
                    data_type: Box::new(FenicDType::IntegerType),
                },
            ],
        };

        let json_series = cast_series_to_fenic_dtype(
            &all_null_struct,
            &simple_struct_type,
            &FenicDType::JsonType,
        )
        .unwrap();

        let roundtrip_struct =
            cast_series_to_fenic_dtype(&json_series, &FenicDType::JsonType, &simple_struct_type)
                .unwrap();

        // All fields should remain null after roundtrip
        let roundtrip_fields = roundtrip_struct.struct_().unwrap().fields_as_series();
        assert_eq!(roundtrip_fields[0].null_count(), 2); // name field
        assert_eq!(roundtrip_fields[1].null_count(), 2); // age field
    }

    #[test]
    fn test_cast_deeply_nested_struct_to_json_leaf() {
        // Create a deeply nested struct that will be converted to a JSON string
        let deep_tags = vec![
            Some(Series::new("".into(), vec!["rust", "polars", "json"])),
            Some(Series::new("".into(), vec!["test", "nested", "struct"])),
        ];
        let deep_tags_series = Series::new("tags".into(), deep_tags);

        let deep_value_series = create_string_series(
            "value",
            vec![Some("deep_nested_val"), Some("another_deep_val")],
        );

        // Deep inner struct
        let deep_inner_struct = StructChunked::from_series(
            "deep_metadata".into(),
            2,
            [deep_value_series, deep_tags_series].iter(),
        )
        .unwrap()
        .into_series();

        let middle_id_series = create_string_series("id", vec![Some("mid_1"), Some("mid_2")]);

        // Middle struct containing the deep struct
        let middle_struct = StructChunked::from_series(
            "middle".into(),
            2,
            [middle_id_series, deep_inner_struct].iter(),
        )
        .unwrap()
        .into_series();

        let outer_name_series =
            create_string_series("name", vec![Some("outer_1"), Some("outer_2")]);

        // Outer struct containing the middle struct (which contains the deep struct)
        let outer_struct = StructChunked::from_series(
            "outer".into(),
            2,
            [outer_name_series, middle_struct].iter(),
        )
        .unwrap()
        .into_series();

        // Source type: deeply nested struct
        let src_fields = vec![
            StructField {
                name: "name".to_string(),
                data_type: Box::new(FenicDType::StringType),
            },
            StructField {
                name: "middle".to_string(),
                data_type: Box::new(FenicDType::StructType {
                    struct_fields: vec![
                        StructField {
                            name: "id".to_string(),
                            data_type: Box::new(FenicDType::StringType),
                        },
                        StructField {
                            name: "deep_metadata".to_string(),
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
                    ],
                }),
            },
        ];

        // Destination type: convert the entire middle struct to a JSON string
        let dest_fields = vec![
            StructField {
                name: "name".to_string(),
                data_type: Box::new(FenicDType::StringType),
            },
            StructField {
                name: "middle".to_string(),
                data_type: Box::new(FenicDType::JsonType), // Convert entire middle struct to JSON
            },
        ];

        let src_type = FenicDType::StructType {
            struct_fields: src_fields,
        };
        let dest_type = FenicDType::StructType {
            struct_fields: dest_fields,
        };

        let result = cast_series_to_fenic_dtype(&outer_struct, &src_type, &dest_type).unwrap();
        assert_eq!(result.len(), 2);

        let struct_ca = result.struct_().unwrap();
        let fields = struct_ca.fields_as_series();

        // Check that name field is preserved
        let name_field = &fields[0];
        assert_eq!(name_field.str().unwrap().get(0), Some("outer_1"));
        assert_eq!(name_field.str().unwrap().get(1), Some("outer_2"));

        // Check that middle struct was converted to JSON strings
        let json_field = &fields[1];
        assert!(json_field.str().is_ok());

        // Parse the JSON to verify it contains the nested structure
        let json_str_1 = json_field.str().unwrap().get(0).unwrap();
        let json_val_1: serde_json::Value = serde_json::from_str(json_str_1).unwrap();

        assert_eq!(json_val_1["id"], "mid_1");
        assert_eq!(json_val_1["deep_metadata"]["value"], "deep_nested_val");
        assert_eq!(json_val_1["deep_metadata"]["tags"][0], "rust");
        assert_eq!(json_val_1["deep_metadata"]["tags"][1], "polars");
        assert_eq!(json_val_1["deep_metadata"]["tags"][2], "json");

        let json_str_2 = json_field.str().unwrap().get(1).unwrap();
        let json_val_2: serde_json::Value = serde_json::from_str(json_str_2).unwrap();

        assert_eq!(json_val_2["id"], "mid_2");
        assert_eq!(json_val_2["deep_metadata"]["value"], "another_deep_val");
        assert_eq!(json_val_2["deep_metadata"]["tags"][0], "test");
        assert_eq!(json_val_2["deep_metadata"]["tags"][1], "nested");
        assert_eq!(json_val_2["deep_metadata"]["tags"][2], "struct");
    }

    #[test]
    fn test_cast_string_to_markdown() {
        let s = create_string_series(
            "markdown",
            vec![
                Some("# Header\n\nSome **bold** text"),
                Some("## Another header\n\n- List item 1\n- List item 2"),
                None,
            ],
        );

        let result =
            cast_series_to_fenic_dtype(&s, &FenicDType::StringType, &FenicDType::MarkdownType)
                .unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(
            result.str().unwrap().get(0),
            Some("# Header\n\nSome **bold** text")
        );
        assert_eq!(
            result.str().unwrap().get(1),
            Some("## Another header\n\n- List item 1\n- List item 2")
        );
        assert_eq!(result.str().unwrap().get(2), None);
    }

    #[test]
    fn test_cast_markdown_to_string() {
        let s = create_string_series(
            "markdown",
            vec![
                Some("# Title\n\nParagraph with *italic* text"),
                Some("> Blockquote\n\n```rust\nfn main() {}\n```"),
                None,
            ],
        );

        let result =
            cast_series_to_fenic_dtype(&s, &FenicDType::MarkdownType, &FenicDType::StringType)
                .unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(
            result.str().unwrap().get(0),
            Some("# Title\n\nParagraph with *italic* text")
        );
        assert_eq!(
            result.str().unwrap().get(1),
            Some("> Blockquote\n\n```rust\nfn main() {}\n```")
        );
        assert_eq!(result.str().unwrap().get(2), None);
    }

    #[test]
    fn test_cast_integer_to_markdown() {
        let s = create_int_series("numbers", vec![Some(42), Some(-100), None]);

        let result =
            cast_series_to_fenic_dtype(&s, &FenicDType::IntegerType, &FenicDType::MarkdownType)
                .unwrap();

        assert_eq!(result.len(), 3);
        // The integer should be converted to string first, then treated as markdown
        assert_eq!(result.str().unwrap().get(0), Some("42"));
        assert_eq!(result.str().unwrap().get(1), Some("-100"));
        assert_eq!(result.str().unwrap().get(2), None);
    }

    #[test]
    fn test_cast_markdown_to_integer() {
        let s = create_string_series(
            "markdown_numbers",
            vec![Some("42"), Some("-100"), Some("not a number"), None],
        );

        let result =
            cast_series_to_fenic_dtype(&s, &FenicDType::MarkdownType, &FenicDType::IntegerType)
                .unwrap();

        assert_eq!(result.len(), 4);
        // Should parse valid numbers and handle invalid ones appropriately
        assert_eq!(result.i64().unwrap().get(0), Some(42));
        assert_eq!(result.i64().unwrap().get(1), Some(-100));
        // "not a number" should result in null
        assert_eq!(result.i64().unwrap().get(2), None);
        assert_eq!(result.i64().unwrap().get(3), None);
    }

    #[test]
    fn test_cast_embedding_to_array() {
        let embedding_data = vec![
            Some(Series::new("".into(), vec![1.0f32, 2.0f32, 3.0f32])),
            Some(Series::new("".into(), vec![4.5f32, 5.5f32, 6.5f32])),
            None,
        ];

        // Create series with Array(Float32, 3) type (simulating embedding)
        let embedding_series = Series::new("embeddings".into(), embedding_data)
            .cast(&DataType::Array(Box::new(DataType::Float32), 3))
            .unwrap();

        let result = cast_series_to_fenic_dtype(
            &embedding_series,
            &FenicDType::EmbeddingType {
                dimensions: 3,
                embedding_model: "oai-small".to_string(),
            },
            &FenicDType::ArrayType {
                element_type: Box::new(FenicDType::FloatType),
            },
        )
        .unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result.list().unwrap().get_as_series(0).unwrap().len(), 3);
        assert_eq!(result.list().unwrap().get_as_series(1).unwrap().len(), 3);
        let third_embedding = result.list().unwrap().get_as_series(2);
        assert!(third_embedding.is_none() || third_embedding.unwrap().null_count() > 0);
    }
}
