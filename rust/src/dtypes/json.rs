use crate::dtypes::types::FenicDType;
use polars::chunked_array::builder::get_list_builder;
use polars::prelude::*;
use polars_arrow::{
    array::{FixedSizeListArray, Float32Array},
    bitmap::Bitmap,
};
use serde_json::Value;

/// Cast a string series to JSON type by validating each string.
/// Invalid JSON strings are replaced with null values.
pub fn cast_string_to_json(s: &Series) -> PolarsResult<Series> {
    let ca = s.str()?;
    let validated: Vec<Option<&str>> = ca
        .into_iter()
        .map(|opt_str| {
            opt_str.and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok().map(|_| s))
        })
        .collect();

    Ok(StringChunked::from_iter_options(ca.name().clone(), validated.into_iter()).into_series())
}

/// Cast any logical type series to JSON strings.
/// This is a unified function that handles all types by using anyvalue_to_json_value.
pub fn cast_fenic_dtype_to_json(s: &Series) -> PolarsResult<Series> {
    let mut json_strings = Vec::with_capacity(s.len());

    for i in 0..s.len() {
        let json_opt = match s.get(i)? {
            AnyValue::Null => None,
            value => {
                let json_value = anyvalue_to_json_value(value);
                serde_json::to_string(&json_value).ok()
            }
        };
        json_strings.push(json_opt);
    }

    Ok(Series::new(s.name().clone(), json_strings))
}

// Convert a JSON string series to any Fenic type
pub fn cast_json_to_fenic_dtype(s: &Series, target_type: &FenicDType) -> PolarsResult<Series> {
    let ca = s.str()?;

    // First, parse all JSON strings into Values
    let mut json_values = Vec::with_capacity(ca.len());
    for opt_json_str in ca.into_iter() {
        let value = match opt_json_str {
            Some(json_str) => match serde_json::from_str::<Value>(json_str) {
                Ok(json_val) => json_val,
                Err(_) => unreachable!(
                    "Internal error: JSON parsing failed unexpectedly. Input should have already been validated before this point."
                ),
            },
            None => Value::Null,
        };
        json_values.push(value);
    }

    cast_json_to_logical_type_helper(&json_values, target_type, s.name().clone())
}

/// Convert a JSON string series to any Fenic type.
fn cast_json_to_logical_type_helper(
    json_array: &[Value],
    target_type: &FenicDType,
    name: PlSmallStr,
) -> PolarsResult<Series> {
    match target_type {
        FenicDType::StringType => {
            let mut builder = StringChunkedBuilder::new(name, json_array.len());
            for json_val in json_array {
                append_string_value(&mut builder, json_val)?;
            }
            Ok(builder.finish().into_series())
        }

        FenicDType::IntegerType => {
            let mut builder = PrimitiveChunkedBuilder::<Int64Type>::new(name, json_array.len());
            for json_val in json_array {
                append_int_value(&mut builder, json_val)?;
            }
            Ok(builder.finish().into_series())
        }

        FenicDType::FloatType => {
            let mut builder = PrimitiveChunkedBuilder::<Float32Type>::new(name, json_array.len());
            for json_val in json_array {
                append_float_value(&mut builder, json_val)?;
            }
            Ok(builder.finish().into_series())
        }

        FenicDType::DoubleType => {
            let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new(name, json_array.len());
            for json_val in json_array {
                append_double_value(&mut builder, json_val)?;
            }
            Ok(builder.finish().into_series())
        }

        FenicDType::BooleanType => {
            let mut builder = BooleanChunkedBuilder::new(name, json_array.len());
            for json_val in json_array {
                append_bool_value(&mut builder, json_val)?;
            }
            Ok(builder.finish().into_series())
        }

        FenicDType::ArrayType { element_type } => {
            let mut builder = get_list_builder(
                &element_type.canonical_polars_type(),
                json_array.len() * 10, // Estimate inner capacity
                json_array.len(),
                name,
            );

            for json_val in json_array {
                append_array_value(&mut builder, json_val, element_type)?;
            }

            Ok(builder.finish().into_series())
        }

        FenicDType::EmbeddingType { dimensions, .. } => {
            // Use fixed-size array for embeddings
            let mut values = Vec::with_capacity(json_array.len() * dimensions);
            let mut validity = Vec::with_capacity(json_array.len());

            for json_val in json_array {
                match json_val {
                    Value::Array(arr) if arr.len() == *dimensions => {
                        let mut all_valid = true;
                        let start_idx = values.len();

                        for item in arr {
                            match item.as_f64() {
                                Some(f) => values.push(f as f32),
                                None => {
                                    all_valid = false;
                                    break;
                                }
                            }
                        }

                        if all_valid {
                            validity.push(true);
                        } else {
                            // Invalidate and backfill with 0.0
                            values.truncate(start_idx);
                            values.extend(std::iter::repeat_n(0.0f32, *dimensions));
                            validity.push(false);
                        }
                    }
                    // Anything that's not an array (or array of wrong length or with non-floats) → null
                    _ => {
                        values.extend(std::iter::repeat_n(0.0f32, *dimensions));
                        validity.push(false);
                    }
                }
            }

            // Create the inner Float32 array
            let values_array = Float32Array::from_vec(values);

            // Create the FixedSizeListArray
            let dtype = ArrowDataType::FixedSizeList(
                Box::new(ArrowField::new("item".into(), ArrowDataType::Float32, true)),
                *dimensions,
            );

            let validity_bitmap = if validity.iter().all(|&v| v) {
                None
            } else {
                Some(Bitmap::from_iter(validity))
            };

            let array = FixedSizeListArray::new(
                dtype,
                json_array.len(),
                Box::new(values_array),
                validity_bitmap,
            );

            let chunked = ArrayChunked::with_chunk(name, array);
            Ok(chunked.into_series())
        }

        FenicDType::StructType { struct_fields } => {
            // Create builders for each field
            let mut field_series = Vec::with_capacity(struct_fields.len());

            for field in struct_fields {
                let field_json: Vec<Value> = json_array
                    .iter()
                    .map(|json_val| match json_val {
                        Value::Object(obj) => obj.get(&field.name).cloned().unwrap_or(Value::Null),
                        Value::Null => Value::Null,
                        _ => Value::Null,
                    })
                    .collect();

                let field_series_result = cast_json_to_logical_type_helper(
                    &field_json,
                    &field.data_type,
                    field.name.as_str().into(),
                )?;

                field_series.push(field_series_result);
            }

            let struct_chunked =
                StructChunked::from_series(name, json_array.len(), field_series.iter())?;

            Ok(struct_chunked.into_series())
        }

        FenicDType::JsonType => {
            let mut builder = StringChunkedBuilder::new(name, json_array.len());
            for json_val in json_array {
                match json_val {
                    Value::Null => builder.append_null(),
                    _ => match serde_json::to_string(json_val) {
                        Ok(json_str) => builder.append_value(&json_str),
                        Err(_) => builder.append_null(),
                    },
                }
            }
            Ok(builder.finish().into_series())
        }

        _ => todo!("Builder not implemented for this type"),
    }
}

/// Convert a Polars AnyValue to a serde_json Value.
pub fn anyvalue_to_json_value(val: AnyValue) -> Value {
    match val {
        AnyValue::Null => Value::Null,
        AnyValue::Boolean(b) => Value::Bool(b),
        AnyValue::UInt8(u) => Value::from(u),
        AnyValue::UInt16(u) => Value::from(u),
        AnyValue::UInt32(u) => Value::from(u),
        AnyValue::UInt64(u) => Value::from(u),
        AnyValue::Int8(i) => Value::from(i),
        AnyValue::Int16(i) => Value::from(i),
        AnyValue::Int32(i) => Value::from(i),
        AnyValue::Int64(i) => Value::from(i),
        AnyValue::Float32(f) => serde_json::Number::from_f64(f as f64)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        AnyValue::Float64(f) => serde_json::Number::from_f64(f)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        AnyValue::String(s) => Value::String(s.into()),
        AnyValue::StringOwned(s) => Value::String(s.as_str().into()),
        AnyValue::Binary(b) => Value::String(String::from_utf8_lossy(b).into()),
        AnyValue::BinaryOwned(b) => Value::String(String::from_utf8_lossy(&b).to_string()),
        AnyValue::List(s) => {
            let mut arr = Vec::with_capacity(s.len());
            for val in s.iter() {
                arr.push(anyvalue_to_json_value(val));
            }
            Value::Array(arr)
        }
        AnyValue::Array(slice, _) => {
            let mut arr = Vec::with_capacity(slice.len());
            for val in slice.iter() {
                arr.push(anyvalue_to_json_value(val));
            }
            Value::Array(arr)
        }
        // For now, we clone struct field values using `.into_static()` to get owned `AnyValue<'static>` instances.
        // This approach is simple and safe, allowing us to walk each row without worrying about lifetimes tied to Arrow arrays.
        // Although cloning incurs some allocation overhead (especially for strings, binaries, and nested types), it provides correctness and ease of use.
        // The ideal method to extract `AnyValue` from an Arrow array at a given index is `arr_to_any_value`, but this function is currently `pub(crate)`
        // and not accessible outside the Polars crate. Similarly, the iterator `_iter_struct_av` on `AnyValue::Struct` is private (marked with an underscore).
        // To achieve zero-copy, lifetime-safe access without cloning, we would need to either copy and maintain this private logic ourselves,
        // or maintain a local patched fork of Polars where these methods are made public.
        // For now, cloning with `.into_static()` is a practical tradeoff for prototyping or simpler usage, with the option to optimize later.
        AnyValue::Struct(_, _, _) | AnyValue::StructOwned(_) => {
            let static_val = val.into_static();
            if let AnyValue::StructOwned(payload) = static_val {
                let mut map = serde_json::Map::new();
                let (values, fields) = payload.as_ref();

                for (field, value) in fields.iter().zip(values.iter()) {
                    map.insert(
                        field.name().to_string(),
                        anyvalue_to_json_value(value.clone()),
                    );
                }
                Value::Object(map)
            } else {
                Value::Null
            }
        }
        _ => Value::Null,
    }
}

// Helper functions for appending specific value types

fn append_string_value(builder: &mut StringChunkedBuilder, json_val: &Value) -> PolarsResult<()> {
    match json_val {
        Value::String(s) => builder.append_value(s),
        Value::Null => builder.append_null(),
        _ => {
            // Convert other types to string, fallback to null if conversion fails
            match serde_json::to_string(json_val) {
                Ok(json_str) => builder.append_value(&json_str),
                Err(_) => builder.append_null(),
            }
        }
    }
    Ok(())
}

fn append_int_value(
    builder: &mut PrimitiveChunkedBuilder<Int64Type>,
    json_val: &Value,
) -> PolarsResult<()> {
    match json_val {
        Value::Number(n) => match n.as_i64() {
            Some(i) => builder.append_value(i),
            None => builder.append_null(),
        },
        Value::Null => builder.append_null(),
        _ => builder.append_null(),
    }
    Ok(())
}

fn append_float_value(
    builder: &mut PrimitiveChunkedBuilder<Float32Type>,
    json_val: &Value,
) -> PolarsResult<()> {
    match json_val {
        Value::Number(n) => match n.as_f64() {
            Some(f) => builder.append_value(f as f32),
            None => builder.append_null(),
        },
        Value::Null => builder.append_null(),
        _ => builder.append_null(),
    }
    Ok(())
}

fn append_double_value(
    builder: &mut PrimitiveChunkedBuilder<Float64Type>,
    json_val: &Value,
) -> PolarsResult<()> {
    match json_val {
        Value::Number(n) => match n.as_f64() {
            Some(f) => builder.append_value(f),
            None => builder.append_null(),
        },
        Value::Null => builder.append_null(),
        _ => builder.append_null(),
    }
    Ok(())
}

fn append_bool_value(builder: &mut BooleanChunkedBuilder, json_val: &Value) -> PolarsResult<()> {
    match json_val {
        Value::Bool(b) => builder.append_value(*b),
        Value::Null => builder.append_null(),
        _ => builder.append_null(),
    }
    Ok(())
}

fn append_array_value<T>(
    builder: &mut T,
    json_val: &Value,
    element_type: &FenicDType,
) -> PolarsResult<()>
where
    T: ListBuilderTrait,
{
    match json_val {
        Value::Array(arr) => {
            let inner_series =
                cast_json_to_logical_type_helper(arr, element_type, PlSmallStr::from(""))?;
            builder.append_series(&inner_series)?;
        }
        _ => {
            builder.append_null();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::dtypes::StructField;
    use serde_json::json;

    use super::*;

    #[test]
    fn test_cast_string_to_json_valid() {
        let s = Series::new(
            "test".into(),
            vec![
                Some(r#"{"key": "value"}"#),
                Some(r#"[1, 2, 3]"#),
                Some(r#""simple string""#),
                None,
            ],
        );

        let result = cast_string_to_json(&s).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result.str().unwrap().get(0), Some(r#"{"key": "value"}"#));
        assert_eq!(result.str().unwrap().get(1), Some(r#"[1, 2, 3]"#));
        assert_eq!(result.str().unwrap().get(2), Some(r#""simple string""#));
        assert_eq!(result.str().unwrap().get(3), None);
    }

    #[test]
    fn test_cast_string_to_json_invalid() {
        let s = Series::new(
            "test".into(),
            vec![
                Some(r#"{"key": "value"}"#), // valid
                Some("invalid json"),        // invalid
                Some(""),                    // invalid
            ],
        );

        let result = cast_string_to_json(&s).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result.str().unwrap().get(0), Some(r#"{"key": "value"}"#));
        assert_eq!(result.str().unwrap().get(1), None); // invalid becomes null
        assert_eq!(result.str().unwrap().get(2), None); // invalid becomes null
    }

    #[test]
    fn test_anyvalue_to_json_value_primitives() {
        // Test primitive types
        assert_eq!(anyvalue_to_json_value(AnyValue::Null), json!(null));
        assert_eq!(anyvalue_to_json_value(AnyValue::Boolean(true)), json!(true));
        assert_eq!(anyvalue_to_json_value(AnyValue::Int64(42)), json!(42));
        assert_eq!(anyvalue_to_json_value(AnyValue::Float64(3.14)), json!(3.14));
        assert_eq!(
            anyvalue_to_json_value(AnyValue::String("test")),
            json!("test")
        );

        // Test integer types
        assert_eq!(anyvalue_to_json_value(AnyValue::UInt8(255)), json!(255));
        assert_eq!(anyvalue_to_json_value(AnyValue::Int32(-42)), json!(-42));

        // Test float edge cases
        assert_eq!(
            anyvalue_to_json_value(AnyValue::Float32(f32::NAN)),
            json!(null)
        );
        assert_eq!(
            anyvalue_to_json_value(AnyValue::Float64(f64::INFINITY)),
            json!(null)
        );
    }

    #[test]
    fn test_anyvalue_to_json_value_binary() {
        // Test binary data
        let binary_data = b"hello";
        assert_eq!(
            anyvalue_to_json_value(AnyValue::Binary(binary_data)),
            json!("hello")
        );

        // Test invalid UTF-8 binary
        let invalid_utf8 = &[0xFF, 0xFE, 0xFD];
        let result = anyvalue_to_json_value(AnyValue::Binary(invalid_utf8));
        // Should handle lossy conversion
        assert!(result.is_string());
    }

    #[test]
    fn test_cast_json_to_string() {
        let json_values = vec![
            json!("hello"),
            json!(42),
            json!(true),
            json!(null),
            json!({"key": "value"}),
        ];

        let json_strings: Vec<String> = json_values
            .iter()
            .map(|v| serde_json::to_string(v).unwrap())
            .collect();

        let s = Series::new("test".into(), json_strings);

        let result = cast_json_to_fenic_dtype(&s, &FenicDType::StringType).unwrap();
        assert_eq!(result.len(), 5);
        assert_eq!(result.str().unwrap().get(0), Some("hello"));
        assert_eq!(result.str().unwrap().get(1), Some("42"));
        assert_eq!(result.str().unwrap().get(2), Some("true"));
        assert_eq!(result.str().unwrap().get(3), None);
        assert!(result.str().unwrap().get(4).unwrap().contains("key"));
    }

    #[test]
    fn test_cast_json_to_integer() {
        let json_values = vec![
            json!(42),
            json!(3.14), // Should become null (not an integer)
            json!("not a number"),
            json!(null),
            json!(-100),
        ];

        let json_strings: Vec<String> = json_values
            .iter()
            .map(|v| serde_json::to_string(v).unwrap())
            .collect();

        let s = Series::new("test".into(), json_strings);

        let result = cast_json_to_fenic_dtype(&s, &FenicDType::IntegerType).unwrap();
        assert_eq!(result.len(), 5);
        assert_eq!(result.i64().unwrap().get(0), Some(42));
        assert_eq!(result.i64().unwrap().get(1), None); // Float becomes null
        assert_eq!(result.i64().unwrap().get(2), None); // Invalid conversion
        assert_eq!(result.i64().unwrap().get(3), None); // Null
        assert_eq!(result.i64().unwrap().get(4), Some(-100));
    }

    #[test]
    fn test_cast_json_to_float() {
        let json_values = vec![
            json!(42.5),
            json!(42), // Integer should convert to float
            json!("not a number"),
            json!(null),
            json!(-3.14),
        ];

        let json_strings: Vec<String> = json_values
            .iter()
            .map(|v| serde_json::to_string(v).unwrap())
            .collect();

        let s = Series::new("test".into(), json_strings);

        let result = cast_json_to_fenic_dtype(&s, &FenicDType::FloatType).unwrap();
        assert_eq!(result.len(), 5);
        assert_eq!(result.f32().unwrap().get(0), Some(42.5));
        assert_eq!(result.f32().unwrap().get(1), Some(42.0));
        assert_eq!(result.f32().unwrap().get(2), None); // Invalid conversion
        assert_eq!(result.f32().unwrap().get(3), None); // Null
        assert_eq!(result.f32().unwrap().get(4), Some(-3.14));
    }

    #[test]
    fn test_cast_json_to_boolean() {
        let json_values = vec![
            json!(true),
            json!(false),
            json!("not a boolean"),
            json!(null),
            json!(1), // Should become null (not a boolean)
        ];

        let json_strings: Vec<String> = json_values
            .iter()
            .map(|v| serde_json::to_string(v).unwrap())
            .collect();

        let s = Series::new("test".into(), json_strings);

        let result = cast_json_to_fenic_dtype(&s, &FenicDType::BooleanType).unwrap();
        assert_eq!(result.len(), 5);
        assert_eq!(result.bool().unwrap().get(0), Some(true));
        assert_eq!(result.bool().unwrap().get(1), Some(false));
        assert_eq!(result.bool().unwrap().get(2), None); // Invalid conversion
        assert_eq!(result.bool().unwrap().get(3), None); // Null
        assert_eq!(result.bool().unwrap().get(4), None); // Number becomes null
    }

    #[test]
    fn test_cast_json_to_array_strings() {
        let json_values = vec![
            json!(["hello", "world"]),
            json!([1, 2, 3]), // Numbers should convert to strings
            json!("not an array"),
            json!(null),
            json!([]), // Empty array
        ];

        let json_strings: Vec<String> = json_values
            .iter()
            .map(|v| serde_json::to_string(v).unwrap())
            .collect();

        let s = Series::new("test".into(), json_strings);

        let array_type = FenicDType::ArrayType {
            element_type: Box::new(FenicDType::StringType),
        };

        let result = cast_json_to_fenic_dtype(&s, &array_type).unwrap();
        assert_eq!(result.len(), 5);

        let list_ca = result.list().unwrap();

        // First array should have 2 elements
        let first_array = list_ca.get_as_series(0).unwrap();
        assert_eq!(first_array.len(), 2);
        assert_eq!(first_array.str().unwrap().get(0), Some("hello"));
        assert_eq!(first_array.str().unwrap().get(1), Some("world"));

        // Second array should convert numbers to strings
        let second_array = list_ca.get_as_series(1).unwrap();
        assert_eq!(second_array.len(), 3);
        assert_eq!(second_array.str().unwrap().get(0), Some("1"));
        assert_eq!(second_array.str().unwrap().get(1), Some("2"));
        assert_eq!(second_array.str().unwrap().get(2), Some("3"));

        // Invalid array should be null
        assert!(list_ca.get_as_series(2).is_none());
        assert!(list_ca.get_as_series(3).is_none());

        // Empty array should work
        let empty_array = list_ca.get_as_series(4).unwrap();
        assert_eq!(empty_array.len(), 0);
    }

    #[test]
    fn test_cast_json_to_array_integers() {
        let json_values = vec![
            json!([1, 2, 3]),
            json!([1.5, 2.7]), // Floats should become null in integer array
            json!(["1", "2"]), // Strings should become null
            json!([null, 5]),  // Mixed with nulls
        ];

        let json_strings: Vec<String> = json_values
            .iter()
            .map(|v| serde_json::to_string(v).unwrap())
            .collect();

        let s = Series::new("test".into(), json_strings);

        let array_type = FenicDType::ArrayType {
            element_type: Box::new(FenicDType::IntegerType),
        };

        let result = cast_json_to_fenic_dtype(&s, &array_type).unwrap();
        assert_eq!(result.len(), 4);

        let list_ca = result.list().unwrap();

        // First array should work
        let first_array = list_ca.get_as_series(0).unwrap();
        assert_eq!(first_array.len(), 3);
        assert_eq!(first_array.i64().unwrap().get(0), Some(1));
        assert_eq!(first_array.i64().unwrap().get(1), Some(2));
        assert_eq!(first_array.i64().unwrap().get(2), Some(3));

        // Second array should have nulls for floats
        let second_array = list_ca.get_as_series(1).unwrap();
        assert_eq!(second_array.len(), 2);
        assert_eq!(second_array.i64().unwrap().get(0), None); // 1.5 becomes null
        assert_eq!(second_array.i64().unwrap().get(1), None); // 2.7 becomes null

        // Third array should have nulls for strings
        let third_array = list_ca.get_as_series(2).unwrap();
        assert_eq!(third_array.len(), 2);
        assert_eq!(third_array.i64().unwrap().get(0), None); // "1" becomes null
        assert_eq!(third_array.i64().unwrap().get(1), None); // "2" becomes null

        // Fourth array mixed with nulls
        let fourth_array = list_ca.get_as_series(3).unwrap();
        assert_eq!(fourth_array.len(), 2);
        assert_eq!(fourth_array.i64().unwrap().get(0), None); // null
        assert_eq!(fourth_array.i64().unwrap().get(1), Some(5)); // 5
    }

    #[test]
    fn test_cast_json_to_struct_complete() {
        let json_values = vec![
            json!({"name": "Alice", "age": 30, "active": true}),
            json!({"name": "Bob", "age": 25, "active": false}),
        ];

        let json_strings: Vec<String> = json_values
            .iter()
            .map(|v| serde_json::to_string(v).unwrap())
            .collect();

        let s = Series::new("test".into(), json_strings);

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

        let result = cast_json_to_fenic_dtype(&s, &struct_type).unwrap();
        assert_eq!(result.len(), 2);

        let struct_ca = result.struct_().unwrap();
        let fields = struct_ca.fields_as_series();

        assert_eq!(fields.len(), 3);

        let name_series = &fields[0];
        let age_series = &fields[1];
        let active_series = &fields[2];

        // First object
        assert_eq!(name_series.str().unwrap().get(0), Some("Alice"));
        assert_eq!(age_series.i64().unwrap().get(0), Some(30));
        assert_eq!(active_series.bool().unwrap().get(0), Some(true));

        // Second object
        assert_eq!(name_series.str().unwrap().get(1), Some("Bob"));
        assert_eq!(age_series.i64().unwrap().get(1), Some(25));
        assert_eq!(active_series.bool().unwrap().get(1), Some(false));
    }

    #[test]
    fn test_cast_json_to_struct_missing_fields() {
        let json_values = vec![
            json!({"name": "Alice", "age": 30}), // Missing active field
            json!({"name": "Bob"}),              // Missing age and active
            json!({"age": 25}),                  // Missing name and active
            json!("not an object"),              // Invalid
            json!(null),
        ];

        let json_strings: Vec<String> = json_values
            .iter()
            .map(|v| serde_json::to_string(v).unwrap())
            .collect();

        let s = Series::new("test".into(), json_strings);

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

        let result = cast_json_to_fenic_dtype(&s, &struct_type).unwrap();
        assert_eq!(result.len(), 5);

        let struct_ca = result.struct_().unwrap();
        let fields = struct_ca.fields_as_series();

        let name_series = &fields[0];
        let age_series = &fields[1];
        let active_series = &fields[2];

        // First object: has name and age, missing active
        assert_eq!(name_series.str().unwrap().get(0), Some("Alice"));
        assert_eq!(age_series.i64().unwrap().get(0), Some(30));
        assert_eq!(active_series.bool().unwrap().get(0), None);

        // Second object: has name only
        assert_eq!(name_series.str().unwrap().get(1), Some("Bob"));
        assert_eq!(age_series.i64().unwrap().get(1), None);
        assert_eq!(active_series.bool().unwrap().get(1), None);

        // Third object: has age only
        assert_eq!(name_series.str().unwrap().get(2), None);
        assert_eq!(age_series.i64().unwrap().get(2), Some(25));
        assert_eq!(active_series.bool().unwrap().get(2), None);

        // Invalid objects should have all nulls
        assert_eq!(name_series.str().unwrap().get(3), None);
        assert_eq!(age_series.i64().unwrap().get(3), None);
        assert_eq!(active_series.bool().unwrap().get(3), None);

        assert_eq!(name_series.str().unwrap().get(4), None);
        assert_eq!(age_series.i64().unwrap().get(4), None);
        assert_eq!(active_series.bool().unwrap().get(4), None);
    }

    fn create_string_series(name: &str, values: Vec<Option<&str>>) -> Series {
        Series::new(name.into(), values)
    }

    fn create_int_series(name: &str, values: Vec<Option<i64>>) -> Series {
        Series::new(name.into(), values)
    }

    #[test]
    fn test_cast_struct_to_json() {
        let name_series =
            create_string_series("name", vec![Some("Alice"), Some("Bob"), None, None]);
        let age_series = create_int_series("age", vec![Some(30), None, Some(40), None]);

        let struct_series =
            StructChunked::from_series("person".into(), 4, [name_series, age_series].iter())
                .unwrap()
                .into_series();

        let result = cast_fenic_dtype_to_json(&struct_series).unwrap();
        assert_eq!(result.len(), 4);

        // First entry: both fields present
        let json1: serde_json::Value =
            serde_json::from_str(result.str().unwrap().get(0).unwrap()).unwrap();
        assert_eq!(json1["name"], "Alice");
        assert_eq!(json1["age"], 30);

        // Second entry: name present, age null
        let json2: serde_json::Value =
            serde_json::from_str(result.str().unwrap().get(1).unwrap()).unwrap();
        assert_eq!(json2["name"], "Bob");
        assert_eq!(json2["age"], serde_json::Value::Null);

        // Third entry: name null, age present
        let json3: serde_json::Value =
            serde_json::from_str(result.str().unwrap().get(2).unwrap()).unwrap();
        assert_eq!(json3["name"], serde_json::Value::Null);
        assert_eq!(json3["age"], 40);

        // Fourth entry: both fields null (but struct exists)
        let json4: serde_json::Value =
            serde_json::from_str(result.str().unwrap().get(3).unwrap()).unwrap();
        assert_eq!(json4["name"], serde_json::Value::Null);
        assert_eq!(json4["age"], serde_json::Value::Null);
    }

    #[test]
    fn test_cast_array_to_json() {
        // Create an array of strings
        let string_arrays = vec![
            Some(Series::new("".into(), vec!["hello", "world"])),
            Some(Series::new("".into(), vec!["foo", "bar", "baz"])),
            None,
        ];
        let array_series = Series::new("test".into(), string_arrays);

        let result = cast_fenic_dtype_to_json(&array_series).unwrap();
        assert_eq!(result.len(), 3);

        // Check first array becomes JSON array
        let json_str_1 = result.str().unwrap().get(0).unwrap();
        let json_val_1: serde_json::Value = serde_json::from_str(json_str_1).unwrap();
        assert_eq!(json_val_1, json!(["hello", "world"]));

        // Check second array
        let json_str_2 = result.str().unwrap().get(1).unwrap();
        let json_val_2: serde_json::Value = serde_json::from_str(json_str_2).unwrap();
        assert_eq!(json_val_2, json!(["foo", "bar", "baz"]));

        // Check null array
        assert_eq!(result.str().unwrap().get(2), None);
    }
}
