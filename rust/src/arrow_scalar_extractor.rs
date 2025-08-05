use minijinja::Value as JinjaValue;
use polars::prelude::*;
use polars_arrow::array::*;
use polars_arrow::datatypes::ArrowDataType;
use serde_json::Value as JsonValue;
use std::collections::BTreeMap;

trait ArrowToValue: Sized {
    // Conversion for primitive types
    fn from_null() -> Self;
    fn from_str(s: &str) -> Self;
    fn from_bool(b: bool) -> Self;
    fn from_i8(i: i8) -> Self;
    fn from_i16(i: i16) -> Self;
    fn from_i32(i: i32) -> Self;
    fn from_i64(i: i64) -> Self;
    fn from_u8(u: u8) -> Self;
    fn from_u16(u: u16) -> Self;
    fn from_u32(u: u32) -> Self;
    fn from_u64(u: u64) -> Self;
    fn from_f32(f: f32) -> Self;
    fn from_f64(f: f64) -> Self;

    // Container conversions
    fn from_struct(fields: Vec<(String, Self)>) -> Self;
    fn from_list(values: Vec<Self>) -> Self;
}

impl ArrowToValue for JinjaValue {
    fn from_null() -> Self {
        JinjaValue::from(())
    }

    fn from_str(s: &str) -> Self {
        JinjaValue::from(s)
    }

    fn from_bool(b: bool) -> Self {
        JinjaValue::from(b)
    }

    fn from_i8(i: i8) -> Self {
        JinjaValue::from(i)
    }

    fn from_i16(i: i16) -> Self {
        JinjaValue::from(i)
    }

    fn from_i32(i: i32) -> Self {
        JinjaValue::from(i)
    }

    fn from_i64(i: i64) -> Self {
        JinjaValue::from(i)
    }

    fn from_u8(u: u8) -> Self {
        JinjaValue::from(u)
    }

    fn from_u16(u: u16) -> Self {
        JinjaValue::from(u)
    }

    fn from_u32(u: u32) -> Self {
        JinjaValue::from(u)
    }

    fn from_u64(u: u64) -> Self {
        JinjaValue::from(u)
    }

    fn from_f32(f: f32) -> Self {
        JinjaValue::from(f)
    }

    fn from_f64(f: f64) -> Self {
        JinjaValue::from(f)
    }

    fn from_struct(fields: Vec<(String, Self)>) -> Self {
        let mut map = BTreeMap::new();
        for (key, value) in fields {
            map.insert(key, value);
        }
        JinjaValue::from(map)
    }

    fn from_list(values: Vec<Self>) -> Self {
        JinjaValue::from(values)
    }
}

impl ArrowToValue for JsonValue {
    fn from_null() -> Self {
        JsonValue::Null
    }

    fn from_str(s: &str) -> Self {
        JsonValue::from(s)
    }

    fn from_bool(b: bool) -> Self {
        JsonValue::from(b)
    }

    fn from_i8(i: i8) -> Self {
        JsonValue::from(i)
    }

    fn from_i16(i: i16) -> Self {
        JsonValue::from(i)
    }

    fn from_i32(i: i32) -> Self {
        JsonValue::from(i)
    }

    fn from_i64(i: i64) -> Self {
        JsonValue::from(i)
    }

    fn from_u8(u: u8) -> Self {
        JsonValue::from(u)
    }

    fn from_u16(u: u16) -> Self {
        JsonValue::from(u)
    }

    fn from_u32(u: u32) -> Self {
        JsonValue::from(u)
    }

    fn from_u64(u: u64) -> Self {
        JsonValue::from(u)
    }

    fn from_f32(f: f32) -> Self {
        JsonValue::from(f)
    }

    fn from_f64(f: f64) -> Self {
        JsonValue::from(f)
    }

    fn from_struct(fields: Vec<(String, Self)>) -> Self {
        let map: serde_json::Map<String, JsonValue> = fields.into_iter().collect();
        JsonValue::Object(map)
    }

    fn from_list(values: Vec<Self>) -> Self {
        JsonValue::Array(values)
    }
}

// Macro for downcasting arrays
macro_rules! downcast_array {
    ($array:expr, $type:ty) => {
        $array.as_any().downcast_ref::<$type>().ok_or_else(|| {
            PolarsError::ComputeError(format!("Failed to downcast to {}", stringify!($type)).into())
        })?
    };
}

pub struct ArrowScalarConverter;

impl ArrowScalarConverter {
    // Public API methods that specify the concrete type
    pub fn to_jinja(&self, array: &dyn Array, row_idx: usize) -> PolarsResult<JinjaValue> {
        self.convert(array, row_idx)
    }

    pub fn to_json(&self, array: &dyn Array, row_idx: usize) -> PolarsResult<JsonValue> {
        self.convert(array, row_idx)
    }

    // Single generic implementation that works for both JinjaValue and JsonValue
    fn convert<V: ArrowToValue>(&self, array: &dyn Array, row_idx: usize) -> PolarsResult<V> {
        if array.is_null(row_idx) {
            return Ok(V::from_null());
        }

        match array.dtype() {
            ArrowDataType::Utf8 => {
                let str_array = downcast_array!(array, Utf8Array<i32>);
                Ok(V::from_str(str_array.value(row_idx)))
            }
            ArrowDataType::LargeUtf8 => {
                let str_array = downcast_array!(array, Utf8Array<i64>);
                Ok(V::from_str(str_array.value(row_idx)))
            }
            ArrowDataType::Utf8View => {
                let str_array = downcast_array!(array, Utf8ViewArray);
                Ok(V::from_str(str_array.value(row_idx)))
            }
            ArrowDataType::Binary => {
                let binary_array = downcast_array!(array, BinaryArray<i32>);
                Ok(V::from_str(
                    String::from_utf8_lossy(binary_array.value(row_idx)).as_ref(),
                ))
            }
            ArrowDataType::BinaryView => {
                let binary_array = downcast_array!(array, BinaryViewArray);
                Ok(V::from_str(
                    String::from_utf8_lossy(binary_array.value(row_idx)).as_ref(),
                ))
            }
            ArrowDataType::Boolean => {
                let bool_array = downcast_array!(array, BooleanArray);
                Ok(V::from_bool(bool_array.value(row_idx)))
            }
            ArrowDataType::Int8 => {
                let int_array = downcast_array!(array, PrimitiveArray<i8>);
                Ok(V::from_i8(int_array.value(row_idx)))
            }
            ArrowDataType::Int16 => {
                let int_array = downcast_array!(array, PrimitiveArray<i16>);
                Ok(V::from_i16(int_array.value(row_idx)))
            }
            ArrowDataType::Int32 => {
                let int_array = downcast_array!(array, PrimitiveArray<i32>);
                Ok(V::from_i32(int_array.value(row_idx)))
            }
            ArrowDataType::Int64 => {
                let int_array = downcast_array!(array, PrimitiveArray<i64>);
                Ok(V::from_i64(int_array.value(row_idx)))
            }
            ArrowDataType::UInt8 => {
                let uint_array = downcast_array!(array, PrimitiveArray<u8>);
                Ok(V::from_u8(uint_array.value(row_idx)))
            }
            ArrowDataType::UInt16 => {
                let uint_array = downcast_array!(array, PrimitiveArray<u16>);
                Ok(V::from_u16(uint_array.value(row_idx)))
            }
            ArrowDataType::UInt32 => {
                let uint_array = downcast_array!(array, PrimitiveArray<u32>);
                Ok(V::from_u32(uint_array.value(row_idx)))
            }
            ArrowDataType::UInt64 => {
                let uint_array = downcast_array!(array, PrimitiveArray<u64>);
                Ok(V::from_u64(uint_array.value(row_idx)))
            }
            ArrowDataType::Float32 => {
                let float_array = downcast_array!(array, PrimitiveArray<f32>);
                Ok(V::from_f32(float_array.value(row_idx)))
            }
            ArrowDataType::Float64 => {
                let float_array = downcast_array!(array, PrimitiveArray<f64>);
                Ok(V::from_f64(float_array.value(row_idx)))
            }
            ArrowDataType::Struct(_) => {
                let struct_array = downcast_array!(array, StructArray);
                self.convert_struct(struct_array, row_idx)
            }
            ArrowDataType::List(_) => {
                let list_array = downcast_array!(array, ListArray<i32>);
                self.convert_list(list_array, row_idx)
            }
            ArrowDataType::LargeList(_) => {
                let list_array = downcast_array!(array, ListArray<i64>);
                self.convert_large_list(list_array, row_idx)
            }
            ArrowDataType::FixedSizeList(_, _) => {
                let list_array = downcast_array!(array, FixedSizeListArray);
                self.convert_fixed_size_list(list_array, row_idx)
            }
            _ => Err(PolarsError::ComputeError(
                format!(
                    "Unsupported Arrow data type for conversion: {:?}",
                    array.dtype()
                )
                .into(),
            )),
        }
    }

    fn convert_struct<V: ArrowToValue>(
        &self,
        struct_array: &StructArray,
        row_idx: usize,
    ) -> PolarsResult<V> {
        // Struct is not null, convert each field normally
        let mut fields = Vec::new();
        for (field, array) in struct_array.fields().iter().zip(struct_array.values()) {
            let value: V = self.convert(array.as_ref(), row_idx)?;
            fields.push((field.name.clone().to_string(), value));
        }
        Ok(V::from_struct(fields))
    }

    fn convert_list<V: ArrowToValue>(
        &self,
        list_array: &ListArray<i32>,
        row_idx: usize,
    ) -> PolarsResult<V> {
        let list_slice = list_array.value(row_idx);
        let mut values = Vec::new();

        for item_idx in 0..list_slice.len() {
            let item_value = self.convert(list_slice.as_ref(), item_idx)?;
            values.push(item_value);
        }

        Ok(V::from_list(values))
    }

    fn convert_large_list<V: ArrowToValue>(
        &self,
        list_array: &ListArray<i64>,
        row_idx: usize,
    ) -> PolarsResult<V> {
        let list_slice = list_array.value(row_idx);
        let mut values = Vec::new();

        for item_idx in 0..list_slice.len() {
            let item_value = self.convert(list_slice.as_ref(), item_idx)?;
            values.push(item_value);
        }

        Ok(V::from_list(values))
    }

    fn convert_fixed_size_list<V: ArrowToValue>(
        &self,
        list_array: &FixedSizeListArray,
        row_idx: usize,
    ) -> PolarsResult<V> {
        let list_size = list_array.size();

        let values_array = list_array.values();
        let start_idx = row_idx * list_size;

        let mut values = Vec::new();
        for i in 0..list_size {
            let item_value = self.convert(values_array.as_ref(), start_idx + i)?;
            values.push(item_value);
        }

        Ok(V::from_list(values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minijinja::Value as JinjaValue;
    use polars_arrow::datatypes::ArrowDataType;
    use polars_arrow::offset::OffsetsBuffer;
    use serde_json::json;

    #[test]
    fn test_primitives_to_jinja_and_json() {
        let converter = ArrowScalarConverter;

        // Boolean
        let bool_array = BooleanArray::from_slice([true, false]);
        assert_eq!(
            converter.to_jinja(&bool_array, 0).unwrap(),
            JinjaValue::from(true)
        );
        assert_eq!(converter.to_json(&bool_array, 1).unwrap(), json!(false));

        // Int8
        let int_array = PrimitiveArray::<i8>::from_slice([42, 7]);
        assert_eq!(
            converter.to_jinja(&int_array, 0).unwrap(),
            JinjaValue::from(42)
        );
        assert_eq!(converter.to_json(&int_array, 1).unwrap(), json!(7));

        // Int16
        let int_array = PrimitiveArray::<i16>::from_slice([42, 7]);
        assert_eq!(
            converter.to_jinja(&int_array, 0).unwrap(),
            JinjaValue::from(42)
        );
        assert_eq!(converter.to_json(&int_array, 1).unwrap(), json!(7));

        // Int32
        let int_array = PrimitiveArray::<i32>::from_slice([42, 7]);
        assert_eq!(
            converter.to_jinja(&int_array, 0).unwrap(),
            JinjaValue::from(42)
        );
        assert_eq!(converter.to_json(&int_array, 1).unwrap(), json!(7));

        // Int64
        let int_array = PrimitiveArray::<i64>::from_slice([42, 7]);
        assert_eq!(
            converter.to_jinja(&int_array, 0).unwrap(),
            JinjaValue::from(42)
        );
        assert_eq!(converter.to_json(&int_array, 1).unwrap(), json!(7));

        // UInt8
        let uint_array = PrimitiveArray::<u8>::from_slice([8, 9]);
        assert_eq!(
            converter.to_jinja(&uint_array, 0).unwrap(),
            JinjaValue::from(8)
        );
        assert_eq!(converter.to_json(&uint_array, 1).unwrap(), json!(9));

        // UInt16
        let uint_array = PrimitiveArray::<u16>::from_slice([8, 9]);
        assert_eq!(
            converter.to_jinja(&uint_array, 0).unwrap(),
            JinjaValue::from(8)
        );
        assert_eq!(converter.to_json(&uint_array, 1).unwrap(), json!(9));

        // UInt32
        let uint_array = PrimitiveArray::<u32>::from_slice([8, 9]);
        assert_eq!(
            converter.to_jinja(&uint_array, 0).unwrap(),
            JinjaValue::from(8)
        );
        assert_eq!(converter.to_json(&uint_array, 1).unwrap(), json!(9));

        // UInt64
        let uint_array = PrimitiveArray::<u64>::from_slice([8, 9]);
        assert_eq!(
            converter.to_jinja(&uint_array, 0).unwrap(),
            JinjaValue::from(8)
        );
        assert_eq!(converter.to_json(&uint_array, 1).unwrap(), json!(9));

        // Float32
        let float_array = PrimitiveArray::<f32>::from_slice([3.0, 2.0]);
        assert_eq!(
            converter.to_jinja(&float_array, 0).unwrap(),
            JinjaValue::from(3.0)
        );
        assert_eq!(converter.to_json(&float_array, 1).unwrap(), json!(2.0));

        // Float64
        let float_array = PrimitiveArray::<f64>::from_slice([3.0, 2.0]);
        assert_eq!(
            converter.to_jinja(&float_array, 0).unwrap(),
            JinjaValue::from(3.0)
        );
        assert_eq!(converter.to_json(&float_array, 1).unwrap(), json!(2.0));

        // Utf8
        let utf8_array = Utf8Array::<i32>::from_slice(&["hello", "world"]);
        assert_eq!(
            converter.to_jinja(&utf8_array, 0).unwrap(),
            JinjaValue::from("hello")
        );
        assert_eq!(converter.to_json(&utf8_array, 1).unwrap(), json!("world"));

        // Utf8View
        let utf8_view_array = Utf8ViewArray::from_slice_values(&["hello", "world"]);
        assert_eq!(
            converter.to_jinja(&utf8_view_array, 0).unwrap(),
            JinjaValue::from("hello")
        );
        assert_eq!(
            converter.to_json(&utf8_view_array, 1).unwrap(),
            json!("world")
        );

        // LargeUtf8
        let large_utf8_array = Utf8Array::<i64>::from_slice(&["hello", "world"]);
        assert_eq!(
            converter.to_jinja(&large_utf8_array, 0).unwrap(),
            JinjaValue::from("hello")
        );
        assert_eq!(
            converter.to_json(&large_utf8_array, 1).unwrap(),
            json!("world")
        );

        // Binary
        let binary_array = BinaryArray::<i32>::from_slice(&[b"hello", b"world"]);
        assert_eq!(
            converter.to_jinja(&binary_array, 0).unwrap(),
            JinjaValue::from("hello")
        );
        assert_eq!(converter.to_json(&binary_array, 1).unwrap(), json!("world"));

        // BinaryView
        let binary_view_array = BinaryViewArray::from_slice_values(&[b"hello", b"world"]);
        assert_eq!(
            converter.to_jinja(&binary_view_array, 0).unwrap(),
            JinjaValue::from("hello")
        );
        assert_eq!(
            converter.to_json(&binary_view_array, 1).unwrap(),
            json!("world")
        );
    }

    #[test]
    fn test_struct_to_jinja_and_json() {
        let converter = ArrowScalarConverter;

        // Create the field arrays
        let field1 = PrimitiveArray::<i32>::from_slice([123, 456]);
        let field2 = BooleanArray::from_slice([true, false]);

        // Create the fields metadata
        let fields = vec![
            polars_arrow::datatypes::Field::new("a".into(), ArrowDataType::Int32, true),
            polars_arrow::datatypes::Field::new("b".into(), ArrowDataType::Boolean, true),
        ];

        // Create the struct array
        let struct_array = StructArray::new(
            ArrowDataType::Struct(fields.clone()),
            2,
            vec![Box::new(field1), Box::new(field2)],
            Some([true, false].into()),
        );

        let expected_jinja = {
            let mut map = std::collections::BTreeMap::new();
            map.insert("a".to_string(), JinjaValue::from(123));
            map.insert("b".to_string(), JinjaValue::from(true));
            JinjaValue::from(map)
        };

        let expected_json = json!({"a": 123, "b": true});

        assert_eq!(
            converter.to_jinja(&struct_array, 0).unwrap(),
            expected_jinja
        );
        assert_eq!(converter.to_json(&struct_array, 0).unwrap(), expected_json);

        // Second struct should have null fields
        let expected_jinja_2 = JinjaValue::from(());
        let expected_json_2 = json!(null);

        assert_eq!(
            converter.to_jinja(&struct_array, 1).unwrap(),
            expected_jinja_2
        );
        assert_eq!(
            converter.to_json(&struct_array, 1).unwrap(),
            expected_json_2
        );
    }

    #[test]
    fn test_list_to_jinja_and_json() {
        let converter = ArrowScalarConverter;

        // Create values array with 6 elements (for 2 lists of 3 elements each)
        let values = PrimitiveArray::<i32>::from_slice([1, 2, 3]);

        // Offsets: [0, 3, 3] means one list with 3 elements, and one list with 0 elements
        let offsets = vec![0i32, 3, 3];

        let data_type = ArrowDataType::List(Box::new(polars_arrow::datatypes::Field::new(
            "item".into(),
            ArrowDataType::Int32,
            false,
        )));

        let list_array = ListArray::<i32>::new(
            data_type,
            OffsetsBuffer::try_from(offsets).unwrap(),
            Box::new(values),
            Some([true, false].into()), // Second list is null
        );

        // First list should have values
        let expected_jinja = JinjaValue::from(vec![
            JinjaValue::from(1),
            JinjaValue::from(2),
            JinjaValue::from(3),
        ]);
        let expected_json = json!([1, 2, 3]);

        assert_eq!(converter.to_jinja(&list_array, 0).unwrap(), expected_jinja);
        assert_eq!(converter.to_json(&list_array, 0).unwrap(), expected_json);

        // Second list should have null values
        let expected_jinja_2 = JinjaValue::from_null();
        let expected_json_2 = json!(null);

        assert_eq!(
            converter.to_jinja(&list_array, 1).unwrap(),
            expected_jinja_2
        );
        assert_eq!(converter.to_json(&list_array, 1).unwrap(), expected_json_2);
    }

    #[test]
    fn test_large_list_to_jinja_and_json() {
        let converter = ArrowScalarConverter;

        // Create values array
        let values = PrimitiveArray::<i32>::from_slice([4, 5, 6]);

        // Offsets: [0, 3, 3] means one list with 3 elements, and one list with 0 elements
        let offsets = vec![0i64, 3, 3];

        let data_type = ArrowDataType::LargeList(Box::new(polars_arrow::datatypes::Field::new(
            "item".into(),
            ArrowDataType::Int32,
            false,
        )));

        let list_array = ListArray::<i64>::new(
            data_type,
            polars_arrow::offset::OffsetsBuffer::try_from(offsets).unwrap(),
            Box::new(values),
            Some([true, false].into()), // Second list is null
        );

        // First list should have values
        let expected_jinja = JinjaValue::from(vec![
            JinjaValue::from(4),
            JinjaValue::from(5),
            JinjaValue::from(6),
        ]);
        let expected_json = json!([4, 5, 6]);

        assert_eq!(converter.to_jinja(&list_array, 0).unwrap(), expected_jinja);
        assert_eq!(converter.to_json(&list_array, 0).unwrap(), expected_json);

        // Second list should have null values
        let expected_jinja_2 = JinjaValue::from_null();
        let expected_json_2 = json!(null);

        assert_eq!(
            converter.to_jinja(&list_array, 1).unwrap(),
            expected_jinja_2
        );
        assert_eq!(converter.to_json(&list_array, 1).unwrap(), expected_json_2);
    }

    #[test]
    fn test_fixed_size_list_to_jinja_and_json() {
        let converter = ArrowScalarConverter;

        // For fixed size list of size 2, with 2 lists
        let values = PrimitiveArray::<i32>::from_slice([7, 8, 0, 0]);

        let data_type = ArrowDataType::FixedSizeList(
            Box::new(polars_arrow::datatypes::Field::new(
                "item".into(),
                ArrowDataType::Int32,
                false,
            )),
            2,
        );

        let fixed_list_array = FixedSizeListArray::new(
            data_type,
            2, // length (number of lists)
            Box::new(values),
            Some([true, false].into()), // Second list is null
        );

        // First list should have values
        let expected_jinja = JinjaValue::from(vec![JinjaValue::from(7), JinjaValue::from(8)]);
        let expected_json = json!([7, 8]);

        assert_eq!(
            converter.to_jinja(&fixed_list_array, 0).unwrap(),
            expected_jinja
        );
        assert_eq!(
            converter.to_json(&fixed_list_array, 0).unwrap(),
            expected_json
        );

        // Second list should have null values
        let expected_jinja_2 = JinjaValue::from(());
        let expected_json_2 = json!(null);

        assert_eq!(
            converter.to_jinja(&fixed_list_array, 1).unwrap(),
            expected_jinja_2
        );
        assert_eq!(
            converter.to_json(&fixed_list_array, 1).unwrap(),
            expected_json_2
        );
    }

    #[test]
    fn test_nested_list_of_structs() {
        let converter = ArrowScalarConverter;

        // Create struct fields for each item in the list
        // We'll have a list of 2 structs, each with fields "name" (string) and "age" (int32)

        // Create arrays for the struct fields
        let names = Utf8Array::<i32>::from_slice(&["Alice", "Bob"]);
        let ages = PrimitiveArray::<i32>::from_slice([30, 25]);

        // Define the struct fields
        let struct_fields = vec![
            polars_arrow::datatypes::Field::new("name".into(), ArrowDataType::Utf8, false),
            polars_arrow::datatypes::Field::new("age".into(), ArrowDataType::Int32, false),
        ];

        // Create the struct array
        let struct_array = StructArray::new(
            ArrowDataType::Struct(struct_fields.clone()),
            2, // 2 structs
            vec![Box::new(names), Box::new(ages)],
            None, // No nulls
        );

        // Now create a list containing one element which is the entire struct array
        let offsets = vec![0i32, 2]; // One list containing both structs

        let list_data_type = ArrowDataType::List(Box::new(polars_arrow::datatypes::Field::new(
            "item".into(),
            ArrowDataType::Struct(struct_fields),
            false,
        )));

        let list_array = ListArray::<i32>::new(
            list_data_type,
            OffsetsBuffer::try_from(offsets).unwrap(),
            Box::new(struct_array),
            None,
        );

        // Expected values
        let expected_jinja = {
            let mut alice = BTreeMap::new();
            alice.insert("name".to_string(), JinjaValue::from("Alice"));
            alice.insert("age".to_string(), JinjaValue::from(30));

            let mut bob = BTreeMap::new();
            bob.insert("name".to_string(), JinjaValue::from("Bob"));
            bob.insert("age".to_string(), JinjaValue::from(25));

            JinjaValue::from(vec![JinjaValue::from(alice), JinjaValue::from(bob)])
        };

        let expected_json = json!([
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ]);

        assert_eq!(converter.to_jinja(&list_array, 0).unwrap(), expected_jinja);
        assert_eq!(converter.to_json(&list_array, 0).unwrap(), expected_json);
    }

    #[test]
    fn test_struct_with_nested_list() {
        let converter = ArrowScalarConverter;

        // Create a struct with a field that is a list
        // E.g., {"name": "Alice", "hobbies": ["reading", "coding"]}

        let names = Utf8Array::<i32>::from_slice(&["Alice"]);

        // Create the hobbies list
        let hobbies_values = Utf8Array::<i32>::from_slice(&["reading", "coding"]);
        let hobbies_offsets = vec![0i32, 2]; // One list with 2 elements

        let hobbies_list = ListArray::<i32>::new(
            ArrowDataType::List(Box::new(polars_arrow::datatypes::Field::new(
                "item".into(),
                ArrowDataType::Utf8,
                false,
            ))),
            OffsetsBuffer::try_from(hobbies_offsets).unwrap(),
            Box::new(hobbies_values),
            None,
        );

        // Create the struct
        let struct_fields = vec![
            polars_arrow::datatypes::Field::new("name".into(), ArrowDataType::Utf8, false),
            polars_arrow::datatypes::Field::new(
                "hobbies".into(),
                ArrowDataType::List(Box::new(polars_arrow::datatypes::Field::new(
                    "item".into(),
                    ArrowDataType::Utf8,
                    false,
                ))),
                false,
            ),
        ];

        let struct_array = StructArray::new(
            ArrowDataType::Struct(struct_fields),
            1, // 1 struct
            vec![Box::new(names), Box::new(hobbies_list)],
            None,
        );

        // Expected values
        let expected_jinja = {
            let mut map = BTreeMap::new();
            map.insert("name".to_string(), JinjaValue::from("Alice"));
            map.insert(
                "hobbies".to_string(),
                JinjaValue::from(vec![
                    JinjaValue::from("reading"),
                    JinjaValue::from("coding"),
                ]),
            );
            JinjaValue::from(map)
        };

        let expected_json = json!({
            "name": "Alice",
            "hobbies": ["reading", "coding"]
        });

        assert_eq!(
            converter.to_jinja(&struct_array, 0).unwrap(),
            expected_jinja
        );
        assert_eq!(converter.to_json(&struct_array, 0).unwrap(), expected_json);
    }

    #[test]
    fn test_null_primitive_values() {
        let converter = ArrowScalarConverter;

        // Test null handling for primitive types
        // Create an array with null values
        let values: Vec<Option<i32>> = vec![Some(42), None, Some(7)];
        let int_array = PrimitiveArray::<i32>::from_iter(values);

        // First element should be 42
        assert_eq!(
            converter.to_jinja(&int_array, 0).unwrap(),
            JinjaValue::from(42)
        );
        assert_eq!(converter.to_json(&int_array, 0).unwrap(), json!(42));

        // Second element should be null
        assert_eq!(
            converter.to_jinja(&int_array, 1).unwrap(),
            JinjaValue::from(())
        );
        assert_eq!(converter.to_json(&int_array, 1).unwrap(), json!(null));

        // Third element should be 7
        assert_eq!(
            converter.to_jinja(&int_array, 2).unwrap(),
            JinjaValue::from(7)
        );
        assert_eq!(converter.to_json(&int_array, 2).unwrap(), json!(7));
    }
}
