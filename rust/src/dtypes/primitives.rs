use crate::dtypes::types::FenicDType;
use polars::prelude::*;

/// Cast between primitive types using Polars' standard casting.
///
/// This function handles casting between basic types like integers, floats,
/// strings, and booleans. For more complex types, it falls back to using
/// the canonical Polars type.
pub fn cast_primitive_to_primitive(
    s: &Series,
    _src_type: &FenicDType,
    dest_type: &FenicDType,
) -> PolarsResult<Series> {
    // For primitive types, we can use Polars' standard casting
    let polars_dtype = dest_type.canonical_polars_type();
    s.cast(&polars_dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_int_series(name: &str, values: Vec<Option<i64>>) -> Series {
        Series::new(name.into(), values)
    }

    #[test]
    fn test_cast_primitive_types() {
        // Test basic type casting between primitives
        let int_series = create_int_series("test", vec![Some(42), Some(-10), None]);

        // Int to Float
        let result = cast_primitive_to_primitive(
            &int_series,
            &FenicDType::IntegerType,
            &FenicDType::FloatType,
        )
        .unwrap();
        assert_eq!(result.f32().unwrap().get(0), Some(42.0));
        assert_eq!(result.f32().unwrap().get(1), Some(-10.0));
        assert_eq!(result.f32().unwrap().get(2), None);

        // Int to String
        let result = cast_primitive_to_primitive(
            &int_series,
            &FenicDType::IntegerType,
            &FenicDType::StringType,
        )
        .unwrap();
        assert_eq!(result.str().unwrap().get(0), Some("42"));
        assert_eq!(result.str().unwrap().get(1), Some("-10"));
        assert_eq!(result.str().unwrap().get(2), None);
    }
}
