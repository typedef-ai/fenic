// Module structure
pub mod cast;
pub mod collections;
pub mod json;
pub mod primitives;
pub mod types;

// Re-export main types and functions
pub use cast::cast_series_to_fenic_dtype;
pub use types::{FenicDType, StructField};

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

/// Keyword arguments passed into the `cast_expr` function from the user.
///
/// The `dtype_str` field is expected to be a JSON string
/// representing a Fenic logical type (LfDType).
#[derive(Deserialize, Debug)]
pub struct CastKwargs {
    source_dtype: String,
    dest_dtype: String,
}

/// This function is used to compute the output type of the `cast_expr`
/// based on the provided `dtype_str`. It deserializes the JSON
/// into an `LfDType` and converts it into a corresponding Polars `DataType`.
///
/// This function is used only for schema inference and not for actual casting.
fn fenic_dtype_str_to_polars_dtype(
    _input_fields: &[Field],
    kwargs: CastKwargs,
) -> PolarsResult<Field> {
    // If deserialization fails, it's a critical bug in upstream logic.
    let fenic_type =
        serde_json::from_str::<FenicDType>(&kwargs.dest_dtype).expect("Invalid Fenic type string");

    Ok(Field::new(
        "casted".into(),
        fenic_type.canonical_polars_type(),
    ))
}

/// A Polars expression function that casts a Series to a custom Fenic logical type.
///
#[polars_expr(output_type_func_with_kwargs=fenic_dtype_str_to_polars_dtype)]
fn cast_expr(inputs: &[Series], kwargs: CastKwargs) -> PolarsResult<Series> {
    let src_type = serde_json::from_str::<FenicDType>(&kwargs.source_dtype)
        .expect("Invalid Fenic type string");
    let dest_type =
        serde_json::from_str::<FenicDType>(&kwargs.dest_dtype).expect("Invalid Fenic type string");

    cast_series_to_fenic_dtype(&inputs[0], &src_type, &dest_type)
}
