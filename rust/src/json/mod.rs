pub mod jq;
use jaq_core::{Ctx, RcIter};
use jaq_json::Val;
use polars::prelude::*;
use polars_arrow::array::ValueSize;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use serde_json::Value;

#[pyfunction]
pub fn py_validate_jq_query(query: &str) -> PyResult<()> {
    match jq::build_jq_query(query) {
        Ok(_) => Ok(()),
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[derive(Deserialize)]
struct JqKwargs {
    query: String,
}

fn jq_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "jq".into(),
        DataType::List(Box::new(DataType::String)),
    ))
}

/// Runs a jq filter expression on each JSON string in the input column.
///
/// # Arguments
///
/// * `inputs` - A slice of input `Series`. The first Series must be a UTF8 string column
///              containing JSON-formatted strings.
/// * `kwargs` - Keyword arguments containing the jq query string (`kwargs.query`).
///
/// # Returns
///
/// Returns a new `Series` of type `List<String>`, where each row is a list of JSON strings
/// representing the results of applying the jq filter to the corresponding input JSON string.
///
/// - If the jq filter produces multiple results, all results are included as strings in the list.
/// - If no results are produced, the corresponding output is a null value.
/// - If input is null, or JSON parsing or jq evaluation fails, the output is null for that row.
///
/// # Errors
///
/// Returns a `PolarsError::ComputeError` if building the jq filter fails.
///
/// # Behavior
///
/// For each JSON string in the input:
/// 1. Parses the string into a JSON value.
/// 2. Runs the jq filter on the parsed JSON value.
/// 3. Collects all jq results as strings.
/// 4. Appends these results as a list to the output builder.
/// 5. Appends null if any error occurs or if the input is null.
///
/// # Example
///
/// ```ignore
/// // Suppose inputs is a Series with JSON strings:
/// // ["{\"foo\": 1}", "{\"foo\": 2}", null]
/// // And kwargs.query = ".foo"
/// // The output is a List<String> Series:
/// // [["1"], ["2"], null]
/// ```
#[polars_expr(output_type_func=jq_output)]
fn jq_expr(inputs: &[Series], kwargs: JqKwargs) -> PolarsResult<Series> {
    // Build the jq filter from the query string in kwargs
    let filter = jq::build_jq_query(&kwargs.query)
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let jq_inputs = RcIter::new(core::iter::empty());

    // Extract the string column from inputs
    let ca = inputs[0].str()?;

    // Prepare builder for output List<String>, estimate capacity:
    // ca.len() = number of rows
    // ca.get_values_size() = total number of strings in all lists
    let mut builder =
        ListStringChunkedBuilder::new("jq".into(), ca.len(), ca.get_values_size() * 5);

    // Iterate over each string in the column
    for opt_str in ca.into_iter() {
        if let Some(s) = opt_str {
            // Parse input JSON string
            match serde_json::from_str::<Value>(s) {
                Ok(val) => {
                    let v: Val = val.into();

                    // Run the jq filter in context; empty Ctx + inputs slice as empty since unused
                    let results = filter
                        .run((Ctx::new([], &jq_inputs), v))
                        .collect::<Result<Vec<_>, _>>();

                    match results {
                        Ok(values) => {
                            if values.is_empty() {
                                builder.append_null();
                            } else {
                                let result_strings: Vec<String> =
                                    values.iter().map(|v| v.to_string()).collect();
                                builder
                                    .append_values_iter(result_strings.iter().map(|s| s.as_str()));
                            }
                        }
                        Err(e) => {
                            return Err(PolarsError::ComputeError(
                                format!(
                                    "jq query execution failed: {}. Query: '{}'",
                                    e, kwargs.query
                                )
                                .into(),
                            ));
                        }
                    }
                }
                Err(e) => {
                    unreachable!("Invalid JSON encountered in jq operation. This should have been caught by type validation. JSON: '{}', Error: {}", s, e);
                }
            }
        } else {
            builder.append_null();
        }
    }

    Ok(builder.finish().into_series())
}
