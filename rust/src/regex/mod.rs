use polars::datatypes::DataType;
use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use regex::Regex;
use std::collections::HashMap;

#[pyfunction]
pub fn py_validate_regex(regex: &str) -> PyResult<()> {
    match Regex::new(regex) {
        Ok(_) => Ok(()),
        Err(error) => Err(PyValueError::new_err(error.to_string())),
    }
}

#[polars_expr(output_type=Int32)]
fn regexp_instr(inputs: &[Series]) -> PolarsResult<Series> {
    let text_series = inputs[0].str()?;
    let pattern_series = inputs[1].str()?;
    let idx_series = inputs[2].i64()?;

    let len = text_series.len();
    let mut regex_cache: HashMap<String, Regex> = HashMap::new();

    // Handle broadcasting: if a series has length 1, it's a literal that should be broadcast
    let pattern_is_literal = pattern_series.len() == 1;
    let idx_is_literal = idx_series.len() == 1;

    let mut result_vec = Vec::with_capacity(len);

    for i in 0..len {
        let text_opt = text_series.get(i);
        // Use index 0 for literals (they'll be broadcast), otherwise use i
        let pattern_opt = pattern_series.get(if pattern_is_literal { 0 } else { i });
        let idx_opt = idx_series.get(if idx_is_literal { 0 } else { i });

        let value = match (text_opt, pattern_opt, idx_opt) {
            (Some(text), Some(pattern), Some(idx)) => {
                // Validate index is non-negative
                if idx < 0 {
                    Some(0)
                } else {
                    // Get or compile regex, return error if invalid
                    if !regex_cache.contains_key(pattern) {
                        match Regex::new(pattern) {
                            Ok(re) => {
                                regex_cache.insert(pattern.to_string(), re);
                            }
                            Err(e) => {
                                return Err(PolarsError::ComputeError(
                                    format!("Invalid regex pattern '{}': {}", pattern, e).into(),
                                ));
                            }
                        }
                    }

                    let regex = regex_cache.get(pattern).unwrap();

                    // Try to find a match
                    if let Some(captures) = regex.captures(text) {
                        let idx_usize = idx as usize;
                        // idx=0 is whole match, idx=1+ are capture groups
                        if let Some(matched) = captures.get(idx_usize) {
                            // Return 1-based position (PySpark compatibility)
                            Some((matched.start() as i32) + 1)
                        } else {
                            // No match for this group
                            Some(0)
                        }
                    } else {
                        // No match
                        Some(0)
                    }
                }
            }
            _ => None, // If any input is null, return null
        };

        result_vec.push(value);
    }

    Ok(Int32Chunked::from_iter_options(PlSmallStr::EMPTY, result_vec.into_iter()).into_series())
}

#[polars_expr(output_type_func=extract_all_output_type)]
fn regexp_extract_all(inputs: &[Series]) -> PolarsResult<Series> {
    let text_series = inputs[0].str()?;
    let pattern_series = inputs[1].str()?;
    let idx_series = inputs[2].i64()?;

    let len = text_series.len();
    let mut regex_cache: HashMap<String, Regex> = HashMap::new();

    // Handle broadcasting: if a series has length 1, it's a literal that should be broadcast
    let pattern_is_literal = pattern_series.len() == 1;
    let idx_is_literal = idx_series.len() == 1;

    let mut result_vec = Vec::with_capacity(len);

    for i in 0..len {
        let text_opt = text_series.get(i);
        // Use index 0 for literals (they'll be broadcast), otherwise use i
        let pattern_opt = pattern_series.get(if pattern_is_literal { 0 } else { i });
        let idx_opt = idx_series.get(if idx_is_literal { 0 } else { i });

        let value = match (text_opt, pattern_opt, idx_opt) {
            (Some(text), Some(pattern), Some(idx)) => {
                // Validate index is non-negative
                if idx < 0 {
                    Some(Series::new_empty(PlSmallStr::EMPTY, &DataType::String))
                } else {
                    // Get or compile regex, return error if invalid
                    if !regex_cache.contains_key(pattern) {
                        match Regex::new(pattern) {
                            Ok(re) => {
                                regex_cache.insert(pattern.to_string(), re);
                            }
                            Err(e) => {
                                return Err(PolarsError::ComputeError(
                                    format!("Invalid regex pattern '{}': {}", pattern, e).into(),
                                ));
                            }
                        }
                    }

                    let regex = regex_cache.get(pattern).unwrap();
                    let idx_usize = idx as usize;
                    let mut matches = Vec::new();

                    // Find all matches
                    for captures in regex.captures_iter(text) {
                        // idx=0 is whole match, idx=1+ are capture groups
                        if let Some(matched) = captures.get(idx_usize) {
                            matches.push(matched.as_str());
                        }
                    }

                    // Return as Series
                    Some(Series::from_iter(matches))
                }
            }
            _ => None, // If any input is null, return null
        };

        result_vec.push(value);
    }

    let list_chunked: ListChunked = result_vec.into_iter().collect();
    Ok(list_chunked.into_series())
}

fn extract_all_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    Ok(Field::new(
        field.name().clone(),
        DataType::List(Box::new(DataType::String)),
    ))
}
