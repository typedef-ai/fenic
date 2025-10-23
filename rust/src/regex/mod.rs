use polars::datatypes::DataType;
use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use regex::Regex;
use std::collections::hash_map::Entry;
use std::collections::HashMap;

#[pyfunction]
pub fn py_validate_regex(regex: &str) -> PyResult<()> {
    match Regex::new(regex) {
        Ok(_) => Ok(()),
        Err(error) => Err(PyValueError::new_err(error.to_string())),
    }
}

/// Get or compile a regex from the cache.
/// Uses the Entry API for efficient single-lookup caching.
fn get_or_compile_regex<'a>(
    regex_cache: &'a mut HashMap<String, Regex>,
    pattern: &str,
) -> PolarsResult<&'a Regex> {
    match regex_cache.entry(pattern.to_string()) {
        Entry::Occupied(entry) => Ok(entry.into_mut()),
        Entry::Vacant(entry) => {
            let regex = Regex::new(pattern).map_err(|e| {
                PolarsError::ComputeError(
                    format!("Invalid regex pattern '{}': {}", pattern, e).into(),
                )
            })?;
            Ok(entry.insert(regex))
        }
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
                    let regex = get_or_compile_regex(&mut regex_cache, pattern)?;

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
                    let regex = get_or_compile_regex(&mut regex_cache, pattern)?;
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
                    Some(
                        StringChunked::from_iter_values(PlSmallStr::EMPTY, matches.into_iter())
                            .into_series(),
                    )
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

#[cfg(test)]
mod tests {
    use super::*;

    // Pure Rust tests - these can run with `cargo test`
    #[test]
    fn test_get_or_compile_regex_valid_pattern() {
        let mut cache = HashMap::new();
        let pattern = r"\d+";

        let result = get_or_compile_regex(&mut cache, pattern);
        assert!(result.is_ok());

        let regex = result.unwrap();
        assert!(regex.is_match("123"));
        assert!(!regex.is_match("abc"));
    }

    #[test]
    fn test_get_or_compile_regex_caches() {
        let mut cache = HashMap::new();
        let pattern = r"[a-z]+";

        // First call should compile and cache
        let result1 = get_or_compile_regex(&mut cache, pattern);
        assert!(result1.is_ok());
        assert_eq!(cache.len(), 1);

        // Second call should use cache (verify cache size doesn't change)
        let result2 = get_or_compile_regex(&mut cache, pattern);
        assert!(result2.is_ok());
        assert_eq!(cache.len(), 1);

        // Different pattern should add to cache
        let result3 = get_or_compile_regex(&mut cache, r"\d+");
        assert!(result3.is_ok());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_get_or_compile_regex_invalid_pattern() {
        let mut cache = HashMap::new();
        let pattern = r"[invalid(";

        let result = get_or_compile_regex(&mut cache, pattern);
        assert!(result.is_err());

        // Verify error message format
        let err = result.unwrap_err();
        let err_msg = err.to_string();
        assert!(err_msg.contains("Invalid regex pattern"));
        assert!(err_msg.contains("[invalid("));
    }

    #[test]
    fn test_get_or_compile_regex_special_patterns() {
        let mut cache = HashMap::new();

        // Test email pattern
        let email_pattern = r"(\w+)@(\w+)\.(\w+)";
        let result = get_or_compile_regex(&mut cache, email_pattern);
        assert!(result.is_ok());
        assert!(result.unwrap().is_match("test@example.com"));

        // Test word boundary pattern
        let word_boundary = r"\bword\b";
        let result = get_or_compile_regex(&mut cache, word_boundary);
        assert!(result.is_ok());
        assert!(result.unwrap().is_match("a word here"));
    }

    // PyO3 tests - these are tested via Python integration tests
    // Note: py_validate_regex is tested in tests/_backends/local/functions/test_regexp_functions.py
    // because standalone PyO3 tests require Python runtime linking
}
