pub mod arrow_scalar_extractor;
pub mod chunking;
pub mod dtypes;
pub mod fuzz;
pub mod jinja;
pub mod json;
pub mod markdown_json;
pub mod transcript;

use pyo3::prelude::*;

#[pymodule]
fn _polars_plugins(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(json::py_validate_jq_query, m)?)?;
    Ok(())
}
