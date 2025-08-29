use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use regex::Regex;

#[pyfunction]
pub fn py_validate_regex(regex: &str) -> PyResult<()> {
    match Regex::new(regex) {
        Ok(_) => Ok(()),
        Err(error) => Err(PyValueError::new_err(error.to_string())),
    }
}
