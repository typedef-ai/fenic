pub mod render;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::jinja::render::render;

#[derive(Deserialize)]
struct JinjaKwargs {
    template: String,
}

fn jinja_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new("jinja".into(), DataType::String))
}

/// Renders a Jinja template for each row in the input struct column.
///
/// # Arguments
///
/// * `inputs` - A slice of input `Series`. The first Series must be a struct column
///              containing the template variables as fields.
/// * `kwargs` - Keyword arguments containing the template string.
///
/// # Returns
///
/// Returns a new `Series` of type `String`, where each row is the rendered template
/// for the corresponding input struct.
///
/// # Behavior
///
/// For each struct in the input:
/// 1. Extracts values from ALL struct fields
/// 2. Builds a context object with field names as keys
/// 3. Renders the Jinja template with the context
/// 4. Returns the rendered string
///
/// # Example
///
/// ```ignore
/// // Suppose inputs is a struct Series with fields "name" and "age"
/// // And kwargs.template = "Hello {{ name }}! You are {{ age }} years old."
/// // The output is a String Series with rendered templates for each row
/// ```
#[polars_expr(output_type_func=jinja_output)]
fn jinja_render(inputs: &[Series], kwargs: JinjaKwargs) -> PolarsResult<Series> {
    let template = kwargs.template;
    render(inputs, &template)
}
