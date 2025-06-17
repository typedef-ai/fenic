pub mod converter;
pub mod types;

use converter::MdAstConverter;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[derive(Debug)]
pub enum MarkdownToJsonError {
    ParseError(String),
    StructureError(String),
    SerdeError(serde_json::Error),
}

impl std::fmt::Display for MarkdownToJsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MarkdownToJsonError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            MarkdownToJsonError::StructureError(msg) => write!(f, "Structure error: {}", msg),
            MarkdownToJsonError::SerdeError(err) => write!(f, "Serialization error: {}", err),
        }
    }
}

impl std::error::Error for MarkdownToJsonError {}

#[polars_expr(output_type=String)]
fn md_to_json_expr(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;

    let mut converter = MdAstConverter::new();

    let out: Result<Vec<Option<String>>, PolarsError> = ca
        .into_iter()
        .map(|opt| match opt {
            Some(markdown) => match converter.convert_markdown(markdown) {
                Ok(json) => Ok(Some(json)),
                Err(e) => Err(PolarsError::ComputeError(
                    format!("Error converting markdown to JSON: {}", e).into(),
                )),
            },
            None => Ok(None),
        })
        .collect();

    Ok(Series::new("md_to_json".into(), out?))
}
