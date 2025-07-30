mod parsers;
mod types;

pub use parsers::ParserRegistry;
pub use types::{FormatParser, ParseError, UnifiedTranscriptEntry};

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct TranscriptFormatKwargs {
    format: String,
}

fn convert_format_to_struct(
    _input_fields: &[Field],
    kwargs: TranscriptFormatKwargs,
) -> PolarsResult<Field> {
    // Check that the format is one of the allowed formats
    let format = kwargs.format;
    let registry = ParserRegistry::default();
    if !registry.list_parsers().contains(&&format) {
        return Err(PolarsError::ComputeError(
            format!("Format {} not found", format).into(),
        ));
    }

    // Return unified schema regardless of format
    Ok(Field::new(
        "output".into(),
        DataType::Struct(vec![
            Field::new("index".into(), DataType::Int64), // Nullable
            Field::new("speaker".into(), DataType::String), // Nullable
            Field::new("start_time".into(), DataType::Float64), // Required
            Field::new("end_time".into(), DataType::Float64), // Nullable
            Field::new("duration".into(), DataType::Float64), // Nullable
            Field::new("content".into(), DataType::String), // Required
            Field::new("format".into(), DataType::String), // Required
        ]),
    ))
}

#[polars_expr(output_type_func_with_kwargs=convert_format_to_struct)]
fn ts_parse_expr(inputs: &[Series], kwargs: TranscriptFormatKwargs) -> PolarsResult<Series> {
    use polars::prelude::*;

    // Get the input Series as a string column (raw transcript text)
    let ca = inputs[0].str()?;
    // Get a parser from the registry
    let registry = ParserRegistry::default();
    // Get the format from the kwargs
    let format = kwargs.format;
    // Check that the format is one of the allowed formats
    if !registry.list_parsers().contains(&&format) {
        return Err(PolarsError::ComputeError(
            format!("Format {} not found", format).into(),
        ));
    }

    // Process each row: if parsing fails (or returns no entries), return an empty list.
    let out_values: Result<Vec<Option<AnyValue<'_>>>, PolarsError> = ca
        .into_iter()
        .map(|opt| {
            match opt {
                Some(val) => {
                    let result = registry.parse(&format, val);
                    match result {
                        Ok(entries) => {
                            // If no entries were parsed, return an empty List.
                            if entries.is_empty() {
                                return Ok(Some(AnyValue::List(Series::new(
                                    "".into(),
                                    Vec::<AnyValue>::new(),
                                ))));
                            }

                            // Convert all entries to unified struct format
                            let mut struct_values = Vec::new();
                            for entry in entries {
                                let fields = vec![
                                    Field::new("index".into(), DataType::Int64),
                                    Field::new("speaker".into(), DataType::String),
                                    Field::new("start_time".into(), DataType::Float64),
                                    Field::new("end_time".into(), DataType::Float64),
                                    Field::new("duration".into(), DataType::Float64),
                                    Field::new("content".into(), DataType::String),
                                    Field::new("format".into(), DataType::String),
                                ];
                                let values = vec![
                                    entry.index.map_or(AnyValue::Null, AnyValue::Int64),
                                    entry.speaker.as_ref().map_or(AnyValue::Null, |s| {
                                        AnyValue::StringOwned(s.clone().into())
                                    }),
                                    AnyValue::Float64(entry.start_time),
                                    entry.end_time.map_or(AnyValue::Null, AnyValue::Float64),
                                    entry.duration.map_or(AnyValue::Null, AnyValue::Float64),
                                    AnyValue::StringOwned(entry.content.clone().into()),
                                    AnyValue::StringOwned(entry.format.clone().into()),
                                ];
                                struct_values
                                    .push(AnyValue::StructOwned(Box::new((values, fields))));
                            }
                            Ok(Some(AnyValue::List(Series::new("".into(), struct_values))))
                        }
                        // If parsing fails for this row, return null.
                        Err(_e) => Ok(None),
                    }
                }
                None => Ok(None),
            }
        })
        .collect();

    let out_values: Vec<AnyValue> = out_values?
        .into_iter()
        .map(|opt| opt.unwrap_or(AnyValue::Null))
        .collect();

    Ok(Series::new("ts_parse".into(), out_values))
}
