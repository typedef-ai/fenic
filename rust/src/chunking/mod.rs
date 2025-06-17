pub mod splitter;

use polars::prelude::*;
use polars_arrow::array::ValueSize;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use splitter::{RecursiveChunkLengthFunction, RecursiveChunkingCharacters, TextSplitter};
use std::str::FromStr;

#[derive(Deserialize)]
pub struct TextChunkKwargs {
    desired_chunk_size: usize,
    chunk_overlap: usize,
    chunk_length_function_name: String,
    chunking_character_set_name: String,
    chunking_character_set_custom_characters: Option<Vec<String>>,
}

fn chunk_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "chunks".into(),
        DataType::List(Box::new(DataType::String)),
    ))
}

#[polars_expr(output_type_func=chunk_output)]
fn text_chunk_expr(inputs: &[Series], kwargs: TextChunkKwargs) -> PolarsResult<Series> {
    let chunk_length_function =
        RecursiveChunkLengthFunction::from_str(&kwargs.chunk_length_function_name)
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;

    let chunking_chars = match &kwargs.chunking_character_set_custom_characters {
        Some(custom) => RecursiveChunkingCharacters::Custom(custom.clone()),
        None => RecursiveChunkingCharacters::from_str(&kwargs.chunking_character_set_name)
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?,
    };

    let chunker = TextSplitter::new(
        chunking_chars,
        chunk_length_function,
        kwargs.desired_chunk_size,
        kwargs.chunk_overlap,
    )
    .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;

    let ca = inputs[0].str()?;

    let total_original_chars = ca.get_values_size() as f64;
    let overlap_percentage = kwargs.chunk_overlap as f64 / kwargs.desired_chunk_size as f64;
    let expansion_factor = 1.0 / (1.0 - overlap_percentage);
    let capacity = total_original_chars * expansion_factor;

    // Ensure capacity is not zero or negative if desired_chunk_size is less than or equal to chunk_overlap.
    // Also, make sure it's at least the original size.
    let capacity = if capacity.is_finite() && capacity > 0.0 {
        capacity as usize
    } else {
        // Fallback to original size if calculation is invalid (e.g., 100% overlap)
        // or just a default large enough number.
        total_original_chars as usize * 2 // A safe fallback if math goes wrong
    };

    let mut builder = ListStringChunkedBuilder::new("chunks".into(), ca.len(), capacity);

    for opt_val in ca {
        match opt_val {
            Some(val) => {
                let chunks = chunker
                    .recursively_chunk_text(val)
                    .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;

                builder.append_values_iter(chunks.iter().map(|s| s.as_str()));
            }
            None => {
                builder.append_null();
            }
        }
    }

    Ok(builder.finish().into_series())
}

#[polars_expr(output_type=UInt32)]
fn count_tokens(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let chunker = TextSplitter::new(
        RecursiveChunkingCharacters::Ascii,
        RecursiveChunkLengthFunction::TokenCount,
        1, // dummy values to construct a TextSplitter to just access the tokenizer
        0,
    )
    .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let counts = ca
        .into_iter()
        .map(|opt| match opt {
            Some(s) => chunker
                .chunk_length(s)
                .map(|val: usize| Some(val as u32))
                .map_err(|e| PolarsError::ComputeError(e.to_string().into())),
            None => Ok(None),
        })
        .collect::<Result<Vec<Option<u32>>, _>>()?;

    Ok(Series::new("counts".into(), counts))
}
