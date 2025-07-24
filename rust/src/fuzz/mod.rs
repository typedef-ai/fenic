use polars::chunked_array::ops::arity;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rapidfuzz::distance::{damerau_levenshtein, hamming, indel, jaro, jaro_winkler, levenshtein};

#[polars_expr(output_type=Float64)]
fn normalized_indel_similarity(inputs: &[Series]) -> PolarsResult<Series> {
    // Use broadcast_binary_elementwise for string operations since it handles nulls safely.
    // binary_elementwise_values could process garbage data at null positions (not sure what Arrow does
    // when iterating over arrays with nulls - probably does unsafe access), which may not be
    // safe for string operations that expect valid UTF-8. We can benchmark this and decide whether
    // eliminating branching is worth it.
    let left = inputs[0].str()?;
    let right = inputs[1].str()?;
    let similarity: Float64Chunked = arity::broadcast_binary_elementwise(
        left,
        right,
        |left: Option<&str>, right: Option<&str>| match (left, right) {
            (Some(left), Some(right)) => {
                Some(indel::normalized_similarity(left.chars(), right.chars()) * 100.0)
            }
            _ => None,
        },
    );
    Ok(similarity.into_series())
}

#[polars_expr(output_type=Float64)]
fn normalized_levenshtein_similarity(inputs: &[Series]) -> PolarsResult<Series> {
    let left = inputs[0].str()?;
    let right = inputs[1].str()?;
    let similarity: Float64Chunked = arity::broadcast_binary_elementwise(
        left,
        right,
        |left: Option<&str>, right: Option<&str>| match (left, right) {
            (Some(left), Some(right)) => {
                Some(levenshtein::normalized_similarity(left.chars(), right.chars()) * 100.0)
            }
            _ => None,
        },
    );
    Ok(similarity.into_series())
}

#[polars_expr(output_type=Float64)]
fn normalized_damerau_levenshtein_similarity(inputs: &[Series]) -> PolarsResult<Series> {
    let left = inputs[0].str()?;
    let right = inputs[1].str()?;
    let similarity: Float64Chunked = arity::broadcast_binary_elementwise(
        left,
        right,
        |left: Option<&str>, right: Option<&str>| match (left, right) {
            (Some(left), Some(right)) => Some(
                damerau_levenshtein::normalized_similarity(left.chars(), right.chars()) * 100.0,
            ),
            _ => None,
        },
    );
    Ok(similarity.into_series())
}

#[polars_expr(output_type=Float64)]
fn normalized_jarowinkler_similarity(inputs: &[Series]) -> PolarsResult<Series> {
    let left = inputs[0].str()?;
    let right = inputs[1].str()?;
    let similarity: Float64Chunked = arity::broadcast_binary_elementwise(
        left,
        right,
        |left: Option<&str>, right: Option<&str>| match (left, right) {
            (Some(left), Some(right)) => {
                Some(jaro_winkler::normalized_similarity(left.chars(), right.chars()) * 100.0)
            }
            _ => None,
        },
    );
    Ok(similarity.into_series())
}

#[polars_expr(output_type=Float64)]
fn normalized_jaro_similarity(inputs: &[Series]) -> PolarsResult<Series> {
    let left = inputs[0].str()?;
    let right = inputs[1].str()?;
    let similarity: Float64Chunked = arity::broadcast_binary_elementwise(
        left,
        right,
        |left: Option<&str>, right: Option<&str>| match (left, right) {
            (Some(left), Some(right)) => {
                Some(jaro::normalized_similarity(left.chars(), right.chars()) * 100.0)
            }
            _ => None,
        },
    );
    Ok(similarity.into_series())
}

#[polars_expr(output_type=Float64)]
fn normalized_hamming_similarity(inputs: &[Series]) -> PolarsResult<Series> {
    let left = inputs[0].str()?;
    let right = inputs[1].str()?;
    let similarity: Float64Chunked = arity::broadcast_binary_elementwise(
        left,
        right,
        |left: Option<&str>, right: Option<&str>| match (left, right) {
            (Some(left), Some(right)) => Some(
                hamming::normalized_similarity_with_args(
                    left.chars(),
                    right.chars(),
                    &hamming::Args::default().pad(true),
                ) * 100.0,
            ),
            _ => None,
        },
    );
    Ok(similarity.into_series())
}
