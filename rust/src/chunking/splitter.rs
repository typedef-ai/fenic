use serde::Deserialize;
use std::fmt;
use tiktoken_rs::CoreBPE;

/// A list of strings that can be used to recursively split text into chunks.
/// These are ordered from most to least preferred.
const RECURSIVE_ASCII_ONLY_CHUNK_SPLIT_STRINGS: &[&str] = &[
    "\n\n", // Double newline (paragraph break)
    "\n",   // Single newline
    ".",    // End of sentence
    ",",    // Comma
    ";",    // Semicolon
    ":",    // Colon
    " ",    // Space
    "-",    // Hyphen
    "",     // Empty string (character by character if needed)
];

const RECURSIVE_UNICODE_CHUNK_SPLIT_STRINGS: &[&str] = &[
    "\n\n",     // Double newline (paragraph break)
    "\n",       // Single newline
    ".",        // End of sentence
    ",",        // Comma
    "\u{200b}", // Zero-width space
    "\u{ff0c}", // Fullwidth comma
    "\u{3001}", // Ideographic comma
    "\u{ff0e}", // Fullwidth full stop
    "\u{3002}", // Ideographic full stop
    ";",        // Semicolon
    ":",        // Colon
    " ",        // Space
    "-",        // Hyphen
    "",         // Empty string (character by character if needed)
];

#[derive(Debug)]
pub enum RecursiveChunkingCharacters {
    Ascii,
    Unicode,
    Custom(Vec<String>),
}

#[derive(Debug)]
pub enum ChunkingError {
    InvalidChunkingCharacters,
    InvalidChunkLengthFunction,
    CustomCharactersRequired,
    ChunkOverlapTooLarge,
    TokenizerError(String),
    NoTokenizerFound,
    NoSeparatorFound,
}

impl fmt::Display for ChunkingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChunkingError::InvalidChunkingCharacters => write!(f, "Invalid chunking characters!"),
            ChunkingError::InvalidChunkLengthFunction => {
                write!(f, "Invalid chunk length function!")
            }
            ChunkingError::CustomCharactersRequired => {
                write!(f, "Must instantiate with custom characters")
            }
            ChunkingError::ChunkOverlapTooLarge => {
                write!(f, "Chunk overlap cannot be larger than chunk size!")
            }
            ChunkingError::TokenizerError(e) => write!(f, "Tokenizer error: {}", e),
            ChunkingError::NoTokenizerFound => write!(f, "No tokenizer found"),
            ChunkingError::NoSeparatorFound => {
                write!(f, "No valid separator remaining for recursive chunking!")
            }
        }
    }
}

impl std::error::Error for ChunkingError {}

impl std::str::FromStr for RecursiveChunkingCharacters {
    type Err = ChunkingError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "ASCII" => RecursiveChunkingCharacters::Ascii,
            "UNICODE" => RecursiveChunkingCharacters::Unicode,
            "CUSTOM" => return Err(ChunkingError::CustomCharactersRequired),
            _ => return Err(ChunkingError::InvalidChunkingCharacters),
        })
    }
}

#[derive(Debug, Deserialize)]
pub enum RecursiveChunkLengthFunction {
    CharacterCount,
    WordCount,
    TokenCount,
}

impl std::str::FromStr for RecursiveChunkLengthFunction {
    type Err = ChunkingError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "CHARACTER" => RecursiveChunkLengthFunction::CharacterCount,
            "WORD" => RecursiveChunkLengthFunction::WordCount,
            "TOKEN" => RecursiveChunkLengthFunction::TokenCount,
            _ => return Err(ChunkingError::InvalidChunkLengthFunction),
        })
    }
}

pub struct TextSplitter {
    chunking_characters: RecursiveChunkingCharacters,
    chunk_length_function: RecursiveChunkLengthFunction,
    chunk_size: usize,
    chunk_overlap: usize,
    tokenizer: Option<CoreBPE>,
}

impl TextSplitter {
    pub fn new(
        chunking_characters: RecursiveChunkingCharacters,
        chunk_length_function: RecursiveChunkLengthFunction,
        chunk_size: usize,
        chunk_overlap: usize,
    ) -> Result<Self, ChunkingError> {
        if chunk_overlap >= chunk_size {
            return Err(ChunkingError::ChunkOverlapTooLarge);
        }
        match chunk_length_function {
            RecursiveChunkLengthFunction::TokenCount => {
                let tokenizer = tiktoken_rs::cl100k_base();
                match tokenizer {
                    Ok(tokenizer) => Ok(Self {
                        chunking_characters,
                        chunk_length_function,
                        chunk_size,
                        chunk_overlap,
                        tokenizer: Some(tokenizer),
                    }),
                    Err(e) => Err(ChunkingError::TokenizerError(e.to_string())),
                }
            }
            _ => Ok(Self {
                chunking_characters,
                chunk_length_function,
                chunk_size,
                chunk_overlap,
                tokenizer: None,
            }),
        }
    }

    /// Recursively splits text by the given separators, progressively splitting the text into smaller
    /// and smaller chunks until each is under the expected chunk size.
    pub fn recursively_chunk_text(&self, text: &str) -> Result<Vec<String>, ChunkingError> {
        // Trim the input text to remove leading/trailing whitespace
        let trimmed_text = text.trim();

        // If the text is empty after trimming, return an empty vector
        if trimmed_text.is_empty() {
            return Ok(vec![]);
        }

        // Get the separators to use
        let separators = match &self.chunking_characters {
            RecursiveChunkingCharacters::Ascii => RECURSIVE_ASCII_ONLY_CHUNK_SPLIT_STRINGS.to_vec(),
            RecursiveChunkingCharacters::Unicode => RECURSIVE_UNICODE_CHUNK_SPLIT_STRINGS.to_vec(),
            RecursiveChunkingCharacters::Custom(chars) => {
                chars.iter().map(|s| s.as_str()).collect()
            }
        };

        // Split the text recursively
        let chunks = self.split_text_recursive(trimmed_text, &separators)?;

        // Post-process the chunks
        let mut result = chunks;
        result.retain(|chunk| !chunk.trim().is_empty());
        result.iter_mut().for_each(|chunk| {
            *chunk = chunk.trim().to_string();
        });

        Ok(result)
    }

    /// Helper method for recursive text chunking
    fn split_text_recursive(
        &self,
        text: &str,
        separators: &[&str],
    ) -> Result<Vec<String>, ChunkingError> {
        let mut final_chunks = Vec::new();

        // If we've exhausted all separators, return the text as a single chunk
        if separators.is_empty() {
            return Ok(vec![text.to_string()]);
        }

        // Get the appropriate separator to use, choosing the first one that appears in the text,
        // to avoid splits that do not perform any work.
        let mut separator = match separators.first() {
            Some(s) => s,
            None => return Err(ChunkingError::NoSeparatorFound),
        };
        let mut new_separators = Vec::new();
        for (i, s) in separators.iter().enumerate() {
            if s.is_empty() {
                separator = s;
                break;
            }

            if text.contains(s) {
                separator = s;
                new_separators = separators[i + 1..].to_vec();
                break;
            }
        }

        // Split the text by the separator
        let splits: Vec<&str> = if separator.is_empty() {
            // If separator is empty, we need to handle character-by-character splitting
            // We'll collect the characters into a Vec of string slices
            // This avoids allocating a new String for each character
            let mut result = Vec::with_capacity(text.len());
            let mut start = 0;

            for (i, c) in text.char_indices() {
                // For each character, create a slice of just that character
                // This is more efficient than converting each char to a String
                result.push(&text[start..i + c.len_utf8()]);
                start = i + c.len_utf8();
            }

            result
        } else {
            // For non-empty separators, we can use split which returns string slices
            text.split(separator).collect()
        };

        // Filter out empty splits
        let splits: Vec<&str> = splits.into_iter().filter(|s| !s.is_empty()).collect();

        // Now go merging things, recursively splitting longer texts
        let mut valid_splits: Vec<(String, usize)> = Vec::new();

        for s in splits {
            let split_length = self.chunk_length(s)?;

            if split_length < self.chunk_size {
                // Store the split and its length as a tuple
                valid_splits.push((s.to_string(), split_length));
            } else {
                // If we have good splits, merge them and add to final chunks
                if !valid_splits.is_empty() {
                    let merged_text = self.merge_splits(&valid_splits, separator)?;
                    final_chunks.extend(merged_text);
                    valid_splits.clear();
                }

                // If no more separators, add the split as is
                if new_separators.is_empty() {
                    final_chunks.push(s.to_string());
                } else {
                    // Otherwise, recursively split with the remaining separators
                    let other_chunks = self.split_text_recursive(s, &new_separators)?;
                    final_chunks.extend(other_chunks);
                }
            }
        }

        // Don't forget the remaining good splits
        if !valid_splits.is_empty() {
            let merged_text = self.merge_splits(&valid_splits, separator)?;
            final_chunks.extend(merged_text);
        }

        Ok(final_chunks)
    }

    /// Helper method to merge chunk_contents into a single chunk
    fn merge_chunk(&self, chunk_contents: &[String], separator: &str) -> Option<String> {
        let text = chunk_contents.join(separator);
        let text = text.trim();
        if text.is_empty() {
            None
        } else {
            Some(text.to_string())
        }
    }

    /// Merge valid splits (under the chunk size), greedily into larger chunks, to ensure
    /// chunks are as large as possible
    fn merge_splits(
        &self,
        splits: &[(String, usize)],
        separator: &str,
    ) -> Result<Vec<String>, ChunkingError> {
        let mut merged_chunks = Vec::new();
        let mut current_chunk_contents: Vec<String> = Vec::new();
        let mut current_chunk_length = 0;

        // Get separator length once
        let separator_len = if !separator.is_empty() {
            self.chunk_length(separator)?
        } else {
            0
        };

        for (s, split_length) in splits {
            // Check if adding this split would exceed chunk_size
            let separator_needed = if !current_chunk_contents.is_empty() {
                separator_len
            } else {
                0
            };
            // If adding the next split would take the chunk over the chunk size,
            // finalize the current chunk and create an overlap chunk.
            if current_chunk_length + split_length + separator_needed > self.chunk_size
                && !current_chunk_contents.is_empty()
            {
                if let Some(merged_chunk) = self.merge_chunk(&current_chunk_contents, separator) {
                    merged_chunks.push(merged_chunk);
                }

                // Create overlap chunk by removing splits from the current chunk until it is under
                // the chunk overlap size. This is greedy, so it might remove more overlap than
                // is technically required, but is a lot more performant than walking backwards through
                // the result
                while current_chunk_length > self.chunk_overlap
                    || (current_chunk_length + split_length + separator_needed > self.chunk_size
                        && current_chunk_length > 0)
                {
                    if let Some(first) = current_chunk_contents.first() {
                        let first_len = self.chunk_length(first)?;
                        let separator_len_for_first = if current_chunk_contents.len() > 1 {
                            separator_len
                        } else {
                            0
                        };
                        current_chunk_length -= first_len + separator_len_for_first;
                        current_chunk_contents.remove(0);
                    } else {
                        break;
                    }
                }
            }

            // Add the split to the current document
            current_chunk_contents.push(s.clone());

            // Update the total length
            let separator_len_for_total = if current_chunk_contents.len() > 1 {
                separator_len
            } else {
                0
            };
            current_chunk_length += split_length + separator_len_for_total;
        }

        // Don't forget the last document
        if !current_chunk_contents.is_empty() {
            if let Some(doc) = self.merge_chunk(&current_chunk_contents, separator) {
                merged_chunks.push(doc);
            }
        }

        Ok(merged_chunks)
    }

    pub fn chunk_length(&self, chunk: &str) -> Result<usize, ChunkingError> {
        match self.chunk_length_function {
            RecursiveChunkLengthFunction::CharacterCount => Ok(chunk.len()),
            RecursiveChunkLengthFunction::WordCount => Ok(chunk.split_whitespace().count()),
            RecursiveChunkLengthFunction::TokenCount => {
                if let Some(tokenizer) = &self.tokenizer {
                    let encoding = tokenizer.encode_ordinary(chunk);
                    let length = encoding.len();
                    Ok(length)
                } else {
                    Err(ChunkingError::NoTokenizerFound)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::fixture;
    use rstest::rstest;

    const CORPUS_URL: &str =
        "https://typedef-assets.s3.us-west-2.amazonaws.com/example_texts/pride_and_prejudice";

    #[fixture]
    fn chunking_corpus() -> String {
        let response = reqwest::blocking::get(CORPUS_URL).unwrap();
        let body = response.text();
        match body {
            Ok(body) => body,
            Err(e) => {
                panic!("Failed to download corpus: {}", e);
            }
        }
    }

    use super::*;
    #[rstest]
    fn test_recursively_chunk_corpus_by_word_count(chunking_corpus: String) {
        let chunker = TextSplitter::new(
            RecursiveChunkingCharacters::Ascii,
            RecursiveChunkLengthFunction::WordCount,
            500,
            50,
        );
        let chunks = chunker
            .unwrap()
            .recursively_chunk_text(chunking_corpus.as_str());
        assert!(chunks.is_ok());
        assert_eq!(290, chunks.unwrap().len());
    }

    #[rstest]
    fn test_recursively_chunk_corpus_by_character_count(chunking_corpus: String) {
        let chunker = TextSplitter::new(
            RecursiveChunkingCharacters::Ascii,
            RecursiveChunkLengthFunction::CharacterCount,
            2500,
            250,
        );
        let chunks = chunker
            .unwrap()
            .recursively_chunk_text(chunking_corpus.as_str());
        assert!(chunks.is_ok());
        assert_eq!(344, chunks.unwrap().len());
    }

    #[rstest]
    fn test_recursively_chunk_corpus_by_token_count(chunking_corpus: String) {
        let chunker = TextSplitter::new(
            RecursiveChunkingCharacters::Ascii,
            RecursiveChunkLengthFunction::TokenCount,
            500,
            50,
        );
        let chunks = chunker
            .unwrap()
            .recursively_chunk_text(chunking_corpus.as_str());
        assert!(chunks.is_ok());
        assert_eq!(438, chunks.unwrap().len());
    }

    #[rstest]
    fn test_recursively_chunk_corpus_w_custom_characters_by_word_count(chunking_corpus: String) {
        let chunker = TextSplitter::new(
            RecursiveChunkingCharacters::Custom(vec![
                ".".to_string(),
                ",".to_string(),
                " ".to_string(),
                "".to_string(),
            ]),
            RecursiveChunkLengthFunction::WordCount,
            500,
            50,
        );
        let chunks = chunker
            .unwrap()
            .recursively_chunk_text(chunking_corpus.as_str());
        assert!(chunks.is_ok());
        assert_eq!(308, chunks.unwrap().len());
    }
}
