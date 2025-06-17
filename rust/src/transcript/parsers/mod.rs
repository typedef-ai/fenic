mod generic;
mod srt;

pub use generic::GenericTranscriptParser;
pub use srt::SrtParser;

use crate::transcript::types::{FormatParser, ParseError, UnifiedTranscriptEntry};
use std::collections::HashMap;

/// Returns true if the given line appears to be a header line.
/// We assume that a header line contains a '(' followed by a digit.
pub fn is_header_line(line: &str) -> bool {
    line.rfind('(')
        .and_then(|open_idx| line.get(open_idx + 1..)?.chars().next())
        .map(|c| c.is_ascii_digit())
        .unwrap_or(false)
}

/// Returns true if the line contains only digits (after trimming), i.e. looks like a subtitle index.
pub fn looks_like_subtitle_index(line: &str) -> bool {
    line.trim().parse::<usize>().is_ok()
}

/// Returns true if the line contains the arrow sequence expected in timestamp lines.
pub fn looks_like_timestamp(line: &str) -> bool {
    // Using simple substring search here.
    line.contains("-->")
}

/// A registry to hold and expose parsers.
/// Using a hash map with keys (for example, file extension or format names)
pub struct ParserRegistry {
    parsers: HashMap<String, Box<dyn FormatParser>>,
}

impl ParserRegistry {
    /// Create a new registry.
    pub fn new() -> Self {
        Self {
            parsers: HashMap::new(),
        }
    }

    /// Register a new parser with the given key.
    pub fn register_parser(&mut self, key: &str, parser: Box<dyn FormatParser>) {
        self.parsers.insert(key.to_string(), parser);
    }

    /// Retrieve and run the parser for the given key on the provided input.
    pub fn parse(&self, key: &str, input: &str) -> Result<Vec<UnifiedTranscriptEntry>, ParseError> {
        if let Some(parser) = self.parsers.get(key) {
            parser.parse(input)
        } else {
            Err(ParseError::UnknownTranscriptFormat(format!(
                "Unknown Format: {}",
                key
            )))
        }
    }

    /// List the keys of all registered parsers.
    pub fn list_parsers(&self) -> Vec<&String> {
        self.parsers.keys().collect()
    }
}

impl Default for ParserRegistry {
    fn default() -> Self {
        let mut registry = ParserRegistry::new();
        registry.register_parser("srt", Box::new(SrtParser));
        registry.register_parser("generic", Box::new(GenericTranscriptParser));
        registry
    }
}
