/// Unified transcript entry structure that works for all transcript formats.
/// This provides a consistent schema regardless of the source format.
#[derive(Debug, Clone)]
pub struct UnifiedTranscriptEntry {
    /// Entry index/number (auto-generated for formats that don't have it)
    pub index: Option<i64>,
    /// Speaker name/identifier (None for formats like SRT that don't have speakers)
    pub speaker: Option<String>,
    /// Start time in seconds from beginning of conversation
    pub start_time: f64,
    /// End time in seconds (computed for formats that only have start time)
    pub end_time: Option<f64>,
    /// Duration in seconds (computed from end_time - start_time when available)
    pub duration: Option<f64>,
    /// The actual text content
    pub content: String,
    /// Which format parser was used ("srt", "generic", etc.)
    pub format: String,
}

#[derive(Debug)]
pub enum ParseError {
    InvalidTranscriptFormat(String),
    UnknownTranscriptFormat(String),
    Utf8Error(std::str::Utf8Error),
}

impl From<std::str::Utf8Error> for ParseError {
    fn from(err: std::str::Utf8Error) -> Self {
        ParseError::Utf8Error(err)
    }
}

pub trait FormatParser {
    fn parse(&self, input: &str) -> Result<Vec<UnifiedTranscriptEntry>, ParseError>;
}
