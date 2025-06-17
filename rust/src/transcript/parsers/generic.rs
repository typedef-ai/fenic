use super::is_header_line;
use crate::transcript::types::{FormatParser, ParseError, UnifiedTranscriptEntry};
use memchr::memchr;

/// Parse generic format timestamps: "00:01.451" or "1:23:45.678"
/// Generic format typically uses dot as decimal separator
fn parse_generic_timestamp(timestamp: &str) -> Result<f64, ParseError> {
    let timestamp = timestamp.trim();

    // Split by colons to get time components
    let parts: Vec<&str> = timestamp.split(':').collect();

    match parts.len() {
        1 => {
            // Just seconds: "45.123"
            timestamp.parse::<f64>().map_err(|_| {
                ParseError::InvalidTranscriptFormat(format!(
                    "Invalid generic timestamp: {}",
                    timestamp
                ))
            })
        }
        2 => {
            // Minutes:seconds: "01:45.123"
            let minutes = parts[0].parse::<f64>().map_err(|_| {
                ParseError::InvalidTranscriptFormat(format!(
                    "Invalid minutes in generic timestamp: {}",
                    timestamp
                ))
            })?;
            let seconds = parts[1].parse::<f64>().map_err(|_| {
                ParseError::InvalidTranscriptFormat(format!(
                    "Invalid seconds in generic timestamp: {}",
                    timestamp
                ))
            })?;
            Ok(minutes * 60.0 + seconds)
        }
        3 => {
            // Hours:minutes:seconds: "1:23:45.678"
            let hours = parts[0].parse::<f64>().map_err(|_| {
                ParseError::InvalidTranscriptFormat(format!(
                    "Invalid hours in generic timestamp: {}",
                    timestamp
                ))
            })?;
            let minutes = parts[1].parse::<f64>().map_err(|_| {
                ParseError::InvalidTranscriptFormat(format!(
                    "Invalid minutes in generic timestamp: {}",
                    timestamp
                ))
            })?;
            let seconds = parts[2].parse::<f64>().map_err(|_| {
                ParseError::InvalidTranscriptFormat(format!(
                    "Invalid seconds in generic timestamp: {}",
                    timestamp
                ))
            })?;
            Ok(hours * 3600.0 + minutes * 60.0 + seconds)
        }
        _ => Err(ParseError::InvalidTranscriptFormat(format!(
            "Invalid generic timestamp format: {}",
            timestamp
        ))),
    }
}

/// Compute duration from start and end times
fn compute_duration(start_time: f64, end_time: Option<f64>) -> Option<f64> {
    end_time.map(|end| end - start_time)
}

pub struct GenericTranscriptParser;

impl FormatParser for GenericTranscriptParser {
    fn parse(&self, input: &str) -> Result<Vec<UnifiedTranscriptEntry>, ParseError> {
        let bytes = input.as_bytes();
        let len = bytes.len();
        let mut raw_entries = Vec::new();
        let mut pos = 0;

        // First pass: extract raw entries with string timestamps
        while pos < len {
            // Skip any leading whitespace or newlines.
            while pos < len && (bytes[pos] == b'\n' || bytes[pos] == b'\r' || bytes[pos] == b' ') {
                pos += 1;
            }
            if pos >= len {
                break;
            }

            // --- Parse header line ---
            let header_start = pos;
            let header_end = match memchr(b'\n', &bytes[pos..]) {
                Some(idx) => pos + idx,
                None => len,
            };
            let header_line = std::str::from_utf8(&bytes[header_start..header_end])?;
            pos = header_end + 1; // Advance past the newline

            // Reverse-scan the header line for the last '(' and ')'.
            let (speaker, timestamp) = if let (Some(open_idx), Some(close_idx)) =
                (header_line.rfind('('), header_line.rfind(')'))
            {
                if open_idx < close_idx {
                    (
                        header_line[..open_idx].trim(),
                        header_line[open_idx + 1..close_idx].trim(),
                    )
                } else {
                    return Err(ParseError::InvalidTranscriptFormat(format!(
                        "Invalid header format: {}",
                        header_line
                    )));
                }
            } else {
                return Err(ParseError::InvalidTranscriptFormat(format!(
                    "No timestamp found in header: {}",
                    header_line
                )));
            };

            // --- Accumulate content ---
            let content_start = pos;
            let mut content_end = pos;
            while pos < len {
                // Find the next newline.
                let line_end = match memchr(b'\n', &bytes[pos..]) {
                    Some(idx) => pos + idx,
                    None => {
                        // Last line - include it in content
                        content_end = len;
                        pos = len;
                        break;
                    }
                };
                let line = std::str::from_utf8(&bytes[pos..line_end])?;
                // If this line looks like a header, break out of the content loop.
                if is_header_line(line) {
                    break;
                }
                content_end = line_end + 1;
                pos = line_end + 1;
            }
            // Ensure content_end doesn't exceed bounds
            let content_end = content_end.min(len);
            let content = if content_start <= content_end {
                std::str::from_utf8(&bytes[content_start..content_end])?.trim_end()
            } else {
                ""
            };

            // Store raw entry data for processing
            raw_entries.push((
                speaker.to_string(),
                timestamp.to_string(),
                content.to_string(),
            ));
        }

        // Second pass: convert to unified entries with look-ahead for end times
        let mut unified_entries = Vec::new();
        for (i, (speaker, timestamp_str, content)) in raw_entries.iter().enumerate() {
            // Parse start time
            let start_time = parse_generic_timestamp(timestamp_str)?;

            // Look ahead to next entry for end time
            let end_time = if i + 1 < raw_entries.len() {
                let next_timestamp = &raw_entries[i + 1].1;
                Some(parse_generic_timestamp(next_timestamp)?)
            } else {
                None
            };

            // Compute duration
            let duration = compute_duration(start_time, end_time);

            unified_entries.push(UnifiedTranscriptEntry {
                index: Some((i + 1) as i64), // Auto-generate 1-based index
                speaker: if speaker.is_empty() {
                    None
                } else {
                    Some(speaker.clone())
                },
                start_time,
                end_time,
                duration,
                content: content.clone(),
                format: "generic".to_string(),
            });
        }

        Ok(unified_entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generic_transcript_parser() {
        let transcript_text = r#"Nitay (00:01.451)
Apurva great to have you today here with us in the podcast. Why don't we start with, give us a bit of your background, work history.
Apurva Mehta (00:09.006)
Yeah, thanks, Nitay and Kostas.
"#;
        let entries = GenericTranscriptParser.parse(transcript_text).unwrap();
        assert_eq!(entries.len(), 2);

        // Check first entry
        assert_eq!(entries[0].speaker, Some("Nitay".to_string()));
        assert_eq!(entries[0].start_time, 1.451);
        assert_eq!(entries[0].end_time, Some(9.006)); // Look-ahead to next entry
        assert!((entries[0].duration.unwrap() - 7.555).abs() < 0.001); // 9.006 - 1.451 ≈ 7.555
        assert_eq!(entries[0].content, "Apurva great to have you today here with us in the podcast. Why don't we start with, give us a bit of your background, work history.");
        assert_eq!(entries[0].format, "generic");

        // Check second entry
        assert_eq!(entries[1].speaker, Some("Apurva Mehta".to_string()));
        assert_eq!(entries[1].start_time, 9.006);
        assert_eq!(entries[1].end_time, None); // Last entry
        assert_eq!(entries[1].duration, None);
        assert_eq!(entries[1].content, "Yeah, thanks, Nitay and Kostas.");
        assert_eq!(entries[1].format, "generic");
    }

    #[test]
    fn test_single_entry_generic() {
        let transcript_text = "Speaker (00:05.0)\nSingle entry content.";
        let result = GenericTranscriptParser.parse(transcript_text);
        assert!(result.is_ok());
        let entries = result.unwrap();
        assert_eq!(entries.len(), 1);

        let entry = &entries[0];
        assert_eq!(entry.speaker, Some("Speaker".to_string()));
        assert_eq!(entry.start_time, 5.0);
        assert_eq!(entry.end_time, None); // No next entry to look ahead to
        assert_eq!(entry.duration, None);
        assert!(!entry.content.is_empty(), "Content should not be empty");
        assert!(entry.content.contains("Single entry content"));
    }

    #[test]
    fn test_invalid_generic_parsing() {
        let invalid_generic = "No timestamp format here";
        assert!(GenericTranscriptParser.parse(invalid_generic).is_err());
    }
}
