use super::{looks_like_subtitle_index, looks_like_timestamp};
use crate::transcript::types::{FormatParser, ParseError, UnifiedTranscriptEntry};
use memchr::memchr;

/// Parse SRT format timestamps: "00:00:20,000"
/// SRT format uses comma as decimal separator and is always HH:MM:SS,mmm
fn parse_srt_timestamp(timestamp: &str) -> Result<f64, ParseError> {
    let timestamp = timestamp.trim();

    // Replace comma with dot for parsing
    let normalized = timestamp.replace(',', ".");

    // Split by colons to get time components
    let parts: Vec<&str> = normalized.split(':').collect();

    if parts.len() != 3 {
        return Err(ParseError::InvalidTranscriptFormat(format!(
            "SRT timestamp must be in HH:MM:SS,mmm format: {}",
            timestamp
        )));
    }

    let hours = parts[0].parse::<f64>().map_err(|_| {
        ParseError::InvalidTranscriptFormat(format!(
            "Invalid hours in SRT timestamp: {}",
            timestamp
        ))
    })?;
    let minutes = parts[1].parse::<f64>().map_err(|_| {
        ParseError::InvalidTranscriptFormat(format!(
            "Invalid minutes in SRT timestamp: {}",
            timestamp
        ))
    })?;
    let seconds = parts[2].parse::<f64>().map_err(|_| {
        ParseError::InvalidTranscriptFormat(format!(
            "Invalid seconds in SRT timestamp: {}",
            timestamp
        ))
    })?;

    Ok(hours * 3600.0 + minutes * 60.0 + seconds)
}

/// Compute duration from start and end times
fn compute_duration(start_time: f64, end_time: Option<f64>) -> Option<f64> {
    end_time.map(|end| end - start_time)
}

pub struct SrtParser;

impl FormatParser for SrtParser {
    fn parse(&self, input: &str) -> Result<Vec<UnifiedTranscriptEntry>, ParseError> {
        let bytes = input.as_bytes();
        let len = bytes.len();
        let mut pos = 0;
        let mut entries = Vec::new();

        while pos < len {
            // Skip any leading whitespace, newlines, or carriage returns.
            while pos < len && (bytes[pos] == b'\n' || bytes[pos] == b'\r' || bytes[pos] == b' ') {
                pos += 1;
            }
            if pos >= len {
                break;
            }

            // --- Determine Header: Index line (optional) and Timestamp line ---
            // Use memchr to find the end of the current line.
            let header_end = match memchr(b'\n', &bytes[pos..]) {
                Some(idx) => pos + idx,
                None => len,
            };
            let header_line = std::str::from_utf8(&bytes[pos..header_end])?;
            let mut entry_index: Option<usize> = None;
            let timestamp_line: &str;

            // Heuristic: if the line looks like an index, then use it;
            // otherwise, if it looks like a timestamp, assume the index is missing.
            if looks_like_subtitle_index(header_line) {
                // Parse the index.
                entry_index = header_line.trim().parse::<usize>().ok();
                pos = header_end + 1; // Advance past the index line.

                // Next line should be the timestamp line.
                let ts_line_end = match memchr(b'\n', &bytes[pos..]) {
                    Some(idx) => pos + idx,
                    None => len,
                };
                timestamp_line = std::str::from_utf8(&bytes[pos..ts_line_end])?;
                if !looks_like_timestamp(timestamp_line) {
                    return Err(ParseError::InvalidTranscriptFormat(format!(
                        "Invalid timestamp line: {}",
                        timestamp_line
                    )));
                }
                pos = ts_line_end + 1; // Advance past the timestamp line.
            } else if looks_like_timestamp(header_line) {
                // No index; the header line is actually the timestamp line.
                timestamp_line = header_line;
                pos = header_end + 1; // Advance past the timestamp line.
            } else {
                // Neither an index nor a timestamp was detected.
                return Err(ParseError::InvalidTranscriptFormat(format!(
                    "Invalid header line: {}",
                    header_line
                )));
            }

            // --- Parse Timestamp Line ---
            // Expected format: "00:00:20,000 --> 00:00:24,400"
            let parts: Vec<&str> = timestamp_line.split("-->").collect();
            if parts.len() != 2 {
                return Err(ParseError::InvalidTranscriptFormat(format!(
                    "Invalid timestamp line: {}",
                    timestamp_line
                )));
            }
            let start = parts[0].trim();
            let end = parts[1].trim();

            // --- Accumulate Subtitle Text ---
            let text_start = pos;
            let mut text_end = pos;
            // Read lines until we hit either:
            // 1. A blank line, or
            // 2. A line that looks like the next header (index or timestamp).
            while pos < len {
                let line_end = match memchr(b'\n', &bytes[pos..]) {
                    Some(idx) => pos + idx,
                    None => len,
                };
                let line = std::str::from_utf8(&bytes[pos..line_end])?;
                let trimmed = line.trim();

                // If the line is blank or it looks like the start of a new entry, break out.
                if trimmed.is_empty()
                    || looks_like_subtitle_index(trimmed)
                    || looks_like_timestamp(trimmed)
                {
                    // If the line is blank, consume it.
                    if trimmed.is_empty() {
                        pos = line_end + 1;
                    }
                    break;
                }
                // Otherwise, extend the text for the current entry.
                text_end = line_end;
                pos = line_end + 1;
            }
            let text = std::str::from_utf8(&bytes[text_start..text_end])?.trim_end();

            // Parse timestamps to seconds
            let start_time = parse_srt_timestamp(start)?;
            let end_time = parse_srt_timestamp(end)?;
            let duration = compute_duration(start_time, Some(end_time));

            entries.push(UnifiedTranscriptEntry {
                index: entry_index.map(|i| i as i64),
                speaker: None, // SRT format doesn't typically have speakers
                start_time,
                end_time: Some(end_time),
                duration,
                content: text.to_string(),
                format: "srt".to_string(),
            });
        }
        Ok(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_srt1() {
        let srt_data = r#"1
00:00:01,000 --> 00:00:04,000
Hello, world!

2
00:00:05,000 --> 00:00:07,000
This is a subtitle.

3
00:00:08,000 --> 00:00:10,000
Third subtitle line 1.
Third subtitle line 2.
"#;
        let result = SrtParser.parse(srt_data);
        assert!(result.is_ok());
        let entries = result.unwrap();
        assert_eq!(entries.len(), 3);

        // Check first entry
        assert_eq!(entries[0].index, Some(1));
        assert_eq!(entries[0].start_time, 1.0); // 00:00:01,000 = 1 second
        assert_eq!(entries[0].end_time, Some(4.0)); // 00:00:04,000 = 4 seconds
        assert_eq!(entries[0].duration, Some(3.0)); // 4 - 1 = 3 seconds
        assert!(entries[0].content.contains("Hello, world!"));
        assert_eq!(entries[0].format, "srt");

        // Check third entry (multiline)
        assert_eq!(entries[2].index, Some(3));
        assert_eq!(entries[2].start_time, 8.0); // 00:00:08,000 = 8 seconds
        assert_eq!(entries[2].end_time, Some(10.0)); // 00:00:10,000 = 10 seconds
        assert!(entries[2].content.contains("Third subtitle line 1."));
        assert!(entries[2].content.contains("Third subtitle line 2."));
    }

    #[test]
    fn test_invalid_srt_parsing() {
        let invalid_srt = "This is not a valid SRT format at all";
        assert!(SrtParser.parse(invalid_srt).is_err());
    }

    #[test]
    fn test_malformed_srt_cases() {
        // Missing arrow in timestamp
        let missing_arrow = r#"1
00:00:01,000 00:00:04,000
Hello world"#;
        assert!(SrtParser.parse(missing_arrow).is_err());

        // Different separator (dot instead of comma) - should still work
        let dot_separator = r#"1
00:00:01.000 --> 00:00:04.000
Hello world"#;
        let result = SrtParser.parse(dot_separator);
        assert!(result.is_ok());
        let entries = result.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].start_time, 1.0);
        assert_eq!(entries[0].end_time, Some(4.0));

        // Missing timestamp entirely
        let missing_timestamp = r#"1
Hello world"#;
        assert!(SrtParser.parse(missing_timestamp).is_err());

        // Invalid index (not a number)
        let invalid_index = r#"abc
00:00:01,000 --> 00:00:04,000
Hello world"#;
        // This should still parse (index is optional) but treat "abc" as timestamp
        assert!(SrtParser.parse(invalid_index).is_err());

        // Incomplete timestamp (missing end time)
        let incomplete_timestamp = r#"1
00:00:01,000 -->
Hello world"#;
        assert!(SrtParser.parse(incomplete_timestamp).is_err());

        // Malformed time values
        let malformed_time = r#"1
25:99:99,999 --> 00:00:04,000
Hello world"#;
        // This might parse but produce incorrect values - depends on implementation
        let result = SrtParser.parse(malformed_time);
        // Should either error or parse with unexpected values
        if result.is_ok() {
            // If it parses, the time should be very large due to 99 minutes/seconds
            let entries = result.unwrap();
            assert_eq!(entries.len(), 1);
            // 25*3600 + 99*60 + 99.999 = very large number
            assert!(entries[0].start_time > 90000.0);
        }

        // Empty content after timestamp
        let empty_content = r#"1
00:00:01,000 --> 00:00:04,000

2
00:00:05,000 --> 00:00:08,000
Next subtitle"#;
        let result = SrtParser.parse(empty_content);
        assert!(result.is_ok());
        let entries = result.unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].content, ""); // Empty content should be preserved
        assert_eq!(entries[1].content, "Next subtitle");
    }

    #[test]
    fn test_parse_srt2() {
        let srt_data = r#"1
00:00:03,400 --> 00:00:06,177
In this lesson, we're going to
be talking about finance. And

2
00:00:06,177 --> 00:00:10,009
one of the most important aspects
of finance is interest.

3
00:00:10,009 --> 00:00:13,655
When I go to a bank or some
other lending institution

4
00:00:13,655 --> 00:00:17,720
to borrow money, the bank is happy
to give me that money. But then I'm

5
00:00:17,900 --> 00:00:21,480
going to be paying the bank for the
privilege of using their money. And that

6
00:00:21,660 --> 00:00:26,440
amount of money that I pay the bank is
called interest. Likewise, if I put money

7
00:00:26,620 --> 00:00:31,220
in a savings account or I purchase a
certificate of deposit, the bank just

8
00:00:31,300 --> 00:00:35,800
doesn't put my money in a little box
and leave it there until later. They take"#;

        let result = SrtParser.parse(srt_data);
        assert!(result.is_ok());
        let entries = result.unwrap();
        assert_eq!(entries.len(), 8);

        // Entry 1
        assert_eq!(entries[0].index, Some(1));
        assert_eq!(entries[0].start_time, 3.4);
        assert_eq!(entries[0].end_time, Some(6.177));
        assert!((entries[0].duration.unwrap() - 2.777).abs() < 0.001);
        assert_eq!(
            entries[0].content,
            "In this lesson, we're going to\nbe talking about finance. And"
        );
        assert_eq!(entries[0].format, "srt");

        // Entry 2
        assert_eq!(entries[1].index, Some(2));
        assert_eq!(entries[1].start_time, 6.177);
        assert_eq!(entries[1].end_time, Some(10.009));
        assert!((entries[1].duration.unwrap() - 3.832).abs() < 0.001);
        assert_eq!(
            entries[1].content,
            "one of the most important aspects\nof finance is interest."
        );

        // Entry 3
        assert_eq!(entries[2].index, Some(3));
        assert_eq!(entries[2].start_time, 10.009);
        assert_eq!(entries[2].end_time, Some(13.655));
        assert!((entries[2].duration.unwrap() - 3.646).abs() < 0.001);
        assert_eq!(
            entries[2].content,
            "When I go to a bank or some\nother lending institution"
        );

        // Entry 4
        assert_eq!(entries[3].index, Some(4));
        assert_eq!(entries[3].start_time, 13.655);
        assert_eq!(entries[3].end_time, Some(17.72));
        assert!((entries[3].duration.unwrap() - 4.065).abs() < 0.001);
        assert_eq!(
            entries[3].content,
            "to borrow money, the bank is happy\nto give me that money. But then I'm"
        );

        // Entry 5
        assert_eq!(entries[4].index, Some(5));
        assert_eq!(entries[4].start_time, 17.9);
        assert_eq!(entries[4].end_time, Some(21.48));
        assert!((entries[4].duration.unwrap() - 3.58).abs() < 0.001);
        assert_eq!(
            entries[4].content,
            "going to be paying the bank for the\nprivilege of using their money. And that"
        );

        // Entry 6
        assert_eq!(entries[5].index, Some(6));
        assert_eq!(entries[5].start_time, 21.66);
        assert_eq!(entries[5].end_time, Some(26.44));
        assert!((entries[5].duration.unwrap() - 4.78).abs() < 0.001);
        assert_eq!(
            entries[5].content,
            "amount of money that I pay the bank is\ncalled interest. Likewise, if I put money"
        );

        // Entry 7
        assert_eq!(entries[6].index, Some(7));
        assert_eq!(entries[6].start_time, 26.62);
        assert_eq!(entries[6].end_time, Some(31.22));
        assert!((entries[6].duration.unwrap() - 4.6).abs() < 0.001);
        assert_eq!(
            entries[6].content,
            "in a savings account or I purchase a\ncertificate of deposit, the bank just"
        );

        // Entry 8
        assert_eq!(entries[7].index, Some(8));
        assert_eq!(entries[7].start_time, 31.3);
        assert_eq!(entries[7].end_time, Some(35.8));
        assert!((entries[7].duration.unwrap() - 4.5).abs() < 0.001);
        assert_eq!(
            entries[7].content,
            "doesn't put my money in a little box\nand leave it there until later. They take"
        );
    }
}
