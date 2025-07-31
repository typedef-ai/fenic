use crate::transcript::types::{FormatParser, ParseError, UnifiedTranscriptEntry};
use regex::Regex;
use std::sync::OnceLock;

// Static regex patterns for WebVTT parsing
pub fn looks_like_timestamp_line(line: &str) -> bool {
    line.contains("-->")
}

pub fn html_tag_regex() -> &'static Regex {
    static HTML_TAG_REGEX: OnceLock<Regex> = OnceLock::new();
    HTML_TAG_REGEX.get_or_init(|| Regex::new(r"<[^>]*>").expect("Failed to compile HTML tag regex"))
}

pub fn speaker_regex() -> &'static Regex {
    static SPEAKER_REGEX: OnceLock<Regex> = OnceLock::new();
    SPEAKER_REGEX.get_or_init(|| {
        Regex::new(r"<v(?:\.[^>]*)?\s+([^>]+)>").expect("Failed to compile speaker regex")
    })
}
pub struct WebVTTParser;

impl FormatParser for WebVTTParser {
    // Parses WebVTT format, extracting:
    // - Cue timestamps (start/end)
    // - Speaker information from <v> tags
    // - Clean text content (stripped of HTML tags and cue settings)
    // - Ignore cue identifiers when indexing the cue entries
    // - Skips NOTE, STYLE, and REGION blocks
    // Reference: https://www.w3.org/TR/webvtt1/
    fn parse(&self, input: &str) -> Result<Vec<UnifiedTranscriptEntry>, ParseError> {
        let lines: Vec<&str> = input.lines().collect();
        let mut entries = Vec::new();
        let mut i = 0;
        let mut cue_index = 0;

        // Parse the first WEBVTT line, allowing for optional BOM character
        let first_line = lines[0].trim().trim_start_matches('\u{FEFF}');
        if !first_line.starts_with("WEBVTT") {
            return Err(ParseError::InvalidTranscriptFormat(
                "No WEBVTT header found".to_string(),
            ));
        }

        i += 1;

        while i < lines.len() {
            // Skip empty lines
            if lines[i].trim().is_empty() {
                i += 1;
                continue;
            }

            // Skip NOTE, STYLE, and REGION blocks
            // technically you shouldn't see STYLE or REGION blocks after the first cue, but we'll just skip them
            let line = lines[i].trim();
            if line.starts_with("NOTE") || line.starts_with("STYLE") || line.starts_with("REGION") {
                i = Self::skip_block(&lines, i);
                continue;
            }

            // Try to parse a cue
            let entry = Self::parse_cue(&lines, &mut i, &mut cue_index)?;
            entries.push(entry);
        }

        Ok(entries)
    }
}

impl WebVTTParser {
    fn skip_block(lines: &[&str], mut start_index: usize) -> usize {
        start_index += 1;

        // Skip until we find an empty line or end of file
        while start_index < lines.len() && !lines[start_index].trim().is_empty() {
            start_index += 1;
        }
        start_index
    }

    fn parse_cue(
        lines: &[&str],
        index: &mut usize,
        cue_index: &mut usize,
    ) -> Result<UnifiedTranscriptEntry, ParseError> {
        let mut current_line = lines[*index].trim();

        // Expected format: "00:00:01.000 --> 00:00:05.000 <optional cue settings>"
        // Check if current line contains a timestamp (cue timing line)
        if looks_like_timestamp_line(current_line) {
            // This line is the timing line, no identifier present
        } else {
            // Current line might be an identifier.  Skip it.
            *index += 1;
            if *index >= lines.len() {
                return Err(ParseError::InvalidTranscriptFormat(
                    "Non-cue line found at end of file".to_string(),
                ));
            }
            current_line = lines[*index].trim();
        }

        // Parse timing line
        let timestamp_parts: Vec<&str> = current_line.split("-->").collect();
        if timestamp_parts.len() < 2 {
            return Err(ParseError::InvalidTranscriptFormat(format!(
                "Invalid timestamp line: {}.  Expected format: HH:MM:SS.mmm --> HH:MM:SS.mmm or MM:SS.mmm --> MM:SS.mmm",
                current_line
            )));
        }
        let end_timestamp_parts: Vec<&str> = timestamp_parts[1].split(".").collect();
        if end_timestamp_parts.len() < 2 || end_timestamp_parts[1].len() < 3 {
            return Err(ParseError::InvalidTranscriptFormat(format!(
                "Invalid timestamp line: {}.  Expected format: HH:MM:SS.mmm --> HH:MM:SS.mmm or MM:SS.mmm --> MM:SS.mmm",
                current_line
            )));
        }

        let start_str: &str = timestamp_parts[0].trim();
        let end_str: &str = timestamp_parts[1][..end_timestamp_parts[0].len() + 4].trim(); // 3 for milliseconds, 1 for the '.'
        let start_time = Self::parse_timestamp(start_str)?;
        let end_time = Self::parse_timestamp(end_str)?;
        let duration = end_time - start_time;

        *index += 1;

        // Collect cue payload (text content)
        let mut content_lines = Vec::new();
        while *index < lines.len() && !lines[*index].trim().is_empty() {
            content_lines.push(lines[*index]);
            *index += 1;
        }

        let raw_content = content_lines.join("\n");
        let (speaker, cleaned_content) = Self::parse_payload_and_speaker(&raw_content);

        *cue_index += 1;

        Ok(UnifiedTranscriptEntry {
            index: Some(*cue_index as i64), // auto-generate 1-based index
            speaker,
            start_time,
            end_time: Some(end_time),
            duration: Some(duration),
            content: cleaned_content,
            format: "webvtt".to_string(),
        })
    }

    fn parse_timestamp(timestamp: &str) -> Result<f64, ParseError> {
        let parts: Vec<&str> = timestamp.split(':').collect();
        if parts.len() != 3 && parts.len() != 2 {
            return Err(ParseError::InvalidTranscriptFormat(format!(
                "Invalid timestamp format: {}.  Expected MM:SS.mmm or HH:MM:SS.mmm",
                timestamp
            )));
        }

        // parse the hours, minutes, and seconds.milliseconds
        let (hours, minutes, seconds_part) = if parts.len() == 3 {
            // Format: HH:MM:SS.mmm
            (
                parts[0].parse::<f64>().map_err(|_| {
                    ParseError::InvalidTranscriptFormat(format!(
                        "Invalid hours in timestamp: {}",
                        parts[0]
                    ))
                })?,
                parts[1].parse::<f64>().map_err(|_| {
                    ParseError::InvalidTranscriptFormat(format!(
                        "Invalid minutes in timestamp: {}",
                        parts[1]
                    ))
                })?,
                parts[2],
            )
        } else {
            // Format: MM:SS.mmm (no hours)
            (
                0.0,
                parts[0].parse::<f64>().map_err(|_| {
                    ParseError::InvalidTranscriptFormat(format!(
                        "Invalid minutes in timestamp: {}",
                        parts[0]
                    ))
                })?,
                parts[1],
            )
        };

        // parse the seconds and drop the milliseconds
        let seconds_and_millis: Vec<&str> = seconds_part.split('.').collect();

        if seconds_and_millis.len() != 2 {
            return Err(ParseError::InvalidTranscriptFormat(format!(
                "Invalid timestamp format: {}.  Expected MM:SS.mmm or HH:MM:SS.mmm",
                timestamp
            )));
        }

        let seconds = seconds_and_millis[0].parse::<f64>().map_err(|_| {
            ParseError::InvalidTranscriptFormat(format!(
                "Invalid seconds: {}",
                seconds_and_millis[0]
            ))
        })?;
        let milliseconds = seconds_and_millis[1].parse::<f64>().map_err(|_| {
            ParseError::InvalidTranscriptFormat(format!(
                "Invalid milliseconds: {}",
                seconds_and_millis[1]
            ))
        })?;

        Ok(hours * 3600.0 + minutes * 60.0 + seconds + milliseconds / 1000.0)
    }

    fn parse_payload_and_speaker(content: &str) -> (Option<String>, String) {
        let mut speaker = None;
        let mut processed_content = content.to_string();

        // Extract speaker from voice tags like <v Speaker Name>
        // If there are multiple speaker tags, use the first one
        if let Some(captures) = speaker_regex().captures(content) {
            speaker = Some(captures.get(1).unwrap().as_str().trim().to_string());
        }

        // Remove all HTML tags
        processed_content = html_tag_regex()
            .replace_all(&processed_content, "")
            .to_string();

        // Clean up extra whitespace
        processed_content = processed_content
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join(" ");

        (speaker, processed_content)
    }
}
#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_basic_webvtt_parsing() {
        let webvtt_content = r#"WEBVTT

1
00:00:01.000 --> 00:00:05.000
Hello, world!

2
00:00:06.000 --> 00:00:10.000
<v Alice>This is Alice speaking.

NOTE This is a note and should be ignored

cue-3
00:00:11.500 --> 00:00:15.750
<b>Bold text</b> and <i>italic text</i>.
"#;

        let entries = WebVTTParser.parse(webvtt_content).unwrap();

        assert_eq!(entries.len(), 3);

        assert_eq!(entries[0].index, Some(1));
        assert_eq!(entries[0].content, "Hello, world!");
        assert_eq!(entries[0].start_time, 1.0);
        assert_eq!(entries[0].end_time, Some(5.0));
        assert_eq!(entries[0].duration, Some(4.0));
        assert_eq!(entries[0].speaker, None);

        assert_eq!(entries[1].index, Some(2));
        assert_eq!(entries[1].content, "This is Alice speaking.");
        assert_eq!(entries[1].start_time, 6.0);
        assert_eq!(entries[1].end_time, Some(10.0));
        assert_eq!(entries[1].duration, Some(4.0));
        assert_eq!(entries[1].speaker, Some("Alice".to_string()));

        assert_eq!(entries[2].index, Some(3));
        assert_eq!(entries[2].content, "Bold text and italic text.");
        assert_eq!(entries[2].start_time, 11.5);
        assert_eq!(entries[2].end_time, Some(15.75));
        assert_eq!(entries[2].duration, Some(4.25));
    }

    #[test]
    fn test_webvtt_with_bom() {
        // concat the BOM character
        let webvtt_content = format!(
            "\u{FEFF}{}",
            r#"WEBVTT
00:00:01.000 --> 00:00:05.000
No cue identifier or speaker here.

00:00:06.000 --> 00:00:10.000
<v Bob>Bob speaking without cue identifier.
"#
        );

        let entries = WebVTTParser.parse(&webvtt_content).unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].index, Some(1));
        assert_eq!(entries[0].content, "No cue identifier or speaker here.");
        assert_eq!(entries[0].speaker, None);
        assert_eq!(entries[0].start_time, 1.0);
        assert_eq!(entries[0].end_time, Some(5.0));
        assert_eq!(entries[0].duration, Some(4.0));
        assert_eq!(entries[0].format, "webvtt");

        assert_eq!(entries[1].index, Some(2));
        assert_eq!(entries[1].speaker, Some("Bob".to_string()));
        assert_eq!(entries[1].content, "Bob speaking without cue identifier.");
        assert_eq!(entries[1].start_time, 6.0);
        assert_eq!(entries[1].end_time, Some(10.0));
        assert_eq!(entries[1].duration, Some(4.0));
        assert_eq!(entries[1].format, "webvtt");
    }

    #[test]
    fn test_webvtt_with_no_hours() {
        // concat the BOM character
        let webvtt_content = format!(
            r#"WEBVTT
00:01.000 --> 00:05.000
No hours here.

01:00:06.000 --> 01:00:10.000
<v Bob>Bob speaking an hour later.
"#
        );

        let entries = WebVTTParser.parse(&webvtt_content).unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].index, Some(1));
        assert_eq!(entries[0].content, "No hours here.");
        assert_eq!(entries[0].speaker, None);
        assert_eq!(entries[0].start_time, 1.0);
        assert_eq!(entries[0].end_time, Some(5.0));
        assert_eq!(entries[0].duration, Some(4.0));
        assert_eq!(entries[0].format, "webvtt");

        assert_eq!(entries[1].index, Some(2));
        assert_eq!(entries[1].speaker, Some("Bob".to_string()));
        assert_eq!(entries[1].content, "Bob speaking an hour later.");
        assert_eq!(entries[1].start_time, 3606.0);
        assert_eq!(entries[1].end_time, Some(3610.0));
        assert_eq!(entries[1].duration, Some(4.0));
        assert_eq!(entries[1].format, "webvtt");
    }

    #[test]
    fn test_webvtt_with_cue_settings() {
        let webvtt_content = r#"WEBVTT

1
00:00:01.000 --> 00:00:05.000 align:center line:50%
Text with cue settings

bla bla
00:00:06.000 --> 00:00:10.000
And with random text for identifiers

cue-2
00:00:11.000 --> 00:00:15.000
Alphanumeric identifier

00:00:16.000 --> 00:00:20.000
No identifier here
"#;

        let entries = WebVTTParser.parse(webvtt_content).unwrap();

        assert_eq!(entries.len(), 4);
        assert_eq!(entries[0].content, "Text with cue settings");
        assert_eq!(entries[0].start_time, 1.0);
        assert_eq!(entries[0].end_time, Some(5.0));
        assert_eq!(entries[0].duration, Some(4.0));
        assert_eq!(entries[0].speaker, None);
        assert_eq!(entries[0].index, Some(1));
        assert_eq!(entries[0].format, "webvtt");

        assert_eq!(entries[1].index, Some(2));
        assert_eq!(entries[1].content, "And with random text for identifiers");
        assert_eq!(entries[1].start_time, 6.0);
        assert_eq!(entries[1].end_time, Some(10.0));
        assert_eq!(entries[1].duration, Some(4.0));
        assert_eq!(entries[1].speaker, None);
        assert_eq!(entries[1].index, Some(2));
        assert_eq!(entries[1].format, "webvtt");

        assert_eq!(entries[2].index, Some(3));
        assert_eq!(entries[2].content, "Alphanumeric identifier");
        assert_eq!(entries[2].start_time, 11.0);
        assert_eq!(entries[2].end_time, Some(15.0));
        assert_eq!(entries[2].duration, Some(4.0));
        assert_eq!(entries[2].speaker, None);
        assert_eq!(entries[2].index, Some(3));
        assert_eq!(entries[2].format, "webvtt");

        assert_eq!(entries[3].index, Some(4));
        assert_eq!(entries[3].content, "No identifier here");
        assert_eq!(entries[3].start_time, 16.0);
        assert_eq!(entries[3].end_time, Some(20.0));
        assert_eq!(entries[3].duration, Some(4.0));
        assert_eq!(entries[3].speaker, None);
        assert_eq!(entries[3].index, Some(4));
        assert_eq!(entries[3].format, "webvtt");
    }

    #[test]
    fn test_malformed_webvtt_cases() {
        // Missing WEBVTT header
        let missing_header = r#"1
00:00:01.000 --> 00:00:04.000
Hello world"#;
        assert!(WebVTTParser.parse(missing_header).is_err());

        // Malformed WEBVTT header
        let missing_header = r#"something before WEBVTT
00:00:01.000 --> 00:00:04.000
Hello world"#;
        assert!(WebVTTParser.parse(missing_header).is_err());

        // Missing arrow in timestamp
        let missing_arrow = r#"WEBVTT
1
00:00:01.000 00:00:04.000

Hello world"#;
        assert!(WebVTTParser.parse(missing_arrow).is_err());

        // Different separator (dot instead of comma) - should still work
        let comma_separator = r#"WEBVTT
1
00:00:01,000 --> 00:00:04,000
Hello world"#;
        assert!(WebVTTParser.parse(comma_separator).is_err());

        // Missing timestamp entirely
        let missing_timestamp = r#"1
WEBVTT

Hello world"#;
        assert!(WebVTTParser.parse(missing_timestamp).is_err());

        // Malformed blocks
        let malformed_blocks = r#"1
WEBVTT
00:00:01.000 --> 00:00:04.000
Hello world

Some other text

00:00:05.000 --> 00:00:08.000
Next subtitle"#;
        assert!(WebVTTParser.parse(malformed_blocks).is_err());

        // Incomplete timestamp (missing end time)
        let incomplete_timestamp = r#"WEBVTT
1
00:00:01.000 -->
Hello world"#;
        assert!(WebVTTParser.parse(incomplete_timestamp).is_err());

        // Malformed time values
        let malformed_time = r#"WEBVTT

25:99:99.999 --> 00:00:04.000
Hello world"#;
        // This might parse but produce incorrect values - depends on implementation
        let result = WebVTTParser.parse(malformed_time);
        // Should either error or parse with unexpected values
        if result.is_ok() {
            // If it parses, the time should be very large due to 99 minutes/seconds
            let entries = result.unwrap();
            assert_eq!(entries.len(), 1);
            // 25*3600 + 99*60 + 99.999 = very large number
            assert!(entries[0].start_time > 90000.0);
        }

        // Empty content after timestamp
        let empty_content = r#"WEBVTT
1
00:00:01.000 --> 00:00:04.000

2
00:00:05.000 --> 00:00:08.000
Next subtitle"#;
        let result = WebVTTParser.parse(empty_content);
        assert!(result.is_ok());
        let entries = result.unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].content, ""); // Empty content should be preserved
        assert_eq!(entries[1].content, "Next subtitle");
    }
}
