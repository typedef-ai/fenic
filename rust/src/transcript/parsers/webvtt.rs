use super::srt::SrtWebVttParser;
use crate::transcript::types::{FormatParser, ParseError, UnifiedTranscriptEntry};

/// Parse WebVTT format timestamps: "00:00:20.000"
/// WebVTT format uses dot as decimal separator and is always HH:MM:SS.mmm
fn parse_webvtt_timestamp(timestamp: &str) -> Result<f64, ParseError> {
    let timestamp = timestamp.trim();

    // Split by colons to get time components
    let parts: Vec<&str> = timestamp.split(':').collect();

    // WebVTT timestamp must be in HH:MM:SS.mmm format
    if parts.len() != 3 {
        return Err(ParseError::InvalidTranscriptFormat(format!(
            "WebVTT timestamp must be in HH:MM:SS.mmm format: {}",
            timestamp
        )));
    }

    if parts[2].contains(',') {
        return Err(ParseError::InvalidTranscriptFormat(format!(
            "WebVTT timestamp must be in HH:MM:SS.mmm format: {}",
            timestamp
        )));
    }

    let hours = parts[0].parse::<f64>().map_err(|_| {
        ParseError::InvalidTranscriptFormat(format!(
            "Invalid hours in WebVTT timestamp: {}",
            timestamp
        ))
    })?;
    let minutes = parts[1].parse::<f64>().map_err(|_| {
        ParseError::InvalidTranscriptFormat(format!(
            "Invalid minutes in WebVTT timestamp: {}",
            timestamp
        ))
    })?;
    let seconds = parts[2].parse::<f64>().map_err(|_| {
        ParseError::InvalidTranscriptFormat(format!(
            "Invalid seconds in WebVTT timestamp: {}",
            timestamp
        ))
    })?;

    Ok(hours * 3600.0 + minutes * 60.0 + seconds)
}

pub struct WebVttParser;

impl FormatParser for WebVttParser {
    // WebVTT format is similar to SRT, except
    // 1. The timestamp format is slightly different,
    // 2. WebVTT has a header section
    // 3. WebVTT supports rich styling and HTML5
    // We can ignore the header, use our own timestamp parser, and reuse the SRT parser, keeping the rich styling in the output.
    fn parse(&self, input: &str) -> Result<Vec<UnifiedTranscriptEntry>, ParseError> {
        let bytes = input.as_bytes();
        let len = bytes.len();
        let mut pos = 0;

        // Skip any leading whitespace, newlines, or carriage returns.
        while pos < len && (bytes[pos] == b'\n' || bytes[pos] == b'\r' || bytes[pos] == b' ') {
            pos += 1;
        }

        // Check for 'WEBVTT' header at the current position
        let webvtt_header = b"WEBVTT";
        if pos + webvtt_header.len() > len
            || &bytes[pos..pos + webvtt_header.len()] != webvtt_header
        {
            return Err(ParseError::InvalidTranscriptFormat(
                "Missing 'WEBVTT' header at start of file".to_string(),
            ));
        }
        pos += webvtt_header.len();

        SrtWebVttParser.parse_format(&input[pos..], "webvtt", parse_webvtt_timestamp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_webvtt1() {
        let webvtt_data = r#"
WEBVTT
1
00:00:01.000 --> 00:00:03.000
This is <b>bold</b> and <i>italic</i> text.

2
00:00:04.000 --> 00:00:06.000
Tabbed:\tItem A\nNext Line Here

3
00:00:07.000 --> 00:00:09.000
<font color="green">Green colored font text</font>

4
00:00:10.000 --> 00:00:12.000
Normal line with no styling.
"#;
        let result = WebVttParser.parse(webvtt_data);
        assert!(result.is_ok());
        let entries = result.unwrap();
        assert_eq!(entries.len(), 4);

        // Check first entry
        assert_eq!(entries[0].index, Some(1));
        assert_eq!(entries[0].start_time, 1.0); // 00:00:01.000 = 1 second
        assert_eq!(entries[0].end_time, Some(3.0)); // 00:00:03.000 = 3 seconds
        assert_eq!(entries[0].duration, Some(2.0)); // 3 - 1 = 2 seconds
        assert!(entries[0]
            .content
            .contains("This is <b>bold</b> and <i>italic</i> text."));
        assert_eq!(entries[0].format, "webvtt");

        // Check second entry (multiline HTML5 with tab and newline)
        assert_eq!(entries[1].index, Some(2));
        assert_eq!(entries[1].start_time, 4.0); // 00:00:04.000 = 4 seconds
        assert_eq!(entries[1].end_time, Some(6.0)); // 00:00:06.000 = 6 seconds
        assert_eq!(entries[1].duration, Some(2.0)); // 6 - 4 = 2 seconds
        println!("entries[1].content: {}", entries[1].content);
        assert!(entries[1]
            .content
            .contains("Tabbed:\\tItem A\\nNext Line Here"));
        assert_eq!(entries[1].format, "webvtt");

        // Check third entry (HTML5 with styling tags
        assert_eq!(entries[2].index, Some(3));
        assert_eq!(entries[2].start_time, 7.0); // 00:00:07.000 = 7 seconds
        assert_eq!(entries[2].end_time, Some(9.0)); // 00:00:09.000 = 9 seconds
        assert!(entries[2]
            .content
            .contains("<font color=\"green\">Green colored font text</font>"));
        assert_eq!(entries[2].format, "webvtt");
    }

    #[test]
    fn test_invalid_webvtt_parsing() {
        let invalid_srt = "This is not a valid SRT format at all";
        assert!(WebVttParser.parse(invalid_srt).is_err());
    }

    #[test]
    fn test_malformed_webvtt_cases() {
        // Missing WEBVTT header
        let missing_header = r#"1
00:00:01.000 --> 00:00:04.000
Hello world"#;
        assert!(WebVttParser.parse(missing_header).is_err());

        // Missing arrow in timestamp
        let missing_arrow = r#"1
WEBVTT
00:00:01.000 00:00:04.000
Hello world"#;
        assert!(WebVttParser.parse(missing_arrow).is_err());

        // Missing timestamp entirely
        let missing_timestamp = r#"1
WEBVTT
Hello world"#;
        assert!(WebVttParser.parse(missing_timestamp).is_err());

        // Invalid index (not a number)
        let invalid_index = r#"WEBVTT
1
abc
00:00:01.000 --> 00:00:04.000
Hello world"#;
        // This should still parse (index is optional) but treat "abc" as timestamp
        assert!(WebVttParser.parse(invalid_index).is_err());

        // Incomplete timestamp (missing end time)
        let incomplete_timestamp = r#"WEBVTT
1
00:00:01.000 -->
Hello world"#;
        assert!(WebVttParser.parse(incomplete_timestamp).is_err());

        // Malformed time values
        let malformed_time = r#"WEBVTT
1
25:99:99.999 --> 00:00:04.000
Hello world"#;
        // This might parse but produce incorrect values - depends on implementation
        let result = WebVttParser.parse(malformed_time);
        // Should either error or parse with unexpected values
        if result.is_ok() {
            // If it parses, the time should be very large due to 99 minutes/seconds
            let entries = result.unwrap();
            assert_eq!(entries.len(), 1);
            // 25*3600 + 99*60 + 99.999 = very large number
            assert!(entries[0].start_time > 90000.0);
        }

        // comma separator (SRT timestamp format)
        let srt_timestamp = r#"WEBVTT
1
00:00:01,000 --> 00:00:04,000
Hello world"#;
        assert!(WebVttParser.parse(srt_timestamp).is_err());

        // Empty content after timestamp
        let empty_content = r#"WEBVTT
1
00:00:01.000 --> 00:00:04.000

2
00:00:05.000 --> 00:00:08.000
Next subtitle"#;
        let result = WebVttParser.parse(empty_content);
        assert!(result.is_ok());
        let entries = result.unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].content, ""); // Empty content should be preserved
        assert_eq!(entries[1].content, "Next subtitle");
    }

    #[test]
    fn test_parse_webvtt2() {
        let webvtt_data = r#"WEBVTT

1
00:00:03.400 --> 00:00:06.177
<v Alice>In this lesson, we're going to
be talking about finance. And

2
00:00:06.177 --> 00:00:10.009
one of the most important aspects
<c.important>of finance is interest.</c.important>

3
00:00:10.009 --> 00:00:13.655
When I go to a bank or some
other lending institution

4
00:00:13.655 --> 00:00:17.720
to borrow money, the bank is happy
<v Bob>to give me that money. But then I'm

5
00:00:17.900 --> 00:00:21.480
going to be paying the bank for the
privilege of using their money. And that

6
00:00:21.660 --> 00:00:26.440
amount of money that I pay the bank is
called interest. Likewise, if I put money

7
00:00:26.620 --> 00:00:31.220
in a savings account or I purchase a
certificate of deposit, the bank just

8
00:00:31.300 --> 00:00:35.800
doesn't put my money in a little box
<c.final>and leave it there until later. They take</c.final>"#;

        let result = WebVttParser.parse(webvtt_data);
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
            "<v Alice>In this lesson, we're going to\nbe talking about finance. And"
        );
        assert_eq!(entries[0].format, "webvtt");

        // Entry 2
        assert_eq!(entries[1].index, Some(2));
        assert_eq!(entries[1].start_time, 6.177);
        assert_eq!(entries[1].end_time, Some(10.009));
        assert!((entries[1].duration.unwrap() - 3.832).abs() < 0.001);
        assert_eq!(
            entries[1].content,
            "one of the most important aspects\n<c.important>of finance is interest.</c.important>"
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
            "to borrow money, the bank is happy\n<v Bob>to give me that money. But then I'm"
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
            "doesn't put my money in a little box\n<c.final>and leave it there until later. They take</c.final>"
        );
    }
}
