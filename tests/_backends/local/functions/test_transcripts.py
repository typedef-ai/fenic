from fenic import col, text


def test_parse_transcript_srt_format(local_session):
    """Test parsing SRT format with unified schema."""
    srt_content = """1
00:00:01,000 --> 00:00:04,000
Hello, world!

2
00:00:05,000 --> 00:00:08,000
This is a test."""

    df = local_session.create_dataframe({"transcript": [srt_content]})
    result = df.select(text.parse_transcript(col("transcript"), "srt")).to_polars()

    entries = result.to_series().to_list()[0]
    assert len(entries) == 2

    # Check first entry with unified schema
    entry1 = entries[0]
    assert entry1["index"] == 1
    assert entry1["speaker"] is None  # SRT doesn't have speakers
    assert entry1["start_time"] == 1.0  # 00:00:01,000 = 1 second
    assert entry1["end_time"] == 4.0   # 00:00:04,000 = 4 seconds
    assert entry1["duration"] == 3.0   # 4 - 1 = 3 seconds
    assert entry1["content"] == "Hello, world!"
    assert entry1["format"] == "srt"

    # Check second entry
    entry2 = entries[1]
    assert entry2["index"] == 2
    assert entry2["start_time"] == 5.0
    assert entry2["end_time"] == 8.0
    assert entry2["duration"] == 3.0
    assert entry2["content"] == "This is a test."
    assert entry2["format"] == "srt"


def test_parse_transcript_webvtt_format(local_session):
    """Test parsing WebVTT format with unified schema."""
    webvtt_content = """WEBVTT

REGION: 100 100 100 100

STYLE: 100% 100% 100% 100%

1
00:00:01.000 --> 00:00:04.000
Hello, world!

00:00:05.000 --> 00:00:08.000
<v User1>This is a test.</v>

NOTE: note blocks should be ignored

cue 55
00:00:09.000 --> 00:00:12.000 <c.bite>
<i>italics</i> and <b>bold</b>
multiline

"""


    df = local_session.create_dataframe({"transcript": [webvtt_content]})
    result = df.select(text.parse_transcript(col("transcript"), "webvtt")).to_polars()

    entries = result.to_series().to_list()[0]
    assert len(entries) == 3

    # Check first entry with unified schema
    entry1 = entries[0]
    assert entry1["index"] == 1
    assert entry1["speaker"] is None
    assert entry1["start_time"] == 1.0  # 00:00:01,000 = 1 second
    assert entry1["end_time"] == 4.0   # 00:00:04,000 = 4 seconds
    assert entry1["duration"] == 3.0   # 4 - 1 = 3 seconds
    assert entry1["content"] == "Hello, world!"
    assert entry1["format"] == "webvtt"

    # Check second entry
    entry2 = entries[1]
    assert entry2["index"] == 2
    assert entry2["speaker"] == "User1"
    assert entry2["start_time"] == 5.0
    assert entry2["end_time"] == 8.0
    assert entry2["duration"] == 3.0
    assert entry2["content"] == "This is a test."
    assert entry2["format"] == "webvtt"

    # Check third entry
    entry3 = entries[2]
    assert entry3["index"] == 3
    assert entry3["speaker"] is None
    assert entry3["start_time"] == 9.0
    assert entry3["end_time"] == 12.0
    assert entry3["duration"] == 3.0
    assert entry3["content"] == "italics and bold multiline"

def test_parse_transcript_generic_format(local_session):
    """Test parsing generic conversation format with unified schema."""
    generic_content = """Nitay (00:01.451)
Great to have you here today.

Apurva Mehta (00:09.006)
Thanks, looking forward to it."""

    df = local_session.create_dataframe({"transcript": [generic_content]})
    result = df.select(text.parse_transcript(col("transcript"), "generic")).to_polars()

    entries = result.to_series().to_list()[0]
    assert len(entries) == 2

    # Check first entry with unified schema
    entry1 = entries[0]
    assert entry1["index"] == 1  # Auto-generated index
    assert entry1["speaker"] == "Nitay"
    assert entry1["start_time"] == 1.451
    assert entry1["end_time"] == 9.006  # Look-ahead to next entry
    assert abs(entry1["duration"] - 7.555) < 0.001  # 9.006 - 1.451
    assert entry1["content"] == "Great to have you here today."
    assert entry1["format"] == "generic"

    # Check second entry (last entry has no end time)
    entry2 = entries[1]
    assert entry2["index"] == 2
    assert entry2["speaker"] == "Apurva Mehta"
    assert entry2["start_time"] == 9.006
    assert entry2["end_time"] is None  # No next entry to look ahead to
    assert entry2["duration"] is None
    assert entry2["content"] == "Thanks, looking forward to it."
    assert entry2["format"] == "generic"


def test_parse_transcript_empty_content(local_session):
    """Test parsing transcript with empty content returns empty array."""
    df = local_session.create_dataframe({"transcript": [""]})
    result = df.select(text.parse_transcript(col("transcript"), "srt")).to_polars()

    # Empty content should return empty array
    assert result.to_series().to_list()[0] == []


def test_parse_transcript_invalid_content(local_session):
    """Test parsing invalid transcript content returns null."""
    df = local_session.create_dataframe({"transcript": ["Not a valid transcript format"]})
    result = df.select(text.parse_transcript(col("transcript"), "srt")).to_polars()

    # Invalid content should return null
    assert result.to_series().to_list()[0] is None


def test_parse_transcript_multiline_content(local_session):
    """Test parsing SRT with multiline content."""
    srt_content = """1
00:00:01,000 --> 00:00:04,000
Line one of subtitle.
Line two of subtitle."""

    df = local_session.create_dataframe({"transcript": [srt_content]})
    result = df.select(text.parse_transcript(col("transcript"), "srt")).to_polars()

    entries = result.to_series().to_list()[0]
    assert len(entries) == 1

    entry = entries[0]
    assert "Line one of subtitle." in entry["content"]
    assert "Line two of subtitle." in entry["content"]
    # Content should preserve newlines
    assert "\n" in entry["content"]
