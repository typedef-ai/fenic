# Podcast Summarization with Fenic

[View in Github](https://github.com/typedef-ai/fenic/blob/main/examples/podcast_summarization/README.md)

This example demonstrates comprehensive podcast transcript summarization using Fenic's semantic operations and unstructured data processing capabilities.

## Overview

This pipeline processes a Lex Fridman podcast episode featuring the Cursor team, showcasing:

- **Extractive & Abstractive Summarization**: Multiple summarization techniques
- **Recursive Summarization**: Chunked processing for long-form content
- **Role-Specific Analysis**: Tailored summaries for host vs guests
- **Unstructured Data Processing**: JSON transcript parsing and analysis

## Data

- **Episode**: "#447 Cursor Team: Future of Programming with AI"
- **Duration**: 2:37:38
- **Participants**: Lex Fridman (host), Michael Truell, Arvid Lunnemark, Aman Sanger, Sualeh Asif
- **Format**: JSON transcript with word-level timing and speaker diarization

## Pipeline Steps

### 1. Data Loading & Processing

- Load JSON files as raw text strings (showcasing unstructured data handling)
- Parse metadata using JSON operations and type casting
- Extract word-level and segment-level data from transcript

### 2. Speaker Identification

- Filter out noise speakers (ads, intro music) using duration thresholds
- Map anonymous speaker IDs to actual participant names
- Create speaker statistics and validation

### 3. Multi-Level Summarization

#### **Full Transcript Summary**

- Chunked recursive summarization for long-form content
- Uses `fc.text.recursive_word_chunk()` for optimal processing
- Combines chunk summaries into cohesive final summary

#### **Host-Specific Analysis (Lex Fridman)**

- Focuses on thought-provoking questions and insights
- Captures interviewing technique and philosophical depth
- Ignores basic facilitation to highlight intellectual contributions

#### **Individual Guest Summaries**

- Technical expertise and product insights
- Unique contributions to Cursor development
- Personal experiences and innovation perspectives

## Key Fenic Features Demonstrated

### **Unstructured Data Processing**

```python
# JSON type casting and extraction
fc.col("content").cast(fc.JsonType)
fc.json.jq(fc.col("json_data"), '.speaker').get_item(0).cast(fc.StringType)
```

### **Text Processing & Aggregation**

```python
# Proper aggregation with array operations
fc.collect_list("segment_text").alias("speech_segments")
fc.text.array_join(fc.col("speech_segments"), " ")
```

### **Semantic Operations**

```python
# Semantic mapping with placeholders
fc.semantic.map(
    "Analyze this guest's contributions... Guest: {{guest}}. Speech: {{speech}}",
    guest=fc.col("guest_name"),
    speech=fc.col("full_speech")
)
```

### **Advanced Filtering & Mapping**

```python
# Complex conditional mapping
fc.when(fc.col("speaker") == "SPEAKER_05", fc.lit("Lex Fridman"))
.when(fc.col("speaker") == "SPEAKER_02", fc.lit("Michael Truell"))
.otherwise(fc.lit("Unknown"))
```

## Technical Highlights

- **Chunked Processing**: Handles 2.5+ hour transcript efficiently
- **Speaker Filtering**: Removes noise using time-based thresholds
- **JSON Extraction**: Processes complex nested transcript data
- **Role-Aware Prompting**: Different analysis for host vs guests
- **Type Safety**: Proper casting for JSON-extracted fields

## Usage

```bash
# Ensure you have OpenAI API key configured
export OPENAI_API_KEY="your-api-key"

# Run the summarization pipeline
python podcast_summarization.py
```

## Output

The pipeline generates:

1. **Full Episode Summary**: Comprehensive overview of key themes and insights
2. **Host Analysis**: Lex Fridman's interviewing mastery and intellectual contributions
3. **Guest Summaries**: Individual analyses for each Cursor team member
4. **Speaker Statistics**: Speaking time, segment counts, and participation metrics

## Learning Outcomes

This example teaches:

- Working with real-world unstructured data (JSON transcripts)
- Combining multiple summarization approaches
- Handling long-form content with chunking strategies
- Creating role-specific semantic operations
- Building robust data pipelines with filtering and validation

Perfect for understanding how Fenic handles complex text processing workflows in production scenarios.
