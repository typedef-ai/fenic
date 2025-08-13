# JSON Processing with Fenic

<p>
  <a href="https://colab.research.google.com/github/typedef-ai/fenic/blob/main/examples/json_processing/json_processing.ipynb">
    <img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
  </a>
</p>

This example demonstrates comprehensive JSON processing capabilities using fenic's JSON type system and JQ integration. It processes a complex nested JSON file containing whisper transcription data and transforms it into multiple structured DataFrames for analysis.

## What This Example Shows

### Core JSON Processing Features

- **JSON Type Casting**: Loading string data and converting to Fenic's JsonType
- **JQ Integration**: Complex queries for nested data extraction and aggregation
- **Array Operations**: Processing nested arrays and extracting scalar values
- **Type Conversion**: Converting JSON data to appropriate DataFrame types
- **Hybrid Processing**: Combining JSON-native operations with traditional DataFrame analytics

### Real-World Use Case

The example processes audio transcription data from OpenAI's Whisper, demonstrating how to handle:

- Complex nested JSON structures
- Multiple levels of data granularity
- Time-series data with speaker attribution
- Confidence scores and quality metrics

## Data Structure

The input JSON contains:

```json
{
  "language": "en",
  "segments": [
    {
      "text": "Let me ask you about AI.",
      "start": 2.94,
      "end": 4.48,
      "words": [
        {
          "word": " Let",
          "start": 2.94,
          "end": 3.12,
          "speaker": "SPEAKER_01",
          "probability": 0.69384765625
        }
        // ... more words
      ]
    }
    // ... more segments
  ]
}
```

## Output DataFrames

The pipeline creates three complementary DataFrames:

### 1. Words DataFrame (Granular Analysis)

**Purpose**: Individual word-level analysis with timing and confidence

- `word_text`: The spoken word
- `speaker`: Speaker identifier (SPEAKER_00, SPEAKER_01)
- `start_time`, `end_time`: Word timing in seconds
- `duration`: Calculated word duration
- `probability`: Speech recognition confidence score

**Key Techniques**:

- Nested JQ traversal with variable binding
- Array explosion from nested structures
- Type casting for numeric operations

### 2. Segments DataFrame (Content Analysis)

**Purpose**: Conversation segments with aggregated metrics

- `segment_text`: Full text of the conversation segment
- `start_time`, `end_time`, `duration`: Segment timing
- `word_count`: Number of words in segment
- `average_confidence`: Average recognition confidence

**Key Techniques**:

- JQ array aggregations (`length`, `add / length`)
- In-query statistical calculations
- Mixed JSON and DataFrame operations

### 3. Speaker Summary DataFrame (Analytics)

**Purpose**: High-level speaker analytics and patterns

- `speaker`: Speaker identifier
- `total_words`: Total words spoken
- `total_speaking_time`: Total time speaking
- `average_confidence`: Overall speech quality
- `first_speaking_time`, `last_speaking_time`: Speaking timeframe
- `word_rate`: Words per minute

**Key Techniques**:

- Traditional DataFrame aggregations (`group_by`, `agg`)
- Multiple aggregation functions (count, avg, min, max, sum)
- Calculated metrics from aggregated data

## Data Extraction Approaches

### Struct Casting + Unnest (Recommended for Simple Fields)

For efficient extraction of simple fields, the example uses struct casting with unnest:

```python
# Define schema for structured extraction
word_schema = fc.StructType([
    fc.StructField("word", fc.StringType),
    fc.StructField("speaker", fc.StringType),
    fc.StructField("start", fc.FloatType),
    fc.StructField("end", fc.FloatType),
    fc.StructField("probability", fc.FloatType)
])

# Cast and unnest to extract all fields automatically
words_df.select(
    fc.col("word_data").cast(word_schema).alias("word_struct")
).unnest("word_struct")
```

**Benefits:**

- More efficient than multiple JQ queries
- Automatic type handling through schema
- Single operation extracts all fields
- Better performance for simple field extraction

### JQ Queries (Best for Complex Operations)

Complex array operations still use JQ for maximum power:

```jq
# Nested array traversal with variable binding
'.segments[] as $seg | $seg.words[] | {...}'

# Array length calculation
'.words | length'

# Array aggregation for averages
'[.words[].probability] | add / length'

# Object construction with mixed data levels
'{word: .word, segment_start: $seg.start, ...}'
```

**When to Use JQ:**

- Array aggregations (`length`, `add / length`)
- Complex transformations
- Variable binding and nested operations
- Mathematical operations within queries

## Running the Example

### Prerequisites

1. Set your OpenAI API key:

   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. Ensure you have the whisper-transcript.json file in the same directory

### Execute

```bash
python json_processing.py
```

### Expected Output

The script will display:

1. Raw JSON data loading and casting
2. Word-level extraction (showing ~4000+ individual words)
3. Segment-level analysis (showing conversation segments)
4. Speaker summary analytics (showing 2 speakers with statistics)
5. Comprehensive pipeline summary

## Technical Highlights

### JSON Type System

- Demonstrates proper JSON type casting from strings
- Shows how to work with Fenic's JsonType for complex operations

### JQ Query Complexity

- Simple field extraction: `.field`
- Nested traversal: `.segments[].words[]`
- Variable binding: `.segments[] as $seg`
- Array operations: `length`, `add / length`
- Object construction with mixed data sources

### Hybrid Processing Pattern

- JSON extraction for granular data access
- DataFrame aggregations for analytical operations
- Type conversion between JSON and DataFrame types
- Calculated fields using both JSON and DataFrame operations

## Learning Outcomes

After studying this example, you'll understand:

1. **How to load and cast JSON data** in Fenic
2. **Struct casting + unnest** for efficient field extraction
3. **Complex JQ queries** for advanced data manipulation
4. **When to choose** struct casting vs JQ for different use cases
5. **Array processing** and aggregation within JSON
6. **Type conversion** between JSON and DataFrame types
7. **Hybrid workflows** combining multiple extraction approaches
8. **Performance optimization** for JSON processing pipelines
9. **Real-world data processing** patterns for audio/transcript analysis

This example serves as a comprehensive reference for JSON processing in Fenic, showcasing both the power of JQ integration and the seamless bridge to traditional DataFrame analytics.
