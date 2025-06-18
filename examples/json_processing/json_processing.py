"""JSON Processing with Fenic.

This example demonstrates comprehensive JSON processing capabilities using Fenic's
JSON type system and JQ integration. It processes a complex nested JSON file containing
whisper transcription data and transforms it into multiple structured DataFrames for
analysis.

Key Features Demonstrated:
- JSON type casting from string data
- Complex JQ queries for nested data extraction
- Array operations and aggregations within JSON
- Type conversion and calculated fields
- Hybrid processing combining JSON extraction with DataFrame operations

The example creates three complementary DataFrames:
1. Words DataFrame: Individual word-level data with timing and confidence
2. Segments DataFrame: Conversation segments with aggregated metrics
3. Speaker Summary DataFrame: High-level speaker analytics and patterns

This showcases how Fenic bridges JSON processing with traditional DataFrame
analytics for real-world use cases like audio transcript analysis.
"""

import fenic as fc


def main():
    """Process whisper transcript JSON data into structured DataFrames.

    This function demonstrates a complete JSON processing pipeline that:
    1. Loads a complex nested JSON file as a string
    2. Casts it to Fenic's JsonType for processing
    3. Uses JQ queries to extract data at multiple granularities
    4. Creates three DataFrames with different analytical perspectives
    5. Applies both JSON-native operations and traditional DataFrame aggregations

    The pipeline processes whisper transcription data containing:
    - Language metadata
    - Conversation segments with text and timing
    - Individual words with speaker, timing, and confidence scores

    Returns:
        None: Prints results and analysis to console
    """
    # Configure session with semantic capabilities
    config = fc.SessionConfig(
        app_name="json_processing",
        semantic=fc.SemanticConfig(
            language_models={
                "mini": fc.OpenAIModelConfig(
                    model_name="gpt-4o-mini",
                    rpm=500,
                    tpm=200_000
                )
            }
        )
    )

    # Create session
    session = fc.Session.get_or_create(config)

    print("🔧 JSON Processing with Fenic")
    print("=" * 50)

    try:
        # Try reading from source root first
        with open("examples/json_processing/whisper-transcript.json", "r") as f:
            json_content = f.read()
    except FileNotFoundError:
        # Fall back to current directory
        with open("whisper-transcript.json", "r") as f:
            json_content = f.read()

    # Create dataframe with the JSON string
    df = session.create_dataframe([{"json_string": json_content}])

    print(f"Loaded JSON file as string with {df.count()} rows")
    df.show(1)

    # Cast the JSON string to JSON type
    df_json = df.select(
        fc.col("json_string").cast(fc.JsonType).alias("json_data")
    )

    print("\nJSON data cast to JSON type:")
    df_json.show(1)

    # 1. Create Words DataFrame (Most Granular)
    print("\n🔤 Creating Words DataFrame...")

    # Extract all words from all segments using JQ
    # This demonstrates nested array traversal and variable binding in JQ
    words_df = df_json.select(
        fc.json.jq(
            fc.col("json_data"),
            # JQ query explanation:
            # - '.segments[] as $seg' iterates through segments, binding each to $seg
            # - '$seg.words[]' iterates through words in each segment
            # - Constructs object with both word-level and segment-level data
            '.segments[] as $seg | $seg.words[] | {word: .word, speaker: .speaker, start: .start, end: .end, probability: .probability, segment_start: $seg.start, segment_end: $seg.end, segment_text: $seg.text}'
        ).alias("word_data")
    ).explode("word_data")  # Convert array of word objects into separate rows

    print(f"Extracted {words_df.count()} individual words")
    words_df.show(3)

    # Extract scalar values using struct casting and unnest - more efficient than JQ + get_item(0)
    # Define schema for word-level data structure
    word_schema = fc.StructType([
        fc.StructField("word", fc.StringType),
        fc.StructField("speaker", fc.StringType),
        fc.StructField("start", fc.FloatType),
        fc.StructField("end", fc.FloatType),
        fc.StructField("probability", fc.FloatType),
        fc.StructField("segment_start", fc.FloatType),
        fc.StructField("segment_end", fc.FloatType)
    ])

    # Cast to struct and unnest to automatically extract all fields
    words_clean_df = words_df.select(
        fc.col("word_data").cast(word_schema).alias("word_struct")
    ).unnest("word_struct").select(
        # Rename fields for clarity
        fc.col("word").alias("word_text"),
        fc.col("speaker"),
        fc.col("start").alias("start_time"),
        fc.col("end").alias("end_time"),
        fc.col("probability"),
        fc.col("segment_start"),
        fc.col("segment_end")
    )

    print("\nScalar extracted fields:")
    words_clean_df.show(3)

    # Add calculated fields - types are already correct from struct schema
    # This demonstrates arithmetic operations on struct-extracted data
    words_final_df = words_clean_df.select(
        "*",
        # Calculate duration: end_time - start_time (demonstrates arithmetic on struct data)
        (fc.col("end_time") - fc.col("start_time")).alias("duration")
    )

    print("\n📊 Words DataFrame with calculated duration:")

    words_final_df.show(10)

    # 2. Create Segments DataFrame (Content-focused)
    print("\n📝 Creating Segments DataFrame...")

    # Extract segment-level data using JQ
    # This demonstrates extracting data at a different granularity level
    segments_df = df_json.select(
        fc.json.jq(
            fc.col("json_data"),
            # Extract segment objects with their text, timing, and nested words array
            '.segments[] | {text: .text, start: .start, end: .end, words: .words}'
        ).alias("segment_data")
    ).explode("segment_data")  # Convert segments array into separate rows

    print(f"Extracted {segments_df.count()} segments")
    segments_df.show(3)

    # Extract segment fields using hybrid approach: struct casting + JQ for complex aggregations
    # Define schema for basic segment fields (text, start, end)
    segment_basic_schema = fc.StructType([
        fc.StructField("text", fc.StringType),
        fc.StructField("start", fc.FloatType),
        fc.StructField("end", fc.FloatType)
    ])

    # First extract basic fields using struct casting, then add complex JQ aggregations
    segments_clean_df = segments_df.select(
        # Extract basic segment data using struct casting (more efficient)
        fc.col("segment_data").cast(segment_basic_schema).alias("segment_struct"),
        # Complex array aggregations still use JQ (best tool for this)
        fc.json.jq(fc.col("segment_data"), '.words | length').get_item(0).cast(fc.IntegerType).alias("word_count"),
        fc.json.jq(fc.col("segment_data"), '[.words[].probability] | add / length').get_item(0).cast(fc.FloatType).alias("average_confidence")
    ).unnest("segment_struct").select(
        # Rename for clarity
        fc.col("text").alias("segment_text"),
        fc.col("start").alias("start_time"),
        fc.col("end").alias("end_time"),
        fc.col("word_count"),
        fc.col("average_confidence")
    ).select(
        "segment_text",
        "start_time",
        "end_time",
        # Calculate segment duration using DataFrame arithmetic
        (fc.col("end_time") - fc.col("start_time")).alias("duration"),
        "word_count",
        "average_confidence"
    )

    print("\n📊 Segments DataFrame with calculated metrics:")
    segments_clean_df.show(5)

    # 3. Create Speaker Summary DataFrame (Aggregated)
    print("\n🎤 Creating Speaker Summary DataFrame...")

    # Use traditional DataFrame aggregations on JSON-extracted data
    # This demonstrates hybrid processing: JSON extraction + DataFrame analytics
    speaker_summary_df = words_final_df.group_by("speaker").agg(
        fc.count("*").alias("total_words"),                    # Count words per speaker
        fc.avg("probability").alias("average_confidence"),     # Average speech confidence
        fc.min("start_time").alias("first_speaking_time"),     # When speaker first appears
        fc.max("end_time").alias("last_speaking_time"),        # When speaker last appears
        fc.sum("duration").alias("total_speaking_time")        # Total time speaking
    ).select(
        "speaker",
        "total_words",
        "total_speaking_time",
        "average_confidence",
        "first_speaking_time",
        "last_speaking_time",
        # Calculate derived metric: words per minute
        (fc.col("total_words") / (fc.col("total_speaking_time") / 60.0)).alias("word_rate")
    )

    print("\n📊 Speaker Summary DataFrame:")
    speaker_summary_df.show()

    # Summary of what we accomplished
    print("\n🎯 JSON Processing Pipeline Summary:")
    print("=" * 60)
    print("📁 Input: Single JSON file (whisper-transcript.json)\n")
    print("📊 Output: 3 structured DataFrames")
    print()
    print("1. 🔤 Words DataFrame:")
    print(f"   - {words_final_df.count()} individual words extracted")
    print("   - Fields: word_text, speaker, timing, probability, duration")
    print("   - Demonstrates: JQ nested array extraction, type casting")
    print()
    print("2. 📝 Segments DataFrame:")
    print(f"   - {segments_clean_df.count()} conversation segments")
    print("   - Fields: text, timing, word_count, average_confidence")
    print("   - Demonstrates: JQ aggregations, array operations")
    print()
    print("3. 🎤 Speaker Summary DataFrame:")
    print(f"   - {speaker_summary_df.count()} speakers analyzed")
    print("   - Fields: totals, averages, speaking patterns, word rates")
    print("   - Demonstrates: DataFrame aggregations on JSON-extracted data")
    print()
    print("🔧 Key Fenic JSON Features Used:")
    print("   ✓ JSON type casting from strings")
    print("   ✓ JQ queries for complex nested extraction")
    print("   ✓ Array operations and aggregations")
    print("   ✓ Type conversion and calculated fields")
    print("   ✓ Traditional DataFrame operations on JSON data")

    # Clean up
    session.stop()

    print("\n✅ JSON processing complete!")


if __name__ == "__main__":
    # Note: Ensure you have set your OpenAI API key:
    # export OPENAI_API_KEY="your-api-key-here"
    main()
