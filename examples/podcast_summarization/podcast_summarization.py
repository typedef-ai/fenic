"""Podcast summarization example using fenic semantic operations.

This example demonstrates how to perform both extractive and abstractive summarization
on podcast transcripts using Fenic's semantic capabilities, including zero-shot
and few-shot prompting, and recursive summarization techniques.
"""

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

import fenic as fc


class PodcastSpeakersSchema(BaseModel):
    """Schema for extracting speaker information from the podcast description."""
    host_name: str = Field(description="The name of the podcast host (usually Lex Fridman)")
    guest_names: List[str] = Field(description="List of all guest names mentioned in the podcast description")
    guest_roles: List[str] = Field(description="List of professional roles, titles, or affiliations of the guests")

def main(config: Optional[fc.SessionConfig] = None):
    """Process podcast transcript to generate various types of summaries."""
    # 1. Configure session with semantic capabilities
    config = config or fc.SessionConfig(
        app_name="podcast_summarization",
        semantic=fc.SemanticConfig(
            language_models={
                "mini": fc.OpenAIModelConfig(
                    model_name="gpt-4o-mini",
                    rpm=500,
                    tpm=200_000,
                )
            }
        ),
    )

    # Create session
    session = fc.Session.get_or_create(config)

    print("Podcast Summarization Pipeline")
    print("=" * 50)
    print("Initializing session with semantic capabilities...")

    # 2. Load podcast data as raw text strings
    data_dir = Path(__file__).parent / "data"

    # Read metadata file as text
    with open(data_dir / "lex_ai_cursor_team_meta.json", "r") as f:
        meta_text = f.read()

    # Read transcript file as text
    with open(data_dir / "lex_ai_cursor_team.json", "r") as f:
        transcript_text = f.read()

    print("Loaded raw JSON files as text strings")
    print(f"Metadata text length: {len(meta_text)} characters")
    print(f"Transcript text length: {len(transcript_text)} characters")

    # 3. Create DataFrames with raw text content
    # Metadata DataFrame - single row with raw JSON text
    meta_data = [{
        "file_name": "lex_ai_cursor_team_meta.json",
        "content": meta_text,
        "content_type": "metadata"
    }]
    meta_df = session.create_dataframe(meta_data)

    # Transcript DataFrame - single row with raw JSON text
    transcript_data = [{
        "file_name": "lex_ai_cursor_team.json",
        "content": transcript_text,
        "content_type": "transcript"
    }]
    transcript_df = session.create_dataframe(transcript_data)

    print("\nDataFrames created with raw text content:")
    print("- Metadata DataFrame: 1 row with JSON text")
    print("- Transcript DataFrame: 1 row with JSON text")

    # Show content lengths
    print("\nContent overview:")
    meta_df.select(
        fc.col("file_name"),
        fc.col("content_type"),
        fc.text.length(fc.col("content")).alias("content_length")
    ).show()

    transcript_df.select(
        fc.col("file_name"),
        fc.col("content_type"),
        fc.text.length(fc.col("content")).alias("content_length")
    ).show()

    # 4. Process metadata: Convert string to JSON type and extract fields
    print("\n=== Step 4: Processing Metadata ===")

    # Cast content string to JSON type
    meta_json_df = meta_df.select(
        fc.col("file_name"),
        fc.col("content_type"),
        fc.col("content").cast(fc.JsonType).alias("json_data")
    )

    print("Metadata converted to JSON type:")
    meta_json_df.show(1)

    # Define metadata struct type
    metadata_struct = fc.StructType([
        fc.StructField("title", fc.StringType),
        fc.StructField("published", fc.StringType),
        fc.StructField("description", fc.StringType),
        fc.StructField("duration", fc.StringType),
        fc.StructField("audio_url", fc.StringType),
        fc.StructField("link", fc.StringType)
    ])

    # Cast entire JSON blob to struct
    meta_struct_df = meta_json_df.select(
        fc.col("file_name"),
        fc.col("json_data").cast(metadata_struct).alias("metadata")
    )

    print("JSON cast to struct type:")
    meta_struct_df.show(1)

    # Extract fields from struct
    meta_extracted_df = meta_struct_df.select(
        fc.col("file_name"),
        fc.col("metadata").title.alias("title"),
        fc.col("metadata").published.alias("published"),
        fc.col("metadata").description.alias("description"),
        fc.col("metadata").duration.alias("duration"),
        fc.col("metadata").audio_url.alias("audio_url"),
        fc.col("metadata").link.alias("link")
    )

    print("\nExtracted metadata fields:")
    meta_extracted_df.select(
        fc.col("title"),
        fc.col("duration"),
        fc.col("description"),
        fc.text.length(fc.col("description")).alias("description_length")
    ).show()

    print("\nMetadata processing complete - ready for summarization!")

    # 5. Process transcript: Create word-level DataFrame
    print("\n=== Step 5: Creating Word-Level DataFrame ===")

    # Cast transcript content to JSON type
    transcript_json_df = transcript_df.select(
        fc.col("file_name"),
        fc.col("content_type"),
        fc.col("content").cast(fc.JsonType).alias("json_data")
    )

    print("Transcript converted to JSON type:")
    transcript_json_df.show(1)

    # Extract all words from all segments using simplified JQ
    words_raw_df = transcript_json_df.select(
        fc.json.jq(
            fc.col("json_data"),
            # Much simpler: just get all words from all segments
            '.segments[] | .words[]'
        ).alias("word_data")
    ).explode("word_data")  # Convert array of word objects into separate rows

    print(f"Extracted word objects: {words_raw_df.count()} words")
    words_raw_df.show(3)

    # Extract and cast individual word fields to proper types
    words_df = words_raw_df.select(
        # Extract basic word-level fields only
        fc.json.jq(fc.col("word_data"), '.word').get_item(0).cast(fc.StringType).alias("word_text"),
        fc.json.jq(fc.col("word_data"), '.speaker').get_item(0).cast(fc.StringType).alias("speaker"),
        fc.json.jq(fc.col("word_data"), '.start').get_item(0).cast(fc.FloatType).alias("start_time"),
        fc.json.jq(fc.col("word_data"), '.end').get_item(0).cast(fc.FloatType).alias("end_time"),
        fc.json.jq(fc.col("word_data"), '.score').get_item(0).cast(fc.FloatType).alias("confidence_score")
    ).select(
        # Add calculated fields
        "*",
        (fc.col("end_time") - fc.col("start_time")).alias("duration")
    )

    print("\n📊 Word-Level DataFrame with calculated fields:")
    words_df.show(10)

    print(f"\nWord-level extraction complete: {words_df.count()} words processed")

    # 6. Process transcript: Create segment-level DataFrame
    print("\n=== Step 6: Creating Segment-Level DataFrame ===")

    # Extract segments using JQ
    segments_raw_df = transcript_json_df.select(
        fc.json.jq(
            fc.col("json_data"),
            # Extract segment objects with their text, timing, and word arrays
            '.segments[]'
        ).alias("segment_data")
    ).explode("segment_data")  # Convert segments array into separate rows

    print(f"Extracted segment objects: {segments_raw_df.count()} segments")
    segments_raw_df.show(3)

    # Extract segment fields and calculate aggregated metrics
    segments_df = segments_raw_df.select(
        # Extract basic segment data
        fc.json.jq(fc.col("segment_data"), '.text').get_item(0).cast(fc.StringType).alias("segment_text"),
        fc.json.jq(fc.col("segment_data"), '.start').get_item(0).cast(fc.FloatType).alias("start_time"),
        fc.json.jq(fc.col("segment_data"), '.end').get_item(0).cast(fc.FloatType).alias("end_time"),
        # Extract speaker directly from segment
        fc.json.jq(fc.col("segment_data"), '.speaker').get_item(0).cast(fc.StringType).alias("speaker"),
        # Calculate word count using JQ array length
        fc.json.jq(fc.col("segment_data"), '.words | length').get_item(0).cast(fc.IntegerType).alias("word_count"),
        # Calculate average confidence using JQ array aggregation
        fc.json.jq(fc.col("segment_data"), '[.words[].score] | add / length').get_item(0).cast(fc.FloatType).alias("average_confidence")
    ).select(
        # Add calculated fields
        "*",
        (fc.col("end_time") - fc.col("start_time")).alias("duration")
    )

    print("\n📊 Segment-Level DataFrame with calculated metrics:")
    segments_df.show(5)

    print(f"\nSegment-level extraction complete: {segments_df.count()} segments processed")

    # 7. Extract host and guest names using semantic operations
    print("\n=== Step 7: Extracting Host and Guest Names ===")

    # Apply semantic extraction to the description field
    speakers_extracted_df = meta_extracted_df.select(
        "*",
        fc.semantic.extract(fc.col("description"), PodcastSpeakersSchema).alias("speakers_info")
    )

    print("Semantic extraction of speakers applied")

    # Extract speaker information into clean columns
    speakers_df = speakers_extracted_df.select(
        fc.col("title"),
        fc.col("duration"),
        speakers_extracted_df.speakers_info.host_name.alias("host_name"),
        speakers_extracted_df.speakers_info.guest_names.alias("guest_names"),
        speakers_extracted_df.speakers_info.guest_roles.alias("guest_roles")
    )

    print("\n📊 Extracted Speaker Information:")
    speakers_df.show()

    print("\nSpeaker extraction complete!")

    # 8. Identify speakers by analyzing their speech patterns
    print("\n=== Step 8: Identifying Speaker Names ===")

    # Aggregate all speech by speaker
    speaker_aggregated_df = segments_df.group_by("speaker").agg(
        fc.collect_list("segment_text").alias("speech_segments"),
        fc.min("start_time").alias("first_speaking_time"),
        fc.max("end_time").alias("last_speaking_time"),
        fc.count("*").alias("segment_count"),
        fc.sum("duration").alias("total_speaking_time")
    ).select(
        "*",
        fc.text.array_join(fc.col("speech_segments"), " ").alias("full_speech")
    )

    print("Aggregated speech by speaker:")
    speaker_aggregated_df.select(
        fc.col("speaker"),
        fc.col("first_speaking_time"),
        fc.col("segment_count"),
        fc.col("total_speaking_time"),
        fc.text.length(fc.col("full_speech")).alias("speech_length")
    ).sort("first_speaking_time").show()

    # Filter out speakers with minimal speaking time (< 60 seconds) to remove ads/noise
    speaker_filtered_df = speaker_aggregated_df.filter(
        fc.col("total_speaking_time") >= 60.0  # At least 1 minute of speaking
    )

    print("\nFiltered out speakers with < 60 seconds of speech")
    print("Remaining speakers:")
    speaker_filtered_df.select(
        fc.col("speaker"),
        fc.col("first_speaking_time"),
        fc.col("total_speaking_time")
    ).sort("first_speaking_time").show()

    # Sort by first speaking time to see who spoke first
    speaker_sorted_df = speaker_filtered_df.sort("first_speaking_time")

    print("\nApplying manual speaker mapping...")

    # Create speaker mapping based on provided assignments
    speaker_mapping_df = speaker_sorted_df.select(
        fc.col("speaker"),
        fc.col("first_speaking_time"),
        fc.col("total_speaking_time"),
        # Map speakers to actual names
        fc.when(fc.col("speaker") == "SPEAKER_05", fc.lit("Lex Fridman"))
        .when(fc.col("speaker") == "SPEAKER_02", fc.lit("Michael Truell"))
        .when(fc.col("speaker") == "SPEAKER_03", fc.lit("Arvid Lunnemark"))
        .when(fc.col("speaker") == "SPEAKER_01", fc.lit("Aman Sanger"))
        .when(fc.col("speaker") == "SPEAKER_04", fc.lit("Sualeh Asif"))
        .otherwise(fc.lit("Unknown")).alias("identified_name"),
        # Map speakers to roles
        fc.when(fc.col("speaker") == "SPEAKER_05", fc.lit("HOST"))
        .otherwise(fc.lit("GUEST")).alias("role")
    ).sort("first_speaking_time")

    print("\n📊 Speaker Identification Results:")
    speaker_mapping_df.show()

    print("\nSpeaker identification complete!")

    # 9. Chunked Recursive Summarization
    print("\n=== Step 9: Chunked Recursive Summarization ===")

    # First, combine all segments into full transcript text
    full_transcript_df = segments_df.agg(
        fc.collect_list("segment_text").alias("segment_list")
    ).select(
        "*",
        fc.text.array_join(fc.col("segment_list"), " ").alias("full_transcript_text")
    )

    print("Full transcript assembled")

    # Step 1: Chunk the transcript into manageable pieces (using word chunking)
    chunked_df = full_transcript_df.select(
        fc.text.recursive_word_chunk(
            fc.col("full_transcript_text"),
            chunk_size=1500,  # ~5-7 minutes of speech
            chunk_overlap_percentage=10
        ).alias("chunks")
    ).explode("chunks").select(
        fc.col("chunks").alias("chunk_text")
    )

    print(f"Transcript split into {chunked_df.count()} chunks")
    chunked_df.select(fc.text.length(fc.col("chunk_text")).alias("chunk_length")).show(5)

    # Step 2: Summarize each chunk independently
    print("\nStep 2: Summarizing individual chunks...")

    chunk_summaries_df = chunked_df.select(
        "*",
        fc.semantic.map(
            "Summarize this portion of a Lex Fridman podcast with the Cursor team. Focus on key technical insights, product decisions, and important discussion points. Keep the summary concise but capture the main ideas. Chunk: {chunk_text}"
        ).alias("chunk_summary")
    )

    print("Individual chunk summaries created")
    chunk_summaries_df.select(fc.text.length(fc.col("chunk_summary")).alias("summary_length")).show(5)

    # Step 3: Combine chunk summaries for recursive summarization
    print("\nStep 3: Recursive combination of summaries...")

    combined_summaries_df = chunk_summaries_df.agg(
        fc.collect_list("chunk_summary").alias("summary_list")
    ).select(
        "*",
        fc.text.array_join(fc.col("summary_list"), " ").alias("combined_summaries")
    )

    # Step 4: Create final summary from combined summaries
    final_summary_df = combined_summaries_df.select(
        "*",
        fc.semantic.map(
            "Create a comprehensive summary of this Lex Fridman podcast episode with the Cursor team (Michael Truell, Arvid Lunnemark, Aman Sanger, Sualeh Asif). Synthesize the key themes, technical insights, product vision, and important discussion points from these chunk summaries. Structure it as a cohesive narrative that captures the essence of the conversation. Combined summaries: {combined_summaries}"
        ).alias("final_summary")
    )

    print("\n📋 Final Podcast Summary:")
    print("=" * 80)
    final_summary_df.select(fc.col("final_summary")).show()

    print("\n✅ Chunked recursive summarization complete!")

    # 10. Host-Specific Summarization (Lex Fridman)
    print("\n=== Step 10: Host-Specific Summarization ===")

    # Filter segments for the host only (SPEAKER_05 = Lex Fridman)
    host_segments_df = segments_df.filter(fc.col("speaker") == "SPEAKER_05")

    print(f"Host segments: {host_segments_df.count()} segments")

    # Get host speaking time
    host_time_df = host_segments_df.agg(fc.sum('duration').alias('total_duration'))
    print("Host total speaking time:")
    host_time_df.show()

    # Aggregate all host speech
    host_speech_df = host_segments_df.agg(
        fc.collect_list("segment_text").alias("host_segments_list")
    ).select(
        "*",
        fc.text.array_join(fc.col("host_segments_list"), " ").alias("host_full_speech")
    )

    print("Host speech aggregated")

    # Create role-specific host summary focusing on his contributions as interviewer/thought leader
    host_summary_df = host_speech_df.select(
        "*",
        fc.semantic.map(
            "Analyze Lex Fridman's role as host in this podcast conversation with the Cursor team. Focus on: 1) His most thought-provoking and insightful questions that drove meaningful discussion, 2) Personal insights, experiences, and expertise he shared, 3) How he guided the conversation toward deeper philosophical or technical topics, 4) Broader connections he made between ideas, technology, and humanity, 5) His unique perspective on AI, programming, and the future. Ignore basic facilitation, simple acknowledgments, and routine transitions. Capture his intellectual contributions and interviewing mastery. Host speech: {host_full_speech}"
        ).alias("host_analysis")
    )

    print("\n🎙️ Host Analysis - Lex Fridman's Contributions:")
    print("=" * 80)
    host_summary_df.select(fc.col("host_analysis")).show()

    print("\n✅ Host-specific summarization complete!")

    # 11. Individual Guest Summaries
    print("\n=== Step 11: Individual Guest Summaries ===")

    # Create a mapping of guest speakers and their names for the summaries

    # Filter guest segments and aggregate speech for each guest
    guest_segments_df = segments_df.filter(
        (fc.col("speaker") != "SPEAKER_05") & (fc.col("speaker") != "null")  # Exclude host and null speakers
    )

    print(f"Total guest segments: {guest_segments_df.count()} segments")

    # Group by speaker and aggregate their speech
    guest_speech_df = guest_segments_df.group_by("speaker").agg(
        fc.collect_list("segment_text").alias("speech_segments"),
        fc.count("*").alias("segment_count"),
        fc.sum("duration").alias("total_speaking_time")
    ).select(
        "*",
        fc.text.array_join(fc.col("speech_segments"), " ").alias("full_speech")
    )

    # Add guest names to the dataframe
    guest_with_names_df = guest_speech_df.select(
        "*",
        fc.when(fc.col("speaker") == "SPEAKER_02", fc.lit("Michael Truell"))
        .when(fc.col("speaker") == "SPEAKER_03", fc.lit("Arvid Lunnemark"))
        .when(fc.col("speaker") == "SPEAKER_01", fc.lit("Aman Sanger"))
        .when(fc.col("speaker") == "SPEAKER_04", fc.lit("Sualeh Asif"))
        .alias("guest_name")
    )

    # filter out noisy speakers
    guest_with_names_df = guest_with_names_df.filter(
        fc.col("segment_count") > 10
    )

    print("\nGuest speaking statistics:")
    guest_with_names_df.select(
        fc.col("guest_name"),
        fc.col("segment_count"),
        fc.col("total_speaking_time")
    ).show()

    # Create guest-specific summaries focusing on their expertise and contributions
    guest_summaries_df = guest_with_names_df.select(
        "*",
        fc.semantic.map(
            "Analyze this guest's contributions to the Lex Fridman podcast about Cursor. Focus on: 1) Their specific technical expertise and insights shared, 2) Product vision and development perspectives they brought, 3) Unique experiences and stories they told, 4) Their role and contributions to the Cursor team/company, 5) Technical innovations or solutions they discussed, 6) Their perspective on AI-assisted programming and the future of coding. Capture their individual voice and expertise. Guest: {guest_name}. Speech: {full_speech}"
        ).alias("guest_analysis")
    )

    print("\n👥 Individual Guest Analyses:")
    print("=" * 80)

    # Show each guest's summary
    guest_summaries_df.select(
        fc.col("guest_name"),
        fc.col("guest_analysis")
    ).show()

    print("\n✅ Individual guest summaries complete!")

    # Clean up
    session.stop()
    print("\n✅ All processing complete with comprehensive summarization pipeline!")


if __name__ == "__main__":
    main()
