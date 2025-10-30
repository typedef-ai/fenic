#!/usr/bin/env python3
"""
Fenic script to parse a PDF into markdown and run a complex semantic map.
The semantic map uses a sophisticated Jinja prompt designed to maximize LLM processing time.
"""

import argparse
import sys
from pathlib import Path

from fenic import col, semantic
from fenic.api.session.config import (
    AnthropicLanguageModel,
    GoogleDeveloperLanguageModel,
    OpenAILanguageModel,
    OpenRouterLanguageModel,
    SemanticConfig,
    SessionConfig,
)
from fenic.api.session.session import Session

# Complex multi-step analysis prompt using Jinja2 template
COMPLEX_PROMPT = """
Perform a comprehensive, multi-layered analysis of the following document:

{{ markdown_content }}

Please complete ALL of the following tasks in great detail:

## Task 1: Line-by-Line Analysis
For EVERY single line in the document above, provide:
- A 5-10 word summarization capturing the essence of that line
- The primary topic or theme of that line
- Any key entities (people, places, organizations, concepts) mentioned
- The sentiment or tone of that line (positive, negative, neutral, analytical, etc.)
- The informational density (high, medium, low)

## Task 2: Pattern Identification
After analyzing every line:
- Identify ALL recurring themes across lines
- List every commonality you can find between different lines
- Detect any patterns in writing style, structure, or content progression
- Note any contradictions or inconsistencies between lines
- Find all implicit connections between non-adjacent lines

Please be extremely thorough and detailed in your analysis. Do not skip any line or any analytical step. Take your time to think deeply about each aspect before providing your response.
"""

# Additional tasks to add more complexity to the prompt:
"""
## Task 3: Hierarchical Clustering
Group the lines into:
- Thematic clusters (group lines with similar topics)
- Structural clusters (group lines by their function: introduction, evidence, conclusion, etc.)
- Semantic clusters (group lines by meaning similarity)
- Temporal clusters (if any time-based progression exists)
- Importance clusters (critical, important, supporting, minor details)

## Task 4: Relationship Mapping
For each identified cluster:
- Explain how lines within the cluster relate to each other
- Identify the central line that best represents the cluster
- Describe inter-cluster relationships and dependencies
- Map cause-and-effect relationships between lines
- Identify prerequisite knowledge chains
 
## Task 5: Deep Semantic Analysis
- Extract the deepest underlying meaning of the document
- Identify implicit assumptions in each section
- Uncover unstated implications
- Detect rhetorical devices and their effects
- Analyze the logical flow and argumentation structure

## Task 6: Comparative Analysis
- Compare the beginning, middle, and end sections
- Identify evolution of ideas throughout the document
- Note any shifts in tone, style, or focus
- Detect any circular references or recursive themes
- Analyze information density distribution

## Task 7: Synthesis and Meta-Analysis
- Create a comprehensive summary that captures ALL nuances
- Generate alternative interpretations of ambiguous sections
- Propose questions that each section raises but doesn't answer
- Identify gaps in the information presented
- Suggest logical extensions of the ideas presented

## Task 8: Granular Statistics
Provide exact counts for:
- Total number of lines analyzed
- Number of unique themes identified
- Number of named entities found
- Number of commonalities discovered
- Number of patterns detected
- Number of relationships mapped

## Task 9: Cross-Reference Matrix
Create a detailed cross-reference showing:
- Which lines reference similar concepts
- Which lines build upon previous lines
- Which lines contradict or qualify other lines
- Which lines provide evidence for claims in other lines

## Task 10: Final Comprehensive Report
Synthesize all previous analyses into a final report that:
- Integrates all findings holistically
- Highlights the most significant discoveries
- Provides a multi-dimensional understanding of the document
- Offers actionable insights based on the analysis
- Suggests areas for further investigation

Please be extremely thorough and detailed in your analysis. Do not skip any line or any analytical step. Take your time to think deeply about each aspect before providing your response.
"""


def main():
    """Main execution function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Parse a PDF and run a complex semantic analysis using the specified language model."
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        nargs="?",
        default="sample.pdf",
        help="Path to the PDF file to process (default: sample.pdf)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["openai", "gemini", "anthropic", "openrouter"],
        default="openai",
        help="Language model to use for semantic analysis (default: openai)"
    )
    
    args = parser.parse_args()
    
    pdf_path = Path(args.pdf_path)
    model_choice = args.model
    
    if not pdf_path.exists():
        sys.exit(f"Error: PDF file not found at {pdf_path}")
    
    # Map user-friendly model names to internal model aliases
    model_alias_map = {
        "openai": "openai",
        "gemini": "gemini",
        "anthropic": "anthropic",
        "openrouter": "openrouter-thinking"
    }
    model_alias = model_alias_map[model_choice]
    
    # Create session with semantic capabilities
    config = SessionConfig(
        app_name="pdf_semantic_analyzer",
        semantic=SemanticConfig(
            language_models={
                # Configure Claude for complex reasoning tasks
                "openai": OpenAILanguageModel(
                    model_name="gpt-5",
                    rpm=100,
                    tpm=1000000,
                    profiles={
                        "default": OpenAILanguageModel.Profile(
                            reasoning_effort="high",
                            verbosity="medium",
                        )
                    },
                    default_profile="default"
                ),
                "gemini": GoogleDeveloperLanguageModel(
                    model_name="gemini-2.5-pro",
                    rpm=100,
                    tpm=1000000,
                    profiles={
                        "default": GoogleDeveloperLanguageModel.Profile(
                            thinking_token_budget=8192
                        )
                    },
                    default_profile="default"
                ),
                "anthropic": AnthropicLanguageModel(
                    model_name="claude-sonnet-4-5",
                    rpm=100,
                    input_tpm=1000000,
                    output_tpm=1000000,
                    profiles={
                        "default": AnthropicLanguageModel.Profile(
                            thinking_token_budget=8192
                        )
                    },
                    default_profile="default"
                ),
                "openrouter-parser": OpenRouterLanguageModel(
                    model_name="google/gemini-2.0-flash-lite-001",
                    rpm=100,
                    tpm=1000000,
                    profiles={
                        "default": OpenRouterLanguageModel.Profile(
                            parsing_engine="pdf-text"
                        )
                    },
                    default_profile="default"
                ),
                "openrouter-thinking": OpenRouterLanguageModel(
                    model_name="google/gemini-2.5-pro",
                    rpm=100,
                    tpm=1000000,
                    profiles={
                        "default": OpenRouterLanguageModel.Profile(
                            reasoning_effort="high"
                        )
                    },
                    default_profile="default"
                )
            },
            default_language_model=model_alias
        )
    )
    
    session = Session.get_or_create(config)
    existing_tables = session.catalog.list_tables()

    if "parsed_df" in existing_tables:
        parsed_df = session.table("parsed_df")
    else:
        
        # Create a DataFrame with the PDF path
        pdf_df = session.create_dataframe({
            "file_path": [str(pdf_path)]
        })

        # Parse PDF to markdown with detailed page separation
        parsed_df = pdf_df.select(
            col("file_path"),
            semantic.parse_pdf(
                col("file_path"),
                page_separator="\n\n--- PAGE {page} ---\n\n",
                describe_images=True,  # Include image descriptions for completeness
                model_alias="openrouter-parser",
            ).alias("markdown_content")
        ).cache()

        parsed_df.write.save_as_table("parsed_df", mode="overwrite")
    # Apply the complex semantic map designed to maximize processing time
    analyzed_df = parsed_df.select(
        col("file_path"),
        col("markdown_content"),
        semantic.map(
            COMPLEX_PROMPT,
            markdown_content=col("markdown_content"),
            temperature=1.0,  # Add some creativity to the analysis
            max_output_tokens=55808,  # Allow for very long responses
            request_timeout=600,
            model_alias=model_alias,
        ).alias("complex_analysis")
    )
    
    # Collect and save results
    result = analyzed_df.collect("polars")
    print(result.data)
    breakpoint()
  


if __name__ == "__main__":
    main()