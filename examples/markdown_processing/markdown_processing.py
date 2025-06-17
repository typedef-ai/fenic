"""Markdown Processing with Fenic.

This example demonstrates how to process academic papers and structured markdown documents
using fenic's specialized markdown functions, JSON processing, and text extraction capabilities.

We'll use the "Attention Is All You Need" paper to show:
1. Generating automatic table of contents from markdown structure
2. Extracting and structuring document sections into DataFrame rows
3. Filtering specific sections (References) using traditional text processing
4. Converting markdown to JSON and using jq for complex document navigation
5. Template-based text extraction to parse structured citation data
"""

from pathlib import Path

import fenic as fc


def main():
    """Process academic paper markdown content using fenic's specialized functions.

    This function demonstrates the key capabilities of fenic for processing structured markdown:

    1. Loading and configuring a fenic session
    2. Reading markdown content into DataFrames
    3. Generating table of contents from document structure
    4. Extracting and structuring document sections
    5. Filtering and processing specific sections like References
    6. Converting markdown to JSON for complex querying
    7. Template-based extraction of structured data

    The example uses the "Attention Is All You Need" paper to show real-world document processing.
    """
    # Configure session with semantic capabilities (not used in this example but shows proper setup)
    config = fc.SessionConfig(
        app_name="markdown_processing",
        semantic=fc.SemanticConfig(
            language_models= {
                "mini": fc.OpenAIModelConfig(
                    model_name="gpt-4o-mini",
                    rpm=500,
                    tpm=200_000
                )
            }
        )
    )

    # Initialize fenic session
    session = fc.Session.get_or_create(config)

    # Load the academic paper markdown content from file
    paper_path = Path(__file__).parent / "attention_is_all_you_need.md"
    with open(paper_path, 'r', encoding='utf-8') as f:
        paper_content = f.read()

    # Create DataFrame with the paper content as a single row
    df = session.create_dataframe({
        "paper_title": ["Attention Is All You Need"],
        "content": [paper_content]
    })

    # Cast content to MarkdownType to enable markdown-specific functions
    df = df.select(
        fc.col("paper_title"),
        fc.col("content").cast(fc.MarkdownType).alias("markdown")
    )

    print("=== PAPER LOADED ===")
    result = df.select(fc.col('paper_title')).to_polars()
    print(f"Paper: {result['paper_title'][0]}")
    print()

    # 1. Generate Table of Contents using markdown.generate_toc()
    print("=== 1. GENERATING TABLE OF CONTENTS ===")
    toc_df = df.select(
        fc.col("paper_title"),
        fc.markdown.generate_toc(fc.col("markdown")).alias("toc")
    )

    print("Table of Contents DataFrame:")
    toc_df.show()
    print()

    # 2. Extract all document sections and convert to structured DataFrame
    print("=== 2. EXTRACTING SECTIONS INTO DATAFRAME ===")
    sections_df = df.select(
        fc.col("paper_title"),
        fc.markdown.generate_toc(fc.col("markdown")).alias("toc"),
        # Extract sections up to level 2 headers, returning array of section objects
        fc.markdown.extract_header_chunks(fc.col("markdown"), header_level=2).alias("sections")
    ).explode("sections").unnest("sections")  # Convert array to rows and flatten struct

    print("Sections DataFrame (each row is a document section):")
    sections_df.show()
    print()

    # 3. Filter for specific section (References) and parse its content
    print("=== 3. EXTRACTING REFERENCES SECTION ===")
    references_df = sections_df.filter(
        fc.col("heading").contains("References")
    )

    print("Individual references extracted by splitting on citation numbers:")
    # Split references content on [1], [2], etc. patterns to separate individual citations
    references_df.select(
        fc.text.split(fc.col("content"), r"\[\d+\]").alias("references")
    ).explode("references").show()
    print()

    # 4. Extract references using JSON + jq approach
    print("=== 4. EXTRACTING REFERENCES WITH JSON + JQ ===")
    # Convert the original document to JSON structure
    document_json_df = df.select(
        fc.col("paper_title"),
        fc.markdown.to_json(fc.col("markdown")).alias("document_json")
    )

    # Extract individual references using pure jq
    # References are nested under "7 Conclusion" -> "References" heading
    individual_refs_df = document_json_df.select(
        fc.col("paper_title"),
        fc.json.jq(
            fc.col("document_json"),
            # Navigate to References section and split text into individual citations
            '.children[-1].children[] | select(.type == "heading" and (.content[0].text == "References")) | .children[0].content[0].text | split("\\n") | .[]'
        ).alias("reference_text")
    ).explode("reference_text").select(
        fc.col("paper_title"),
        fc.col("reference_text").cast(fc.StringType).alias("reference_text")
    ).filter(
        fc.col("reference_text") != ""
    )

    print("Individual references extracted using JSON + jq:")
    individual_refs_df.show()
    print()

    # Extract reference number and content using text.extract() with template
    print("Extracting reference numbers and content using text.extract():")
    parsed_refs_df = individual_refs_df.select(
        fc.col("paper_title"),
        fc.text.extract(
            fc.col("reference_text"),
            "[${ref_number:none}] ${content:none}"
        ).alias("parsed_ref")
    ).select(
        fc.col("paper_title"),
        fc.col("parsed_ref").get_item("ref_number").alias("reference_number"),
        fc.col("parsed_ref").get_item("content").alias("citation_content")
    )

    print("References with separated numbers and content:")
    parsed_refs_df.show()
    print()

    # Clean up session resources
    session.stop()

    print("=== ANALYSIS COMPLETE ===")
    print("This example demonstrated:")
    print("✓ Loading markdown documents into fenic DataFrames")
    print("✓ Generating table of contents with markdown.generate_toc()")
    print("✓ Extracting structured sections with markdown.extract_header_chunks()")
    print("✓ Converting arrays to rows with explode() and unnest()")
    print("✓ Filtering DataFrames to find specific sections")
    print("✓ Text processing with split() and regex patterns")
    print("✓ Converting markdown to JSON with markdown.to_json()")
    print("✓ Querying JSON structures with json.jq() and complex queries")
    print("✓ Template-based text extraction with text.extract()")
    print("✓ Structured citation parsing into separate fields")
if __name__ == "__main__":
    main()
