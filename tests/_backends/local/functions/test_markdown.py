
from textwrap import dedent

from fenic import (
    ArrayType,
    ColumnField,
    IntegerType,
    JsonType,
    MarkdownType,
    StringType,
    StructField,
    StructType,
    col,
    markdown,
)


def test_md_to_json(local_session):
    df = local_session.create_dataframe(
        {
            "string_col": [
                "# hello\n\nSome text\n\n# goodbye",
                "# Title\nSome content",
                "## Subtitle\nMore content",
                "# Another Title\nEven more content",
            ]
        }
    )
    df = df.select(
        markdown.to_json(col("string_col").cast(MarkdownType)).alias("markdown_as_json")
    )
    assert df.schema.column_fields == [
        ColumnField("markdown_as_json", JsonType)
    ]
    result = df.to_polars()
    assert result["markdown_as_json"].to_list() == [
        '{"type":"document","children":[{"type":"heading","level":1,"content":[{"type":"text","text":"hello"}],"children":[{"type":"paragraph","content":[{"type":"text","text":"Some text"}]}]},{"type":"heading","level":1,"content":[{"type":"text","text":"goodbye"}],"children":[]}]}',
        '{"type":"document","children":[{"type":"heading","level":1,"content":[{"type":"text","text":"Title"}],"children":[{"type":"paragraph","content":[{"type":"text","text":"Some content"}]}]}]}',
        '{"type":"document","children":[{"type":"heading","level":2,"content":[{"type":"text","text":"Subtitle"}],"children":[{"type":"paragraph","content":[{"type":"text","text":"More content"}]}]}]}',
        '{"type":"document","children":[{"type":"heading","level":1,"content":[{"type":"text","text":"Another Title"}],"children":[{"type":"paragraph","content":[{"type":"text","text":"Even more content"}]}]}]}',
    ]

def test_md_get_code_blocks(local_session):
    df = local_session.create_dataframe(
        {
            "string_col": [
                # 1. Simple Rust code block under a header
                dedent("""
                    # Example in Rust

                    Here's a quick Rust example:

                    ```rust
                    fn main() {
                        println!("Hello, world!");
                    }
                    ```
                """).strip(),

                # 2. Python with nested code blocks inside list items
                dedent("""
                    # Python Tutorial

                    Steps:

                    1. Install Python
                    2. Write a script:

                    ```python
                    def greet(name):
                        return f"Hello, {name}"
                    ```

                    3. Run it:

                    ```python
                    print(greet("Alice"))
                    ```
                """).strip(),

                # 3. JavaScript with code inside nested sections
                dedent("""
                    # JavaScript Guide

                    ## Setup

                    First, set up your environment.

                    ## Example Code

                    ```javascript
                    function add(a, b) {
                      return a + b;
                    }
                    ```

                    ### Usage

                    ```javascript
                    console.log(add(2, 3));
                    ```
                """).strip(),

                # 4. Mixed content with deeply nested Rust and Python
                dedent("""
                    # Multi-Language Doc

                    ## Rust Section

                    Intro:

                    > Rust is fast.

                    ```rust
                    struct Point {
                        x: i32,
                        y: i32,
                    }
                    ```

                    ### More Rust

                    ```rust
                    impl Point {
                        fn new(x: i32, y: i32) -> Self {
                            Self { x, y }
                        }
                    }
                    ```

                    ## Python Section

                    ```python
                    from dataclasses import dataclass

                    @dataclass
                    class Point:
                        x: int
                        y: int
                    ```
                """).strip(),

                # 5. No code blocks
                "# No Code Blocks\n\nThis document has no code blocks.",
            ]
        }
    )
    df = df.select(col("string_col").cast(MarkdownType).alias("markdown"))
    all_code_blocks = df.select(markdown.get_code_blocks(col("markdown")).alias("code_blocks"))

    assert all_code_blocks.schema.column_fields == [
        ColumnField(
            "code_blocks",
            ArrayType(
                StructType([
                    StructField("language", StringType),
                    StructField("code", StringType)
                ])
            )
        )
    ]

    result = all_code_blocks.to_polars()
    result_list = result["code_blocks"].to_list()

    expected = [
        [{'language': 'rust', 'code': 'fn main() {\n    println!("Hello, world!");\n}'}],
        [
            {'language': 'python', 'code': 'def greet(name):\n    return f"Hello, {name}"'},
            {'language': 'python', 'code': 'print(greet("Alice"))'}
        ],
        [
            {'language': 'javascript', 'code': 'function add(a, b) {\n  return a + b;\n}'},
            {'language': 'javascript', 'code': 'console.log(add(2, 3));'}
        ],
        [
            {'language': 'rust', 'code': 'struct Point {\n    x: i32,\n    y: i32,\n}'},
            {'language': 'rust', 'code': 'impl Point {\n    fn new(x: i32, y: i32) -> Self {\n        Self { x, y }\n    }\n}'},
            {'language': 'python', 'code': 'from dataclasses import dataclass\n\n@dataclass\nclass Point:\n    x: int\n    y: int'}
        ],
        [],
    ]

    assert result_list == expected

    rust_code_blocks = df.select(markdown.get_code_blocks(col("markdown"), language_filter="rust").alias("code_blocks"))
    result = rust_code_blocks.to_polars()
    result_list = result["code_blocks"].to_list()

    expected = [
        [{'language': 'rust', 'code': 'fn main() {\n    println!("Hello, world!");\n}'}],
        [],
        [],
        [{'language': 'rust', 'code': 'struct Point {\n    x: i32,\n    y: i32,\n}'}, {'language': 'rust', 'code': 'impl Point {\n    fn new(x: i32, y: i32) -> Self {\n        Self { x, y }\n    }\n}'}],
        [],
    ]

    assert result_list == expected


def test_md_generate_toc(local_session):
    test_markdown = dedent("""
        # Introduction

        This is the introduction paragraph.

        ## Background

        Some background information here.

        ### Historical Context

        Details about history...

        ### Current State

        Information about the current state...

        ## Methodology

        Our approach involves...

        ### Data Collection

        How we collected data...

        #### Survey Design

        Details about the survey...

        #### Sampling Method

        How we sampled...

        ### Analysis Techniques

        Statistical methods used...

        # Results

        Our findings show...

        ## Key Findings

        ### Finding 1

        First major finding...

        ### Finding 2

        Second major finding...

        # Conclusion

        In conclusion...
    """).strip()

    df = local_session.create_dataframe(
        {
            "string_col": [
                test_markdown,
                "# Simple Doc\n\nJust a paragraph.",
                "## Only Level 2\n\n### And Level 3",
            ]
        }
    )
    df = df.select(col("string_col").cast(MarkdownType).alias("markdown"))

    # Test full TOC (all levels)
    full_toc = df.select(markdown.generate_toc(col("markdown")).alias("toc"))
    assert full_toc.schema.column_fields == [
        ColumnField("toc", MarkdownType)
    ]

    result = full_toc.to_polars()

    toc_list = result["toc"].to_list()

    # First document should have all headings
    assert toc_list[0] == dedent("""
        # Introduction
        ## Background
        ### Historical Context
        ### Current State
        ## Methodology
        ### Data Collection
        #### Survey Design
        #### Sampling Method
        ### Analysis Techniques
        # Results
        ## Key Findings
        ### Finding 1
        ### Finding 2
        # Conclusion
    """).strip()

    # Second document - simple
    assert toc_list[1] == "# Simple Doc"

    # Third document - no h1
    assert toc_list[2] == dedent("""
        ## Only Level 2
        ### And Level 3
    """).strip()

    # Test with max_level=3
    toc_level3 = df.select(markdown.generate_toc(col("markdown"), max_level=3).alias("toc"))
    result = toc_level3.to_polars()
    toc_list = result["toc"].to_list()

    # First document should exclude h4 headings
    assert toc_list[0] == dedent("""
        # Introduction
        ## Background
        ### Historical Context
        ### Current State
        ## Methodology
        ### Data Collection
        ### Analysis Techniques
        # Results
        ## Key Findings
        ### Finding 1
        ### Finding 2
        # Conclusion
    """).strip()

    # Test with max_level=2
    toc_level2 = df.select(markdown.generate_toc(col("markdown"), max_level=2).alias("toc"))
    result = toc_level2.to_polars()
    toc_list = result["toc"].to_list()

    # First document should only have h1 and h2
    assert toc_list[0] == dedent("""
        # Introduction
        ## Background
        ## Methodology
        # Results
        ## Key Findings
        # Conclusion
    """).strip()

    # Test with max_level=1
    toc_level1 = df.select(markdown.generate_toc(col("markdown"), max_level=1).alias("toc"))
    result = toc_level1.to_polars()
    toc_list = result["toc"].to_list()

    # First document should only have h1
    assert toc_list[0] == dedent("""
        # Introduction
        # Results
        # Conclusion
    """).strip()

    # Second document still has its h1
    assert toc_list[1] == "# Simple Doc"

    # Third document has no h1, so should be empty
    assert toc_list[2] is None


def test_md_chunk_by_headings(local_session):
    test_markdown = dedent(
        """
        # Introduction
        This is the introduction to our project.

        ## Getting Started
        Here's how to get started with the project.

        ### Prerequisites
        You need Python 3.8 or higher.

        ### Installation
        Run the following command:
        ```bash
        pip install our-package
        ```

        ## API Reference
        This section covers the API.

        ### Functions
        We have many functions available.

        ## Conclusion
        That's all for now!
        """
    )

    df = local_session.create_dataframe({"markdown": [test_markdown]})
    df = df.select(col("markdown").cast(MarkdownType).alias("md_col"))

    chunks_df = df.select(markdown.extract_header_chunks(col("md_col"), header_level=2).alias("chunks"))

    expected_schema = [
        ColumnField(
            "chunks",
            ArrayType(
                StructType([
                    StructField("heading", StringType),
                    StructField("level", IntegerType),
                    StructField("content", StringType),
                    StructField("parent_heading", StringType),
                    StructField("full_path", StringType),
                ])
            )
        )
    ]
    assert chunks_df.schema.column_fields == expected_schema

    result = chunks_df.to_polars()
    chunks = result["chunks"][0]

    # Should have 3 chunks: Getting Started, API Reference, Conclusion
    assert len(chunks) == 3

    # Verify first chunk (Getting Started)
    chunk1 = chunks[0]
    assert chunk1["heading"] == "Getting Started"
    assert chunk1["level"] == 2
    assert chunk1["parent_heading"] == "Introduction"
    assert chunk1["full_path"] == "Introduction > Getting Started"
    assert "get started with the project" in chunk1["content"]
    assert "Prerequisites" in chunk1["content"]  # Should include subsections
    assert "Python 3.8" in chunk1["content"]
    assert "pip install" in chunk1["content"]

    # Verify second chunk (API Reference)
    chunk2 = chunks[1]
    assert chunk2["heading"] == "API Reference"
    assert chunk2["level"] == 2
    assert chunk2["parent_heading"] == "Introduction"
    assert chunk2["full_path"] == "Introduction > API Reference"
    assert "covers the API" in chunk2["content"]
    assert "Functions" in chunk2["content"]  # Should include subsections
    assert "many functions available" in chunk2["content"]

    # Verify third chunk (Conclusion)
    chunk3 = chunks[2]
    assert chunk3["heading"] == "Conclusion"
    assert chunk3["level"] == 2
    assert chunk3["parent_heading"] == "Introduction"
    assert chunk3["full_path"] == "Introduction > Conclusion"
    assert "That's all for now" in chunk3["content"]


def test_md_chunk_by_headings_level1(local_session):
    """Test chunking at level 1 with multiple top-level sections."""
    test_markdown = """# Chapter 1
Introduction to the topic.

## Section A
Content for section A.

# Chapter 2
Second chapter content.

## Section B
Content for section B.

### Subsection B1
Detailed content.

# Chapter 3
Final chapter.
"""

    df = local_session.create_dataframe({"markdown": [test_markdown]})
    df = df.select(col("markdown").cast(MarkdownType).alias("md_col"))

    # Test chunking at level 1
    chunks_df = df.select(markdown.extract_header_chunks(col("md_col"), header_level=1).alias("chunks"))
    result = chunks_df.to_polars()
    chunks = result["chunks"][0]

    # Should have 3 chunks: Chapter 1, Chapter 2, Chapter 3
    assert len(chunks) == 3

    # Verify first chunk
    chunk1 = chunks[0]
    assert chunk1["heading"] == "Chapter 1"
    assert chunk1["level"] == 1
    assert chunk1["parent_heading"] is None  # No parent for top-level
    assert chunk1["full_path"] == "Chapter 1"
    assert "Introduction to the topic" in chunk1["content"]
    assert "Section A" in chunk1["content"]  # Should include subsections

    # Verify second chunk
    chunk2 = chunks[1]
    assert chunk2["heading"] == "Chapter 2"
    assert chunk2["level"] == 1
    assert chunk2["parent_heading"] is None
    assert chunk2["full_path"] == "Chapter 2"
    assert "Second chapter content" in chunk2["content"]
    assert "Section B" in chunk2["content"]
    assert "Subsection B1" in chunk2["content"]  # Should include all nested content

    # Verify third chunk
    chunk3 = chunks[2]
    assert chunk3["heading"] == "Chapter 3"
    assert chunk3["level"] == 1
    assert chunk3["parent_heading"] is None
    assert chunk3["full_path"] == "Chapter 3"
    assert "Final chapter" in chunk3["content"]


def test_md_chunk_by_headings_empty(local_session):
    """Test chunking with no headings at the specified level."""
    test_markdown = """# Main Title
Just some content without level 2 headings.

### Only Level 3
Some content here.
"""

    df = local_session.create_dataframe({"markdown": [test_markdown]})
    df = df.select(col("markdown").cast(MarkdownType).alias("md_col"))

    # Test chunking at level 2 (should return empty array)
    chunks_df = df.select(markdown.extract_header_chunks(col("md_col"), header_level=2).alias("chunks"))
    result = chunks_df.to_polars()
    chunks = result["chunks"][0]

    # Should have no chunks since there are no level 2 headings
    assert len(chunks) == 0


def test_md_chunk_by_headings_complex_content(local_session):
    """Test chunking with rich inline content and deep nesting."""
    test_markdown = dedent(
        """
        # User Guide

        Welcome to our **comprehensive guide** with `inline code` and [links](https://example.com).

        ## Getting Started

        This section has *emphasized text*, **bold text**, and ~~strikethrough~~.

        ### Basic Setup

        1. Install with `pip install package`
        2. Configure using **environment variables**
        3. Run with the following:

        ```python
        import our_package
        our_package.run()
        ```

        #### Deep Nesting

        Even deeply nested content with [complex links](https://docs.example.com "Documentation") should be included.

        > **Note:** This is a blockquote with **bold** and *italic* text.
        >
        > It can span multiple paragraphs.

        ### Advanced Features

        - Feature one with `code`
        - Feature two with **emphasis**
        - Feature three with [link text](https://example.com)

        ## API Documentation

        The API supports various methods:

        | Method | Description | Example |
        |--------|-------------|---------|
        | `GET` | Retrieve data | `api.get()` |
        | `POST` | Send data | `api.post(data)` |

        ### Core Functions

        #### Function One

        This function includes:
        - Parameter `x`: an integer
        - Parameter `y`: a string with **special** formatting

        <div class="warning">
        HTML blocks should also be captured.
        </div>

        ## Troubleshooting

        Common issues and solutions.

        ### Debug Mode

        Enable debug with `DEBUG=true` in your `.env` file.

        ---

        That's all for the guide!
        """
    )

    df = local_session.create_dataframe({"markdown": [test_markdown]})
    df = df.select(col("markdown").cast(MarkdownType).alias("md_col"))

    # Test chunking at level 2
    chunks_df = df.select(markdown.extract_header_chunks(col("md_col"), header_level=2).alias("chunks"))
    result = chunks_df.to_polars()
    chunks = result["chunks"][0]

    # Should have 3 chunks: Getting Started, API Documentation, Troubleshooting
    assert len(chunks) == 3

    # Verify first chunk has rich content
    chunk1 = chunks[0]
    assert chunk1["heading"] == "Getting Started"
    assert chunk1["parent_heading"] == "User Guide"
    assert chunk1["full_path"] == "User Guide > Getting Started"

    # Check various inline elements are captured
    assert "emphasized text" in chunk1["content"]
    assert "bold text" in chunk1["content"]
    assert "strikethrough" in chunk1["content"]
    assert "pip install package" in chunk1["content"]
    assert "environment variables" in chunk1["content"]
    assert "import our_package" in chunk1["content"]  # From code block
    assert "complex links" in chunk1["content"]
    # Note: Link titles are attributes, not text content, so "Documentation" won't appear
    assert "blockquote" in chunk1["content"]
    assert "Feature one" in chunk1["content"]  # From list

    # Verify second chunk
    chunk2 = chunks[1]
    assert chunk2["heading"] == "API Documentation"
    assert chunk2["parent_heading"] == "User Guide"
    assert "supports various methods" in chunk2["content"]
    assert "GET" in chunk2["content"]  # From table
    assert "Retrieve data" in chunk2["content"]
    assert "api.get()" in chunk2["content"]
    assert "Function One" in chunk2["content"]  # H3 content
    assert "special" in chunk2["content"]  # Bold in nested section
    assert "HTML blocks should also be captured" in chunk2["content"]  # HTML block

    # Verify third chunk
    chunk3 = chunks[2]
    assert chunk3["heading"] == "Troubleshooting"
    assert "Debug Mode" in chunk3["content"]  # H3 heading text
    assert "DEBUG=true" in chunk3["content"]  # Inline code
    assert ".env" in chunk3["content"]
    assert "That's all for the guide" in chunk3["content"]  # Content after thematic break


def test_md_chunk_by_headings_breadcrumbs(local_session):
    """Test chunking with simple nested headers to verify breadcrumb paths."""
    test_markdown = dedent(
        """
        # Chapter 1

        Introduction content.

        ## Section A

        Section A content.

        ### Subsection A1

        Subsection A1 content.

        ### Subsection A2

        Subsection A2 content.

        ## Section B

        Section B content.

        ### Subsection B1

        Subsection B1 content.
        """
    )

    df = local_session.create_dataframe({"markdown": [test_markdown]})
    df = df.select(col("markdown").cast(MarkdownType).alias("md_col"))

    # Test chunking at level 3
    chunks_df = df.select(markdown.extract_header_chunks(col("md_col"), header_level=3).alias("chunks"))
    result = chunks_df.to_polars()
    chunks = result["chunks"][0]

    # Should have 3 chunks: Subsection A1, Subsection A2, Subsection B1
    assert len(chunks) == 3

    # Verify first chunk
    chunk1 = chunks[0]
    assert chunk1["heading"] == "Subsection A1"
    assert chunk1["level"] == 3
    assert chunk1["parent_heading"] == "Section A"
    assert chunk1["full_path"] == "Chapter 1 > Section A > Subsection A1"

    # Verify second chunk
    chunk2 = chunks[1]
    assert chunk2["heading"] == "Subsection A2"
    assert chunk2["level"] == 3
    assert chunk2["parent_heading"] == "Section A"
    assert chunk2["full_path"] == "Chapter 1 > Section A > Subsection A2"

    # Verify third chunk
    chunk3 = chunks[2]
    assert chunk3["heading"] == "Subsection B1"
    assert chunk3["level"] == 3
    assert chunk3["parent_heading"] == "Section B"
    assert chunk3["full_path"] == "Chapter 1 > Section B > Subsection B1"


def test_md_chunk_by_headings_no_h1(local_session):
    """Test chunking when document doesn't start with H1."""
    test_markdown = dedent(
        """
        ## Getting Started

        Content for getting started.

        ### Prerequisites

        System requirements.

        ## Configuration

        Setup instructions.
        """
    )

    df = local_session.create_dataframe({"markdown": [test_markdown]})
    df = df.select(col("markdown").cast(MarkdownType).alias("md_col"))

    # Test chunking at level 2
    chunks_df = df.select(markdown.extract_header_chunks(col("md_col"), header_level=2).alias("chunks"))
    result = chunks_df.to_polars()
    chunks = result["chunks"][0]

    # Should have 2 chunks: Getting Started, Configuration
    assert len(chunks) == 2

    # Verify chunks have no parent (since no H1)
    chunk1 = chunks[0]
    assert chunk1["heading"] == "Getting Started"
    assert chunk1["parent_heading"] is None
    assert chunk1["full_path"] == "Getting Started"
    assert "Prerequisites" in chunk1["content"]

    chunk2 = chunks[1]
    assert chunk2["heading"] == "Configuration"
    assert chunk2["parent_heading"] is None
    assert chunk2["full_path"] == "Configuration"
