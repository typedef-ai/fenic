# News Article Bias Detection

[View in Github](https://github.com/typedef-ai/fenic/blob/main/examples/news_analysis/README.md)

A comprehensive demonstration of fenic's semantic classification capabilities for detecting editorial bias and analyzing news articles. This example shows how to use semantic operations to identify bias patterns across multiple news sources and generate AI-powered media profiles.

## Overview

This pipeline performs sophisticated news analysis using fenic's semantic operations in a step-by-step educational format:

- **Language Analysis**: Uses `semantic.extract()` to identify biased, emotional, or sensationalist language patterns
- **Political Bias Classification**: Uses `semantic.classify()` grounded in extracted data for accurate bias detection
- **Topic Classification**: Categorizes articles by subject (politics, technology, business, climate, healthcare)
- **AI-Powered Source Profiling**: Uses `semantic.reduce()` to create comprehensive media profiles for each news source

Available in both **Python script** (`news_analysis.py`) and **Jupyter notebook** (`news_analysis.ipynb`) formats for different learning preferences.

## Key Features

### Two-Stage Analysis Pipeline

**Stage 1 - Information Extraction**: Uses `semantic.extract()` with Pydantic models to identify bias indicators, emotional language, and opinion markers from each article.

**Stage 2 - Grounded Classification**: Uses extracted information as context for `semantic.classify()` to achieve more accurate political bias detection.

### Multi-Dimensional Classification

Simultaneously classifies articles across:

- **Topics**: politics, technology, business, climate, healthcare
- **Political Bias**: far_left, left_leaning, neutral, right_leaning, far_right
- **Journalistic Style**: sensationalist vs informational

### Source Consistency Analysis

Analyzes bias patterns across multiple articles per source to identify editorial consistency and detect sources with mixed editorial perspectives.

### AI-Generated Media Profiles

Uses `semantic.reduce()` to synthesize extracted information into comprehensive, natural language profiles for each news source.

## Dataset

The example includes **25 news articles from 8 sources** covering diverse topics:

- **Politics**: Federal Reserve policy, climate agreements, Supreme Court cases
- **Technology**: AI developments, content moderation, privacy concerns
- **Business**: Corporate earnings, market analysis, economic trends
- **Healthcare**: Medical breakthroughs, drug pricing, treatment access

### Source Types

- **Neutral Sources**: Global Wire Service, National Press Bureau (3 articles each)
- **Left-leaning Sources**: Progressive Voice, Social Justice Today (3 articles each)
- **Right-leaning Sources**: Liberty Herald, Free Market Weekly (3 articles each)
- **Mixed-bias Sources**: Balanced Tribune (4 articles), Independent Monitor (3 articles)

The mixed-bias sources provide realistic examples of sources with inconsistent editorial patterns, demonstrating how fenic handles nuanced content classification across different bias consistency levels.

## Technical Implementation

### Two-Stage Pipeline Implementation

**Stage 1 - Information Extraction:**

```python
# Extract bias indicators and language patterns
enriched_df = df.select(
    # Topic classification
    fc.semantic.classify(
        fc.col("combined_content"),
        ["politics", "technology", "business", "climate", "healthcare"]
    ).alias("primary_topic"),
    # Extract structured information using Pydantic
    fc.semantic.extract(
        fc.col("combined_content"),
        ArticleAnalysis  # bias_indicators, emotional_language, opinion_markers
    ).alias("analysis_metadata")
).unnest("analysis_metadata")
```

**Stage 2 - Grounded Classification:**

```python
# Combine extracted information for context-aware classification
combined_extracts = fc.text.jinja(
    (
        "Primary Topic: {{primary_topic}}\n"
        "Political Bias Indicators: {{bias_indicators}}\n"
        "Emotional Language Summary: {{emotional_language}}\n"
        "Opinion Markers: {{opinion_markers}}"
    ),
    primary_topic=fc.col("primary_topic"),
    bias_indicators=fc.col("bias_indicators"),
    emotional_language=fc.col("emotional_language"),
    opinion_markers=fc.col("opinion_markers")
)

# Classify bias using extracted context
results_df = enriched_df.select(
    "*",
    fc.semantic.classify(
        col("combined_extracts"),
        ["far_left", "left_leaning", "neutral", "right_leaning", "far_right"]
    ).alias("content_bias"),
    fc.semantic.classify(
        col("combined_extracts"),
        ["sensationalist", "informational"]
    ).alias("journalistic_style")
)
```

### AI-Powered Source Profiling

```python
# Generate comprehensive source profiles using semantic.reduce
source_profiles = results_df.group_by("source").agg(
    fc.semantic.reduce(
        """
        You are given a set of article analyses from {{news_outlet}}.
        Create a concise (3-5 sentence) media profile for {{news_outlet}}.
        Summarize the information provided without explicitly referencing it.
        """,
        column=fc.col("article_attributes"),
        group_context = {
            "news_outlet": fc.col("source"),
        },
    ).alias("source_profile"),
).select(fc.col("source"), fc.col("source_profile"))
```

## Output Analysis

The pipeline generates comprehensive analysis including:

- **Multi-dimensional classifications** across topic and political bias spectrum
- **Language pattern analysis** extracting bias indicators, emotional language, and opinion markers
- **AI-generated source profiles** using semantic.reduce to summarize editorial characteristics

## Use Cases

### Media Organizations

- **Content quality assessment** for editorial guidelines
- **Bias detection** in reporter training and content review
- **Audience analytics** understanding reader preferences

### News Aggregators

- **Content categorization** for personalized feeds
- **Bias warnings** for balanced information consumption
- **Source diversity** ensuring multiple perspectives

### Research Applications

- **Media bias studies** analyzing coverage patterns
- **Information quality research** measuring factual content
- **Comparative analysis** across different news sources

### Educational Tools

- **Media literacy training** identifying bias indicators
- **Critical thinking exercises** comparing article perspectives
- **Journalism education** understanding editorial techniques

## Running the Example

### Option 1: Python Script

1. **Setup**: Ensure you have fenic installed with Google Gemini API access
2. **Environment**: Set your `GOOGLE_API_KEY` environment variable.
   a. Alternatively, you can run the example with an OpenAI(`OPENAI_API_KEY`) model by uncommenting the provided additional model configurations.
   b. Using an Anthropic model requires installing fenic with the `anthropic` extra package, and setting the `ANTHROPIC_API_KEY` environment variable
3. **Execute**: Run `python news_analysis.py`

### Option 2: Jupyter Notebook

1. **Setup**: Same API requirements as above
2. **Launch**: Open `news_analysis.ipynb` in Jupyter
3. **Learn**: Step-by-step educational walkthrough with explanations

### Alternative Models

The script includes commented configurations for OpenAI and Anthropic models if you prefer different providers.

**Explore**: Modify the dataset or classification categories to test different scenarios

## Advanced Features Demonstrated

### Grounded Classification Pipeline

Shows how to improve classification accuracy by first extracting relevant information with `semantic.extract()`, then using that context for more informed `semantic.classify()` operations.

### Pydantic Integration

Demonstrates structured data extraction using type-safe Pydantic models with automatic field validation for consistent output formatting.

### Multi-Model Support

Includes configurations for Google Gemini (default), OpenAI, and Anthropic models, showing fenic's flexibility across different LLM providers.

### Semantic Reduction for Profiling

Uses `semantic.reduce()` to synthesize multiple data points into coherent natural language profiles, demonstrating AI-powered summarization capabilities.

### Educational Format

Available in both script and notebook formats, with the notebook providing step-by-step explanations ideal for learning semantic operations.

## Expected Results

### Generated Source Profile

> The Balanced Tribune presents a diverse range of topics, primarily focusing on business, technology, climate, and healthcare. It exhibits a right-leaning bias in its business and technology coverage, emphasizing themes like Wall Street stability and American free enterprise, while adopting a far-left perspective on climate issues, critiquing fossil fuel companies. The publication often employs sensationalist and informational journalistic styles, utilizing emotional language to evoke strong reactions, such as the impact of inflation and the urgency of climate action. Opinion markers frequently reflect a mix of support for innovation and criticism of regulatory frameworks, indicating a complex stance on various issues.
