# PDF Parsing Evaluation Harness

An automated test harness for evaluating the quality of PDF-to-Markdown parsing across multiple LLM models using Fenic's semantic parsing capabilities.

## Overview

This tool orchestrates end-to-end evaluation of PDF parsing by:

1. Processing PDFs with specified LLM models using Fenic's `semantic.parse_pdf()`
2. Evaluating parsing quality against the original PDFs
3. Persisting results and metrics to a Fenic database
4. Generating comparative reports across models

## Quickstart

### Prerequisites

- Fenic installed with `eval` extra: `uv sync --extra "eval" --extra "google"`
- If evaluating Google models, include the google extra
- LLM API keys configured as environment variables (e.g., `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `OPENROUTER_API_KEY`)
- PDF files to evaluate

### Basic Usage

```bash
python eval_models.py \
  --app-name "my_pdf_evaluation" \
  --models "gemini-2.0-flash" "gpt-4o-mini" "openrouter/meta-llama/llama-3.3-70b-instruct" \
  --input-pdfs "/path/to/pdfs/*.pdf" \
  --output-path "./evaluation_results" \
  --test-id "experiment_2025_01_15"
```

### Example Output

```bash
Starting PDF Evaluation Test Harness
Test ID: experiment_2025_01_15
Models to evaluate: gemini-2.0-flash, gpt-4o-mini, openrouter/meta-llama/llama-3.3-70b-instruct
PDF glob: /path/to/pdfs/*.pdf
Output path: ./evaluation_results

######################################################################
# Evaluating Model: gemini-2.0-flash
# Test ID: experiment_2025_01_15
######################################################################

Processing PDFs with model gemini-2.0-flash...
Found 12 PDF files
Parsing PDF content using model: gemini-2.0-flash
Parsed 12 PDFs successfully
✓ PDF processing completed for gemini-2.0-flash

Grading markdown for test experiment_2025_01_15...
✓ Grading completed for experiment_2025_01_15

... [similar output for other models] ...

======================================================================
PROCESSING SUMMARY
======================================================================
✓ gemini-2.0-flash: PDF=✓ Grade=✓
✓ gpt-4o-mini: PDF=✓ Grade=✓
✓ openrouter/meta-llama/llama-3.3-70b-instruct: PDF=✓ Grade=✓

======================================================================
EVALUATION RESULTS FOR TEST_ID: experiment_2025_01_15
======================================================================
┌──────────────────────────────────────────────────┬───────────────┬─────────────┬────────────┬──────────┬─────────┬─────────────────────┬────────────────────┬───────────────┬─────────────┐
│ model_name                                       │ overall_score │ structure_f1│ text_score │ lm_cost  │ time_ms │ uncached_input_tokens│ cached_input_tokens│ output_tokens │ total_pages │
├──────────────────────────────────────────────────┼───────────────┼─────────────┼────────────┼──────────┼─────────┼─────────────────────┼────────────────────┼───────────────┼─────────────┤
│ gemini-2.0-flash                                 │ 0.924         │ 0.891       │ 0.957      │ 0.0234   │ 45230   │ 234567              │ 0                  │ 123456        │ 156         │
│ gpt-4o-mini                                      │ 0.887         │ 0.843       │ 0.931      │ 0.0456   │ 52341   │ 245678              │ 12345              │ 134567        │ 156         │
│ openrouter/meta-llama/llama-3.3-70b-instruct     │ 0.856         │ 0.812       │ 0.901      │ 0.0123   │ 38912   │ 223456              │ 0                  │ 112345        │ 156         │
└──────────────────────────────────────────────────┴───────────────┴─────────────┴────────────┴──────────┴─────────┴─────────────────────┴────────────────────┴───────────────┴─────────────┘

Processing summary saved to: ./evaluation_results/experiment_2025_01_15_summary.json
```

## Features

### Automated Evaluation Pipeline

- **Multi-Model Support**: Evaluate multiple LLM models in a single run (OpenAI, Google, Anthropic, OpenRouter)
- **Batch Processing**: Process multiple PDFs efficiently using Fenic's batch inference
- **Persistent Results**: All evaluation data is automatically persisted to a Fenic database, keyed by test ID and model name
- **Preserved Outputs**: Parsed Markdown documents are saved to disk and not deleted after evaluation

### Database Persistence

The harness stores results in multiple Fenic tables:

- **`eval_results`**: Contains evaluation scores (overall_score, structure_f1, text_score, page counts)
- **`eval_execution_ids`**: Maps test IDs and models to Fenic execution IDs
- **`parsed_output_test_{test_id}_{model}`**: Stores parsed Markdown content for each model/test combination
- **`fenic_system.query_metrics`**: System table with detailed execution metrics (costs, tokens, timing)

Results are automatically joined across these tables to provide comprehensive evaluation reports including:

- Quality scores (text fidelity, structural fidelity, overall)
- Cost metrics (total LM cost, token usage)
- Performance metrics (execution time, cache hits)
- Document metadata (page counts)

### Provider models with direct VLM support

```bash
# Built-in models (configured in pdf_processor.py)
--models "gemini-2.0-flash" "gpt-4o-mini"
```

#### Supported models include

Google

-

## OpenAI

### OpenRouter

The syntax for using OpenRouter is a little different. The model names include the model's underlying original provider.
To use a model through OpenRouter, provide the model name as it appears in the OpenRouter platform. The tool will recognize
the "f{provider}/{model}" syntax and use OpenRouter. You can specify any model in OpenRouter.

```bash
--models "meta-llama/llama-3.3-70b-instruct"
```

#### OpenRouter with custom parsing engine

OpenRouter lets you use any model for analyzing file content by first using one of its tools ('parsing engine')
to parse the file contents. If no engine is specified, it will default to 'native' and try to use the model's
own parsing capabilities.

Specify the parsing engine by adding it to the end of the model name. For e.g., to use '/mistral-ocr'

```bash
--models "meta-llama/llama-3.3-70b-instruct/mistral-ocr"
```

Available parsing engines for OpenRouter models:

- `native`: Fenic's native PDF parsing (default)
- `mistral-ocr`: Mistral's OCR-based parsing
- `pdf-text`: Simple text extraction

## Evaluation Criteria

The harness computes a comprehensive quality score based on two primary metrics:

### 1. Text Fidelity Score

Measures how accurately the parsed Markdown captures the raw text content of the PDF.

- **Algorithm**: Levenshtein distance-based fuzzy matching using Fenic's `text.compute_fuzzy_ratio()` function
- **Process**:
  - Extracts raw text from PDF (reading order normalized by bounding box coordinates)
  - Extracts plain text content from Markdown (stripping formatting)
  - Computes similarity ratio between 0.0 (no match) and 1.0 (perfect match)
- **What it captures**: Completeness and accuracy of text extraction, including handling of special characters, formatting, and reading order

### 2. Document Structure Fidelity Score

Evaluates how well the document structure is preserved using F1 score across structural element types.

- **Structural Elements Analyzed**:
  - Headings (with hierarchy levels 1-6)
  - Paragraphs
  - Lists (bulleted and numbered)
  - Tables
  - Code blocks
  - Images
  - Blockquotes

- **Metrics Computed**:
  - **Precision**: Fraction of Markdown structural elements that correctly match PDF elements
    - `precision = matched_elements / total_markdown_elements`
  - **Recall**: Fraction of PDF structural elements successfully captured in Markdown
    - `recall = matched_elements / total_pdf_elements`
  - **F1 Score**: Harmonic mean of precision and recall
    - `f1 = 2 * (precision * recall) / (precision + recall)`

- **Matching Algorithm**:
  - Fuzzy content matching (85% similarity threshold using sequence matcher)
  - Type and level matching (e.g., H1 in PDF must match H1 in Markdown)
  - Normalized comparison (case-insensitive, whitespace-normalized, truncated to first 100 chars)

- **Per-Type Breakdown**: F1 scores are computed individually for each element type (headings, tables, lists, etc.) to identify specific parsing strengths and weaknesses

### 3. Overall Score

The final score is a weighted combination of the two metrics, normalized by document length:

```python
overall_score = (structure_f1 + text_score) / 2

# Page-weighted aggregation across documents:
weighted_metric = sum(metric_score * page_count) / sum(page_count)
```

All scores are weighted by the number of pages in each document to ensure that larger, more complex documents have proportional influence on the final evaluation.

### Score Interpretation

- **0.90 - 1.00**: Excellent parsing quality
- **0.80 - 0.89**: Good parsing with minor issues
- **0.70 - 0.79**: Acceptable parsing with noticeable gaps
- **Below 0.70**: Poor parsing quality requiring investigation

## Command-Line Options

### `eval_models.py` (Main Harness)

```bash
Required Arguments:
  -a, --app-name        Application name for Fenic session
  -m, --models          List of model names to evaluate (space-separated)
  -i, --input-pdfs      Glob pattern for input PDF files
  -o, --output-path     Base directory for output files
  -t, --test-id         Unique test identifier

Optional Arguments:
  -v, --verbose         Enable verbose output with stack traces
```

### `pdf_processor.py` (Standalone Processing)

```bash
Required Arguments:
  -i, --input-glob-pattern   Glob pattern for input PDF files
  -a, --app-name            Application name for Fenic session
  -m, --model-name          Model name for parsing
  -t, --test-id             Test identifier
  -o, --output-dir          Output directory for Markdown files

Optional Arguments:
  -e, --parse-engine        Parsing engine (mistral-ocr, pdf-text, native)
```

### `grade_md.py` (Standalone Grading)

```bash
Required Arguments:
  --pdf_dir             Directory containing PDF files
  --md_dir              Directory containing Markdown files
  --test_id             Test identifier
  -a, --app_name        Application name
  -m, --model_name      Model name for results
```

## Architecture

### Components

1. **`eval_models.py`**: Main orchestration harness (`PDFEvalTestHarness` class)
   - Coordinates processing and grading for multiple models
   - Manages Fenic session lifecycle
   - Aggregates and displays results

2. **`pdf_processor.py`**: PDF parsing module
   - Uses Fenic's `semantic.parse_pdf()` for LLM-based parsing
   - Persists parsed content to Fenic tables
   - Writes individual Markdown files

3. **`grade_md.py`**: Evaluation module
   - Orchestrates text and structure fidelity grading
   - Computes weighted scores across documents
   - Saves results to `eval_results` table

4. **`grade_md_structure.py`**: Structure analysis module
   - Extracts structural elements from PDFs (using PyMuPDF)
   - Parses Markdown structure (using markdown-it)
   - Calculates precision, recall, and F1 scores

### Data Flow

```bash
Input PDFs
    ↓
[pdf_processor.py]
    ↓ semantic.parse_pdf()
Markdown Files + Fenic Table (parsed_output_test_{test_id}_{model})
    ↓
[grade_md.py]
    ↓ UDFs + fuzzy matching
Scores → eval_results table
    ↓ JOIN with fenic_system.query_metrics
[eval_models.py: retrieve_results_from_fenic()]
    ↓
Display Results + Summary JSON
```

### Output Structure

```bash
{output_path}/
├── {test_id}/
│   ├── {provider}__{model}/
│   │   ├── document1.md
│   │   ├── document2.md
│   │   └── ...
│   └── {another_model}/
│       └── ...
└── {test_id}_summary.json
```

## Advanced Usage

### Comparing Parsing Engines

```bash
# Compare different parsing engines for the same model
python eval_models.py \
  --app-name "engine_comparison" \
  --models \
    "openrouter/meta-llama/llama-3.3-70b-instruct/native" \
    "openrouter/meta-llama/llama-3.3-70b-instruct/mistral-ocr" \
    "openrouter/meta-llama/llama-3.3-70b-instruct/pdf-text" \
  --input-pdfs "test_docs/*.pdf" \
  --output-path "./engine_tests" \
  --test-id "engine_eval_v1"
```

### Running Individual Components

```bash
# 1. Process PDFs only
python pdf_processor.py \
  -i "docs/*.pdf" \
  -a "my_app" \
  -m "gemini-2.0-flash" \
  -t "test_001" \
  -o "./outputs"

# 2. Grade existing outputs
python grade_md.py \
  --pdf_dir "docs/" \
  --md_dir "./outputs/test_001/gemini-2.0-flash" \
  --test_id "test_001" \
  -a "my_app" \
  -m "gemini-2.0-flash"
```

### Querying Results from Fenic

Results persist in the Fenic database and can be queried programmatically:

```python
import fenic as fc

session = fc.Session.get_or_create(fc.SessionConfig(app_name="my_app"))

# Query all results for a test
results = session.table("eval_results").filter(
    fc.col("test_id") == "experiment_2025_01_15"
)
results.show()

# Compare models
comparison = session.table("eval_results").select(
    "model_name",
    "overall_score",
    "structure_f1",
    "text_score"
).order_by(fc.col("overall_score").desc())
comparison.show()

# Access parsed content
parsed = session.table("parsed_output_test_experiment_2025_01_15_gemini-2.0-flash")
parsed.show()
```

## Troubleshooting

### No PDFs Found

- Verify the glob pattern matches your file locations
- Use absolute paths or ensure relative paths are correct
- Check file permissions

### API Rate Limits

- Adjust model configurations in `pdf_processor.py` (`rpm`, `tpm` parameters)
- Reduce the number of PDFs in a single run
- Use models with higher rate limits

### Missing Evaluation Results

- Ensure both processing and grading completed successfully (check for ✓ symbols)
- Verify test_id matches between processing and grading steps
- Check the Fenic database: `session.catalog.list_tables()`

### Memory Issues with Large PDFs

- Process PDFs in smaller batches
- Increase available system memory
- Use lighter-weight models for initial testing

## Dependencies

- `fenic`: Core framework
- `PyMuPDF` (fitz): PDF text and structure extraction
- `markdown-it-py`: Markdown parsing
- `polars`: DataFrames and display formatting

## License

This tool is part of the Fenic project.
