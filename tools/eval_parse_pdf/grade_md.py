#!/usr/bin/env python3
"""
pdf_to_markdown_grader.py

Grades Markdown translations against original PDFs for text fidelity.
Matches PDFs to Markdown files by the PDF metadata 'title' field.
"""

from pathlib import Path

import fitz  # PyMuPDF
from grade_md_structure import grade_structure_fidelity
from markdown_it import MarkdownIt

import fenic as fc


def _extract_pdf_text(pdf_path: Path) -> str:
    """Extract raw text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text_parts = []

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        blocks.sort(key=lambda b: (round(b["bbox"][1], 1), round(b["bbox"][0], 1)))
        for b in blocks:
            if b["type"] == 0:  # text blocks only
                for line in b["lines"]:
                    line_text = " ".join(span["text"] for span in line["spans"])
                    text_parts.append(line_text.strip())
    return " ".join(text_parts)

def _extract_markdown_text(md_path: str) -> str:
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    md = MarkdownIt()
    tokens = md.parse(content)

    # Flatten all inline text content
    texts = []
    for t in tokens:
        if t.type == "inline":
            texts.extend([c.content for c in t.children if c.type == "text"])

    return " ".join(texts)

def grade_pdfs(pdf_dir: str, md_dir: str, test_id: str, app_name: str, model_name: str, session=None):
    """
    Grade a directory of PDFs against Markdown files using the PDF metadata title.

    Returns:
        overall_score (float), results (list of tuples: (title, score, page_count))
    """
    print(f"{'#'*70}")
    print(f"Grading PDFs with model {model_name} for test {test_id}")
    print(f"PDF directory: {pdf_dir}")
    print(f"Markdown directory: {md_dir}")
    print(f"App name: {app_name}")
    print(f"{'#'*70}")
    close_session = True
    if session is None:
        session = fc.Session.get_or_create(fc.SessionConfig(app_name=app_name))
    else:
        close_session = False


    pdf_raw_df = session.read.pdf_metadata(pdf_dir, recursive=True)
    md_raw_df = session.read.docs(md_dir, content_type="markdown", recursive=True)

    # Extract the filenames from md_df and pdf_df
    md_df = md_raw_df.select(
        fc.col("file_path").alias("md_file_path"),
        fc.col("content").alias("md_text")
    ).with_column(
        "title", 
        fc.text.split_part(fc.col("md_file_path"), "/", -1)
    ).with_column(
        "title", fc.text.regexp_replace(fc.col("title"), r"\.md$", "")
    )
    pdf_df = pdf_raw_df.select(
        fc.col("file_path").alias("pdf_file_path"),
        "page_count",
        "title",
    )

    # Join the md_df and pdf_df on the filebase
    grade_md_paths_df = md_df.join(pdf_df, on="title", how="inner")

    # UDFs for extracting raw text and for grading structural fidelity
    udf_extract_raw_md_text = fc.udf(
        _extract_markdown_text,
        return_type=fc.StringType
    )
    udf_extract_raw_pdf_text = fc.udf(
        _extract_pdf_text,
        return_type=fc.StringType
    )
    udf_grade_file_structure = fc.udf(
        grade_structure_fidelity,
        return_type=fc.StructType([
            fc.StructField("precision", fc.DoubleType),
            fc.StructField("recall", fc.DoubleType),  
            fc.StructField("f1", fc.DoubleType),  
            fc.StructField("detailed_metrics", fc.StringType),  #
        ])
    )
    grade_md_paths_df = grade_md_paths_df.with_column(
        "md_text", udf_extract_raw_md_text(fc.col("md_file_path"))
    ).with_column(
        "pdf_text", udf_extract_raw_pdf_text(fc.col("pdf_file_path")),
    ).with_column(
        "structure_fidelity_score_tuple", udf_grade_file_structure(fc.col("pdf_file_path"), fc.col("md_file_path")),
    ).with_column(
        "text_fidelity_score", fc.text.compute_fuzzy_ratio(fc.col("pdf_text"), fc.col("md_text"), method="levenshtein"),
    ).unnest("structure_fidelity_score_tuple").cache()

    grade_md_total_df = grade_md_paths_df.agg(
        fc.sum("page_count").alias("total_pages"),
        fc.sum(fc.col("precision") * fc.col("page_count")).alias("sum_weighted_precision"),
        fc.sum(fc.col("recall") * fc.col("page_count")).alias("sum_weighted_recall"),
        fc.sum(fc.col("f1") * fc.col("page_count")).alias("sum_weighted_f1"),
        fc.sum(fc.col("text_fidelity_score") * fc.col("page_count")).alias("sum_weighted_text_score"),
    ).select(
        "total_pages",
        (fc.col("sum_weighted_precision") / fc.col("total_pages")).alias("structure_precision"),
        (fc.col("sum_weighted_recall") / fc.col("total_pages")).alias("structure_recall"),
        (fc.col("sum_weighted_f1") / fc.col("total_pages")).alias("structure_f1"),
        (fc.col("sum_weighted_text_score") / fc.col("total_pages")).alias("text_score"),
    ).with_column(
        "overall_score", ((fc.col("structure_f1") + fc.col("text_score")) / 2.0)
    ).select(
        '*',
        fc.lit(test_id).alias("test_id"),
        fc.lit(md_dir).alias("md_dir"),
        fc.lit(pdf_dir).alias("pdf_dir"),
        fc.lit(model_name).alias("model_name"),
    ).cache()

    grade_md_total_df.write.save_as_table("eval_results", mode="append")
    grade_md_total_df.drop("pdf_dir", "md_dir", "total_pages").show()

    if close_session:
        session.stop()

 
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grade PDF→Markdown fidelity (text only).")
    parser.add_argument("--pdf_dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--md_dir", required=True, help="Directory containing Markdown files")
    parser.add_argument("--test_id", required=True, help="Test identifier string")
    parser.add_argument("--app_name", "-a", required=True, help="App name")
    parser.add_argument("--model_name", "-m", required=True, help="Model name") # For entering results into correct row

    args = parser.parse_args()
    grade_pdfs(pdf_dir=args.pdf_dir, md_dir=args.md_dir, test_id=args.test_id, app_name=args.app_name, model_name=args.model_name)
