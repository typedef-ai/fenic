#!/usr/bin/env python3
"""
structural_fidelity.py

Module for calculating structural F1 fidelity between PDF and Markdown documents.
Evaluates how well document structure (headings, lists, tables, etc.) is preserved.
"""

import difflib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF
from markdown_it import MarkdownIt


@dataclass
class StructuralElement:
    """Represents a structural element in a document."""
    element_type: str  # 'heading', 'paragraph', 'list', 'table', 'code_block', etc.
    level: int = 0  # For headings (1-6) or list nesting level
    content: str = ""  # Text content for comparison
    
    def __hash__(self):
        return hash((self.element_type, self.level, self.normalize_content(self.content)))
    
    def __eq__(self, other):
        if not isinstance(other, StructuralElement):
            return False
        return (self.element_type == other.element_type and 
                self.level == other.level and
                self.fuzzy_match(self.content, other.content))
    
    @staticmethod
    def normalize_content(text: str) -> str:
        """Normalize text for comparison."""
        text = re.sub(r'\s+', ' ', text.strip().lower())
        text = re.sub(r'[^\w\s]', '', text)
        return text[:100]  # Compare first 100 chars
    
    @staticmethod
    def fuzzy_match(text1: str, text2: str, threshold: float = 0.85) -> bool:
        """Check if two texts are similar enough."""
        norm1 = StructuralElement.normalize_content(text1)
        norm2 = StructuralElement.normalize_content(text2)
        if not norm1 or not norm2:
            return norm1 == norm2
        ratio = difflib.SequenceMatcher(None, norm1, norm2).ratio()
        return ratio >= threshold


def grade_structure_fidelity(pdf_path: str, md_path: str) -> Dict[str, Any]:
    """
    Grade structure fidelity between PDF and Markdown documents.
    """
    pdf_elements = _extract_pdf_structure(pdf_path)
    md_elements = _extract_markdown_structure(md_path)
    precision, recall, f1, detailed_metrics = _calculate_structural_f1(pdf_elements, md_elements)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "detailed_metrics": _format_structural_analysis(md_path, precision, recall, f1, detailed_metrics)
    }


def _format_structural_analysis(title: str, precision: float, recall: float, f1: float, 
                             detailed_metrics: Dict) -> str:
    """
    Format a structural analysis report as a string.
    
    Args:
        title: Document title
        precision: Structural precision score
        recall: Structural recall score
        f1: Structural F1 score
        detailed_metrics: Dictionary with detailed breakdown
        
    Returns:
        Formatted string containing the structural analysis report
    """
    lines = []
    lines.append(f"\n--- Structural Analysis for {title} ---")
    lines.append(f"Overall: Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    lines.append(f"Total Elements - PDF: {detailed_metrics['total_pdf_elements']}, "
                 f"MD: {detailed_metrics['total_md_elements']}, "
                 f"Matches: {detailed_metrics['total_matches']}")
    
    if detailed_metrics['per_type']:
        lines.append("\nPer-type metrics:")
        lines.append(f"{'Element Type':12s} | {'PDF':>4s} | {'MD':>4s} | {'Match':>5s} | {'Prec':>6s} | {'Rec':>6s} | {'F1':>6s}")
        lines.append("-" * 60)
        for elem_type, metrics in sorted(detailed_metrics['per_type'].items()):
            lines.append(f"{elem_type:12s} | {metrics['pdf_count']:4d} | {metrics['md_count']:4d} | "
                        f"{metrics['matches']:5d} | {metrics['precision']:6.3f} | "
                        f"{metrics['recall']:6.3f} | {metrics['f1']:6.3f}")
    
    return "\n".join(lines)


def _extract_pdf_structure(pdf_path: Path) -> List[StructuralElement]:
    """
    Extract structural elements from PDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of StructuralElement objects representing document structure
    """
    doc = fitz.open(pdf_path)
    elements = []
    
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        blocks.sort(key=lambda b: (round(b["bbox"][1], 1), round(b["bbox"][0], 1)))
        
        for block in blocks:
            if block["type"] == 0:  # Text block
                block_text = []
                fonts_sizes = []
                
                for line in block["lines"]:
                    for span in line["spans"]:
                        block_text.append(span["text"])
                        fonts_sizes.append((span.get("size", 0), span.get("flags", 0)))
                
                text = " ".join(block_text).strip()
                if not text:
                    continue
                
                # Determine element type based on font characteristics
                if fonts_sizes:
                    avg_size = sum(fs[0] for fs in fonts_sizes) / len(fonts_sizes)
                    is_bold = any(fs[1] & 2**4 for fs in fonts_sizes)  # Bold flag
                    
                    # Heuristics for structure detection
                    if avg_size > 16 or (avg_size > 14 and is_bold):
                        # Likely a heading
                        level = 1 if avg_size > 20 else 2 if avg_size > 16 else 3
                        elements.append(StructuralElement("heading", level, text))
                    elif re.match(r'^[\•\-\*]\s+', text) or re.match(r'^\d+[\.\)]\s+', text):
                        # List item
                        elements.append(StructuralElement("list", 1, text))
                    elif len(text.split()) < 10 and avg_size > 12:
                        # Short text with larger font might be heading
                        elements.append(StructuralElement("heading", 4, text))
                    else:
                        # Regular paragraph
                        elements.append(StructuralElement("paragraph", 0, text))
            
            elif block["type"] == 1:  # Image block
                elements.append(StructuralElement("image", 0, "image"))
    
    # Detect tables by looking for grid-like patterns
    for page in doc:
        tables = page.find_tables()
        for table in tables:
            if table:
                # Extract table content for comparison
                try:
                    table_data = table.extract()
                    table_text = " ".join(" ".join(str(cell) for cell in row if cell) for row in table_data)
                    elements.append(StructuralElement("table", 0, table_text))
                except Exception as e:
                    print(f"Error extracting table: {e}")
                    elements.append(StructuralElement("table", 0, "table"))
    
    doc.close()
    return elements


def _extract_markdown_structure(md_path: str) -> List[StructuralElement]:
    """
    Extract structural elements from markdown.
    
    Args:
        md_path: Path to the markdown file
        
    Returns:
        List of StructuralElement objects representing document structure
    """
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    elements = []
    md = MarkdownIt()
    tokens = md.parse(content)
    
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        if token.type == "heading_open":
            level = int(token.tag[1])  # h1 -> 1, h2 -> 2, etc.
            # Get the content from the inline token
            if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                content = tokens[i + 1].content
                elements.append(StructuralElement("heading", level, content))
            i += 3  # Skip heading_open, inline, heading_close
            
        elif token.type == "bullet_list_open" or token.type == "ordered_list_open":
            # Collect all list items
            list_items = []
            i += 1
            while i < len(tokens) and tokens[i].type != token.type.replace("_open", "_close"):
                if tokens[i].type == "list_item_open":
                    i += 1
                    if i < len(tokens) and tokens[i].type == "paragraph_open":
                        i += 1
                        if i < len(tokens) and tokens[i].type == "inline":
                            list_items.append(tokens[i].content)
                i += 1
            
            if list_items:
                content = " ".join(list_items)
                elements.append(StructuralElement("list", 1, content))
                
        elif token.type == "table_open":
            # Collect table content
            table_text = []
            i += 1
            while i < len(tokens) and tokens[i].type != "table_close":
                if tokens[i].type == "inline":
                    table_text.append(tokens[i].content)
                i += 1
            
            if table_text:
                content = " ".join(table_text)
                elements.append(StructuralElement("table", 0, content))
                
        elif token.type == "code_block":
            elements.append(StructuralElement("code_block", 0, token.content))
            i += 1
            
        elif token.type == "fence":
            elements.append(StructuralElement("code_block", 0, token.content))
            i += 1
            
        elif token.type == "blockquote_open":
            # Collect blockquote content
            quote_text = []
            i += 1
            while i < len(tokens) and tokens[i].type != "blockquote_close":
                if tokens[i].type == "inline":
                    quote_text.append(tokens[i].content)
                i += 1
            
            if quote_text:
                content = " ".join(quote_text)
                elements.append(StructuralElement("blockquote", 0, content))
                
        elif token.type == "paragraph_open":
            # Get paragraph content
            if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                content = tokens[i + 1].content
                if content.strip():
                    elements.append(StructuralElement("paragraph", 0, content))
            i += 3  # Skip paragraph_open, inline, paragraph_close

        # Skip images    
        #elif token.type == "image":
        #    elements.append(StructuralElement("image", 0, token.content or "image"))
        #    i += 1
            
        else:
            i += 1
    
    return elements



def _calculate_structural_f1(pdf_elements: List[StructuralElement], 
                           md_elements: List[StructuralElement]) -> Tuple[float, float, float, Dict]:
    """
    Calculate F1 score for structural fidelity.
    
    Args:
        pdf_elements: List of structural elements extracted from PDF
        md_elements: List of structural elements extracted from Markdown
    
    Returns:
        Tuple of (precision, recall, f1_score, detailed_metrics)
        - precision: Fraction of markdown elements that match PDF
        - recall: Fraction of PDF elements found in markdown
        - f1_score: Harmonic mean of precision and recall
        - detailed_metrics: Dictionary with per-type breakdowns
    """
    # Create sets for comparison (using fuzzy matching in __eq__)
    pdf_set = set(pdf_elements)
    md_set = set(md_elements)
    
    # Calculate matches
    matches = pdf_set & md_set
    
    # Calculate metrics
    precision = len(matches) / len(md_set) if md_set else 0.0
    recall = len(matches) / len(pdf_set) if pdf_set else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate per-type metrics
    pdf_types = Counter(e.element_type for e in pdf_elements)
    md_types = Counter(e.element_type for e in md_elements)
    match_types = Counter(e.element_type for e in matches)
    
    per_type_metrics = {}
    for elem_type in set(pdf_types.keys()) | set(md_types.keys()):
        type_precision = match_types[elem_type] / md_types[elem_type] if md_types[elem_type] > 0 else 0.0
        type_recall = match_types[elem_type] / pdf_types[elem_type] if pdf_types[elem_type] > 0 else 0.0
        type_f1 = 2 * (type_precision * type_recall) / (type_precision + type_recall) if (type_precision + type_recall) > 0 else 0.0
        
        per_type_metrics[elem_type] = {
            'precision': type_precision,
            'recall': type_recall,
            'f1': type_f1,
            'pdf_count': pdf_types[elem_type],
            'md_count': md_types[elem_type],
            'matches': match_types[elem_type]
        }
    
    detailed_metrics = {
        'total_pdf_elements': len(pdf_elements),
        'total_md_elements': len(md_elements),
        'total_matches': len(matches),
        'per_type': per_type_metrics
    }
    
    return precision, recall, f1, detailed_metrics


