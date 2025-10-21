from __future__ import annotations

import glob
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import fitz  # PyMuPDF
import polars as pl

from fenic._backends.local.utils.io_utils import PathScheme, get_path_scheme
from fenic.core._utils.schema import convert_custom_schema_to_polars_schema
from fenic.core.error import FileLoaderError, ValidationError
from fenic.core.types import ColumnField, Schema
from fenic.core.types.datatypes import (
    BooleanType,
    IntegerType,
    JsonType,
    MarkdownType,
    StringType,
)
from fenic.core.types.enums import DocContentType

logger = logging.getLogger(__name__)


def validate_pages_argument(pages: Optional[Union[int, List[Union[int, List[int]]]]]) -> None:
    """Validate the pages argument.

    Args:
        pages: Either an int, or a list of ints or pairs of ints (ranges), or None

    Raises:
        ValidationError: If the pages argument is invalid
    """
    if pages is None:
        return
    if isinstance(pages, int):
        if pages <= 0:
            raise ValidationError("Page numbers must be positive integers")
    elif isinstance(pages, list):
        for item in pages:
            if isinstance(item, int):
                if item <= 0:
                    raise ValidationError("Page numbers must be positive integers")
            elif isinstance(item, list):
                if len(item) != 2:
                    raise ValidationError("Page ranges must be pairs of two numbers")
                if not all(isinstance(x, int) for x in item):
                    raise ValidationError("Page range values must be integers")
                if item[0] <= 0 or item[1] <= 0:
                    raise ValidationError("Page numbers must be positive integers")
                if item[1] < item[0]:
                    raise ValidationError(f"Invalid page range [{item[0]}, {item[1]}]: end page must be >= start page")
            else:
                raise ValidationError(f"Invalid pages element type: {type(item).__name__}. Expected int or list of two ints")
    else:
        raise ValidationError(f"Invalid pages type: {type(pages).__name__}. Expected int, list, or Column")


def resolve_and_coalesce_pages(pages: Union[int, List[Union[int, List[int]]]], total_pages: int) -> List[Tuple[int, int]]:
    """Resolve and coalesce page specifications into sorted, non-overlapping ranges.

    Converts page numbers and ranges into a sorted list of non-overlapping page ranges.
    All page numbers are 1-indexed as input but converted to 0-indexed ranges for internal use.

    Args:
        pages: Either a single page number (int) or a list of page numbers and/or ranges.
               Page numbers are 1-indexed. Ranges are represented as [start, end] (inclusive).

    Returns:
        List of (start, end) tuples representing 0-indexed page ranges (inclusive).
        Ranges are sorted and non-overlapping.

    Examples:
        >>> resolve_and_coalesce_pages(5)
        [(4, 4)]
        >>> resolve_and_coalesce_pages([1, 3, 5])
        [(0, 0), (2, 2), (4, 4)]
        >>> resolve_and_coalesce_pages([1, [2, 4], 3, 5])
        [(0, 0), (1, 3), (4, 4)]
        >>> resolve_and_coalesce_pages([[1, 3], [2, 5], 7])
        [(0, 4), (6, 6)]
    """
    # Convert to list of (start, end) tuples (0-indexed, inclusive)
    ranges = []
    if isinstance(pages, int):
        # Single page: convert 1-indexed to 0-indexed
        ranges.append((pages - 1, min(pages - 1, total_pages - 1)))
    else:
        for item in pages:
            # Single page: convert 1-indexed to 0-indexed
            # every range is capped by the total number of pages in the document
            ranges.append((item - 1, min(item - 1, total_pages - 1)) if isinstance(item, int) else (item[0] - 1, min(item[1] - 1, total_pages - 1)))

    # Sort by start page
    ranges.sort()

    # Coalesce overlapping ranges
    if not ranges:
        return []
    coalesced = [ranges[0]]
    for start, end in ranges[1:]:
        last_start, last_end = coalesced[-1]
        # Check if ranges overlap or are adjacent
        if start <= last_end + 1:
            # Merge ranges
            coalesced[-1] = (last_start, max(last_end, end))
        else:
            # No overlap, add new range
            coalesced.append((start, end))
    return coalesced

class DocFolderLoader:
    """A class that encapsulates folder traversal and multi-threaded file processing.

    This class provides functionality to:
    1. Traverse folders with glob patterns and regex exclusions
    2. Process files in parallel
    3. Collect results
    4. Handle errors gracefully for individual files
    """

    @staticmethod
    def load_docs_from_folder(
            paths: list[str],
            content_type: DocContentType,
            exclude_pattern: Optional[str] = None,
            recursive: bool = False,
    ) -> pl.DataFrame:
        """Load documents from a folder.

        Args:
            paths: list of paths to the folders to load files from, the paths will be glob patterns.
            content_type: Content type of the files.
            exclude_pattern: A regex pattern to exclude files.
            recursive: Whether to recursively load files from the folder.

        Returns:
            DataFrame: A dataframe containing the files in the folder.

        Notes:
            - The exclude pattern is a regex pattern to exclude files.
            - The recursive flag indicates whether to recursively load files from the paths.
            - The dataframe will be the union of all the files in the folders, plus additional columns for file metadata.
        """
        if not paths:
            raise ValidationError("No paths provided")

        logger.debug(f"Attempting to load files from: {paths}")
        if content_type == "markdown":
            file_extension = "md"
        else:
            file_extension = content_type

        files = DocFolderLoader._enumerate_files(
            paths,
            file_extension,
            exclude_pattern,
            recursive)

        if not files:
            logger.debug(f"No files found in {paths}")
            return DocFolderLoader._build_no_files_dataframe(content_type=content_type)

        # Calculate the batch size to ensure that each worker gets at least one file.
        max_workers = os.cpu_count() + 4
        
        # Process files with the appropriate handler based on extension
        return DocFolderLoader._process_files(files, max_workers, content_type)

    @staticmethod
    def check_file_extensions(file_paths: List[str], valid_file_extension: Literal["md", "json", "pdf"]):
        """Check that the file extensions are valid."""
        _ = DocFolderLoader._enumerate_files(file_paths, valid_file_extension, exclude_pattern=None, recursive=False)


    @staticmethod
    def get_schema(content_type: str = None) -> Schema:
        """Get the schema for the data type.

        Args:
            file_extension: The file extension to determine schema

        Returns:
            Schema: The schema for the data type
        """
        column_fields = [
            ColumnField(name="file_path", data_type=StringType),
            ColumnField(name="error", data_type=StringType),
        ]
        if content_type == "pdf":
            column_fields.extend([
                # additional file metadata fields
                ColumnField(name="size", data_type=IntegerType),
                # PDF metadata fields
                ColumnField(name="title", data_type=StringType),
                ColumnField(name="author", data_type=StringType),
                ColumnField(name="creation_date", data_type=StringType),
                ColumnField(name="mod_date", data_type=StringType),
                ColumnField(name="page_count", data_type=IntegerType),
                ColumnField(name="has_forms", data_type=BooleanType),
                ColumnField(name="has_signature_fields", data_type=BooleanType),
                ColumnField(name="image_count", data_type=IntegerType),
                ColumnField(name="is_encrypted", data_type=BooleanType),
            ])
        else: # load file content directly
            if content_type == "markdown":
                dest_type = MarkdownType
            else:
                dest_type = JsonType
            column_fields.append(ColumnField(name="content", data_type=dest_type))
        return Schema(
            column_fields=column_fields
        )

    @staticmethod
    def validate_paths(
        paths: list[str],
    ):
        """Checks that the path is valid, and returns a list of files that match the include and exclude patterns."""
        if not get_path_scheme(paths[0]) == PathScheme.LOCALFS:
            raise NotImplementedError("S3 and HF paths are not supported yet.")

        # List the paths, this will raise an error if the path is not valid.
        DocFolderLoader._list_paths_local_fs(paths)

    @staticmethod
    def _list_paths_local_fs(
        paths: list[str],
    ) -> list[Path]:
        base_paths = []
        for path_pattern in paths:
            # Extract the base directory (everything before the first wildcard)
            base_path = Path(path_pattern.split("*")[0]).parent
            if not base_path.exists():
                raise ValidationError(f"path does not exist: {base_path}")
            base_paths.append(base_path)
        return base_paths

    @staticmethod
    def _enumerate_files(
        paths: list[str],
        valid_file_extension: str,
        exclude_pattern: Optional[str] = None,
        recursive: bool = False
    ) -> List[str]:
        r"""Enumerate files in a folder based on include and exclude patterns.

        Args:
            paths: paths to the folders to traverse, these will be glob patterns.
            exclude_pattern: Regex pattern to exclude files (e.g., r"\.tmp$", r"temp.*")
            recursive: Whether to traverse subdirectories recursively

        Returns:
            List of Path objects for matching files
        """

        path_scheme = get_path_scheme(paths[0])
        if path_scheme == PathScheme.S3:
            return DocFolderLoader._enumerate_files_s3(paths, valid_file_extension, exclude_pattern, recursive)
        elif path_scheme == PathScheme.HF:
            return DocFolderLoader._enumerate_files_hf(paths, valid_file_extension, exclude_pattern, recursive)
        else:
            return DocFolderLoader._enumerate_files_local_fs(paths, valid_file_extension, exclude_pattern, recursive)

    @staticmethod
    def _process_files(
        files: List[str],
        max_workers: int,
        content_type: str = None,
    ) -> pl.DataFrame:
        """Process files in parallel using a thread pool.

        Args:
            files: List of file paths to process
            max_workers: Number of worker threads
            file_extension: File extension to determine processing type

        Returns:
            DataFrame: A dataframe containing the files in the folder.
        """
        # Determine which processing function and schema to use

        schema = convert_custom_schema_to_polars_schema(DocFolderLoader.get_schema(content_type=content_type))
        if content_type == "pdf":
            process_func = DocFolderLoader._process_single_pdf_metadata
        else:
            process_func = DocFolderLoader._process_single_file
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            it = iter(files)
            pending = {executor.submit(process_func, f)
                       for _, f in zip(range(max_workers), it, strict=False)}

            def results_generator():
                while pending:
                    for future in as_completed(pending):
                        pending.remove(future)
                        yield future.result()
                        try:
                            pending.add(executor.submit(process_func, next(it)))
                        except StopIteration:
                            pass

            # Uses the iterator over the results to build the dataframe.
            return pl.DataFrame(results_generator(), schema=schema)

    @staticmethod
    def _process_single_file(
        file_path: str,
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """Process a single file.

        Args:
            file_path: The path to the file to process
            is_s3_path: Whether the path is an S3 path

        Returns:
            DataFrame: A polars dataframe containing the file content.
        """
        path_scheme = get_path_scheme(file_path)
        logger.debug(f"Processing file: {file_path} - {path_scheme}")
        file_content: Optional[str] = None
        string_error: Optional[str] = None
        try:
            if path_scheme == PathScheme.S3:
                file_content = DocFolderLoader._load_file_s3(file_path)
            elif path_scheme == PathScheme.HF:
                file_content = DocFolderLoader._load_file_hf(file_path)
            else:
                file_content = DocFolderLoader._load_file_local_fs(file_path)

            logger.debug(f"File loaded successfully: {file_path}")
        except Exception as e:
            logger.error(f"Error loading file: {file_path} - {e}")
            string_error = str(e)
        return file_path, string_error, file_content

    @staticmethod
    def _build_no_files_dataframe(content_type: str) -> pl.DataFrame:
        """Build an empty dataframe with the appropriate schema."""
        return pl.DataFrame({}, schema=convert_custom_schema_to_polars_schema(DocFolderLoader.get_schema(content_type=content_type)))

    @staticmethod
    def _enumerate_files_s3(
            paths: list[str],
            valid_file_extension: str,
            exclude_pattern: Optional[str],
            recursive: bool,
    ) -> List[str]:
        """Enumerate files in an S3 bucket based on include/exclude patterns."""
        raise NotImplementedError("S3 file enumeration is not implemented yet.")

    @staticmethod
    def _enumerate_files_hf(
            paths: list[str],
            valid_file_extension: str,
            exclude_pattern: Optional[str],
            recursive: bool,
    ) -> List[str]:
        """Enumerate files in a HuggingFace dataset based on include/exclude patterns."""
        raise NotImplementedError("HF file enumeration is not implemented yet.")

    @staticmethod
    def _enumerate_files_local_fs(
            paths: list[str],
            valid_file_extension: str,
            exclude_pattern: Optional[str],
            recursive: bool = False,
    ) -> List[str]:
        """Enumerate files in a local fs folder based on include/exclude patterns.

        Returns:
            List of file paths that match the include/exclude patterns.

        Raises:
            ValidationError: If the exclude pattern is invalid.
            FileLoaderError: If the file extension is not supported.
            FileNotFoundError: If the path does not exist.
        """
        all_files = []
        for path_pattern in paths:
            split_path_pattern = path_pattern.split("*")
            base_path = Path(split_path_pattern[0])
            if len(split_path_pattern) > 1:
                base_path = base_path.parent
            else:
                if base_path.is_file():
                    all_files.append(path_pattern)
                    continue
                else:
                    logger.debug("Path doesn't have a wildcard, adding a default one.")
                    path_pattern = str(Path.joinpath(base_path, "*" + valid_file_extension))

            if base_path.exists():
                matched_files = glob.glob(path_pattern, recursive=recursive)
                all_files.extend(matched_files)
            else:
                raise FileNotFoundError(f"Path does not exist: {base_path}")

        # Compile exclude regex if provided
        exclude_regex = None
        if exclude_pattern:
            try:
                exclude_regex = re.compile(exclude_pattern)
            except re.error as e:
                raise ValidationError(f"Invalid exclude pattern regex: '{exclude_pattern}'") from e

        # Filter out directories and apply exclude pattern
        matching_files = []
        for file_str in all_files:
            file_path = Path(file_str)
            if not file_path.is_file():
                continue
            if exclude_regex and exclude_regex.search(file_str):
                continue
            if not file_str.endswith(valid_file_extension):
                raise FileLoaderError(f"Only files with the extension '{valid_file_extension}' are supported in this plan.")
            matching_files.append(file_str)

        return matching_files

    @staticmethod
    def _load_file_local_fs(
            file_path: str,
    ) -> str:
        """Load a file from the local filesystem."""
        with open(file_path, 'r') as file:
            return file.read()

    @staticmethod
    def _load_file_s3(file_path: str) -> str:
        """Load a file from S3."""
        raise NotImplementedError("S3 file loading is not implemented yet.")

    @staticmethod
    def _load_file_hf(file_path: str) -> str:
        """Load a file from HuggingFace."""
        raise NotImplementedError("HF file loading is not implemented yet.")

    @staticmethod
    def _process_single_pdf_metadata(file_path: str) -> dict:
        """Process a single PDF file to extract metadata.

        Args:
            file_path: The path to the PDF file to process

        Returns:
            dict: A dictionary containing PDF metadata and error information.
        """
        
        path_scheme = get_path_scheme(file_path)
        logger.debug(f"Processing PDF: {file_path} - {path_scheme}")
        
        # Initialize the flat result dict with default values
        result = {
            "file_path": file_path,
            "error": None,
            "size": 0,
            "title": None,
            "author": None,
            "creation_date": None,
            "mod_date": None,
            "page_count": 0,
            "has_forms": False,
            "has_signature_fields": False,
            "image_count": 0,
            "is_encrypted": False,
        }
        
        try:
            if path_scheme == PathScheme.S3:
                raise NotImplementedError("S3 PDF processing not implemented yet.")
            elif path_scheme == PathScheme.HF:
                raise NotImplementedError("HF PDF processing not implemented yet.")
            else:
                result["size"] = os.path.getsize(file_path)
                doc = fitz.open(file_path)
                
                # Extract basic document info
                doc_metadata = doc.metadata
                result.update({
                    "title": doc_metadata.get("title") or "",
                    "author": doc_metadata.get("author") or "",
                    "creation_date": doc_metadata.get("creationDate") or "",
                    "mod_date": doc_metadata.get("modDate") or "",
                    "page_count": len(doc),
                    "is_encrypted": doc.needs_pass,
                })
                
                # Analyze document structure
                image_count = 0
                has_forms = False
                has_signature_fields = False
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    
                    # Count raster images
                    page_images = page.get_images()
                    if page_images:
                        image_count += len(page_images)
                    
                    # Vector drawings are represented as drawings in PyMuPDF
                    drawings = page.get_drawings()
                    if drawings:
                        image_count += len(drawings)
                    
                    # Check for forms and signature fields
                    if not has_forms or not has_signature_fields:
                        widgets = list(page.widgets())
                        if len(widgets) > 0:
                            has_forms = True
                            for widget in widgets:
                                if widget.field_type == fitz.PDF_WIDGET_TYPE_SIGNATURE:
                                    has_signature_fields = True
                                    break
                
                result.update({
                    "has_forms": has_forms,
                    "has_signature_fields": has_signature_fields,
                    "image_count": image_count,
                })
                
                doc.close()
                logger.debug(f"PDF processed successfully: {file_path}")
                
        except Exception as e:
            logger.warning(f"Error processing PDF {file_path}: {str(e)}")
            result["error"] = str(e)
        
        return result
