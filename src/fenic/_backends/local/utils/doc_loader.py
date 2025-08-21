from __future__ import annotations

import glob
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import polars as pl

from fenic._backends.local.utils.io_utils import PathScheme, get_path_scheme
from fenic.core.error import FileLoaderError, ValidationError
from fenic.core.types import ColumnField, Schema
from fenic.core.types.datatypes import (
    StringType,
)

logger = logging.getLogger(__name__)

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
            valid_file_extension: str,
            exclude_pattern: Optional[str] = None,
            recursive: bool = False,
    ) -> pl.DataFrame:
        """Load documents from a folder.

        Args:
            paths: list of paths to the folders to load files from, the paths will be glob patterns.
            valid_file_extension: Valid file extension for the files to be processed.
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

        files = DocFolderLoader._enumerate_files(
            paths,
            valid_file_extension,
            exclude_pattern,
            recursive)
        
        if not files:
            logger.debug(f"No files found in {paths}")
            return DocFolderLoader._build_no_files_dataframe()

        # Calculate the batch size to ensure that each worker gets at least one file.
        max_workers = os.cpu_count() + 4
        return DocFolderLoader._process_files(files, max_workers)

    @staticmethod
    def get_schema() -> Schema:
        """Get the schema for the data type.

        Args:
            data_type: The data type of the files to load

        Returns:
            Schema: The schema for the data type
        """
        return Schema(
            column_fields=[
                ColumnField(name="file_path", data_type=StringType),
                ColumnField(name="error", data_type=StringType),
                ColumnField(name="content", data_type=StringType),
            ]
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
    ) -> pl.DataFrame:
        """Process files in parallel using a thread pool.

        Args:
            files: List of file paths to process
            max_workers: Number of worker threads

        Returns:
            DataFrame: A dataframe containing the files in the folder.
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(DocFolderLoader._process_single_file, file) for file in files]
            results_generator = (future.result() for future in as_completed(futures))

             # Uses the iterator over the results to build the dataframe.
            return pl.DataFrame(results_generator, schema=DocFolderLoader._get_polars_schema())


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
    def _get_polars_schema() -> pl.Schema:
        return pl.Schema({
            "file_path": pl.Utf8,
            "error": pl.Utf8,
            "content": pl.Utf8,
        })

    @staticmethod
    def _build_no_files_dataframe() -> pl.DataFrame:
        """Build a dataframe from the file content."""
        return pl.DataFrame({}, schema=DocFolderLoader._get_polars_schema())

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
        """Enumerate files in a local fs folder based on include/exclude patterns."""
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
                raise ValueError(f"Path does not exist: {base_path}")

        # Compile exclude regex if provided
        exclude_regex = None
        if exclude_pattern:
            try:
                exclude_regex = re.compile(exclude_pattern)
            except re.error as e:
                raise ValidationError(f"Invalid exclude pattern regex: {exclude_pattern}") from e

        # Filter out directories and apply exclude pattern
        matching_files = []
        for file_str in all_files:
            file_path = Path(file_str)
            if not file_path.is_file():
                continue
            if exclude_regex and exclude_regex.search(file_str):
                continue
            if not file_str.endswith(valid_file_extension):
                raise FileLoaderError(f"Only files with the extension {valid_file_extension} are supported in this plan.")
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
