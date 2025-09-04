"""Tests for the DocFilterLoader class."""

import os
from pathlib import Path
from typing import List

import pytest

from fenic._backends.local.utils.doc_loader import DocFolderLoader
from fenic.core.error import FileLoaderError, ValidationError
from tests.conftest import _save_md_file, _save_pdf_file


class TestDocLoader:
    """Test cases for the DocLoader class."""
    @classmethod
    def get_globbed_path(cls, path: str, file_extension: str) -> list[str]:
        return [str(Path.joinpath(Path(path), file_extension))]

    def test_enumerate_files_basic(self, temp_dir_with_test_files):
        """Enumerate all the files in a folder."""
        files = DocFolderLoader._enumerate_files(
            self.get_globbed_path(temp_dir_with_test_files, "*.md"),
            valid_file_extension=".md")
        assert len(files) == 2
        self._assert_file_in_results(files, temp_dir_with_test_files, "file1.md")
        self._assert_file_in_results(files, temp_dir_with_test_files, "file2.md")

        files_pdf = DocFolderLoader._enumerate_files(
            self.get_globbed_path(temp_dir_with_test_files, "*.pdf"),
            valid_file_extension=".pdf")
        assert len(files_pdf) == 2
        self._assert_file_in_results(files_pdf, temp_dir_with_test_files, "file8.pdf")
        self._assert_file_in_results(files_pdf, temp_dir_with_test_files, "file9.pdf")


    def test_enumerate_files_include_pattern_recursively(self, temp_dir_with_test_files):
        """Enumerate all the files in a folder."""
        files = DocFolderLoader._enumerate_files(
            self.get_globbed_path(temp_dir_with_test_files, "**/*.md"),
            valid_file_extension=".md",
            recursive=True)
        assert len(files) == 5
        self._assert_file_in_results(files, temp_dir_with_test_files, "file1.md")
        self._assert_file_in_results(files, temp_dir_with_test_files, "file2.md")
        self._assert_file_in_results(files, temp_dir_with_test_files, "subdir1/file4.md")
        self._assert_file_in_results(files, temp_dir_with_test_files, "subdir2/file5.md")
        self._assert_file_in_results(files, temp_dir_with_test_files, "temp/temp_file.md")

        files_pdf = DocFolderLoader._enumerate_files(
            self.get_globbed_path(temp_dir_with_test_files, "**/*.pdf"),
            valid_file_extension=".pdf",
            recursive=True)
        assert len(files_pdf) == 5
        self._assert_file_in_results(files_pdf, temp_dir_with_test_files, "file8.pdf")
        self._assert_file_in_results(files_pdf, temp_dir_with_test_files, "file9.pdf")
        self._assert_file_in_results(files_pdf, temp_dir_with_test_files, "subdir1/file6.pdf")
        self._assert_file_in_results(files_pdf, temp_dir_with_test_files, "subdir2/file7.pdf")
        self._assert_file_in_results(files_pdf, temp_dir_with_test_files, "temp/temp_file.pdf")

    def test_enumerate_files_exclude_pattern(self, temp_dir_with_test_files):
        """Enumerate all the files in a folder."""
        files = DocFolderLoader._enumerate_files(
                    self.get_globbed_path(temp_dir_with_test_files, "**/*"),
                    valid_file_extension=".md",
                    exclude_pattern=r"\.pdf$|\.tmp$|\.json$|\.txy$|\backup.md.bak$|\.bak$|temp/.*",
                    recursive=True)
        assert len(files) == 4
        assert not any(f.endswith('.tmp') for f in files)
        assert not any(f.endswith('.bak') for f in files)
        assert not any('temp' in f for f in files)
        
        files_pdf = DocFolderLoader._enumerate_files(
            self.get_globbed_path(temp_dir_with_test_files, "**/*"),
            valid_file_extension=".pdf",
            exclude_pattern=r"\.md$|\.tmp$|\.json$|\.txy$|\backup.md.bak$|\.bak$|temp/.*",
            recursive=True)
        assert len(files_pdf) == 4
        assert not any(f.endswith('.tmp') for f in files_pdf)
        assert not any(f.endswith('.bak') for f in files_pdf)
        assert not any('temp' in f for f in files_pdf)

    def test_load_files_from_folder(self, temp_dir_with_test_files, local_session):
        result_df = local_session.create_dataframe(DocFolderLoader.load_docs_from_folder(
            self.get_globbed_path(temp_dir_with_test_files, "**/*.md"),
            valid_file_extension="md",
            recursive=True))
        assert result_df is not None
        assert result_df.schema == DocFolderLoader.get_schema()
        assert result_df.count() == 5

        result_dict = result_df.to_pydict()
        files_processed = result_dict["file_path"]
        self._assert_file_in_results(files_processed, temp_dir_with_test_files, "file1.md")
        self._assert_file_in_results(files_processed, temp_dir_with_test_files, "file2.md")
        self._assert_file_in_results(files_processed, temp_dir_with_test_files, "subdir1/file4.md")
        self._assert_file_in_results(files_processed, temp_dir_with_test_files, "subdir2/file5.md")
        self._assert_file_in_results(files_processed, temp_dir_with_test_files, "temp/temp_file.md")

    def test_load_files_from_folder_with_error(self, temp_dir_with_test_files, local_session):
        load_local_file = DocFolderLoader._load_file_local_fs
        DocFolderLoader._load_file_local_fs = self._fail_load_file_local_fs

        result_df = local_session.create_dataframe(DocFolderLoader.load_docs_from_folder(
            self.get_globbed_path(temp_dir_with_test_files, "**/*.md"),
            valid_file_extension="md",
            recursive=True))
        assert result_df is not None
        assert result_df.schema == DocFolderLoader.get_schema()
        assert result_df.count() == 5
        result_dict = result_df.to_pydict()
        errors = result_dict["error"]
        for error in errors:
            assert error is not None
            assert "File not found" in error

        DocFolderLoader._load_file_local_fs = load_local_file

    def test_validate_path_ok(self, temp_dir_with_test_files):
        """Test that the validate_path method returns a hash of the files in the path."""
        DocFolderLoader.validate_paths(self.get_globbed_path(temp_dir_with_test_files, "**/*.md"))
        assert True
        
    def test_validate_path_invalid(self):
        """Test that the validate_path method returns a hash of the files in the path."""
        with pytest.raises(ValidationError):
            DocFolderLoader.validate_paths(self.get_globbed_path("/nonexisting/path/in/any/system", "**/*.md"))

    def test_load_docs_with_invalid_extensions(self, temp_dir_with_test_files):
        """Test that the load_docs_from_folder method raises an error if the file extension is not supported."""
        with pytest.raises(FileLoaderError):
            DocFolderLoader.load_docs_from_folder(
                self.get_globbed_path(temp_dir_with_test_files, "**/*.json"),
                valid_file_extension="md",
                recursive=True)

    def test_load_docs_with_no_files(self, local_session, temp_dir_just_one_file):
        """Test that the load_docs_from_folder method returns an empty dataframe if the path is empty."""
        os.remove(os.path.join(temp_dir_just_one_file, "file1.md"))
        result_df = local_session.create_dataframe(DocFolderLoader.load_docs_from_folder(
            self.get_globbed_path(temp_dir_just_one_file, "**/*.md"),
            valid_file_extension="md",
            recursive=True))
        assert result_df is not None
        assert result_df.count() == 0
        assert result_df.schema == DocFolderLoader.get_schema()

    def test_load_docs_with_no_files_pdf(self, local_session, temp_dir_just_one_file):
        """Test that the load_docs_from_folder method returns an empty dataframe if the path is empty."""
        result_df = local_session.create_dataframe(DocFolderLoader.load_docs_from_folder(
            self.get_globbed_path(temp_dir_just_one_file, "**/*.pdf"),
            valid_file_extension="pdf",
            recursive=True))
        assert result_df.count() == 0
        assert result_df.schema == DocFolderLoader.get_schema(file_extension="pdf")

    def test_load_docs_with_200_files(self, local_session, temp_dir_just_one_file):
        """Test that the load_docs_from_folder method returns a dataframe with 200 files."""
        for i in range(2, 201):
            _save_md_file(Path(os.path.join(temp_dir_just_one_file, f"file{i}.md")))
        result_df = local_session.create_dataframe(DocFolderLoader.load_docs_from_folder(
            self.get_globbed_path(temp_dir_just_one_file, "**/*.md"),
            valid_file_extension="md",
            recursive=True))
        assert result_df.count() == 200
        assert result_df.schema == DocFolderLoader.get_schema()

    def test_load_pdf_metadata_from_200_files(self, local_session, temp_dir_just_one_file):
        """Test that the load_docs_from_folder method returns a dataframe with PDF metadata."""
        for i in range(1, 201):
            _save_pdf_file(Path(os.path.join(temp_dir_just_one_file, f"file{i}.pdf")))

        result_df = local_session.create_dataframe(DocFolderLoader.load_docs_from_folder(
            self.get_globbed_path(temp_dir_just_one_file, "**/*.pdf"),
            valid_file_extension="pdf",
            recursive=True))
        assert result_df.count() == 200

        assert result_df.schema == DocFolderLoader.get_schema(file_extension="pdf")

    def _assert_file_in_results(self, files: List[str], temp_dir: Path, file_name: str):
        assert os.path.join(temp_dir, file_name) in files

    @staticmethod
    def _fail_load_file_local_fs(file_path: str):
        raise FileNotFoundError(f"File not found: {file_path}")

