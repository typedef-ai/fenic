#!/usr/bin/env python
"""Export Fenic API documentation as parquet files and upload to HuggingFace.

Usage:
    uv run --extra hf -m fenic_mcp.hf_export                          # latest release
    uv run --extra hf -m fenic_mcp.hf_export --version 0.5.0 0.6.0    # specific versions
    uv run --extra hf -m fenic_mcp.hf_export --all                     # every tag from GitHub
    uv run --extra hf -m fenic_mcp.hf_export --dry-run --version 0.7.0 # local only, skip HF
    uv run --extra hf -m fenic_mcp.hf_export --force --version 0.5.0   # overwrite existing
"""

import argparse
import logging
import os
import subprocess  # nosec: B404 i considered it safe in this context.
import sys
import tempfile
from pathlib import Path

import griffe

import fenic as fc
from fenic_mcp.setup.populate_tables import (
    _populate_api_df,
    _populate_fenic_summary,
    _populate_hierarchy_df,
    _populate_release_metadata,
    _setup_session,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

HF_REPO_ID = "typedef-ai/fenic-codebase"
FENIC_GITHUB_REPO = "typedef-ai/fenic"
TABLE_NAMES = [
    "api_df",
    "hierarchy_df",
    "fenic_summary",
    "fenic_release_metadata",
]


def _get_hf_api():
    """Create an HfApi instance with explicit token from the environment."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable is not set")
    return HfApi(token=token)


README_YAML_TEMPLATE = """\
---
license: apache-2.0
task_categories:
  - text-generation
  - question-answering
language:
  - en
tags:
  - code
  - api-documentation
  - dataframe
  - semantic-ai
  - fenic
pretty_name: Fenic API Documentation
size_categories:
  - 1K<n<10K
configs:
{configs}---
"""

CONFIG_ENTRY_TEMPLATE = """\
  - config_name: "{version}"
    data_files:
      - split: api
        path: "{version}/api_df.parquet"
      - split: hierarchy
        path: "{version}/hierarchy_df.parquet"
      - split: summary
        path: "{version}/fenic_summary.parquet"
      - split: metadata
        path: "{version}/fenic_release_metadata.parquet"
"""


def _get_tags_from_github() -> list[str]:
    """Fetch all version tags from the fenic GitHub repo."""
    result = subprocess.run(  # nosec: B603, B607
        [
            "gh",
            "api",
            f"repos/{FENIC_GITHUB_REPO}/tags",
            "--paginate",
            "-q",
            ".[].name",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    tags = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("v"):
            tags.append(line[1:])  # strip 'v' prefix
    return sorted(tags, key=_version_sort_key)


def _get_latest_tag() -> str:
    """Get the latest release tag from GitHub."""
    result = subprocess.run(  # nosec: B603, B607
        ["gh", "api", f"repos/{FENIC_GITHUB_REPO}/releases/latest", "-q", ".tag_name"],
        capture_output=True,
        text=True,
        check=True,
    )
    tag = result.stdout.strip()
    if tag.startswith("v"):
        tag = tag[1:]
    return tag


def _version_sort_key(version: str) -> tuple[int, ...]:
    """Sort key for semantic version strings."""
    parts = []
    for part in version.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def _clone_fenic(version: str, dest: Path) -> Path:
    """Clone fenic at a specific tag into dest directory."""
    tag = f"v{version}"
    logger.info(f"Cloning fenic at tag {tag}...")
    subprocess.run(  # nosec: B603, B607
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            tag,
            f"https://github.com/{FENIC_GITHUB_REPO}.git",
            str(dest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return dest


def _load_fenic_api_from_source(source_dir: Path) -> griffe.Module:
    """Load the Fenic API using Griffe from a specific source directory."""
    search_path = source_dir / "src"
    logger.info(f"Loading Fenic API with Griffe from {search_path}...")
    loader = griffe.GriffeLoader(search_paths=[str(search_path)])
    return loader.load("fenic")


def _get_source_sha(source_dir: Path) -> str:
    """Return the immutable commit checked out for a cloned release tag."""
    result = subprocess.run(  # nosec: B603, B607
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _export_parquets(session: fc.Session, output_dir: Path) -> None:
    """Export the three tables as parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for table_name in TABLE_NAMES:
        df = session.table(table_name)
        parquet_path = output_dir / f"{table_name}.parquet"
        df.write.parquet(str(parquet_path))
        logger.info(f"Wrote {parquet_path}")


def _get_existing_versions_on_hf() -> set[str]:
    """List versions with every expected artifact present in the HF repo."""
    api = _get_hf_api()
    files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset")
    artifacts_by_version: dict[str, set[str]] = {}
    for filename in files:
        parts = filename.split("/")
        if len(parts) >= 2 and parts[1].endswith(".parquet"):
            artifacts_by_version.setdefault(parts[0], set()).add(parts[1])
    expected_artifacts = {f"{table_name}.parquet" for table_name in TABLE_NAMES}
    return {
        version
        for version, artifacts in artifacts_by_version.items()
        if expected_artifacts <= artifacts
    }


def verify_version_on_hf(version: str) -> None:
    """Download every expected artifact to prove the published version is readable."""
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable is not set")

    revision = _get_hf_api().repo_info(
        repo_id=HF_REPO_ID, repo_type="dataset"
    ).sha
    for table_name in TABLE_NAMES:
        filename = f"{version}/{table_name}.parquet"
        local_path = Path(
            hf_hub_download(  # nosec: B615 revision is the resolved repo commit SHA.
                repo_id=HF_REPO_ID,
                filename=filename,
                repo_type="dataset",
                token=token,
                revision=revision,
                force_download=True,
            )
        )
        if local_path.stat().st_size == 0:
            raise RuntimeError(f"Hugging Face artifact is empty: {filename}")
        logger.info(f"Verified readable Hugging Face artifact: {filename}")


def _upload_to_hf(local_dir: Path, version: str) -> None:
    """Upload a version's parquet files to the HF dataset repo."""
    api = _get_hf_api()
    # Ensure repo exists
    api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)

    version_dir = local_dir / version
    api.upload_folder(
        repo_id=HF_REPO_ID,
        folder_path=str(version_dir),
        path_in_repo=version,
        repo_type="dataset",
    )
    logger.info(f"Uploaded {version} to {HF_REPO_ID}")


def _build_readme(versions: list[str]) -> str:
    """Build the README.md content with YAML front matter for all version configs."""
    # Sort versions descending (newest first)
    sorted_versions = sorted(versions, key=_version_sort_key, reverse=True)
    configs = ""
    for v in sorted_versions:
        configs += CONFIG_ENTRY_TEMPLATE.format(version=v)
    return README_YAML_TEMPLATE.format(configs=configs)


def _update_hf_readme(all_versions: list[str]) -> None:
    """Update the README.md on HF with configs for all versions."""
    api = _get_hf_api()
    readme_content = _build_readme(all_versions)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(readme_content)
        f.flush()
        api.upload_file(
            path_or_fileobj=f.name,
            path_in_repo="README.md",
            repo_id=HF_REPO_ID,
            repo_type="dataset",
        )
    os.unlink(f.name)
    logger.info("Updated README.md on HuggingFace")


def process_version(
    version: str,
    output_base: Path,
    dry_run: bool = False,
    force: bool = False,
) -> bool:
    """Process a single fenic version: clone, extract, export, upload.

    Returns True if the version was processed, False if skipped.
    """
    if not dry_run and not force:
        existing = _get_existing_versions_on_hf()
        if version in existing:
            logger.info(
                f"Version {version} already exists on HF, skipping (use --force to overwrite)"
            )
            return False

    output_dir = output_base / version
    original_cwd = os.getcwd()

    with tempfile.TemporaryDirectory(prefix=f"fenic-clone-{version}-") as clone_tmp:
        clone_dir = Path(clone_tmp) / "fenic"
        _clone_fenic(version, clone_dir)

        # Load API from the cloned source
        fenic_api = _load_fenic_api_from_source(clone_dir)

        # Set up a fenic session in a separate temp work dir.
        # _setup_session calls os.chdir, so we must restore CWD afterwards
        # to avoid breaking subsequent operations after temp dirs are cleaned up.
        with tempfile.TemporaryDirectory(prefix=f"fenic-work-{version}-") as work_tmp:
            session = _setup_session(work_tmp)
            try:
                api_df = _populate_api_df(session, fenic_api)
                _populate_hierarchy_df(api_df)
                _populate_fenic_summary(api_df)
                _populate_release_metadata(
                    session,
                    version,
                    _get_source_sha(clone_dir),
                )
                _export_parquets(session, output_dir)
            finally:
                session.stop()
                os.chdir(original_cwd)

    logger.info(f"Generated parquets for version {version} at {output_dir}")

    if not dry_run:
        _upload_to_hf(output_base, version)

    return True


def main() -> None:
    """Run the command-line exporter or artifact verifier."""
    parser = argparse.ArgumentParser(
        description="Export Fenic API documentation to HuggingFace as parquet files.",
    )
    parser.add_argument(
        "--version",
        nargs="+",
        help="Specific version(s) to export (e.g. 0.5.0 0.6.0)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_versions",
        help="Export every tagged version from GitHub",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate parquets locally but skip HuggingFace upload",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing versions on HuggingFace",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify that each requested version is readable without exporting",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for parquets (default: temp directory)",
    )
    args = parser.parse_args()

    # Determine versions to process
    if args.all_versions:
        versions = _get_tags_from_github()
        logger.info(f"Found {len(versions)} tags: {versions}")
    elif args.version:
        versions = args.version
    else:
        latest = _get_latest_tag()
        versions = [latest]
        logger.info(f"Using latest release: {latest}")

    if not versions:
        logger.error("No versions to process")
        sys.exit(1)

    if args.verify_only:
        for version in versions:
            verify_version_on_hf(version)
        return

    # Set up output directory
    if args.output_dir:
        output_base = args.output_dir
        output_base.mkdir(parents=True, exist_ok=True)
        cleanup_output = False
    elif args.dry_run:
        output_base = Path(tempfile.mkdtemp(prefix="fenic-hf-export-"))
        cleanup_output = False  # keep around for inspection in dry-run
        logger.info(f"Dry-run output directory: {output_base}")
    else:
        output_base = Path(tempfile.mkdtemp(prefix="fenic-hf-export-"))
        cleanup_output = True

    processed = []
    failures = []
    for version in versions:
        try:
            if process_version(version, output_base, args.dry_run, args.force):
                processed.append(version)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone version {version}: {e.stderr}")
            failures.append(version)
        except Exception as e:
            logger.error(f"Failed to process version {version}: {e}")
            failures.append(version)

    # Update README with all version configs (existing + newly processed)
    if not args.dry_run and processed:
        existing = _get_existing_versions_on_hf()
        all_versions = sorted(
            existing | set(processed),
            key=_version_sort_key,
        )
        _update_hf_readme(all_versions)

    logger.info(f"Done. Processed {len(processed)} version(s): {processed}")
    if args.dry_run:
        logger.info(f"Parquets saved to: {output_base}")

    if cleanup_output:
        import shutil

        shutil.rmtree(output_base, ignore_errors=True)

    if failures:
        raise SystemExit(f"Failed to export version(s): {', '.join(failures)}")


if __name__ == "__main__":
    main()
