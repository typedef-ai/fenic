#!/usr/bin/env python3
"""Measure installed package footprint for fenic extra combinations.

The harness creates temporary uv projects that depend on this checkout, syncs
each project, and measures the installed site-packages footprint plus the
largest installed distributions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess  # nosec B404 - developer harness shells out to uv/du with argument lists.
import sys
import tempfile
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

import tomllib

DEFAULT_COMBOS = [
    "core",
    "pdf",
    "cluster",
    "sim-join",
    "pdf,cluster,sim-join",
]


@dataclass(frozen=True)
class PackageSize:
    name: str
    version: str
    bytes: int


@dataclass(frozen=True)
class SizeResult:
    label: str
    extras: list[str]
    package_count: int
    site_packages_bytes: int
    venv_bytes: int
    top_packages: list[PackageSize]
    project_dir: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create temporary uv projects for fenic extra combinations and "
            "measure the resulting installed package footprint."
        )
    )
    parser.add_argument(
        "--fenic-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the fenic checkout to install. Defaults to this repository.",
    )
    parser.add_argument(
        "--combo",
        action="append",
        dest="combos",
        help=(
            "Extra combination to measure. Use 'core' for no extras, comma-separated "
            "extras like 'pdf,cluster', or 'all' for every optional dependency."
        ),
    )
    parser.add_argument(
        "--python",
        default=f"{sys.version_info.major}.{sys.version_info.minor}",
        help="Python version passed to uv. Defaults to the interpreter running this tool.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=8,
        help="Number of largest distributions to show for each combination.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Write machine-readable results to this path.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Write a CSV summary to this path.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep temporary projects after measuring and include their paths in output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated pyproject.toml for each combination without syncing.",
    )
    parser.add_argument(
        "--uv",
        default="uv",
        help="uv executable to run.",
    )
    return parser.parse_args()


def load_optional_extras(fenic_root: Path) -> list[str]:
    with (fenic_root / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)
    return sorted(pyproject["project"].get("optional-dependencies", {}))


def parse_combo(combo: str, all_extras: list[str]) -> tuple[str, list[str]]:
    label = combo.strip()
    if not label or label == "core":
        return "core", []
    if label == "all":
        return "all", all_extras
    extras = [extra.strip() for extra in label.split(",") if extra.strip()]
    if not extras:
        return "core", []
    return ",".join(extras), extras


def dependency_spec(fenic_root: Path, extras: list[str]) -> str:
    root_uri = fenic_root.resolve().as_uri()
    if extras:
        return f"fenic[{','.join(extras)}] @ {root_uri}"
    return f"fenic @ {root_uri}"


def pyproject_for(fenic_root: Path, extras: list[str], python_version: str) -> str:
    dependency = dependency_spec(fenic_root, extras)
    return f"""[project]
name = "fenic-size-probe"
version = "0.0.0"
requires-python = ">={python_version}"
dependencies = [
  "{dependency}",
]
"""


def run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(  # nosec B603 - command is passed as an argument list.
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = "\n".join(
            [
                f"command failed in {cwd}: {' '.join(command)}",
                result.stdout.strip(),
                result.stderr.strip(),
            ]
        ).strip()
        raise RuntimeError(message)
    return result


def venv_python(project_dir: Path) -> Path:
    if sys.platform == "win32":
        return project_dir / ".venv" / "Scripts" / "python.exe"
    return project_dir / ".venv" / "bin" / "python"


def site_packages_path(project_dir: Path, env: dict[str, str]) -> Path:
    code = (
        "import json, site; "
        "print(json.dumps([p for p in site.getsitepackages() if p.endswith('site-packages')][0]))"
    )
    result = run([str(venv_python(project_dir)), "-c", code], cwd=project_dir, env=env)
    return Path(json.loads(result.stdout))


def directory_size_bytes(path: Path) -> int:
    du = shutil.which("du")
    if du:
        result = subprocess.run(  # nosec B603 - du path comes from shutil.which and args are fixed.
            [du, "-sk", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(result.stdout.split()[0]) * 1024

    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def distribution_size_bytes(dist: metadata.Distribution) -> int:
    files = dist.files
    if files is None:
        return 0

    total = 0
    seen: set[Path] = set()
    for package_path in files:
        path = Path(dist.locate_file(package_path))
        if path in seen or not path.exists() or not path.is_file():
            continue
        seen.add(path)
        total += path.stat().st_size
    return total


def installed_packages(site_packages: Path) -> list[PackageSize]:
    packages = []
    for dist in metadata.distributions(path=[str(site_packages)]):
        name = dist.metadata.get("Name") or "unknown"
        version = dist.version or ""
        packages.append(
            PackageSize(
                name=name,
                version=version,
                bytes=distribution_size_bytes(dist),
            )
        )
    return sorted(packages, key=lambda package: package.bytes, reverse=True)


def measure_combo(
    *,
    label: str,
    extras: list[str],
    fenic_root: Path,
    python_version: str,
    top: int,
    uv: str,
    keep: bool,
    dry_run: bool,
    env: dict[str, str],
) -> SizeResult | None:
    temp_dir = Path(tempfile.mkdtemp(prefix=f"fenic-size-{label.replace(',', '-')}-"))
    pyproject = pyproject_for(fenic_root, extras, python_version)
    (temp_dir / "pyproject.toml").write_text(pyproject)

    if dry_run:
        print(f"# {label}: {temp_dir / 'pyproject.toml'}")
        print(pyproject)
        if not keep:
            shutil.rmtree(temp_dir)
        return None

    run(
        [
            uv,
            "sync",
            "--no-dev",
            "--no-install-project",
            "--no-progress",
            "--python",
            python_version,
        ],
        cwd=temp_dir,
        env=env,
    )

    site_packages = site_packages_path(temp_dir, env)
    packages = installed_packages(site_packages)
    result = SizeResult(
        label=label,
        extras=extras,
        package_count=len(packages),
        site_packages_bytes=directory_size_bytes(site_packages),
        venv_bytes=directory_size_bytes(temp_dir / ".venv"),
        top_packages=packages[:top],
        project_dir=str(temp_dir) if keep else None,
    )

    if not keep:
        shutil.rmtree(temp_dir)
    return result


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def print_table(results: list[SizeResult]) -> None:
    if not results:
        return

    baseline = results[0].site_packages_bytes
    rows = []
    for result in results:
        delta = result.site_packages_bytes - baseline
        top = ", ".join(
            f"{package.name} {format_bytes(package.bytes)}"
            for package in result.top_packages
        )
        rows.append(
            [
                result.label,
                str(result.package_count),
                format_bytes(result.site_packages_bytes),
                f"{delta / 1024 / 1024:+.1f} MiB",
                format_bytes(result.venv_bytes),
                top,
            ]
        )

    headers = ["combo", "packages", "site-packages", "delta", ".venv", "top packages"]
    widths = [
        max(len(row[index]) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def write_json(path: Path, results: list[SizeResult]) -> None:
    payload = [
        {
            **asdict(result),
            "top_packages": [asdict(package) for package in result.top_packages],
        }
        for result in results
    ]
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_csv(path: Path, results: list[SizeResult]) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "combo",
                "extras",
                "package_count",
                "site_packages_bytes",
                "venv_bytes",
                "top_packages",
                "project_dir",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "combo": result.label,
                    "extras": ",".join(result.extras),
                    "package_count": result.package_count,
                    "site_packages_bytes": result.site_packages_bytes,
                    "venv_bytes": result.venv_bytes,
                    "top_packages": "; ".join(
                        f"{package.name}:{package.bytes}"
                        for package in result.top_packages
                    ),
                    "project_dir": result.project_dir or "",
                }
            )


def main() -> int:
    args = parse_args()
    fenic_root = args.fenic_root.resolve()
    all_extras = load_optional_extras(fenic_root)
    combos = args.combos or DEFAULT_COMBOS
    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = ".venv"

    results = []
    for combo in combos:
        label, extras = parse_combo(combo, all_extras)
        print(f"measuring {label}...", file=sys.stderr)
        result = measure_combo(
            label=label,
            extras=extras,
            fenic_root=fenic_root,
            python_version=args.python,
            top=args.top,
            uv=args.uv,
            keep=args.keep,
            dry_run=args.dry_run,
            env=env,
        )
        if result is not None:
            results.append(result)

    print_table(results)
    if args.json:
        write_json(args.json, results)
    if args.csv:
        write_csv(args.csv, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
