"""`fenic` umbrella CLI.

Subcommands:
  - check <file>   Validate a fenic script without executing it (lint + dry-run
                   plan construction). See fenic.scripts.fenic_check.

The standalone `fenic-serve` entry point is unchanged; this umbrella is additive.
"""
from __future__ import annotations

import argparse
import sys

from fenic.scripts import fenic_check


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fenic", description="fenic command-line tools.")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser(
        "check",
        help="Validate a fenic script without executing it (no tokens; no key unless it configures semantic models).",
        description="Lint fenic symbol/namespace usage and dry-run the logical-plan "
                    "construction (no materialization), reporting result schema or the precise error as JSON.",
    )
    check.add_argument("file", nargs="?", default="-",
                       help="Path to a .py file, or '-' / omitted to read from stdin.")

    return parser


def main() -> None:
    """Entry point for the `fenic` console script."""
    args = _build_parser().parse_args()
    if args.command == "check":
        sys.exit(fenic_check.run(args.file))


if __name__ == "__main__":
    main()
