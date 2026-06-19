"""`fenic` umbrella CLI.

Subcommands:
  check <file>     Validate a fenic script without executing it (lint + dry-run plan).
  skill install    Install the fenic-mechanics skill into your coding agents'
                   skill directories. Detects which agents are present and asks
                   which to install for and whether to install globally (for your
                   user) or just for the current project.

The standalone `fenic-serve` entry point is unchanged; this umbrella is additive.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fenic.scripts import fenic_assets, fenic_check


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

    skill = sub.add_parser("skill", help="Manage the fenic-mechanics agent skill.")
    skill_sub = skill.add_subparsers(dest="skill_command", required=True)
    install = skill_sub.add_parser(
        "install",
        help="Install the fenic-mechanics skill into your coding agents.",
        description="Copy the bundled fenic-mechanics skill into the skill directories your "
                    "coding agents read. Interactive by default; use flags to script it.",
    )
    install.add_argument("--agents", help="Comma-separated agents (e.g. claude,codex). Default: all detected.")
    install.add_argument("--all", action="store_true", help="Install for all detected agents (no prompt).")
    scope = install.add_mutually_exclusive_group()
    scope.add_argument("--global", dest="global_", action="store_true", help="Install for your user (default).")
    scope.add_argument("--project", action="store_true", help="Install for the current project only.")
    install.add_argument("dir", nargs="?", default=".", help="Project directory for --project (default: current).")

    return parser


def _choose_agents(args, detected: list[str]) -> list[str]:
    if args.all:
        return detected
    if args.agents:
        chosen = [a.strip() for a in args.agents.split(",") if a.strip()]
        return [a for a in chosen if a in detected]
    if sys.stdin.isatty():
        print(f"Detected agents: {', '.join(detected)}")
        raw = input("Install the fenic skill for which? [Enter = all, or comma-separated]: ").strip()
        if not raw:
            return detected
        return [a for a in (x.strip() for x in raw.split(",")) if a in detected]
    return detected  # non-interactive default


def _choose_scope(args, project_dir: Path) -> str:
    if args.project:
        return "project"
    if args.global_:
        return "global"
    is_project = (project_dir / ".git").exists() or (project_dir / "pyproject.toml").exists()
    if sys.stdin.isatty():
        opts = "(g)lobal for your user" + (", or (p)roject-only here" if is_project else "")
        raw = input(f"Install {opts}? [g/p, Enter = g]: ").strip().lower()
        return "project" if raw.startswith("p") else "global"
    return "global"


def _cmd_skill_install(args) -> None:
    detected = fenic_assets.detect_agents()
    if not detected:
        print("No supported coding agents detected (looked for ~/.claude, ~/.codex, ~/.cursor, "
              "~/.gemini, ~/.copilot and their CLIs).")
        return
    agents = _choose_agents(args, detected)
    if not agents:
        print("No agents selected — nothing to do.")
        return
    scope = _choose_scope(args, Path(args.dir))
    written = fenic_assets.install_skill(agents, scope, Path(args.dir))
    for w in written:
        print(f"installed {w}")
    where = "your user" if scope == "global" else f"project {Path(args.dir).resolve()}"
    print(f"\nInstalled the fenic-mechanics skill for {', '.join(agents)} ({where}).")


def main() -> None:
    """Entry point for the `fenic` console script."""
    args = _build_parser().parse_args()
    if args.command == "check":
        sys.exit(fenic_check.run(args.file))
    if args.command == "skill" and args.skill_command == "install":
        _cmd_skill_install(args)


if __name__ == "__main__":
    main()
