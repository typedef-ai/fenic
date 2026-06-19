"""Install the bundled fenic-mechanics skill into the dirs your coding agents read.

`pip install fenic` carries the skill inside the package (under `_agent_assets/`,
produced at build time by tools/bundle_agent_assets.py). No agent scans
site-packages, so `fenic skill install` copies the skill into a skill directory
the agent actually reads:

  - Claude Code:                 ~/.claude/skills/   (or <project>/.claude/skills/)
  - Codex / Cursor / Gemini / Copilot: ~/.agents/skills/  (cross-agent convention;
    Cursor & Copilot also read ~/.claude/skills/)

`~/.agents/skills/` and per-agent native skill support are recent (late 2025 / 2026)
and version-dependent; on an older agent the skill simply won't be picked up.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

import fenic

# Detection: an agent counts as "present" if its config dir exists or its CLI is
# on PATH. (config-dir name, CLI binary)
_AGENTS = {
    "claude": ("~/.claude", "claude"),
    "codex": ("~/.codex", "codex"),
    "cursor": ("~/.cursor", "cursor"),
    "gemini": ("~/.gemini", "gemini"),
    "copilot": ("~/.copilot", "copilot"),
}


def assets_dir() -> Path:
    """Return the bundled `_agent_assets` directory inside the installed fenic package."""
    d = Path(fenic.__file__).resolve().parent / "_agent_assets"
    if not d.exists():
        raise FileNotFoundError(
            f"Bundled agent assets not found at {d}. In a source checkout, run "
            "`python tools/bundle_agent_assets.py` first; a pip-installed fenic ships them."
        )
    return d


def detect_agents() -> List[str]:
    """Return the coding agents present on this machine (config dir exists or CLI on PATH)."""
    found = []
    for name, (cfg, binary) in _AGENTS.items():
        if Path(cfg).expanduser().exists() or shutil.which(binary):
            found.append(name)
    return found


def skill_dir(agent: str, scope: str, project_dir: Optional[Path] = None) -> Path:
    """Return the skill directory `agent` reads for the given scope ('global' or 'project')."""
    home = Path.home()
    base = Path(project_dir or Path.cwd())
    # Claude has its own dir; the rest share the cross-agent ~/.agents (or ./.agents).
    if agent == "claude":
        return home / ".claude" / "skills" if scope == "global" else base / ".claude" / "skills"
    return home / ".agents" / "skills" if scope == "global" else base / ".agents" / "skills"


def install_skill(agents: List[str], scope: str = "global",
                  project_dir: Optional[Path] = None) -> List[Path]:
    """Copy the bundled skills into the (deduplicated) skill dirs for the given agents/scope."""
    src = assets_dir() / "skills"
    target_dirs = sorted({skill_dir(a, scope, project_dir) for a in agents if a in _AGENTS},
                         key=str)
    written: List[Path] = []
    for d in target_dirs:
        for skill in sorted(src.iterdir()):
            if skill.is_dir():
                dst = d / skill.name
                shutil.copytree(skill, dst, dirs_exist_ok=True)
                written.append(dst)
    return written
