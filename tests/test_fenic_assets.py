"""Tests for `fenic skill install` (fenic.scripts.fenic_assets).

The bundled skill is a build artifact (src/fenic/_agent_assets/, gitignored), so
the fixture produces it first if absent. Install tests use project scope into a
tmp dir so they never touch the real ~/.claude or ~/.agents.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from fenic.scripts import fenic_assets

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bundled():
    """Ensure the bundled skill exists (build artifact)."""
    try:
        fenic_assets.assets_dir()
    except FileNotFoundError:
        subprocess.run([sys.executable, str(REPO / "tools" / "bundle_agent_assets.py")], check=True)
    return fenic_assets.assets_dir()


def test_detect_agents_returns_known_names():
    detected = fenic_assets.detect_agents()
    assert isinstance(detected, list)
    assert set(detected) <= {"claude", "codex", "cursor", "gemini", "copilot"}


def test_skill_dir_mapping(tmp_path):
    home = Path.home()
    assert fenic_assets.skill_dir("claude", "global") == home / ".claude" / "skills"
    assert fenic_assets.skill_dir("codex", "global") == home / ".agents" / "skills"  # cross-agent
    assert fenic_assets.skill_dir("claude", "project", tmp_path) == tmp_path / ".claude" / "skills"
    assert fenic_assets.skill_dir("gemini", "project", tmp_path) == tmp_path / ".agents" / "skills"


def test_install_skill_project_scope(bundled, tmp_path):
    fenic_assets.install_skill(["claude", "codex"], "project", tmp_path)
    assert (tmp_path / ".claude" / "skills" / "fenic-mechanics" / "SKILL.md").exists()   # claude
    assert (tmp_path / ".agents" / "skills" / "fenic-mechanics" / "SKILL.md").exists()   # codex
    assert (tmp_path / ".claude" / "skills" / "update-fenic-skill" / "SKILL.md").exists()


def test_install_dedups_shared_agents_dir(bundled, tmp_path):
    # codex/cursor/gemini all map to .agents/skills → one target dir, not three copies.
    written = fenic_assets.install_skill(["codex", "cursor", "gemini"], "project", tmp_path)
    target = tmp_path / ".agents" / "skills"
    assert sorted({w.parent for w in written}) == [target]
    assert (target / "fenic-mechanics" / "SKILL.md").exists()
