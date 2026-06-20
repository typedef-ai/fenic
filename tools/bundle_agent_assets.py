"""Copy the fenic-mechanics skill into the package so it ships in the wheel.

The canonical, committed skill lives at `.claude/skills/*`. This mirrors it into
`src/fenic/_agent_assets/skills/` (a gitignored build artifact) so that
`pip install fenic` carries it and `fenic skill install` can copy it into a
coding agent's skill directory. Run before building a wheel
(`just bundle-agent-assets`) and after editing the skill.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "src" / "fenic" / "_agent_assets"


def main() -> None:
    """Mirror the committed skill into src/fenic/_agent_assets/skills/."""
    if DEST.exists():
        shutil.rmtree(DEST)
    (DEST / "skills").mkdir(parents=True)
    shutil.copytree(ROOT / ".claude/skills/fenic-mechanics", DEST / "skills/fenic-mechanics")
    shutil.copytree(ROOT / ".claude/skills/update-fenic-skill", DEST / "skills/update-fenic-skill")
    print(f"bundled agent skill -> {DEST.relative_to(ROOT)}/skills")


if __name__ == "__main__":
    main()
