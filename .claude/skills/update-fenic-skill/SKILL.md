---
name: update-fenic-skill
description: >-
  Use when fenic has been upgraded to a new version and the `fenic-mechanics`
  skill needs refreshing, or when explicitly asked to regenerate/update the
  fenic API reference. Regenerates the version-stamped reference from the
  installed fenic, surfaces the API deltas, and flags which hand-authored
  judgment sections to re-review.
---

# Update the fenic-mechanics skill for a new fenic version

The `fenic-mechanics` skill is **hybrid**: a _generated_ reference
(`.claude/skills/fenic-mechanics/reference/*.md`, introspected from the installed
fenic — never edit by hand) plus _hand-authored_ judgment (`SKILL.md`,
`gotchas.md` — the namespace law, calling conventions, and silent traps). When
fenic changes version, regenerate the first and re-review the second.

## Procedure

1. **Confirm the installed version changed.**

   ```bash
   uv run --project . python -c "from importlib import metadata; print(metadata.version('fenic'))"
   cat .claude/skills/fenic-mechanics/reference/VERSION
   ```

2. **Regenerate the reference** (deterministic — only real API changes will diff):

   ```bash
   uv run --project . python .claude/skills/fenic-mechanics/scripts/generate_reference.py
   ```

3. **Inspect the API deltas:**

   ```bash
   git diff --stat .claude/skills/fenic-mechanics/reference/
   git diff .claude/skills/fenic-mechanics/reference/
   ```

   Classify each change: **added** symbol, **removed** symbol, **renamed**
   symbol, or **changed signature** (param added/removed/renamed/retyped).

4. **Re-review the hand-authored judgment against the deltas.** Update
   `SKILL.md` / `gotchas.md` only where a delta touches them:
   - **Namespace/import law** — did any namespace move, or did a function become
     a method (or vice-versa)? (e.g. something leaving/entering `fc.arr`,
     `fc.dt`, `fc.semantic`).
   - **The 4 silent traps** — re-verify each still holds: `fc.json.jq` still
     returns `ArrayType(JsonType)`; semantic templates still Jinja2 `{{ }}`;
     `fc.dt.datediff(end, start)` arg order; `fc.dt.to_timestamp` format dialect.
     A changed signature on any of these is high-priority.
   - **Model/config** — provider classes, `input_tpm`/`output_tpm` vs `tpm`,
     `default_*` requirements.
   - **The "wrote X, meant Y" table** — drop rows whose anti-pattern is now
     impossible, and add rows for new traps the signature changes introduce.

5. **Verify every symbol the hand-authored files cite still exists.** For each
   `fc.<something>` mentioned in `SKILL.md`/`gotchas.md`, confirm it resolves in
   the new version (a quick `python -c "import fenic as fc; fc.<symbol>"` per
   cited symbol, or grep the regenerated reference).

6. **Re-run the eval harness** (if present) against the new version to confirm
   the skill still moves blocked→pass; note any regressions.

7. **Commit** the regenerated reference + any judgment edits together, with a
   message noting the fenic version bump.

## Don't

- Don't hand-edit files under `reference/` — they're generated; your edits will
  be overwritten on the next run. Put durable knowledge in `gotchas.md`.
- Don't add signature dumps to `SKILL.md`/`gotchas.md` — those live in the
  generated reference. The hand-authored files hold _judgment_ (what's wrong,
  what's silent, what to do instead).
