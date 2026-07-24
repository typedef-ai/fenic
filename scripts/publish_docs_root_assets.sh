#!/usr/bin/env bash
set -euo pipefail

remote="${1:-origin}"
branch="${2:-gh-pages}"
repo_root="$(git rev-parse --show-toplevel)"
worktree_dir="$(mktemp -d "${TMPDIR:-/tmp}/fenic-docs-root.XXXXXX")"

cleanup() {
	git -C "$repo_root" worktree remove --force "$worktree_dir" >/dev/null 2>&1 || true
	rmdir "$worktree_dir" >/dev/null 2>&1 || true
}
trap cleanup EXIT

git -C "$repo_root" fetch "$remote" "$branch" --depth=1
git -C "$repo_root" worktree add --detach "$worktree_dir" "$remote/$branch"

python3 "$repo_root/scripts/backfill_docs_version_metadata.py" "$worktree_dir"

assets=(
	"llms.txt:llms.txt"
	"llms-full.txt:llms-full.txt"
	"agents/index.md:agents.md"
	"robots.txt:robots.txt"
)
for mapping in "${assets[@]}"; do
	source_asset="${mapping%%:*}"
	root_asset="${mapping#*:}"
	source_path="$worktree_dir/latest/$source_asset"
	if [[ ! -f $source_path ]]; then
		echo "Missing latest documentation asset: $source_path" >&2
		exit 1
	fi
	cp "$source_path" "$worktree_dir/$root_asset"
done

mkdir -p "$worktree_dir/.well-known"
cp \
	"$worktree_dir/latest/agents/index.md" \
	"$worktree_dir/.well-known/agent-instructions.md"

git -C "$worktree_dir" add \
	.well-known/agent-instructions.md \
	agents.md \
	llms-full.txt \
	llms.txt \
	robots.txt
git -C "$worktree_dir" add --update

if git -C "$worktree_dir" diff --cached --quiet; then
	exit 0
fi

git -C "$worktree_dir" commit -m "docs: update root discovery assets"
git -C "$worktree_dir" push "$remote" "HEAD:$branch"
