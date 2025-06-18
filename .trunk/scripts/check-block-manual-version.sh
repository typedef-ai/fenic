#!/bin/bash

# check for version changes in Cargo.toml and pyproject.toml files
# skips the check when running on release-please branches

target="$1"

current_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)

if [[ ${current_branch} == "release-please--branches--main" ]]; then
	exit 0
fi

if [[ "$(basename "${target}")" == "Cargo.toml" ]] || [[ "$(basename "${target}")" == "pyproject.toml" ]]; then
	# In CI, GITHUB_BASE_REF contains the target branch (e.g., "main", "feature-A")
	if [[ -n ${GITHUB_BASE_REF} ]]; then
		base_branch="origin/${GITHUB_BASE_REF}"
	else
		# Fallback for local testing or other contexts
		base_branch="origin/main"
	fi

	# Get merge base and check for version changes in the diff
	merge_base=$(git merge-base HEAD "${base_branch}" 2>/dev/null)
	if [[ -n ${merge_base} ]]; then
		# Check if version lines were added/modified in this branch
		git diff "${merge_base}" -- "${target}" | grep -E "^\+.*version\s*=" --line-number --with-filename 2>/dev/null || true
	else
		# Fallback: check last commit only
		git diff HEAD~1 -- "${target}" | grep -E "^\+.*version\s*=" --line-number --with-filename 2>/dev/null || true
	fi
fi
