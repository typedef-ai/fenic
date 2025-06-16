#!/bin/bash

# check for version fields in Cargo.toml and pyproject.toml files
# skips the check when running on release-please branches

target="$1"

current_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)

if [[ ${current_branch} == "release-please--branches--main" ]]; then
	exit 0
fi

if [[ "$(basename "${target}")" == "Cargo.toml" ]] || [[ "$(basename "${target}")" == "pyproject.toml" ]]; then
	grep -E "^version\s*=" --line-number --with-filename "${target}" 2>/dev/null || true
fi
