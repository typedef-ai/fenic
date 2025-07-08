import 'docs/justfile'

gitRoot := `git rev-parse --show-toplevel`

[private]
default:
  @just --list

helpText := '
see: https://just.systems/man

tips:
- run \`just --list\` to list targets
  - \`just\` works as well as list is the default
- add \`-n\` directly after \`just\` to noop or dry-run the command
'

# print help message
help:
  @echo "{{ helpText }}"
  @just -f {{ gitRoot }}/justfile --list

helpSyncText := '
examples:
  # skip running sync when running tests
  $ just -n sync=false test-cloud
  [ "false" != "false" ] && uv sync --extra=cloud  || true
  uv run pytest -m cloud tests

  # no skip sync
  $ just -n test-cloud
  [ "true" != "false" ] && uv sync --extra=cloud  || true
  uv run pytest -m cloud tests

  # run sync with minimum dependency versions prior to running tests
  $ just -n sync=min test-cloud
  [ "min" != "false" ] && uv sync --extra=cloud --resolution=lowest-direct || true
  uv run pytest -m cloud tests

  # run sync with maximum dependency versions prior to running tests
  $ just -n sync=max test-cloud
  [ "max" != "false" ] && uv sync --extra=cloud --upgrade || true
  uv run pytest -m cloud tests
'

# print help message related to sync
help-sync:
  @echo "{{ helpSyncText }}"
  @just -f {{ gitRoot }}/justfile --list

helpTestText := '
\`test*\` targets will run unit tests
  - by default running uv sync prior
    - set `sync=false` to skip (e.g. \`just sync=false test\`)

- default (\`test\`/\`test-local\`) tests will run without fenic cloud
  dependencies and tests
- \`test-cloud\` will run the fenic cloud related tests
'

# print help message related to test
help-test:
  @echo "{{ helpTestText }}"
  @just -f {{ gitRoot }}/justfile --list

# setup the project
setup: sync sync-rust
  true

sync := "true"
syncMinMaxFlag := if sync == "min" {
  "--resolution=lowest-direct"
} else if sync == "max" {
  "--upgrade"
} else { "" }

# sync project dependencies - set sync=false to skip in other target deps
sync:
  [ "{{ sync }}" != "false" ] && \
  uv sync --extra=google --extra=anthropic {{ syncMinMaxFlag }} || true

alias sync-local := sync

# sync project dependencies related to fenic cloud
sync-cloud:
  [ "{{ sync }}" != "false" ] && \
  uv sync --extra=cloud {{ syncMinMaxFlag }} || true

# sync rust changes (via maturin)
sync-rust:
  uv run maturin develop --uv

# run default tests
test: test-local
  true

# run local tests
test-local: sync
  uv run pytest -m "not cloud" tests

alias test-not-cloud := test-local

# run fenic cloud related tests
test-cloud: sync-cloud
  uv run pytest -m cloud tests

# preview generated docs
preview-docs:
  uv run --group=docs mkdocs serve
