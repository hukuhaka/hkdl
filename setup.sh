#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$repo_root"

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is required" >&2
    exit 1
fi

uv sync \
    --locked \
    --no-editable \
    --refresh-package hkdl \
    --reinstall-package hkdl
.venv/bin/python -c "import hkdl"
.venv/bin/hkdl --help >/dev/null

echo "HKDL is ready. Run: source .venv/bin/activate"
