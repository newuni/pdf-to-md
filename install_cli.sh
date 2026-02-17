#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ./install_cli.sh [--extras LIST] [--bin-dir DIR]

Install the `pdf-to-md` CLI so it can be used from other directories.

Default behavior:
  1) Create/update ./.venv
  2) pip install -e ".[docling,pymupdf4llm]"
  3) Symlink ~/.local/bin/pdf-to-md -> <repo>/.venv/bin/pdf-to-md

Options:
  --extras LIST   Comma-separated extras to install (default: docling,pymupdf4llm)
                 Use "" to install only the base package.
  --bin-dir DIR   Where to place the symlink (default: ~/.local/bin)

Notes:
  - Ensure your shell PATH includes the bin dir (commonly ~/.local/bin).
  - Poppler tools (pdftohtml/pdftotext) are external dependencies for the poppler backend.
USAGE
}

die() {
  echo "Error: $*" >&2
  exit 2
}

repo_dir=$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)

extras=${PDF_TO_MD_EXTRAS:-docling,pymupdf4llm}
bin_dir=${PDF_TO_MD_BIN_DIR:-"$HOME/.local/bin"}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --extras)
      [[ $# -ge 2 ]] || die "--extras requires a value"
      extras=$2
      shift 2
      ;;
    --bin-dir)
      [[ $# -ge 2 ]] || die "--bin-dir requires a value"
      bin_dir=$2
      shift 2
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

command -v python3 >/dev/null 2>&1 || die "python3 is required"

cd -- "$repo_dir"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

venv_python="$repo_dir/.venv/bin/python"
[[ -x "$venv_python" ]] || die "venv python not found at '$venv_python'"

"$venv_python" -m pip install -U pip >/dev/null

if [[ -n "$extras" ]]; then
  "$venv_python" -m pip install -e ".[${extras}]"
else
  "$venv_python" -m pip install -e .
fi

exe_src="$repo_dir/.venv/bin/pdf-to-md"
[[ -x "$exe_src" ]] || die "expected executable not found at '$exe_src'"

mkdir -p "$bin_dir"
ln -sf "$exe_src" "$bin_dir/pdf-to-md"

if [[ ":$PATH:" != *":$bin_dir:"* ]]; then
  echo "Installed, but '$bin_dir' is not on PATH." >&2
  echo "Add it (zsh/bash): export PATH=\"$bin_dir:\$PATH\"" >&2
fi

echo "OK: pdf-to-md installed at '$bin_dir/pdf-to-md'"

