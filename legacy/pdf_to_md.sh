#!/usr/bin/env bash
set -euo pipefail

# Legacy wrapper kept for backwards compatibility.
# Preferred: `pdf-to-md ...` (installed) or `python3 -m pdf_to_md ...`.

root_dir=$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1
  pwd
)

python_bin=python3
if [[ -x "$root_dir/.venv/bin/python" ]]; then
  python_bin="$root_dir/.venv/bin/python"
  export PATH="$root_dir/.venv/bin:$PATH"
fi

export PYTHONPATH="$root_dir${PYTHONPATH:+:$PYTHONPATH}"

exec "$python_bin" -m pdf_to_md "$@"
