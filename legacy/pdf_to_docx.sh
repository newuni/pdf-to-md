#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: pdf_to_docx.sh INPUT.pdf [OUTPUT.docx]

Convert a PDF to DOCX using pandoc. When OUTPUT is omitted the script writes
next to the source file using the same basename.
USAGE
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

if ! command -v pandoc >/dev/null 2>&1; then
  echo "Error: pandoc is required. Install it with 'brew install pandoc' or see https://pandoc.org/install/." >&2
  exit 2
fi

input_pdf=$1

if [[ ! -f $input_pdf ]]; then
  echo "Error: input file '$input_pdf' does not exist." >&2
  exit 3
fi

if [[ $# -eq 2 ]]; then
  output_docx=$2
else
  base_name=${input_pdf%.*}
  output_docx="${base_name}.docx"
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
converter_script="$script_dir/pdf_to_md.sh"

if [[ ! -x $converter_script ]]; then
  echo "Error: expected helper '$converter_script'. Make sure pdf_to_md.sh exists and is executable." >&2
  exit 4
fi

tmp_md=$(mktemp "${TMPDIR:-/tmp}/pdf-to-docx.XXXXXX.md")
cleanup() {
  rm -f "$tmp_md"
}
trap cleanup EXIT

"$converter_script" "$input_pdf" "$tmp_md" >/dev/null
pandoc "$tmp_md" -o "$output_docx"
echo "Converted '$input_pdf' -> '$output_docx' via pdf_to_md.sh + pandoc"
