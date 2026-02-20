# pdf-to-md

Convert PDFs to Markdown with multiple backends.

## Features

- Multiple backends: `poppler`, `pymupdf4llm`, `docling`, `auto`
- Sensible defaults for repo workflows (`input/`, `output/`)
- Clear failures when external dependencies are missing

## Quick Start

### Using uvx (recommended)

Run directly without installing:

```bash
uvx --from git+https://github.com/newuni/pdf-to-md pdf-to-md input/Doc.pdf
```

### Using pip

Install from GitHub:

```bash
pip install git+https://github.com/newuni/pdf-to-md
pdf-to-md input/Doc.pdf
```

Install with optional backends:

```bash
pip install "pdf-to-md[full] @ git+https://github.com/newuni/pdf-to-md"
```

### Using the script directly

```bash
git clone https://github.com/newuni/pdf-to-md
cd pdf-to-md
python3 scripts/convert_pdf.py input/Doc.pdf output/Doc.md
```

### Using Docker (no local installs)

Build a minimal image (includes Poppler tools; does not include heavy optional backends):

```bash
docker build -t pdf-to-md .
```

Convert a PDF from your current directory:

```bash
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD":/work -w /work pdf-to-md input/Doc.pdf output/Doc.md
```

Convert a PDF from any local path:

```bash
PDF="/absolute/path/to/Doc.pdf"
OUT="/absolute/path/to/Doc.md"

docker run --rm \
  -v "$(dirname "$PDF")":/in \
  -v "$(dirname "$OUT")":/out \
  pdf-to-md \
  "/in/$(basename "$PDF")" "/out/$(basename "$OUT")"
```

Optional: build a larger image with extra backends:

```bash
docker build -t pdf-to-md:full --build-arg EXTRAS=full .
```

## Backends

- `auto`: prefers `docling`, then `pymupdf4llm`, then `poppler` (based on availability).
- `poppler`: uses `pdftohtml`/`pdftotext` and a conservative cleanup pass.
- `pymupdf4llm`: direct Markdown via PyMuPDF4LLM.
- `docling`: better layout/structure extraction (tables, reading order).

## Repository Layout

- `input/`: input PDFs (typically internal, do not commit large files).
- `output/`: generated Markdown artifacts.
- `reference/`: curated Markdown (golden expected outputs).
- `samples/`: small PDFs for regression checks.

## Usage

```bash
pdf-to-md --backend auto input/Doc.pdf
pdf-to-md --backend docling input/Doc.pdf output/Doc.docling.md
```

Bulk conversion (directories):

```bash
pdf-to-md batch input --output-dir output --recursive
pdf-to-md --backend docling batch /data/pdfs --output-dir /data/md --workers 2
```

## Installation (recommended)

Editable install:

```bash
python3 -m pip install -e .
```

With optional backends:

```bash
python3 -m pip install -e ".[docling,pymupdf4llm]"
```

## Notes

- `docling` and `poppler` depend on external binaries/environment; the CLI will fail with a clear error if something is missing.
- Scanned PDFs: an OCR pre-step can be added later (for now we only do a lightweight "looks scanned" detection and warn).

## Security

Even for internal PDFs, consider running the conversion in a sandbox (container, no network, resource limits) if you process untrusted files.


## Responsible Use

Use this project only for lawful, authorized purposes.


## Third-Party Services & Trademarks

Third-party names and trademarks belong to their respective owners; no affiliation is implied.
