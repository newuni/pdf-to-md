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

## Installation (recommended)

Editable install:

```bash
python3 -m pip install -e .
```

With optional backends:

```bash
python3 -m pip install -e ".[docling,pymupdf4llm]"
```

Local "install an executable on PATH" helper:

```bash
./install_cli.sh
```

Legacy wrapper (backwards compatibility):

```bash
./legacy/pdf_to_md.sh --backend auto input/Doc.pdf
```

## Notes

- `docling` and `poppler` depend on external binaries/environment; the CLI will fail with a clear error if something is missing.
- Scanned PDFs: an OCR pre-step can be added later (for now we only do a lightweight "looks scanned" detection and warn).

## Security

Even for internal PDFs, consider running the conversion in a sandbox (container, no network, resource limits) if you process untrusted files.
