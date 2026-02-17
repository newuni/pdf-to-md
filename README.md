# pdf-to-md

CLI to convert PDFs to Markdown with multiple backends:

- `poppler` (external): `pdftohtml`/`pdftotext` + cleanup.
- `pymupdf4llm` (Python): direct Markdown via PyMuPDF4LLM.
- `docling` (CLI/Python): better layout/structure extraction.
- `auto`: prefers `docling`, then `pymupdf4llm`, then `poppler` (based on availability).

## Repository Layout

- `input/`: input PDFs (typically internal, do not commit large files).
- `output/`: generated Markdown artifacts.
- `reference/`: curated Markdown (golden expected outputs).
- `samples/`: small PDFs for regression checks.

## Usage

Without installing (from the repo):

```bash
python3 -m pdf_to_md --backend docling input/Doc.pdf output/Doc.docling.md
```

Legacy wrapper:

```bash
./legacy/pdf_to_md.sh --backend auto input/Doc.pdf
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
