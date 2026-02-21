# Changelog

## 1.2.0

- Added default image OCR enrichment (local tesseract) for image-heavy PDFs.
- OCR content is injected near image context using `[OCR_IMAGE ...]` blocks.
- Added `--no-image-ocr` switch to disable image OCR when needed.
- Fixed legacy CLI argument normalization for options that take values.
- Updated Docker image dependencies (`tesseract-ocr`, `tesseract-ocr-eng`,
  `tesseract-ocr-spa`, `libgl1`) and runtime env defaults for compatibility.
- README updated to recommend Docker full image and document OCR behavior.

## 1.1.0

- Repo cleanup: removed obsolete `legacy/` scripts and `install_cli.sh`.
- Documentation updated accordingly.

## 1.0.0

- Initial stable release.
- Python CLI (`pdf-to-md`) with multiple backends: `poppler`, `pymupdf4llm`, `docling`, `auto`.
- Docker support (minimal image with Poppler).
