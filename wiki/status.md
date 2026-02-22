# Implementation Status (pdf-to-md)

## Tracking Rules

- Update this file when closing each implementation task.
- Do not start the next task until the previous task's real status and evidence are recorded here.
- Keep task state aligned with reality at all times (`Completada`, `Parcial`, `Pendiente`, `Bloqueada`) and include the next concrete action.

## Tasks

| ID | Task | Estado | Evidence | Next action |
|---|---|---|---|---|
| T01 | Add status tracking discipline (`wiki/status.md`) | Completada | `AGENTS.md` includes "Status Tracking Discipline"; `wiki/status.md` populated with tasks table | Keep this file updated for every new task |
| T02 | Add `batch` subcommand for directory bulk conversion | Completada | `pdf_to_md/cli.py` adds `batch`; help shows `COMMAND` with `convert` and `batch` | Add basic tests for batch planning and skip rules |
| T03 | Document bulk usage in CLI help + README | Completada | `pdf-to-md --help` and `pdf-to-md batch --help` include examples; `README.md` includes bulk examples | Keep README examples in sync with CLI flags |
| T04 | Validate `batch` end-to-end on repo `input/` | Completada | Ran `pdf-to-md batch input --output-dir /tmp/pdf-to-md-batch-out --recursive --glob "*.pdf" --workers 2` with `converted=10 failed=0` | Add a `--report jsonl` option for CI/auditing |
| T05 | Add default image OCR enrichment and Docker-ready OCR dependencies | Completada | `pdf_to_md/cli.py` now runs image OCR by default and injects `[OCR_IMAGE ...]` blocks; `Dockerfile` installs `tesseract-ocr` + `libgl1`; validated with Docker: `pdf-to-md:full --backend poppler` and `--backend docling` on `input/OpenTok - Análisis técnico.pdf` (`Warning: image OCR inserted 11 block(s)`) | Add automated tests for OCR block insertion and markdown anchoring |
| T06 | Convert external OpenTok technical PDF with advanced backend | Completada | Ran `PATH="/Users/unai/Documents/git/pdf-to-md/.venv/bin:$PATH" .venv/bin/pdf-to-md --backend docling --docling-ocr convert "/Users/unai/Documents/git/kwido-vc-debug/extra-context/opentok-analisis-tecnico.pdf" "/Users/unai/Documents/git/kwido-vc-debug/extra-context/opentok-analisis-tecnico.md"`; output created (`19K`, `459` lines) | Delivered Markdown path to requester |
| T07 | Set Docker as preferred execution mode in agent guidelines | Completada | `AGENTS.md` updated under "Build, Test, and Development Commands" with explicit default preference for Docker execution (`pdf-to-md:full`) and fallback to local `.venv` only when needed | Use Docker-first commands in future conversion runs |
| T08 | Convert external DEPLOYED PDF with Docker-first advanced backend | Completada | Ran Docker conversion with advanced backend and OCR: `docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v /tmp/pdf-to-md-rapidocr-models:/usr/local/lib/python3.12/site-packages/rapidocr/models -v "/Users/unai/Documents/git/kwido-vc-debug/extra-context":/in -v "/Users/unai/Documents/git/kwido-vc-debug/extra-context":/out pdf-to-md:full --backend docling --docling-ocr convert "/in/20251201.127 - DEPLOYED [cmval].pdf" "/out/20251201.127 - DEPLOYED [cmval].md"`; result: `Warning: image OCR inserted 4 block(s)`; output created (`4.3K`, `151` lines) | Reuse same Docker model cache mount for subsequent OCR runs |
| T09 | Fix Docker rootless `docling --ocr` reliability | Completada | `pdf_to_md/cli.py` updated so `--docling-ocr` prefers `--ocr-engine tesseract` by default (`PDF_TO_MD_DOCLING_PREFER_TESSERACT_OCR=1`), avoiding rapidocr permission/download failures under non-root Docker; validated with `docker run --rm -u "$(id -u):$(id -g)" ... pdf-to-md:full --backend docling --docling-ocr "/in/[MEDIA] Precargar Equipo EZIA y Responsable al crear un caso.pdf" "/out/[MEDIA] Precargar Equipo EZIA y Responsable al crear un caso.ocrfix.md"`; conversion finished successfully with OCR blocks inserted | Prepare and publish patch release with changelog notes |

## Releases

| Version | Date (UTC) | Notes |
|---|---:|---|
| v1.2.0 | 2026-02-21 | Default image OCR enrichment (`[OCR_IMAGE ...]`), Docker full OCR/runtime fixes (`tesseract` + `libgl1`), README update |
| v1.1.0 | 2026-02-17 | Cleanup release (removed obsolete legacy scripts) |
| v1.0.0 | 2026-02-17 | Initial stable release |

## Backlog

| ID | Task | Estado | Notes | Next action |
|---|---|---|---|---|
| P01 | Add tests for `batch` (planning, exclude, suffix, skip rules) | Pendiente | Currently tested manually | Add `tests/` with a few small unit tests (no PDFs needed) |
| P02 | Add `--report jsonl` for batch runs | Pendiente | Useful for automation and audits | Emit one JSON object per file + summary |
| P03 | Add `--mirror` option to preserve directory structure under output dir | Pendiente | Current behavior mirrors relative path per input root; file inputs are flat | Decide UX + implement |
