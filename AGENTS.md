# Repository Guidelines

## Status Tracking Discipline (`wiki/status.md`)

- For every implementation task in this project, update `wiki/status.md` immediately after finishing the task with the real result and evidence.
- Do not start the next task until the previous task status is reflected in `wiki/status.md`.
- Keep task state aligned with reality at all times (`Todo`, `Parcial`, `Pendiente`, `Bloqueada`) and include the next concrete action.

## Project Structure & Module Organization
- `pdf_to_md/cli.py` holds the conversion pipeline (Python CLI orchestrating poppler/docling/pymupdf4llm + cleanup). Factor new behaviors into dedicated helper functions rather than expanding the main flow inline.
- Store authoritative PDFs under `input/`. Commit only compact samples needed for regression checks under `samples/` and ignore large files in `.gitignore`.
- Treat generated Markdown as artifacts; keep curated outputs in `reference/` with clear filenames like `sample_contract.md` to showcase expected formatting.

## Build, Test, and Development Commands
- `pdf-to-md input/MyDoc.pdf output/MyDoc.md`: converts a PDF (install with `python3 -m pip install -e .`).
- `python3 -m pdf_to_md input/MyDoc.pdf output/MyDoc.md`: run from the repo without installing.
- `pdftohtml -v` or `pdftotext -v`: confirm conversion backends are available; document their versions in PR descriptions.

## Coding Style & Naming Conventions
- Keep Python code pep8-aligned (4 spaces) and wrap helper functions at 88 characters max.
- Prefer small, testable helpers over long procedural flows inside `main()`.
- Name new scripts `verb_noun.sh` and align temporary files under `/tmp/pdf-to-md-*` to avoid clutter.

## Testing Guidelines
- Exercise the pipeline with a representative file: `pdf-to-md samples/sample.pdf /tmp/out.md`.
- Compare outputs against reference Markdown with `diff -u reference/sample.md /tmp/out.md`; update references only when rendering changes are intended.
- Document edge cases (tables, bullet hierarchies, page separators) in the PR checklist when manual verification is required.

## Commit & Pull Request Guidelines
- Write imperative, descriptive commit subjects (`Add table cleanup step`) and prefix optional scope tags in parentheses when helpful.
- Squash noise commits locally; each PR should summarize the workflow impact and link back to the motivating issue or request.
- Include before/after snippets or line counts in the PR description so reviewers can gauge formatting improvements quickly.

## Security & Configuration Tips
- Never run the converter on untrusted PDFs without sandboxing; note that embedded scripts are stripped, but metadata remains.
- Declare any new external dependency (e.g., Poppler, BeautifulSoup) in `README.md` and provide install commands for macOS and Ubuntu.
- Audit temporary file handling when adding features; prefer in-memory processing or `mktemp` with restricted permissions.
