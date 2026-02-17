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

## Releases

| Version | Date (UTC) | Notes |
|---|---:|---|
| v1.0.0 | 2026-02-17 | Initial stable release |
| v1.1.0 | 2026-02-17 | Cleanup release (removed obsolete legacy scripts) |

## Backlog

| ID | Task | Estado | Notes | Next action |
|---|---|---|---|---|
| P01 | Add tests for `batch` (planning, exclude, suffix, skip rules) | Pendiente | Currently tested manually | Add `tests/` with a few small unit tests (no PDFs needed) |
| P02 | Add `--report jsonl` for batch runs | Pendiente | Useful for automation and audits | Emit one JSON object per file + summary |
| P03 | Add `--mirror` option to preserve directory structure under output dir | Pendiente | Current behavior mirrors relative path per input root; file inputs are flat | Decide UX + implement |
