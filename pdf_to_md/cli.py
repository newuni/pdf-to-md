from __future__ import annotations

import argparse
import contextlib
import html as html_mod
import importlib.util
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from unicodedata import normalize

from . import __version__


def _which(cmd: str) -> str | None:
    return shutil.which(cmd)


def _die(msg: str, code: int = 2) -> int:
    print(f"Error: {msg}", file=sys.stderr)
    return code


def _warn(msg: str) -> None:
    print(f"Warning: {msg}", file=sys.stderr)


def _read_stdin_text() -> str:
    return sys.stdin.read()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=str(path.parent)
    ) as tmp:
        tmp.write(text)
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def _cleanup_poppler_html(raw: str) -> str:
    text = re.sub(r"(?i)<br\s*/?>", "\n", raw)
    text = re.sub(r"(?i)<hr\s*/?>", "\n---\n", text)
    text = re.sub(r"<a[^>]*></a>", "", text)
    text = re.sub(r"</?(?:b|i|strong|em|u|span|font|p)[^>]*>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html_mod.unescape(text)
    return text


def _cleanup_poppler_text(raw: str) -> str:
    return raw.replace("\f", "\n\n---\n\n")


def _cleanup_common(text: str) -> str:
    text = normalize("NFKC", text)
    lines: list[str] = []
    previous_blank = False

    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = line.strip()
        if not stripped:
            if not previous_blank:
                lines.append("")
            previous_blank = True
            continue

        previous_blank = False

        # Drop isolated page numbers / footers commonly produced by poppler tools.
        if stripped.isdigit() or stripped.lower() in {"f", "ff"}:
            continue

        clean = stripped.replace("\u2022", "- ").replace("\u2023", "- ")
        clean = re.sub(r"\s{2,}", " ", clean)
        lines.append(clean)

    out = "\n".join(lines).strip()
    if out:
        out += "\n"
    return out


def convert_with_poppler(input_pdf: Path) -> str:
    if _which("pdftohtml"):
        proc = subprocess.run(
            ["pdftohtml", "-q", "-i", str(input_pdf), "-stdout"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        raw = proc.stdout.decode("utf-8", errors="replace")
        text = _cleanup_poppler_html(raw)
        return _cleanup_common(text)

    if _which("pdftotext"):
        proc = subprocess.run(
            ["pdftotext", "-layout", str(input_pdf), "-"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        raw = proc.stdout.decode("utf-8", errors="replace")
        text = _cleanup_poppler_text(raw)
        return _cleanup_common(text)

    raise RuntimeError("neither pdftohtml nor pdftotext is installed")


@contextlib.contextmanager
def _suppress_output_fds() -> io.StringIO:
    """
    Suppress C-level prints to stdout/stderr while running a library call.
    Returns a buffer with captured Python-level stdout (best-effort).
    """

    buf = io.StringIO()
    devnull = None
    saved_stdout_fd = None
    saved_stderr_fd = None
    try:
        devnull = open(os.devnull, "w", encoding="utf-8")
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        with contextlib.redirect_stdout(buf):
            yield buf
    finally:
        try:
            if saved_stdout_fd is not None:
                os.dup2(saved_stdout_fd, 1)
            if saved_stderr_fd is not None:
                os.dup2(saved_stderr_fd, 2)
        finally:
            if saved_stdout_fd is not None:
                os.close(saved_stdout_fd)
            if saved_stderr_fd is not None:
                os.close(saved_stderr_fd)
            if devnull is not None:
                devnull.close()


def _strip_pymupdf4llm_advisory(md: str) -> str:
    md = re.sub(
        r"\A\s*Consider using the pymupdf_layout package for a greatly improved "
        r"page layout analysis\.\s*\n",
        "",
        md,
    )
    return md


def convert_with_pymupdf4llm(input_pdf: Path) -> str:
    with _suppress_output_fds():
        try:
            import pymupdf4llm  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "missing Python dependency 'pymupdf4llm' "
                "(install with: python3 -m pip install -U pymupdf4llm)"
            ) from exc

        md = pymupdf4llm.to_markdown(str(input_pdf), page_separators=True)

    md = _strip_pymupdf4llm_advisory(md)
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    md = normalize("NFKC", md)

    # Trim trailing whitespace but keep indentation (tables/code blocks).
    md = re.sub(r"[ \t]+(\n)", r"\1", md)

    md = md.strip()
    if md:
        md += "\n"
    return md


def _count_text_chars_first_pages(input_pdf: Path, pages: int) -> int | None:
    if _which("pdftotext"):
        try:
            proc = subprocess.run(
                ["pdftotext", "-f", "1", "-l", str(pages), "-q", str(input_pdf), "-"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return None

        text = proc.stdout.decode("utf-8", errors="replace")
        return sum(1 for ch in text if not ch.isspace())

    # Best-effort fallback via PyMuPDF if available.
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception:
        return None

    try:
        doc = fitz.open(str(input_pdf))
        count = 0
        for i in range(min(pages, doc.page_count)):
            txt = doc.load_page(i).get_text("text") or ""
            count += sum(1 for ch in txt if not ch.isspace())
        return count
    except Exception:
        return None


def pdf_seems_scanned(input_pdf: Path, pages: int, threshold: int) -> bool:
    count = _count_text_chars_first_pages(input_pdf, pages)
    if count is None:
        # Unknown: assume "not scanned" to avoid forcing OCR by default.
        return False
    return count < threshold


def convert_with_docling(
    input_pdf: Path,
    *,
    ocr: bool | None,
    text_sample_pages: int,
    text_threshold: int,
) -> str:
    if not _which("docling"):
        raise RuntimeError(
            "missing command 'docling' (install with: python3 -m pip install -U docling)"
        )

    if ocr is None:
        if pdf_seems_scanned(input_pdf, pages=text_sample_pages, threshold=text_threshold):
            _warn("PDF seems image-only (scanned). Running docling with OCR enabled.")
            ocr = True
        else:
            ocr = False

    with tempfile.TemporaryDirectory(prefix="pdf-to-md-docling-") as outdir:
        cmd = [
            "docling",
            "--from",
            "pdf",
            "--to",
            "md",
            "--output",
            outdir,
            "--image-export-mode",
            "placeholder",
            "--no-enable-remote-services",
            "--ocr" if ocr else "--no-ocr",
            str(input_pdf),
        ]

        subprocess.run(cmd, check=True)

        md_files = sorted(Path(outdir).glob("*.md"))
        if len(md_files) != 1:
            raise RuntimeError(
                f"docling produced {len(md_files)} Markdown files in '{outdir}' (expected 1)"
            )

        return md_files[0].read_text(encoding="utf-8", errors="replace")


def choose_backend_auto() -> str:
    if _which("docling"):
        return "docling"
    if importlib.util.find_spec("pymupdf4llm") is not None:
        return "pymupdf4llm"
    return "poppler"


def resolve_input_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p

    # Convenience: allow passing just the filename when it exists under input/.
    alt = Path("input") / path_str
    if alt.exists():
        return alt

    raise FileNotFoundError(path_str)


def default_output_path(input_pdf: Path) -> Path:
    return Path("output") / f"{input_pdf.stem}.md"


@dataclass(frozen=True)
class BatchResult:
    pdf: Path
    out: Path
    status: str  # converted|skipped|failed
    backend: str
    message: str = ""


def _normalize_argv_for_subcommands(argv: list[str]) -> list[str]:
    """
    Support both subcommand usage and the legacy single-file form.

    Subcommands:
      pdf-to-md convert INPUT.pdf [OUTPUT.md]
      pdf-to-md batch <DIR|PDF>... [batch options]

    Legacy (still supported):
      pdf-to-md [--backend ...] INPUT.pdf [OUTPUT.md]
    """
    if not argv:
        return ["--help"]

    # If user asks for top-level help, keep as-is.
    if any(a in {"-h", "--help"} for a in argv):
        return argv

    # Find first non-option token (argparse treats it as COMMAND).
    first_pos = None
    for i, tok in enumerate(argv):
        if tok == "--":
            if i + 1 < len(argv):
                first_pos = i + 1
            break
        if not tok.startswith("-"):
            first_pos = i
            break

    if first_pos is None:
        return argv

    if argv[first_pos] in {"convert", "batch"}:
        return argv

    return argv[:first_pos] + ["convert"] + argv[first_pos:]


def _iter_pdfs_in_dir(
    root: Path,
    *,
    recursive: bool,
    glob: str,
    exclude: list[str],
    skip_under: list[Path],
) -> list[Path]:
    root = root.resolve()
    skip_under_resolved = [p.resolve() for p in skip_under]

    glob_l = glob.lower()
    exclude_l = [e.lower() for e in exclude]
    match_full_path = "/" in glob_l or "\\" in glob_l

    def is_under_any(p: Path) -> bool:
        pr = p.resolve()
        for s in skip_under_resolved:
            try:
                pr.relative_to(s)
                return True
            except ValueError:
                continue
        return False

    def is_excluded(rel_posix: str, name: str) -> bool:
        for pat in exclude_l:
            if "/" in pat or "\\" in pat:
                if fnmatch(rel_posix, pat):
                    return True
            else:
                if fnmatch(name, pat):
                    return True
        return False

    it = root.rglob("*") if recursive else root.glob("*")
    pdfs: list[Path] = []

    for p in it:
        if not p.is_file():
            continue
        if is_under_any(p):
            continue
        rel_posix = p.relative_to(root).as_posix().lower()
        name = p.name.lower()

        if match_full_path:
            if not fnmatch(rel_posix, glob_l):
                continue
        else:
            if not fnmatch(name, glob_l):
                continue

        if exclude_l and is_excluded(rel_posix, name):
            continue

        if p.suffix.lower() != ".pdf":
            continue

        pdfs.append(p)

    pdfs.sort()
    return pdfs


def _convert_one(
    *,
    input_pdf: Path,
    output_md: Path,
    backend: str,
    force: bool,
    docling_ocr: bool | None,
    text_sample_pages: int,
    text_threshold: int,
) -> None:
    if output_md.exists() and not force:
        raise FileExistsError(str(output_md))

    if backend == "poppler":
        md = convert_with_poppler(input_pdf)
    elif backend == "pymupdf4llm":
        md = convert_with_pymupdf4llm(input_pdf)
    elif backend == "docling":
        md = convert_with_docling(
            input_pdf,
            ocr=docling_ocr,
            text_sample_pages=text_sample_pages,
            text_threshold=text_threshold,
        )
    else:
        raise RuntimeError(f"unknown backend '{backend}'")

    _atomic_write_text(output_md, md)


def build_parser() -> argparse.ArgumentParser:
    default_backend = os.environ.get("PDF_TO_MD_BACKEND", "poppler")
    p = argparse.ArgumentParser(
        prog="pdf-to-md",
        description=(
            "Convert PDF to Markdown using poppler / pymupdf4llm / docling.\n"
            "\n"
            "Convenience behavior:\n"
            "- If INPUT does not exist, it is also searched under ./input/.\n"
            "- If OUTPUT is omitted, it defaults to ./output/<input_stem>.md.\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  pdf-to-md input/Doc.pdf\n"
            "  pdf-to-md --backend docling input/Doc.pdf output/Doc.docling.md\n"
            "  pdf-to-md --backend auto Doc.pdf\n"
            "  pdf-to-md --backend docling --docling-no-ocr Doc.pdf\n"
            "  pdf-to-md batch input --output-dir output --recursive\n"
            "\n"
            "Install backends:\n"
            "  python3 -m pip install -e .\n"
            "  python3 -m pip install -e \".[docling,pymupdf4llm]\"\n"
        ),
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument(
        "--backend",
        choices=["auto", "poppler", "pymupdf4llm", "docling"],
        default=default_backend,
        help=(
            "Backend to use.\n"
            f"- Default: {default_backend!r} (from PDF_TO_MD_BACKEND if set)\n"
            "- auto: docling > pymupdf4llm > poppler (based on availability)\n"
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    p.add_argument(
        "--docling-ocr",
        dest="docling_ocr",
        action="store_true",
        default=None,
        help="Force OCR on for docling (overrides auto detection).",
    )
    p.add_argument(
        "--docling-no-ocr",
        dest="docling_ocr",
        action="store_false",
        default=None,
        help="Force OCR off for docling (overrides auto detection).",
    )
    p.add_argument(
        "--text-sample-pages",
        type=int,
        default=int(os.environ.get("PDF_TO_MD_TEXT_SAMPLE_PAGES", "2")),
        help=(
            "Pages to sample when detecting scanned PDFs (docling auto OCR).\n"
            "Env: PDF_TO_MD_TEXT_SAMPLE_PAGES\n"
        ),
    )
    p.add_argument(
        "--text-threshold",
        type=int,
        default=int(os.environ.get("PDF_TO_MD_TEXT_THRESHOLD", "30")),
        help=(
            "Non-whitespace char threshold below which a PDF is treated as scanned.\n"
            "Env: PDF_TO_MD_TEXT_THRESHOLD\n"
        ),
    )

    sub = p.add_subparsers(dest="command", metavar="COMMAND", required=True)

    convert_p = sub.add_parser(
        "convert",
        help="Convert a single PDF to Markdown (default command).",
    )
    convert_p.add_argument(
        "input_pdf",
        help="Input PDF path (or filename under ./input/).",
    )
    convert_p.add_argument(
        "output_md",
        nargs="?",
        help="Output Markdown path (default: ./output/<input_stem>.md).",
    )

    batch_p = sub.add_parser(
        "batch",
        help="Convert all PDFs found under one or more directories.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  pdf-to-md batch input --output-dir output --recursive\n"
            "  pdf-to-md --backend docling batch /data/pdfs --output-dir /data/md --workers 2\n"
            "  pdf-to-md batch /data --recursive --glob \"*.pdf\" --exclude \"tmp\" --verbose\n"
        ),
    )
    batch_p.add_argument(
        "inputs",
        nargs="+",
        help="Input directories (or individual PDF files).",
    )
    batch_p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory (default: ./output).",
    )
    batch_p.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories (default: false).",
    )
    batch_p.add_argument(
        "--glob",
        default="*.pdf",
        help="Glob pattern (case-insensitive) for PDFs (default: *.pdf).",
    )
    batch_p.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude pattern relative to each input dir (can be repeated).",
    )
    batch_p.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("PDF_TO_MD_WORKERS", "4")),
        help="Max parallel workers (default: 4).",
    )
    batch_p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip PDFs whose output already exists.",
    )
    batch_p.add_argument(
        "--skip-up-to-date",
        action="store_true",
        help="Skip when output exists and is newer than the input PDF.",
    )
    batch_p.add_argument(
        "--suffix-backend",
        action="store_true",
        help="Write outputs as <name>.<backend>.md instead of <name>.md.",
    )
    batch_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the conversion plan without writing outputs.",
    )
    batch_p.add_argument(
        "--verbose",
        action="store_true",
        help="Print one line per file (converted/skipped/failed).",
    )
    return p


def _run_batch(args: argparse.Namespace, *, backend: str) -> int:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Avoid recursively picking up outputs when output_dir is under an input dir.
    skip_under: list[Path] = [output_dir]

    work_items: list[tuple[Path, Path]] = []

    def resolve_batch_input(s: str) -> Path:
        p = Path(s)
        if p.exists():
            return p
        alt = Path("input") / s
        if alt.exists():
            return alt
        return p

    for s in args.inputs:
        inp = resolve_batch_input(s)
        if inp.is_file():
            if inp.suffix.lower() != ".pdf":
                _warn(f"skipping non-PDF file: {inp}")
                continue
            out_name = f"{inp.stem}.md"
            if args.suffix_backend:
                out_name = f"{inp.stem}.{backend}.md"
            work_items.append((inp, output_dir / out_name))
            continue

        if not inp.is_dir():
            _warn(f"skipping missing path: {inp}")
            continue

        base = inp.resolve()
        pdfs = _iter_pdfs_in_dir(
            base,
            recursive=bool(args.recursive),
            glob=str(args.glob),
            exclude=list(args.exclude),
            skip_under=skip_under,
        )
        for pdf in pdfs:
            rel = pdf.relative_to(base)
            out_rel = rel.with_suffix(".md")
            if args.suffix_backend:
                out_rel = out_rel.with_name(f"{rel.stem}.{backend}.md")
            work_items.append((pdf, output_dir / out_rel))

    if not work_items:
        _warn("no PDFs found")
        return 0

    # Deterministic plan order.
    work_items.sort(key=lambda t: str(t[0]))

    planned: list[tuple[Path, Path]] = []
    results: list[BatchResult] = []

    for pdf, out in work_items:
        if out.exists():
            if args.force:
                planned.append((pdf, out))
                continue
            if args.skip_existing:
                results.append(
                    BatchResult(pdf=pdf, out=out, status="skipped", backend=backend, message="exists")
                )
                continue
            if args.skip_up_to_date:
                try:
                    if out.stat().st_mtime >= pdf.stat().st_mtime:
                        results.append(
                            BatchResult(
                                pdf=pdf,
                                out=out,
                                status="skipped",
                                backend=backend,
                                message="up-to-date",
                            )
                        )
                        continue
                except OSError:
                    pass
            results.append(
                BatchResult(
                    pdf=pdf,
                    out=out,
                    status="failed",
                    backend=backend,
                    message="output exists (use --force or --skip-existing)",
                )
            )
            continue

        planned.append((pdf, out))

    if args.dry_run:
        for r in results:
            extra = f" ({r.message})" if r.message else ""
            print(f"{r.status}\t{r.pdf}\t->\t{r.out}{extra}")
        for pdf, out in planned:
            print(f"planned\t{pdf}\t->\t{out}")
        return 0

    workers = max(1, int(args.workers))
    if backend == "docling" and workers > 4:
        _warn("docling backend can be heavy; consider --workers 2-4")

    def task(pdf: Path, out: Path) -> BatchResult:
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            _convert_one(
                input_pdf=pdf,
                output_md=out,
                backend=backend,
                force=bool(args.force),
                docling_ocr=args.docling_ocr,
                text_sample_pages=int(args.text_sample_pages),
                text_threshold=int(args.text_threshold),
            )
            return BatchResult(pdf=pdf, out=out, status="converted", backend=backend)
        except FileExistsError:
            return BatchResult(pdf=pdf, out=out, status="skipped", backend=backend, message="exists")
        except subprocess.CalledProcessError as exc:
            return BatchResult(
                pdf=pdf,
                out=out,
                status="failed",
                backend=backend,
                message=f"backend failed (exit {exc.returncode})",
            )
        except Exception as exc:  # noqa: BLE001
            return BatchResult(pdf=pdf, out=out, status="failed", backend=backend, message=str(exc))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(task, pdf, out): (pdf, out) for pdf, out in planned}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            if args.verbose:
                extra = f" ({r.message})" if r.message else ""
                print(f"{r.status}\t{r.pdf}\t->\t{r.out}{extra}")

    converted = sum(1 for r in results if r.status == "converted")
    failed = sum(1 for r in results if r.status == "failed")
    skipped_n = sum(1 for r in results if r.status == "skipped")

    print(f"Batch complete: converted={converted} skipped={skipped_n} failed={failed}")
    if failed:
        for r in results:
            if r.status == "failed":
                print(f"failed\t{r.pdf}\t->\t{r.out}\t({r.message})", file=sys.stderr)
        return 1

    return 0


def main(argv: list[str] | None = None) -> int:
    argv2 = _normalize_argv_for_subcommands(list(sys.argv[1:] if argv is None else argv))
    args = build_parser().parse_args(argv2)

    backend = args.backend
    if backend == "auto":
        backend = choose_backend_auto()
        _warn(f"auto selected backend '{backend}'")

    if args.command == "batch":
        return _run_batch(args, backend=backend)

    try:
        input_pdf = resolve_input_path(args.input_pdf)
    except FileNotFoundError:
        return _die(f"input file '{args.input_pdf}' does not exist")

    output_md = Path(args.output_md) if args.output_md else default_output_path(input_pdf)

    try:
        _convert_one(
            input_pdf=input_pdf,
            output_md=output_md,
            backend=backend,
            force=bool(args.force),
            docling_ocr=args.docling_ocr,
            text_sample_pages=int(args.text_sample_pages),
            text_threshold=int(args.text_threshold),
        )
    except FileExistsError:
        return _die(f"output file '{output_md}' already exists (use --force)")
    except subprocess.CalledProcessError as exc:
        return _die(f"backend '{backend}' failed (exit {exc.returncode})")
    except Exception as exc:  # noqa: BLE001
        return _die(str(exc))

    print(f"Converted '{input_pdf}' -> '{output_md}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
