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
    p.add_argument("input_pdf", help="Input PDF path (or filename under ./input/).")
    p.add_argument("output_md", nargs="?", help="Output Markdown path (default: ./output/<stem>.md).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        input_pdf = resolve_input_path(args.input_pdf)
    except FileNotFoundError:
        return _die(f"input file '{args.input_pdf}' does not exist")

    backend = args.backend
    if backend == "auto":
        backend = choose_backend_auto()
        _warn(f"auto selected backend '{backend}'")

    output_md = Path(args.output_md) if args.output_md else default_output_path(input_pdf)
    if output_md.exists() and not args.force:
        return _die(f"output file '{output_md}' already exists (use --force)")

    try:
        if backend == "poppler":
            md = convert_with_poppler(input_pdf)
        elif backend == "pymupdf4llm":
            md = convert_with_pymupdf4llm(input_pdf)
        elif backend == "docling":
            md = convert_with_docling(
                input_pdf,
                ocr=args.docling_ocr,
                text_sample_pages=args.text_sample_pages,
                text_threshold=args.text_threshold,
            )
        else:
            return _die(f"unknown backend '{backend}'")
    except subprocess.CalledProcessError as exc:
        return _die(f"backend '{backend}' failed (exit {exc.returncode})")
    except Exception as exc:
        return _die(str(exc))

    _atomic_write_text(output_md, md)
    print(f"Converted '{input_pdf}' -> '{output_md}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
