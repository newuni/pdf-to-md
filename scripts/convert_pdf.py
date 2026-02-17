#!/usr/bin/env python3
"""
pdf-to-md - convenience script

Run the CLI directly from a git checkout without installing:

  python3 scripts/convert_pdf.py input/Doc.pdf output/Doc.md
"""

import os
import sys


repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

from pdf_to_md.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
