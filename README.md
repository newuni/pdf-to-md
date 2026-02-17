# pdf-to-md

CLI para convertir PDFs a Markdown con varios backends:

- `poppler` (externo): `pdftohtml`/`pdftotext` + limpieza.
- `pymupdf4llm` (Python): Markdown directo via PyMuPDF4LLM.
- `docling` (CLI/Python): Markdown con mejor estructura/layout.
- `auto`: elige `docling` si esta disponible, si no `pymupdf4llm`, si no `poppler`.

## Estructura

- `input/`: PDFs de entrada (internos, no versionar si son grandes).
- `output/`: Markdown generado (artefactos).
- `reference/`: Markdown curado (goldens esperados).
- `samples/`: PDFs pequenos para regresion.

## Uso

Sin instalar (desde el repo):

```bash
PYTHONPATH=./src python3 -m pdf_to_md --backend docling input/Documento.pdf output/Documento.docling.md
```

Wrapper legacy:

```bash
./legacy/pdf_to_md.sh --backend auto input/Documento.pdf
```

## Instalacion (recomendada)

Editable:

```bash
python3 -m pip install -e .
```

Con backends opcionales:

```bash
python3 -m pip install -e ".[docling,pymupdf4llm]"
```

## Notas

- `docling` y `poppler` dependen de binarios/entorno; si falta algo el CLI dara un error claro.
- Para PDFs escaneados, mas adelante se puede integrar un paso OCR (por ahora solo se detecta y se avisa).
