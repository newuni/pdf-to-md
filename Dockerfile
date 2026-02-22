FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=1 \
  USER=pdf2md \
  LOGNAME=pdf2md \
  HOME=/tmp \
  XDG_CACHE_HOME=/tmp/.cache \
  TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-spa \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the package first for better layer caching.
COPY pyproject.toml README.md LICENSE /app/
COPY pdf_to_md /app/pdf_to_md

ARG EXTRAS=""
RUN python -m pip install -U pip \
  && if [ -n "$EXTRAS" ]; then python -m pip install ".[${EXTRAS}]"; else python -m pip install .; fi

ENTRYPOINT ["pdf-to-md"]
CMD ["--help"]
