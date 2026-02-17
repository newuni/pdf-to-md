FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=1

RUN apt-get update \
  && apt-get install -y --no-install-recommends poppler-utils \
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

