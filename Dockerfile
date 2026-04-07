FROM ghcr.io/meta-pytorch/openenv-base:latest AS builder

WORKDIR /app

COPY . /app/env

WORKDIR /app/env

RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "openenv-core[core]>=0.2.2" \
    "fastapi>=0.115.0" \
    "uvicorn>=0.24.0" \
    "pydantic>=2.0.0" \
    "requests>=2.31.0" \
    "openai>=1.0.0"

FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

COPY --from=builder /app/env /app/env
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

ENV PYTHONPATH="/app/env:/app/env/server:${PYTHONPATH}"
ENV ENABLE_WEB_INTERFACE=true

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

EXPOSE 7860

CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 7860"]