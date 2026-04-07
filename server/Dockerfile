FROM ghcr.io/meta-pytorch/openenv-base:latest AS builder

WORKDIR /app
COPY . /app/env
WORKDIR /app/env

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

RUN if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app/env

COPY --from=builder /app/env /app/env

ENV PATH="/app/env/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:/app/env/server:${PYTHONPATH}"
ENV ENABLE_WEB_INTERFACE=true

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

EXPOSE 7860

CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 7860"]