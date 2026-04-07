# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Install openenv-core from PyPI + all dependencies
RUN pip install --no-cache-dir \
    "openenv-core[core]>=0.2.2" \
    "fastapi>=0.115.0" \
    "uvicorn>=0.24.0" \
    "pydantic>=2.0.0" \
    "requests>=2.31.0" \
    "openai>=1.0.0"

# Copy environment code into /app/env
COPY . /app/env

# Set working directory to env root so `from models import ...` and
# `from server.X import ...` both resolve without relative imports
WORKDIR /app/env

ENV PYTHONPATH="/app/env:/app/env/server:${PYTHONPATH}"

# Health check against the OpenEnv standard /health route
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]