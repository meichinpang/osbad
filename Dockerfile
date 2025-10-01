# ---------- Builder ----------
FROM python:3.12-slim AS builder

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git build-essential pkg-config python3-dev \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv
WORKDIR /app

# Indexes: PyPI primary; CPU-only torch as extra; allow best match across indexes
ENV UV_INDEX_URL=https://pypi.org/simple
ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV UV_INDEX_STRATEGY=unsafe-best-match
ENV UV_PYTHON_PREFERENCE=only-system
ENV UV_LINK_MODE=copy
ENV UV_HTTP_RETRIES=5 UV_HTTP_TIMEOUT=120 UV_CONCURRENCY=4

# Manifests + VCS (for dynamic versioning)
COPY pyproject.toml uv.lock* ./
COPY .git ./.git

# Source (needed to build the wheel)
COPY . .

# Create venv and install deps + project as NON-EDITABLE (cleaner runtime)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv .venv \
 && uv sync --no-dev --no-editable

# ---------- Runtime ----------
FROM python:3.12-slim AS final
WORKDIR /app

COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/main.py ./main.py

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

CMD ["python", "main.py"]

# ---------- Notebook (Jupyter) ----------
FROM builder AS notebook
WORKDIR /app

RUN uv pip install --python .venv/bin/python --no-cache-dir jupyterlab

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8888

# Jupyter reads JUPYTER_TOKEN at runtime; no token flag here
CMD ["jupyter","lab","--ip=0.0.0.0","--no-browser","--allow-root","--ServerApp.root_dir=/app","--LabApp.default_url=/lab/tree/osbad%23/machine_learning/baseline_models/ml_01_isolation_forest.ipynb"]
