# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# uv for reproducible install (lockfile pinned)
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /usr/local/bin/

WORKDIR /app

# Cache deps layer; sistem Python'u kullan, /opt/venv altinda olustur
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --python /usr/local/bin/python3.12

# App code + artefacts
COPY src ./src
COPY models ./models

# Non-root user (OWASP A05: secure defaults)
RUN groupadd --system app && useradd --system --gid app --home /app --no-create-home app \
    && chown -R app:app /app
USER app

ENV PATH="/opt/venv/bin:$PATH" \
    MODEL_PATH=/app/models/best_model.joblib \
    PREPROCESSOR_PATH=/app/models/preprocessor.joblib \
    POLICY_PATH=/app/models/decision_policy.json \
    LOG_LEVEL=INFO \
    PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3).status==200 else 1)"

CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2", "--proxy-headers"]