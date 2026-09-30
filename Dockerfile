FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/app/.cache/huggingface \
    PATH="/app/.venv/bin:$PATH"

# Dependencies first so code changes don't reinstall torch.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

# Bake the corpus, embedding model and index into the image so startup is fast
# and the container needs no network access except to the LLM.
RUN docsage fetch && docsage ingest

# Hugging Face Spaces and most PaaS hosts set PORT.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn docsage.api:app --host 0.0.0.0 --port ${PORT}"]
