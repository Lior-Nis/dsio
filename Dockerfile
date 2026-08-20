# Local/cloud parity image.
#
# git is installed deliberately, not incidentally: provenance capture shells out to it,
# and without it every run in this image records code_hash=None and becomes unpromotable.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer first so source edits do not invalidate the install.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --extra cpu --no-install-project

COPY src/ ./src/
RUN uv sync --locked --extra cpu

ENV DSIO_RUNS_ROOT=/data/runs \
    DSIO_REGISTRY_ROOT=/data/models \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["uv", "run", "--extra", "cpu", "dsio"]
CMD ["--help"]
