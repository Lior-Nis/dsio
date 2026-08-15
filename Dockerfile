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
# Workspace members must be present for the lockfile to resolve.
COPY pyproject.toml uv.lock README.md ./
COPY lib/dsio/pyproject.toml lib/dsio/README.md ./lib/dsio/
RUN uv sync --locked --no-install-project

COPY lib/ ./lib/
COPY src/ ./src/
RUN uv sync --locked

ENV DSIO_RUNS_ROOT=/data/runs \
    DSIO_REGISTRY_ROOT=/data/models \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["uv", "run", "dsio"]
CMD ["--help"]
