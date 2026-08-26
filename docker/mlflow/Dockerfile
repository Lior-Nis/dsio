# The official ghcr.io/mlflow/mlflow image ships without psycopg2 (mlflow#9513), so an
# unmodified image cannot reach a Postgres backend store. Bake psycopg2-binary in rather
# than pip-installing it at container start, so the image is self-contained and start-up
# does not depend on reaching PyPI.
#
# Pinned to v3.10.0 (not `latest`) so this stack does not silently change under a future
# `docker compose build` — the tag is the newest v3.x release verified to exist on ghcr.io
# at the time this file was written (2026-08-25); bump it deliberately, not by drift.
FROM ghcr.io/mlflow/mlflow:v3.10.0

RUN pip install --no-cache-dir psycopg2-binary
