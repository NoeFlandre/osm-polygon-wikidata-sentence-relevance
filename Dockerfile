# syntax=docker/dockerfile:1.10

# Both images are pinned to multi-platform manifest digests. The digest keeps
# rebuilds on the same published image set while still allowing Docker to pick
# the host architecture (amd64 or arm64).
ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:a116514e19457bcb7af7efe9c3dd0b9b71e85b317694e7882a1c52aa15a78134
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.16@sha256:440fd6477af86a2f1b38080c539f1672cd22acb1b1a47e321dba5158ab08864d

FROM ${PYTHON_IMAGE} AS runtime-base

# Copy the pinned uv binary rather than installing a mutable package-manager
# version during the build.
COPY --from=${UV_IMAGE} /uv /uvx /usr/local/bin/

ENV PATH="/opt/venv/bin:${PATH}" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /opt/venv /home/app/.cache/huggingface \
    && chown -R app:app /opt/venv /home/app

# Keep dependency installation in a separate layer so source-only changes do
# not redownload the lockfile's wheels. The default extras cover the public
# pipeline, Hub publication, Grid'5000 operator, segmentation, and tracking
# commands. A smaller image can opt into a subset with --build-arg
# UV_EXTRAS="--extra hub --extra operator".
ARG UV_EXTRAS="--extra hub --extra operator --extra segmentation --extra tracking"
COPY --chown=app:app pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --locked --no-dev --no-install-project ${UV_EXTRAS}

COPY --chown=app:app src ./src
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --locked --no-dev ${UV_EXTRAS}

FROM runtime-base AS runtime

USER app

# The default is a harmless, deterministic CLI smoke path. Supplying a
# command to `docker run` replaces it.
CMD ["osm-polygon-sentence-relevance", "--help"]

FROM runtime-base AS test

# The test target intentionally includes development tools and test fixtures;
# it is separate from the smaller runtime image.
USER root
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --locked --all-extras --dev
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app tests ./tests
USER app
CMD ["pytest", "-q"]
