# syntax=docker/dockerfile:1.7
#
# idtrackerai validator server: the default install (`pip install .`: Group viewer, Annotate,
# Drive exchange and the analysis tabs that need nothing else), with rclone and ffmpeg.
#
# Build (flyhostel's submodules include private shaliulab repositories cloned over ssh: your ssh
# agent is forwarded for that one step, no key or token ends up in the image):
#
#   eval $(ssh-agent) && ssh-add ~/.ssh/id_rsa      # unless an agent already holds the key
#   DOCKER_BUILDKIT=1 docker build --ssh default \
#       --build-arg VALIDATOR_COMMIT=$(git rev-parse --short HEAD) -t idtrackerai-validator .
#
# Run: see docker-compose.yml and "Install with Docker" in the README.
#
# Build arguments:
#   TORCH_INDEX        pip index of torch / torchvision (default: CPU builds, ~1 GB smaller than CUDA).
#                      For a GPU image: https://download.pytorch.org/whl/cu128 (run with --gpus all).
#   TORCH_VERSION, TORCHVISION_VERSION   the versions the lab's server runs.
#   RCLONE_VERSION     rclone release (the Debian package is too old to sign in to Google Drive).
#   UID, GID           user the server runs as: the owner of the files it writes into the mounted
#                      data folder (match your own: --build-arg UID=$(id -u) --build-arg GID=$(id -g)).
#   VALIDATOR_COMMIT, VALIDATOR_COMMIT_DATE   shown in the app bar (the image has no .git).

ARG PYTHON_VERSION=3.10

# ---------------------------------------------------------------------------------------------
# 1. The web interface, built with node from the client commit setup.py pins
# ---------------------------------------------------------------------------------------------
FROM node:20-bookworm-slim AS frontend
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY setup.py /tmp/setup.py
# CLIENT_COMMIT empty: the "idtrackerai-validator-client" pin of setup.py
ARG CLIENT_COMMIT=""
RUN commit="${CLIENT_COMMIT:-$(sed -n 's/^ *"idtrackerai-validator-client": "\([0-9a-f]\{40\}\)".*/\1/p' /tmp/setup.py)}" \
    && test -n "$commit" \
    && git init -q . \
    && git fetch -q --depth 1 https://github.com/shaliulab/idtrackerai-validator-client.git "$commit" \
    && git checkout -q FETCH_HEAD \
    && echo "$commit" > /CLIENT_COMMIT
RUN --mount=type=cache,target=/root/.npm npm ci --no-audit --no-fund
RUN npm run build && test -f build/index.html

# ---------------------------------------------------------------------------------------------
# 2. The Python packages, in a virtualenv copied into the final image
# ---------------------------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
        git openssh-client build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# GitHub's host key, checked against the fingerprint GitHub publishes
# (https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints)
RUN mkdir -p -m 0700 /root/.ssh \
    && ssh-keyscan -t ed25519 github.com > /root/.ssh/known_hosts 2>/dev/null \
    && ssh-keygen -lf /root/.ssh/known_hosts | grep -q 'SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU' \
    || { echo "github.com host key does not match GitHub's published fingerprint" >&2; exit 1; }

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN --mount=type=cache,target=/root/.cache/pip pip install --upgrade pip setuptools wheel

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.13.0
ARG TORCHVISION_VERSION=0.28.0
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url "$TORCH_INDEX" "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

WORKDIR /src
COPY . .
COPY --from=frontend /src/build idtrackerai_validator_server/frontend
COPY --from=frontend /CLIENT_COMMIT idtrackerai_validator_server/frontend/CLIENT_COMMIT
# flyhostel's submodules are cloned over ssh (the forwarded agent); GIT_SSH_COMMAND keeps the
# known_hosts check strict. Not editable: the image holds only the installed packages.
# idtrackerai requires opencv-python, flyhostel and this server the headless build: both unpack into
# the same cv2/ folder. Keep only the headless one (same version, no libGL / X libraries needed).
RUN --mount=type=ssh --mount=type=cache,target=/root/.cache/pip \
    GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=yes" pip install . \
    && cv="$(pip show opencv-python-headless | sed -n 's/^Version: //p')" \
    && pip uninstall -y opencv-python opencv-python-headless \
    && pip install --no-deps "opencv-python-headless==${cv}" \
    && python -c "import cv2, flyhostel, idtrackerai, imgstore, idtrackerai_validator_server; print('cv2', cv2.__version__)"

# ---------------------------------------------------------------------------------------------
# 3. Runtime
# ---------------------------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm

# ffmpeg (libx264: minimal bundles, sleep bout / PE videos); git for flyhostel's GitPython import.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git ca-certificates curl unzip tini \
    && rm -rf /var/lib/apt/lists/*

ARG RCLONE_VERSION=1.71.2
RUN set -eu; arch="$(dpkg --print-architecture)"; \
    cd /tmp; \
    curl -fsSLO "https://downloads.rclone.org/v${RCLONE_VERSION}/rclone-v${RCLONE_VERSION}-linux-${arch}.zip"; \
    curl -fsSL "https://downloads.rclone.org/v${RCLONE_VERSION}/SHA256SUMS" | grep " rclone-v${RCLONE_VERSION}-linux-${arch}.zip\$" | sha256sum -c -; \
    unzip -q "rclone-v${RCLONE_VERSION}-linux-${arch}.zip"; \
    install -m 0755 "rclone-v${RCLONE_VERSION}-linux-${arch}/rclone" /usr/local/bin/rclone; \
    rm -rf /tmp/rclone-*; \
    rclone version | head -1

COPY --from=builder /opt/venv /opt/venv

ARG UID=1000
ARG GID=1000
RUN groupadd -g "$GID" validator && useradd -m -u "$UID" -g "$GID" -s /bin/bash validator \
    && mkdir -p /data /home/validator/.config /home/validator/.cache \
    && chown validator:validator /data /home/validator/.config /home/validator/.cache

ARG VALIDATOR_COMMIT=""
ARG VALIDATOR_COMMIT_DATE=""
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HOME=/home/validator \
    FLYHOSTEL_VIDEOS=/data \
    BACKEND_HOST=0.0.0.0 \
    BACKEND_PORT=5000 \
    VALIDATOR_COMMIT=${VALIDATOR_COMMIT} \
    VALIDATOR_COMMIT_DATE=${VALIDATOR_COMMIT_DATE}

USER validator
# The app's working directory: it wipes and recreates ./frames at start
WORKDIR /home/validator
# /data: $FLYHOSTEL_VIDEOS (experiments, index.txt; the Annotate tab's validator_annotations.db
# lives in each experiment folder). ~/.config: login secret key, Drive exchange settings,
# rclone.conf (or mount the host's ~/.config/rclone there). ~/.cache: QC / continuity caches.
VOLUME ["/data", "/home/validator/.config", "/home/validator/.cache"]
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import os,urllib.request,ssl; s='https' if os.environ.get('VALIDATOR_CERTFILE') else 'http'; urllib.request.urlopen(s+'://127.0.0.1:'+os.environ.get('BACKEND_PORT','5000')+'/api/auth', timeout=4, context=ssl._create_unverified_context())" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "start-idtrackerai-validator-server", "--no-browser"]
