FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.4.1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONPATH=/app/src

# Allow build/runtime to set ENV (e.g., 'prod') without baking USE_S3 into the image.
ARG ENV=dev
ENV ENV=${ENV}

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ffmpeg \
    curl \
    build-essential \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

WORKDIR /model

COPY pyproject.toml poetry.lock ./

RUN python3.12 -m pip install --no-cache-dir "poetry==${POETRY_VERSION}" \
    && python3.12 -m poetry install --only main --no-root

COPY src ./src
COPY config ./config

ENTRYPOINT ["python3.12", "src/__main__.py"]
CMD ["--help"]