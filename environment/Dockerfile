FROM python:3.11-slim-bookworm@sha256:f06d01d4ba3f8ad5af45a3ec87df19c1d317d8e8961edc0fabda61811af6f7cf

RUN apt-get update \
    && apt-get install -y --no-install-recommends tmux=3.3a-3 procps=2:4.0.2-3 asciinema=2.2.0-1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PATH="/opt/venv/bin:${PATH}" \
    HOME=/home/candidate

COPY requirements.lock /tmp/requirements.lock
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.lock \
    && useradd --uid 1000 --create-home --shell /bin/sh candidate \
    && mkdir -p /app/src \
    && chown -R root:root /opt/venv \
    && chmod -R go-w /opt/venv \
    && chmod 1777 /tmp /var/tmp

WORKDIR /app
COPY src/ /app/src/
COPY run_case.py /app/run_case.py
RUN chown -R root:root /app \
    && chmod -R a=rX /app \
    && chown -R candidate:candidate /app/src \
    && chmod -R u+rwX /app/src

USER root
