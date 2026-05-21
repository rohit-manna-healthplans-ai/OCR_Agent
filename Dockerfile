FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    FLAGS_use_mkldnn=0 \
    FLAGS_enable_mkldnn=0 \
    FLAGS_use_dnnl=0 \
    PORT=8000 \
    WORKERS=1

# Runtime libs for OpenCV + PaddlePaddle on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# IMPORTANT: use requirements-docker.txt (Linux).
# Do NOT use requirements.txt here — it contains Windows-only Paddle wheel links.
COPY requirements-docker.txt /app/requirements-docker.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -r /app/requirements-docker.txt

COPY . /app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"

CMD ["gunicorn", "-c", "gunicorn_conf.py", "app.main:app"]
