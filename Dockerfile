# Azure-ready container for FastAPI OCR
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    FLAGS_use_mkldnn=0 \
    FLAGS_enable_mkldnn=0 \
    FLAGS_use_dnnl=0

# System deps: tesseract + OpenCV runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Use requirements-docker.txt for Linux (HF Spaces); requirements.txt is for Windows
COPY requirements-docker.txt /app/requirements-docker.txt
RUN pip install -r /app/requirements-docker.txt

COPY . /app

# Hugging Face Spaces use port 7860; gunicorn_conf reads PORT
ENV PORT=7860
EXPOSE 7860

CMD ["gunicorn", "-c", "gunicorn_conf.py", "app.main:app"]
