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
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY . /app

# Azure App Service sets $PORT
ENV PORT=8000
EXPOSE 8000

CMD ["gunicorn", "-c", "gunicorn_conf.py", "app.main:app"]
