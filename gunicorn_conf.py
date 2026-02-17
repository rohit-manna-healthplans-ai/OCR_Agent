import os

bind = f"0.0.0.0:{os.environ.get('PORT','8000')}"
worker_class = "uvicorn_worker.UvicornWorker"  # recommended replacement for uvicorn.workers.UvicornWorker
workers = int(os.environ.get("WORKERS", "2"))
timeout = int(os.environ.get("TIMEOUT", "180"))
graceful_timeout = int(os.environ.get("GRACEFUL_TIMEOUT", "60"))
keepalive = int(os.environ.get("KEEPALIVE", "5"))
loglevel = os.environ.get("LOG_LEVEL", "info")
accesslog = "-"
errorlog = "-"
