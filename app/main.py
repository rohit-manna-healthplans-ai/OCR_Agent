from __future__ import annotations

import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ocr.pipeline import run_ocr

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"

app = FastAPI(title="Production OCR Engine (English, PaddleOCR)")

app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/app.js")
def app_js():
    return FileResponse(str(WEB_DIR / "app.js"), media_type="application/javascript")


@app.get("/styles.css")
def styles_css():
    return FileResponse(str(WEB_DIR / "styles.css"), media_type="text/css")


@app.get("/favicon.ico")
def favicon():
    return JSONResponse(status_code=204, content=None)


@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    preset: str = Query(default="auto"),
    dpi: int = Query(default=300, ge=72, le=600),
    max_pages: int = Query(default=10, ge=1, le=200),
    return_debug: bool = Query(default=False),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in [".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"]:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported file type: {suffix}. Upload PDF or image."},
        )

    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td) / (file.filename or f"upload{suffix}")
        content = await file.read()
        tmp_path.write_bytes(content)

        try:
            result = run_ocr(
                input_path=str(tmp_path),
                preset=preset,
                dpi=dpi,
                max_pages=max_pages,
                return_debug=return_debug,
            )
            return JSONResponse(content=result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content={"error": str(e), "type": type(e).__name__},
            )
