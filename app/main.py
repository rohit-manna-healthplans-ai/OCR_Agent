from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response

from ocr.pipeline import run_ocr
from ocr.docx_export import build_docx_from_result


APP_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = APP_DIR / "web"
OUT_DIR = APP_DIR / "out"


app = FastAPI(title="OCR Agent")


@app.get("/", response_class=HTMLResponse)
def home():
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/app.js")
def app_js():
    return FileResponse(WEB_DIR / "app.js", media_type="application/javascript")


@app.get("/styles.css")
def styles_css():
    return FileResponse(WEB_DIR / "styles.css", media_type="text/css")


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    engine: str = Query(default="auto"),
    preset: str = Query(default="auto"),
    dpi: int = Query(default=200, ge=72, le=600),
    max_pages: int = Query(default=5, ge=1, le=200),
    return_debug: bool = Query(default=True),
    return_layout: bool = Query(default=True),
    enable_ollama: bool = Query(default=True),
):
    suffix = Path(file.filename or "upload").suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        # Stream to disk in chunks to avoid loading the whole file in memory
        while True:
            chunk = await file.read(1024 * 1024)  # 1MB
            if not chunk:
                break
            tmp.write(chunk)

    try:
        result = run_ocr(
            input_path=str(tmp_path),
            preset=preset,
            dpi=dpi,
            max_pages=max_pages,
            return_debug=return_debug,
            engine=engine,
            return_layout=return_layout,
            enable_ollama=enable_ollama,
        )
        return JSONResponse(content=result)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass


@app.post("/ocr-batch")
async def ocr_batch_endpoint(
    files: List[UploadFile] = File(...),
    engine: str = Query(default="auto"),
    preset: str = Query(default="auto"),
    dpi: int = Query(default=200, ge=72, le=600),
    max_pages: int = Query(default=5, ge=1, le=200),
    return_debug: bool = Query(default=True),
    return_layout: bool = Query(default=True),
    enable_ollama: bool = Query(default=True),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Batch limit is 10 files. Please upload max 10.")

    results: List[Dict[str, Any]] = []
    tmp_paths: List[Path] = []

    try:
        # Save all first
        for f in files:
            suffix = Path(f.filename or "upload").suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await f.read())
                tmp_paths.append(Path(tmp.name))

        # OCR sequential (stable for PaddleOCR singleton on Windows)
        for tp, f in zip(tmp_paths, files):
            r = run_ocr(
                input_path=str(tp),
                preset=preset,
                dpi=dpi,
                max_pages=max_pages,
                return_debug=return_debug,
                engine=engine,
                return_layout=return_layout,
                enable_ollama=enable_ollama,
            )
            results.append({"filename": f.filename, **r})

        return JSONResponse(content={"count": len(results), "results": results})
    finally:
        for p in tmp_paths:
            try:
                p.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass


@app.post("/ocr-docx")
async def ocr_docx_endpoint(
    file: UploadFile = File(...),
    engine: str = Query(default="auto"),
    preset: str = Query(default="auto"),
    dpi: int = Query(default=200, ge=72, le=600),
    max_pages: int = Query(default=5, ge=1, le=200),
):
    suffix = Path(file.filename or "upload").suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_in = Path(tmp.name)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "ocr_result.docx"

    try:
        result = run_ocr(
            input_path=str(tmp_in),
            preset=preset,
            dpi=dpi,
            max_pages=max_pages,
            return_debug=False,
            engine=engine,
            return_layout=False,
            enable_ollama=False,
        )

        build_docx_from_result(result, str(out_path), title="OCR Output")
        return FileResponse(str(out_path), filename="ocr_result.docx")
    finally:
        try:
            tmp_in.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
