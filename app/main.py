from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response

from ocr.pipeline import run_ocr


APP_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = APP_DIR / "web"


app = FastAPI(title="OCR Agent (LLM Disabled)")


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


def _shape_response(filename: str, raw_json: Dict[str, Any]) -> Dict[str, Any]:
    # formatted extracted text with header/footer/table tags is in structured.cleaned_text_full
    structured = raw_json.get("structured") or {}
    formatted_text = structured.get("cleaned_text_full") or ""

    return {
        "filename": filename,
        "formatted_text": formatted_text,
        "raw_json": raw_json,
        "llm_json": {},  # placeholder (LLM disabled)
    }


@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    engine: str = Query(default="auto"),
    preset: str = Query(default="auto"),
    dpi: int = Query(default=200, ge=72, le=600),
    max_pages: int = Query(default=5, ge=1, le=200),
    return_debug: bool = Query(default=True),
    return_layout: bool = Query(default=True),
):
    suffix = Path(file.filename or "upload").suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)

    try:
        raw = run_ocr(
            input_path=str(tmp_path),
            preset=preset,
            dpi=dpi,
            max_pages=max_pages,
            return_debug=return_debug,
            engine=engine,
            return_layout=return_layout,
        )
        return JSONResponse(content=_shape_response(file.filename or "upload", raw))
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
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Batch limit is 10 files. Please upload max 10.")

    results: List[Dict[str, Any]] = []
    tmp_paths: List[Path] = []

    try:
        for f in files:
            suffix = Path(f.filename or "upload").suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await f.read())
                tmp_paths.append(Path(tmp.name))

        for tp, f in zip(tmp_paths, files):
            raw = run_ocr(
                input_path=str(tp),
                preset=preset,
                dpi=dpi,
                max_pages=max_pages,
                return_debug=return_debug,
                engine=engine,
                return_layout=return_layout,
            )
            results.append(_shape_response(f.filename or "upload", raw))

        return JSONResponse(content={"count": len(results), "results": results})
    finally:
        for p in tmp_paths:
            try:
                p.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
