from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from ocr.pipeline import run_ocr
from ocr.docx_export import build_docx_from_result

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"

app = FastAPI(title="Production OCR Engine (English, Multi-Engine)")

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
    return Response(status_code=204)


def _validate_suffix(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in [".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"]:
        raise ValueError(f"Unsupported file type: {suffix}. Upload PDF or image.")
    return suffix


def _validate_engine(engine: str) -> str:
    eng = (engine or "auto").lower().strip()
    if eng not in {"auto", "paddle", "tesseract"}:
        raise ValueError("engine must be one of: auto, paddle, tesseract")
    return eng


@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    preset: str = Query(default="auto"),
    dpi: int = Query(default=300, ge=72, le=600),
    max_pages: int = Query(default=10, ge=1, le=200),
    engine: str = Query(default="auto"),
    return_debug: bool = Query(default=False),
    return_layout: bool = Query(default=False),
    llm_correct: bool = Query(default=False),
):
    try:
        _validate_suffix(file.filename or "")
        engine = _validate_engine(engine)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td) / (file.filename or "upload.bin")
        tmp_path.write_bytes(await file.read())

        try:
            result = run_ocr(
                input_path=str(tmp_path),
                preset=preset,
                dpi=dpi,
                max_pages=max_pages,
                return_debug=return_debug,
                engine=engine,
                return_layout=return_layout,
                llm_correct=llm_correct,
            )
            result["file_name"] = file.filename or "upload.bin"
            return JSONResponse(content=result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e), "type": type(e).__name__})


@app.post("/ocr-docx")
async def ocr_docx_endpoint(
    file: UploadFile = File(...),
    preset: str = Query(default="auto"),
    dpi: int = Query(default=300, ge=72, le=600),
    max_pages: int = Query(default=10, ge=1, le=200),
    engine: str = Query(default="auto"),
    llm_correct: bool = Query(default=False),
):
    """Returns a .docx with fields, text and detected tables (best-effort)."""
    try:
        _validate_suffix(file.filename or "")
        engine = _validate_engine(engine)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    with tempfile.TemporaryDirectory() as td:
        tmp_in = Path(td) / (file.filename or "upload.bin")
        tmp_in.write_bytes(await file.read())

        try:
            result = run_ocr(
                input_path=str(tmp_in),
                preset=preset,
                dpi=dpi,
                max_pages=max_pages,
                return_debug=False,
                engine=engine,
                return_layout=return_layout,
                llm_correct=llm_correct,
            )
            docx_path = Path(td) / "output.docx"
            build_docx_from_result(result, str(docx_path), title=(file.filename or "OCR Output"))
            out_name = (Path(file.filename or "ocr").stem + ".docx")
            return FileResponse(
                str(docx_path),
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                filename=out_name,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e), "type": type(e).__name__})


@app.post("/ocr-batch")
async def ocr_batch_endpoint(
    files: List[UploadFile] = File(...),
    preset: str = Query(default="auto"),
    dpi: int = Query(default=300, ge=72, le=600),
    max_pages: int = Query(default=10, ge=1, le=200),
    engine: str = Query(default="auto"),
    return_debug: bool = Query(default=False),
    return_layout: bool = Query(default=False),
    llm_correct: bool = Query(default=False),
):
    """Batch OCR: returns JSON list. (No zip)"""
    try:
        engine = _validate_engine(engine)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    results = []
    with tempfile.TemporaryDirectory() as td:
        for f in files:
            try:
                _validate_suffix(f.filename or "")
            except Exception as e:
                results.append({"file_name": f.filename or "unknown", "error": str(e)})
                continue

            tmp_path = Path(td) / (f.filename or "upload.bin")
            tmp_path.write_bytes(await f.read())

            try:
                r = run_ocr(
                    input_path=str(tmp_path),
                    preset=preset,
                    dpi=dpi,
                    max_pages=max_pages,
                    return_debug=return_debug,
                    engine=engine,
                                    return_layout=return_layout,
                    llm_correct=llm_correct,
                )
                r["file_name"] = f.filename or "upload.bin"
                results.append(r)
            except Exception as e:
                import traceback
                traceback.print_exc()
                results.append({"file_name": f.filename or "unknown", "error": str(e), "type": type(e).__name__})

    ok = sum(1 for r in results if "error" not in r)
    return JSONResponse(content={"count": len(results), "ok": ok, "results": results})
