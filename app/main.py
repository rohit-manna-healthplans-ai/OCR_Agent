from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse

from ocr.pipeline import run_ocr
from app.parsers.fast_cpu_parser import FastLocalCPUParser


ALLOWED_EXTENSIONS = {
    ".pdf",
    ".doc", ".docx",
    ".ppt", ".pptx",
    ".jpg", ".jpeg", ".png", ".webp", ".bmp",
    ".tiff", ".tif",
    ".gif",
}
ALLOWED_EXTENSIONS_STR = ", ".join(sorted(ALLOWED_EXTENSIONS))

app = FastAPI(
    title="OCR Agent",
    description="PaddleOCR-powered extraction API. Upload a file, get structured text back.",
    version="2.0.0",
)

# Loads once at startup
FAST_PARSER = FastLocalCPUParser()


def _check_extension(filename: str) -> None:
    suffix = Path(filename or "upload").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS_STR}",
        )


def _shape_response(filename: str, raw_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the API response.

    formatted_text  - zone-tagged, layout-aware human-readable text
    plain_text      - flat unformatted text (all pages joined)
    tables          - all tables extracted across all pages
    layout_summary  - per-page block type counts and column count
    raw_json        - full OCR output with bboxes and confidence scores
    """
    try:
        parser_text = FAST_PARSER.process_data(raw_json)
        parser_error = None
    except Exception as e:
        parser_text = ""
        parser_error = f"{type(e).__name__}: {e}"

    # Collect tables and layout summary across all pages
    all_tables = []
    layout_summary = []
    for page in raw_json.get("pages", []):
        page_tables = page.get("tables", [])
        all_tables.extend(page_tables)
        block_types: Dict[str, int] = {}
        for b in page.get("blocks", []):
            bt = b.get("block_type", "unknown")
            block_types[bt] = block_types.get(bt, 0) + 1
        layout_summary.append({
            "page": page.get("page_index", 0) + 1,
            "columns": len(page.get("columns", [])),
            "block_counts": block_types,
            "table_count": len(page_tables),
        })

    response: Dict[str, Any] = {
        "filename": filename,
        "formatted_text": parser_text,
        "plain_text": raw_json.get("text", ""),
        "tables": all_tables,
        "layout_summary": layout_summary,
        "raw_json": raw_json,
    }
    if parser_error:
        response["parser_error"] = parser_error
    return response


@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok"}


@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    preset: str = Query(default="auto", description="Preprocessing preset: auto | clean_doc | photo | low_light | scan_enhance | printed_hq"),
    dpi: int = Query(default=150, ge=72, le=600, description="DPI for PDF rendering"),
    max_pages: int = Query(default=5, ge=1, le=200, description="Max pages to process"),
    return_debug: bool = Query(default=False, description="Include per-page debug info in response"),
    return_layout: bool = Query(default=True, description="Include layout zone info"),
):
    """
    Extract text from a document or image using PaddleOCR.

    Returns:
    - `formatted_text`: structured text with zone tags (HEADER, BODY, FOOTER, SIDEBAR)
    - `raw_json`: full OCR output with per-page word bounding boxes and confidence scores
    """
    _check_extension(file.filename or "upload")
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
            engine="auto",
            return_layout=return_layout,
        )
        return JSONResponse(content=_shape_response(file.filename or "upload", raw))
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/ocr-batch")
async def ocr_batch_endpoint(
    files: List[UploadFile] = File(...),
    preset: str = Query(default="auto"),
    dpi: int = Query(default=150, ge=72, le=600),
    max_pages: int = Query(default=5, ge=1, le=200),
    return_debug: bool = Query(default=False),
    return_layout: bool = Query(default=True),
):
    """
    Extract text from up to 10 files in one request.

    Returns a list of results in the same order as the uploaded files.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Batch limit is 10 files.")

    for f in files:
        _check_extension(f.filename or "upload")

    tmp_paths: List[Path] = []
    results: List[Dict[str, Any]] = []

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
                engine="auto",
                return_layout=return_layout,
            )
            results.append(_shape_response(f.filename or "upload", raw))

        return JSONResponse(content={"count": len(results), "results": results})
    finally:
        for p in tmp_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
