from __future__ import annotations

"""
Replace the entire app/main.py with this file.

What changed vs the previous version:
- Everything from the old main.py is kept exactly as-is (health, /ocr, /ocr-batch)
- Added: OCR Worker (intermediator) runs as a daemon thread on startup
- Worker does: MongoDB fetch → Azure Blob download → run_ocr() directly → MongoDB save
- No HTTP call between worker and OCR engine — direct Python function call
- Worker is controlled by env vars (all optional — if MONGO_URI is not set, worker stays off)
"""

import os
import re
import time
import json
import logging
import warnings
import tempfile
import datetime as dt
import threading
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Deque, Tuple
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse

from ocr.pipeline import run_ocr
from app.parsers.fast_cpu_parser import FastLocalCPUParser

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

# Auto-load repo root .env so server can be started without --env-file
if load_dotenv is not None:
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ocr_agent")
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("pymongo").setLevel(logging.WARNING)
warnings.filterwarnings(
    "ignore",
    message=r".*connected to a CosmosDB cluster.*",
    category=UserWarning,
)

# ── Shared parser (loaded once at startup) ────────────────────────────
FAST_PARSER = FastLocalCPUParser()

# ── Allowed file extensions for API endpoints ─────────────────────────
ALLOWED_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".ppt", ".pptx",
    ".jpg", ".jpeg", ".png", ".webp", ".bmp",
    ".tiff", ".tif", ".gif",
}
ALLOWED_EXTENSIONS_STR = ", ".join(sorted(ALLOWED_EXTENSIONS))

# ═════════════════════════════════════════════════════════════════════
#  FastAPI app
# ═════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="OCR Agent",
    description="PaddleOCR-powered extraction API with built-in MongoDB worker.",
    version="3.0.0",
)


def _check_extension(filename: str) -> None:
    suffix = Path(filename or "upload").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS_STR}",
        )


# ═════════════════════════════════════════════════════════════════════
#  SETTINGS — all tunable via .env (see .env in repo root)
# ═════════════════════════════════════════════════════════════════════

def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _env_int(
    name: str,
    default: str,
    lo: Optional[int] = None,
    hi: Optional[int] = None,
) -> int:
    value = int(os.getenv(name, default))
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


# API /ocr defaults (query params can still override per request)
_OCR_PRESET            = os.getenv("OCR_PRESET", "auto").strip()
_OCR_CONTENT_TYPE      = os.getenv("OCR_CONTENT_TYPE", "auto").strip()
_OCR_ENGINE            = os.getenv("OCR_ENGINE", "auto").strip()
_OCR_DPI_MIN           = _env_int("OCR_DPI_MIN", "72", 72, 600)
_OCR_DPI_MAX           = _env_int("OCR_DPI_MAX", "600", 72, 600)
_OCR_DPI_DEFAULT       = _env_int("OCR_DPI", "200", _OCR_DPI_MIN, _OCR_DPI_MAX)
_OCR_MAX_PAGES_MAX     = _env_int("OCR_MAX_PAGES_MAX", "200", 1, 500)
_OCR_MAX_PAGES_DEFAULT = _env_int("OCR_MAX_PAGES", "5", 1, _OCR_MAX_PAGES_MAX)
_OCR_RETURN_DEBUG      = _env_bool("OCR_RETURN_DEBUG", "false")
_OCR_RETURN_LAYOUT     = _env_bool("OCR_RETURN_LAYOUT", "true")
_OCR_USE_TABLE_ENGINE  = _env_bool("OCR_USE_TABLE_ENGINE", "false")
_OCR_BATCH_MAX_FILES   = _env_int("OCR_BATCH_MAX_FILES", "10", 1, 50)

_CT_MAP = {
    "form":       "form_grid",
    "screenshot": "screenshot",
    "scan":       "scan_enhance",
    "photo":      "photo",
    "document":   "printed_hq",
}


def _shape_response(filename: str, raw_json: Dict[str, Any]) -> Dict[str, Any]:
    try:
        parser_text = FAST_PARSER.process_data(raw_json)
        parser_error = None
    except Exception as e:
        parser_text = ""
        parser_error = f"{type(e).__name__}: {e}"

    all_tables: List[Any] = []
    layout_summary: List[Any] = []
    for page in raw_json.get("pages", []):
        page_tables = page.get("tables", [])
        all_tables.extend(page_tables)
        block_types: Dict[str, int] = {}
        for b in page.get("blocks", []):
            bt = b.get("block_type", "unknown")
            block_types[bt] = block_types.get(bt, 0) + 1
        layout_summary.append({
            "page":         page.get("page_index", 0) + 1,
            "columns":      len(page.get("columns", [])),
            "block_counts": block_types,
            "table_count":  len(page_tables),
            "table_source": (
                "pp_structure"
                if any(t.get("source") == "pp_structure" for t in page_tables)
                else "layout"
            ),
        })

    response: Dict[str, Any] = {
        "filename":       filename,
        "formatted_text": parser_text,
        "plain_text":     raw_json.get("text", ""),
        "tables":         all_tables,
        "layout_summary": layout_summary,
        "raw_json":       raw_json,
    }
    if parser_error:
        response["parser_error"] = parser_error
    return response


@app.get("/health")
def health():
    """Liveness check — also shows worker status and active .env defaults."""
    return {
        "status": "ok",
        "worker": _worker_status(),
        "settings": {
            "api": {
                "preset": _OCR_PRESET,
                "content_type": _OCR_CONTENT_TYPE,
                "engine": _OCR_ENGINE,
                "dpi": _OCR_DPI_DEFAULT,
                "dpi_min": _OCR_DPI_MIN,
                "dpi_max": _OCR_DPI_MAX,
                "max_pages": _OCR_MAX_PAGES_DEFAULT,
                "max_pages_cap": _OCR_MAX_PAGES_MAX,
                "return_debug": _OCR_RETURN_DEBUG,
                "return_layout": _OCR_RETURN_LAYOUT,
                "use_table_engine": _OCR_USE_TABLE_ENGINE,
                "batch_max_files": _OCR_BATCH_MAX_FILES,
            },
            "worker_ocr": {
                "preset": _W_OCR_PRESET,
                "dpi": _W_OCR_DPI,
                "max_pages": _W_MAX_PAGES,
                "engine": _W_OCR_ENGINE,
                "use_table_engine": _W_USE_TABLE_ENGINE,
            },
        },
    }


@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    preset: str = Query(default=_OCR_PRESET),
    content_type: str = Query(default=_OCR_CONTENT_TYPE),
    dpi: int = Query(default=_OCR_DPI_DEFAULT, ge=_OCR_DPI_MIN, le=_OCR_DPI_MAX),
    max_pages: int = Query(default=_OCR_MAX_PAGES_DEFAULT, ge=1, le=_OCR_MAX_PAGES_MAX),
    return_debug: bool = Query(default=_OCR_RETURN_DEBUG),
    return_layout: bool = Query(default=_OCR_RETURN_LAYOUT),
    use_table_engine: bool = Query(default=_OCR_USE_TABLE_ENGINE),
):
    _check_extension(file.filename or "upload")
    effective_preset = _CT_MAP.get(content_type.lower(), preset)
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
            preset=effective_preset,
            dpi=dpi,
            max_pages=max_pages,
            return_debug=return_debug,
            engine=_OCR_ENGINE,
            return_layout=return_layout,
            use_table_engine=use_table_engine,
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
    preset: str = Query(default=_OCR_PRESET),
    content_type: str = Query(default=_OCR_CONTENT_TYPE),
    dpi: int = Query(default=_OCR_DPI_DEFAULT, ge=_OCR_DPI_MIN, le=_OCR_DPI_MAX),
    max_pages: int = Query(default=_OCR_MAX_PAGES_DEFAULT, ge=1, le=_OCR_MAX_PAGES_MAX),
    return_debug: bool = Query(default=_OCR_RETURN_DEBUG),
    return_layout: bool = Query(default=_OCR_RETURN_LAYOUT),
    use_table_engine: bool = Query(default=_OCR_USE_TABLE_ENGINE),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > _OCR_BATCH_MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Batch limit is {_OCR_BATCH_MAX_FILES} files.",
        )

    effective_preset = _CT_MAP.get(content_type.lower(), preset)
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
                preset=effective_preset,
                dpi=dpi,
                max_pages=max_pages,
                return_debug=return_debug,
                engine=_OCR_ENGINE,
                return_layout=return_layout,
                use_table_engine=use_table_engine,
            )
            results.append(_shape_response(f.filename or "upload", raw))

        return JSONResponse(content={"count": len(results), "results": results})
    finally:
        for p in tmp_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass


# ═════════════════════════════════════════════════════════════════════
#  WORKER CONFIG  (all from env — worker is OFF if MONGO_URI not set)
# ═════════════════════════════════════════════════════════════════════

_W_MONGO_URI         = os.getenv("MONGO_URI", "")
_W_DB_NAME           = os.getenv("MONGO_DB", "IDAI_Web_Database")
_W_COLLECTION        = os.getenv("MONGO_COLLECTION", "screenshots")
_W_STATE_COLLECTION  = os.getenv("STATE_COLLECTION", "ocr_state")
_W_STATE_DOC_ID      = os.getenv("STATE_DOC_ID", "global_state")
_W_VALIDATION_COLLECTION = os.getenv("VALIDATION_COLLECTION", "OCR_Validation_Logs")

_W_BATCH_SIZE        = max(1, min(5, int(os.getenv("BATCH_SIZE", "5"))))
_W_POLL_SEC          = float(os.getenv("POLL_INTERVAL_SEC", "3"))
_W_MAX_CANDIDATES    = int(os.getenv("MAX_PENDING_CANDIDATES", "800"))
_W_COOLDOWN_SEC      = int(os.getenv("COOLDOWN_SEC", "600"))
_W_COOLDOWN_MAX      = int(os.getenv("COOLDOWN_MAX_ITEMS", "5000"))
_W_RETRY_UNTIL_SUCCESS = os.getenv("WORKER_RETRY_UNTIL_SUCCESS", "true").lower() == "true"
_W_RETRY_BASE_SEC    = max(5, int(os.getenv("WORKER_RETRY_BASE_SEC", "15")))
_W_RETRY_MAX_SEC     = max(_W_RETRY_BASE_SEC, int(os.getenv("WORKER_RETRY_MAX_SEC", "120")))
_W_GIVE_UP_AFTER     = int(os.getenv("WORKER_GIVE_UP_AFTER", "0"))  # 0 = never stop retrying

_W_DOWNLOAD_TIMEOUT  = float(os.getenv("DOWNLOAD_TIMEOUT_SEC", "45"))
_W_MAX_IMG_BYTES     = int(os.getenv("MAX_IMAGE_BYTES", str(12 * 1024 * 1024)))
_W_HTTP_RETRIES      = int(os.getenv("DOWNLOAD_HTTP_RETRIES", "3"))

_W_AZURE_CONN_STR    = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "").strip()
_W_AZURE_SAS_TOKEN   = os.getenv("AZURE_BLOB_SAS_TOKEN", "").strip()
_W_AZURE_CONTAINER   = os.getenv("AZURE_STORAGE_CONTAINER", "screenshots").strip() or "screenshots"

# Worker OCR (screenshots) — defaults from .env
_W_OCR_PRESET        = os.getenv("WORKER_OCR_PRESET", os.getenv("OCR_PRESET", "screenshot")).strip()
_W_OCR_DPI           = _env_int("WORKER_DPI", os.getenv("OCR_DPI", "200"), _OCR_DPI_MIN, _OCR_DPI_MAX)
_W_USE_TABLE_ENGINE  = _env_bool("WORKER_USE_TABLE_ENGINE", os.getenv("OCR_USE_TABLE_ENGINE", "false"))
_W_MAX_PAGES         = _env_int("WORKER_MAX_PAGES", "1", 1, _OCR_MAX_PAGES_MAX)
_W_OCR_ENGINE        = os.getenv("WORKER_ENGINE", os.getenv("OCR_ENGINE", "auto")).strip()
_W_WORKER_RETURN_DEBUG   = _env_bool("WORKER_RETURN_DEBUG", os.getenv("OCR_RETURN_DEBUG", "false"))
_W_WORKER_RETURN_LAYOUT  = _env_bool("WORKER_RETURN_LAYOUT", os.getenv("OCR_RETURN_LAYOUT", "true"))

# ── Worker state (thread-safe via GIL for simple reads/writes) ────────
_worker_state: Dict[str, Any] = {
    "status":   "off",   # off | starting | running | idle | error
    "processed": 0,
    "attempted": 0,
    "errors":    0,
    "pending_count": 0,
    "validation_failure_count": 0,
    "cooldown_active_count": 0,
    "retry_scheduled_count": 0,
    "last_run":  None,
}


def _worker_status() -> Dict[str, Any]:
    return dict(_worker_state)


# ═════════════════════════════════════════════════════════════════════
#  WORKER — MongoDB helpers
# ═════════════════════════════════════════════════════════════════════

def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _cooldown_until(seconds: int) -> str:
    return (
        dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=seconds)
    ).isoformat().replace("+00:00", "Z")


def _schedule_retry(
    item_id: str,
    cooldown: Dict[str, str],
    retry_counts: Dict[str, int],
) -> bool:
    """
    Schedule retry with short exponential backoff (retry-until-success mode).
    Returns False only if WORKER_GIVE_UP_AFTER > 0 and limit reached.
    """
    n = int(retry_counts.get(item_id, 0)) + 1
    retry_counts[item_id] = n

    if _W_GIVE_UP_AFTER > 0 and n >= _W_GIVE_UP_AFTER:
        logger.error(
            "Giving up after %d attempts (item_id=%s) — set WORKER_GIVE_UP_AFTER=0 to never give up",
            n,
            item_id,
        )
        cooldown.pop(item_id, None)
        return False

    if _W_RETRY_UNTIL_SUCCESS:
        sec = min(_W_RETRY_BASE_SEC * (2 ** min(n - 1, 5)), _W_RETRY_MAX_SEC)
    else:
        sec = _W_COOLDOWN_SEC

    cooldown[item_id] = _cooldown_until(sec)
    return True


def _clear_retry_state(
    item_id: str,
    cooldown: Dict[str, str],
    retry_counts: Dict[str, int],
) -> None:
    cooldown.pop(item_id, None)
    retry_counts.pop(item_id, None)


def _pending_filter() -> Dict[str, Any]:
    """Pending = no ocr_text, empty ocr_text, or null — plus valid screenshot_url."""
    return {
        "$and": [
            {
                "$or": [
                    {"ocr_text": {"$exists": False}},
                    {"ocr_text": None},
                    {"ocr_text": ""},
                ]
            },
            {"screenshot_url": {"$regex": r"^https?://"}},
        ]
    }


def _pending_save_filter(doc_id: Any) -> Dict[str, Any]:
    """Allow save only when ocr_text is not already present with real content."""
    return {
        "_id": doc_id,
        "$or": [
            {"ocr_text": {"$exists": False}},
            {"ocr_text": None},
            {"ocr_text": ""},
        ],
    }


def _fetch_pending(col: Any, limit: int) -> List[Dict[str, Any]]:
    return list(
        col.find(
            _pending_filter(),
            {"_id": 1, "user_id": 1, "screenshot_id": 1, "screenshot_url": 1, "ts": 1},
        )
        .sort([("ts", 1), ("_id", 1)])
        .limit(limit)
    )


def _save_ocr_text(
    col: Any,
    doc_id: Any,
    text: str,
    extraction_blob_path: Optional[str] = None,
) -> None:
    set_doc: Dict[str, Any] = {"ocr_text": text}
    if extraction_blob_path:
        set_doc["extraction_blob_path"] = extraction_blob_path
    col.update_one(_pending_save_filter(doc_id), {"$set": set_doc})


def _load_state(state_col: Any) -> Dict[str, Any]:
    return state_col.find_one({"_id": _W_STATE_DOC_ID}) or {}


def _prune_cooldown(cooldown: Dict[str, str]) -> Dict[str, str]:
    """Drop expired cooldown entries so failed items become eligible again."""
    now = _utc_now_iso()
    pruned: Dict[str, str] = {}
    for key, until in cooldown.items():
        if isinstance(until, str) and until > now:
            pruned[key] = until
    return pruned


def _trim_cooldown(cooldown: Dict[str, str]) -> Dict[str, str]:
    """Keep only the soonest-to-expire active cooldown entries when over limit."""
    if len(cooldown) <= _W_COOLDOWN_MAX:
        return cooldown
    items = sorted(cooldown.items(), key=lambda kv: kv[1])
    return dict(items[:_W_COOLDOWN_MAX])


def _update_reconciliation_stats(
    col: Any,
    validation_col: Any,
    cooldown: Dict[str, str],
) -> None:
    """Refresh queue visibility metrics (used by /health)."""
    now = _utc_now_iso()
    try:
        _worker_state["pending_count"] = col.count_documents(_pending_filter())
        _worker_state["validation_failure_count"] = validation_col.count_documents(
            {"status": "failure"}
        )
        _worker_state["cooldown_active_count"] = sum(
            1 for v in cooldown.values() if isinstance(v, str) and v > now
        )
        _worker_state["retry_scheduled_count"] = len(cooldown)
    except Exception as e:
        logger.debug("Reconciliation stats update failed: %s", e)


def _fetch_failed_validation_ids(validation_col: Any, limit: int = 1000) -> set:
    """screenshot_ids that failed previously — retry these before new pending items."""
    try:
        rows = validation_col.find(
            {"status": "failure"},
            {"_id": 1},
        ).limit(limit)
        return {str(r["_id"]) for r in rows if r.get("_id")}
    except Exception as e:
        logger.debug("Failed validation id fetch failed: %s", e)
        return set()


def _sort_candidates_for_retry(
    docs: List[Dict[str, Any]],
    failed_ids: set,
    retry_counts: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Prioritize validation failures, then most attempts, then oldest ts."""

    def sort_key(doc: Dict[str, Any]) -> tuple:
        ck = _cooldown_key(doc)
        is_retry = ck in failed_ids
        attempts = int(retry_counts.get(ck, 0))
        ts = doc.get("ts")
        return (0 if is_retry else 1, -attempts, ts if ts is not None else "")

    return sorted(docs, key=sort_key)


def _save_state(
    state_col: Any,
    mode: str,
    cooldown: Dict[str, str],
    retry_counts: Optional[Dict[str, int]] = None,
) -> None:
    cooldown = _trim_cooldown(_prune_cooldown(cooldown))
    payload: Dict[str, Any] = {
        "mode": mode,
        "cooldown": cooldown,
        "updated_at": _utc_now_iso(),
    }
    if retry_counts is not None:
        payload["retry_counts"] = retry_counts
    state_col.update_one(
        {"_id": _W_STATE_DOC_ID},
        {"$set": payload},
        upsert=True,
    )


def _save_state_with_batch(
    state_col: Any,
    mode: str,
    cooldown: Dict[str, str],
    batch_status: str,
    batch_total: int,
    batch_success: int,
    batch_failure: int,
    last_failed_ids: Optional[List[str]] = None,
    retry_counts: Optional[Dict[str, int]] = None,
) -> None:
    cooldown = _trim_cooldown(_prune_cooldown(cooldown))
    payload: Dict[str, Any] = {
        "mode": mode,
        "cooldown": cooldown,
        "updated_at": _utc_now_iso(),
        "last_batch": {
            "status": batch_status,
            "total": batch_total,
            "success_count": batch_success,
            "failure_count": batch_failure,
            "ts": _utc_now_iso(),
        },
    }
    if retry_counts is not None:
        payload["retry_counts"] = retry_counts
    if last_failed_ids:
        payload["last_failed_screenshot_ids"] = last_failed_ids[:50]
    state_col.update_one(
        {"_id": _W_STATE_DOC_ID},
        {"$set": payload},
        upsert=True,
    )


def _validation_upsert(
    validation_col: Any,
    screenshot_id: str,
    user_id: str,
    stage_events: Dict[str, Dict[str, str]],
) -> None:
    stage_order = ("picked", "downloaded", "ocr_extracted", "stored")
    stages = [stage_events[s] for s in stage_order if s in stage_events]
    final_status = "success" if stage_events.get("stored", {}).get("status") == "success" else "failure"
    now = _utc_now_iso()
    validation_col.update_one(
        {"_id": screenshot_id},
        {
            "$set": {
                "screenshot_id": screenshot_id,
                "user_id": user_id,
                "status": final_status,
                "stages": stages,
                "last_updated_at": now,
            }
        },
        upsert=True,
    )


# ═════════════════════════════════════════════════════════════════════
#  WORKER — Download helpers (from intermediator, unchanged)
# ═════════════════════════════════════════════════════════════════════

_dl_session = requests.Session()
_HTTP_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


def _sanitize_url(url: str) -> str:
    if not isinstance(url, str):
        return ""
    u = url.strip()
    for _ in range(4):
        if u.endswith("%22"):
            u = u[:-3].rstrip()
        elif u.endswith('"'):
            u = u[:-1].rstrip()
        else:
            break
    return u


def _is_azure_host(url: str) -> bool:
    try:
        return (urlparse(url).hostname or "").lower().endswith(".blob.core.windows.net")
    except Exception:
        return False


def _account_from_url(url: str) -> Optional[str]:
    host = (urlparse(url).hostname or "").lower()
    if not host.endswith(".blob.core.windows.net"):
        return None
    return host.split(".")[0]


def _conn_str_matches(url: str) -> bool:
    if not _W_AZURE_CONN_STR:
        return False
    acc = _account_from_url(url)
    if not acc:
        return False
    return f"AccountName={acc}" in _W_AZURE_CONN_STR or \
           f"AccountName={acc}" in _W_AZURE_CONN_STR.lower()


def _maybe_add_sas(url: str) -> str:
    if not _W_AZURE_SAS_TOKEN or not _is_azure_host(url):
        return url
    if "sig=" in url.lower():
        return url
    token = _W_AZURE_SAS_TOKEN.lstrip("?&")
    return f"{url}{'&' if '?' in url else '?'}{token}"


def _path_variants(url: str) -> List[str]:
    out: List[str] = []
    for pattern in (
        r"(\d{4}/\d{2}/\d{2})/screenshots/",
        r"(\d{4}/[A-Za-z]+/\d{1,2})/screenshots/",
    ):
        m = re.search(pattern, url)
        if m:
            alt = url.replace(m.group(0), f"{m.group(1)}/", 1)
            if alt != url and alt not in out:
                out.append(alt)
    return out


def _parse_blob_url(url: str) -> Optional[Tuple[str, str]]:
    p = urlparse(_sanitize_url(url))
    if not (p.hostname or "").lower().endswith(".blob.core.windows.net"):
        return None
    path = (p.path or "").lstrip("/")
    if "/" not in path:
        return None
    container, blob = path.split("/", 1)
    container = container.strip("/")
    blob = blob.strip("/")
    if not container or not blob:
        return None
    return container, blob


def _build_extraction_blob_path(item: Dict[str, Any], source_url: str) -> str:
    """
    Build extraction JSON blob path.
    Preferred: derive from screenshot URL by replacing /screenshots/ with /extraction/
    and converting extension to .json.
    Fallback: userId/YYYY/MM/DD/extraction/userId_screenshotId.json using metadata.
    """
    parsed = _parse_blob_url(source_url)
    if parsed:
        _, blob_path = parsed
        if "/screenshots/" in blob_path:
            p = blob_path.replace("/screenshots/", "/extraction/", 1)
        elif blob_path.endswith("/screenshots"):
            p = blob_path[:-len("/screenshots")] + "/extraction"
        else:
            p = blob_path
        p = re.sub(r"\.[A-Za-z0-9]+$", ".json", p)
        if p.endswith("/.json"):
            screenshot_id = str(item.get("item_id", "unknown"))
            user_id = str(item.get("user_id", "unknown")) or "unknown"
            p = p[:-6] + f"/{user_id}_{screenshot_id}.json"
        return p

    user_id = str(item.get("user_id", "unknown")) or "unknown"
    screenshot_id = str(item.get("item_id", "unknown")) or "unknown"
    ts_val = item.get("ts")
    dt_obj = None
    if isinstance(ts_val, dt.datetime):
        dt_obj = ts_val.astimezone(dt.timezone.utc)
    elif isinstance(ts_val, str):
        t = ts_val.strip()
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        try:
            dt_obj = dt.datetime.fromisoformat(t).astimezone(dt.timezone.utc)
        except Exception:
            dt_obj = None
    if dt_obj is None:
        dt_obj = dt.datetime.now(dt.timezone.utc)
    yyyy = f"{dt_obj.year:04d}"
    mm = f"{dt_obj.month:02d}"
    dd = f"{dt_obj.day:02d}"
    return f"{user_id}/{yyyy}/{mm}/{dd}/extraction/{user_id}_{screenshot_id}.json"


def _upload_extraction_json(source_url: str, blob_path: str, payload: Dict[str, Any]) -> None:
    from azure.storage.blob import BlobClient

    container = _W_AZURE_CONTAINER
    parsed = _parse_blob_url(source_url)
    if parsed and parsed[0]:
        container = parsed[0]

    if not _W_AZURE_CONN_STR:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING is required for extraction upload")

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    client = BlobClient.from_connection_string(
        conn_str=_W_AZURE_CONN_STR,
        container_name=container,
        blob_name=blob_path,
    )
    client.upload_blob(body, overwrite=True, content_type="application/json")


def _download_sdk(container: str, blob_path: str) -> bytes:
    from azure.storage.blob import BlobClient
    client = BlobClient.from_connection_string(
        conn_str=_W_AZURE_CONN_STR,
        container_name=container,
        blob_name=blob_path,
    )
    data = client.download_blob(timeout=int(_W_DOWNLOAD_TIMEOUT)).readall()
    if len(data) > _W_MAX_IMG_BYTES:
        raise ValueError(f"Image too large ({len(data)} bytes)")
    return data


def _download_http(url: str) -> bytes:
    delay = 0.6
    headers = {"User-Agent": _HTTP_UA, "Accept": "image/*,*/*;q=0.8"}
    for attempt in range(max(1, _W_HTTP_RETRIES)):
        try:
            with _dl_session.get(
                url, stream=True, timeout=_W_DOWNLOAD_TIMEOUT, headers=headers
            ) as r:
                if r.status_code in (429, 503) and attempt + 1 < _W_HTTP_RETRIES:
                    time.sleep(delay)
                    delay *= 2
                    continue
                r.raise_for_status()
                buf = bytearray()
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        buf.extend(chunk)
                    if len(buf) > _W_MAX_IMG_BYTES:
                        raise ValueError("Image too large")
                return bytes(buf)
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if code in (429, 503) and attempt + 1 < _W_HTTP_RETRIES:
                time.sleep(delay)
                delay *= 2
                continue
            raise
        except requests.RequestException:
            if attempt + 1 < _W_HTTP_RETRIES:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("HTTP download retries exhausted")


def _download_image(url: str) -> bytes:
    """
    Download image bytes from Azure Blob or any HTTP URL.
    Tries SDK first (private blobs), then HTTP with SAS, then path variants.
    """
    url = _sanitize_url(url)

    # 1. Private blob via SDK
    if _W_AZURE_CONN_STR and _conn_str_matches(url):
        parsed = _parse_blob_url(url)
        if parsed:
            container, blob = parsed
            paths = [blob] + [
                _parse_blob_url(v)[1]
                for v in _path_variants(url)
                if _parse_blob_url(v)
            ]
            last: Optional[BaseException] = None
            for bp in paths:
                try:
                    return _download_sdk(container, bp)
                except Exception as e:
                    last = e
            if last:
                logger.debug("SDK download failed, falling back to HTTP: %s", last)

    # 2. HTTP (with SAS appended if available, plus path variants)
    candidates = [_maybe_add_sas(url), url]
    for v in _path_variants(url):
        candidates.append(_maybe_add_sas(v))
        candidates.append(v)
    # Deduplicate preserving order
    seen: set = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    last = None
    for candidate in unique:
        try:
            return _download_http(candidate)
        except Exception as e:
            last = e
    raise last or RuntimeError("All download attempts failed")


def _guess_ext(url: str) -> str:
    try:
        path = urlparse(url).path
        if "." in path:
            ext = path.rsplit(".", 1)[-1].lower()
            if ext in ("jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"):
                return "." + ext
    except Exception:
        pass
    return ".jpg"


def _cooldown_key(doc: Dict[str, Any]) -> str:
    sid = doc.get("screenshot_id")
    if isinstance(sid, str) and sid.strip():
        return sid.strip()
    return str(doc.get("_id", ""))


# ═════════════════════════════════════════════════════════════════════
#  WORKER — Batch scheduling (round-robin per user)
# ═════════════════════════════════════════════════════════════════════

def _build_queues(
    docs: List[Dict[str, Any]],
    cooldown: Dict[str, str],
) -> Dict[str, Deque[Dict[str, Any]]]:
    now = _utc_now_iso()
    queues: Dict[str, Deque[Dict[str, Any]]] = {}
    for doc in docs:
        url = _sanitize_url(doc.get("screenshot_url", ""))
        if not url.startswith("http"):
            continue
        ck = _cooldown_key(doc)
        cool = cooldown.get(ck)
        if isinstance(cool, str) and cool and cool > now:
            continue
        ukey = str(doc.get("user_id", ""))
        if ukey not in queues:
            queues[ukey] = deque()
        queues[ukey].append(
            {
                "item_id": ck,
                "url": url,
                "doc_id": doc["_id"],
                "user_id": ukey,
                "ts": doc.get("ts"),
            }
        )
    return queues


def _round_robin(
    queues: Dict[str, Deque[Dict[str, Any]]],
    batch_size: int,
) -> List[Dict[str, Any]]:
    users = deque(u for u, q in queues.items() if q)
    batch: List[Dict[str, Any]] = []
    while users and len(batch) < batch_size:
        u = users.popleft()
        q = queues.get(u)
        if not q:
            continue
        batch.append(q.popleft())
        if q:
            users.append(u)
    return batch


# ═════════════════════════════════════════════════════════════════════
#  WORKER — OCR processing (direct call — no HTTP)
# ═════════════════════════════════════════════════════════════════════

def _process_item(
    item: Dict[str, Any],
    col: Any,
    validation_col: Any,
    cooldown: Dict[str, str],
    retry_counts: Dict[str, int],
) -> bool:
    """
    Download one image, run OCR directly via run_ocr(), save result to MongoDB.
    Returns True on success, False on failure (item goes to cooldown).
    """
    url     = item["url"]
    item_id = item["item_id"]
    doc_id  = item["doc_id"]
    user_id = str(item.get("user_id", ""))
    stage_events: Dict[str, Dict[str, str]] = {}

    def _record_stage(stage: str, status: str) -> None:
        stage_events[stage] = {"stage": stage, "status": status, "ts": _utc_now_iso()}
        _validation_upsert(validation_col, item_id, user_id, stage_events)

    try:
        _record_stage("picked", "success")
    except Exception as e:
        logger.warning("Validation log failed (item_id=%s): %s", item_id, e)

    # 1. Download image
    try:
        img_bytes = _download_image(url)
        try:
            _record_stage("downloaded", "success")
        except Exception as e:
            logger.warning("Validation stage failed (item_id=%s): %s", item_id, e)
    except Exception as e:
        logger.warning("Download failed (item_id=%s): %s", item_id, e)
        try:
            _record_stage("downloaded", "failure")
        except Exception as ve:
            logger.warning("Validation stage failed (item_id=%s): %s", item_id, ve)
        _schedule_retry(item_id, cooldown, retry_counts)
        return False

    # 2. Write to temp file (run_ocr expects a file path)
    ext = _guess_ext(url)
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name
    except Exception as e:
        logger.warning("Temp file write failed (item_id=%s): %s", item_id, e)
        try:
            _record_stage("ocr_extracted", "failure")
        except Exception as ve:
            logger.warning("Validation stage failed (item_id=%s): %s", item_id, ve)
        _schedule_retry(item_id, cooldown, retry_counts)
        return False

    # 3. Run OCR — direct function call, no HTTP
    try:
        raw = run_ocr(
            input_path=tmp_path,
            preset=_W_OCR_PRESET,
            dpi=_W_OCR_DPI,
            max_pages=_W_MAX_PAGES,
            return_debug=_W_WORKER_RETURN_DEBUG,
            engine=_W_OCR_ENGINE,
            return_layout=_W_WORKER_RETURN_LAYOUT,
            use_table_engine=_W_USE_TABLE_ENGINE,
        )
        result  = _shape_response(item_id + ext, raw)
        ocr_text = result.get("formatted_text", "") or result.get("plain_text", "")
        if not (ocr_text and str(ocr_text).strip()):
            logger.warning("OCR returned empty text (item_id=%s)", item_id)
            try:
                _record_stage("ocr_extracted", "failure")
            except Exception as ve:
                logger.warning("Validation stage failed (item_id=%s): %s", item_id, ve)
            _schedule_retry(item_id, cooldown, retry_counts)
            return False
        try:
            _record_stage("ocr_extracted", "success")
        except Exception as e:
            logger.warning("Validation stage failed (item_id=%s): %s", item_id, e)
    except Exception as e:
        logger.warning("OCR failed (item_id=%s): %s", item_id, e)
        try:
            _record_stage("ocr_extracted", "failure")
        except Exception as ve:
            logger.warning("Validation stage failed (item_id=%s): %s", item_id, ve)
        _schedule_retry(item_id, cooldown, retry_counts)
        return False
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass

    # 4. Save to MongoDB
    try:
        extraction_blob_path = _build_extraction_blob_path(item, url)
        _upload_extraction_json(url, extraction_blob_path, raw)
        _save_ocr_text(col, doc_id, ocr_text, extraction_blob_path=extraction_blob_path)
        try:
            _record_stage("stored", "success")
        except Exception as e:
            logger.warning("Validation stage failed (item_id=%s): %s", item_id, e)
        _clear_retry_state(item_id, cooldown, retry_counts)
        logger.debug("OCR saved (item_id=%s, chars=%d)", item_id, len(ocr_text))
        return True
    except Exception as e:
        logger.warning("MongoDB save failed (item_id=%s): %s", item_id, e)
        try:
            _record_stage("stored", "failure")
        except Exception as ve:
            logger.warning("Validation stage failed (item_id=%s): %s", item_id, ve)
        _schedule_retry(item_id, cooldown, retry_counts)
        return False


def _process_batch(
    batch: List[Dict[str, Any]],
    col: Any,
    validation_col: Any,
    cooldown: Dict[str, str],
    retry_counts: Dict[str, int],
) -> Tuple[int, int, List[str]]:
    """Process a batch of items. Returns (success_count, failure_count, failed_ids)."""
    updated = 0
    failed = 0
    failed_ids: List[str] = []
    for item in batch:
        _worker_state["attempted"] = int(_worker_state.get("attempted", 0)) + 1
        ok = _process_item(item, col, validation_col, cooldown, retry_counts)
        if ok:
            updated += 1
            _worker_state["processed"] += 1
        else:
            failed += 1
            _worker_state["errors"] += 1
            failed_ids.append(str(item.get("item_id", "")))
    return updated, failed, failed_ids


# ═════════════════════════════════════════════════════════════════════
#  WORKER — Main loop (runs in daemon thread)
# ═════════════════════════════════════════════════════════════════════

def _worker_loop() -> None:
    from pymongo import MongoClient

    _worker_state["status"] = "starting"
    logger.info("OCR Worker starting | DB=%s | Collection=%s", _W_DB_NAME, _W_COLLECTION)

    try:
        client = MongoClient(_W_MONGO_URI, serverSelectionTimeoutMS=8000)
        client.admin.command("ping")
        logger.info("OCR Worker connected to MongoDB")
    except Exception as e:
        _worker_state["status"] = "error"
        logger.error("OCR Worker — MongoDB connection failed: %s", e)
        return

    col       = client[_W_DB_NAME][_W_COLLECTION]
    state_col = client[_W_DB_NAME][_W_STATE_COLLECTION]
    validation_col = client[_W_DB_NAME][_W_VALIDATION_COLLECTION]

    state    = _load_state(state_col)
    cooldown = state.get("cooldown") if isinstance(state.get("cooldown"), dict) else {}
    retry_counts: Dict[str, int] = {}
    if isinstance(state.get("retry_counts"), dict):
        retry_counts = {str(k): int(v) for k, v in state["retry_counts"].items()}
    cooldown = _prune_cooldown(cooldown)
    logger.info(
        "OCR Worker loaded | cooldown=%d retry_tracking=%d retry_until_success=%s base=%ss max=%ss",
        len(cooldown),
        len(retry_counts),
        _W_RETRY_UNTIL_SUCCESS,
        _W_RETRY_BASE_SEC,
        _W_RETRY_MAX_SEC,
    )

    while True:
        try:
            cooldown = _prune_cooldown(cooldown)
            _update_reconciliation_stats(col, validation_col, cooldown)

            candidates = _fetch_pending(col, _W_MAX_CANDIDATES)

            if not candidates:
                _worker_state["status"] = "idle"
                _save_state(state_col, "IDLE", cooldown, retry_counts)
                time.sleep(_W_POLL_SEC)
                continue

            failed_validation_ids = _fetch_failed_validation_ids(validation_col)
            candidates = _sort_candidates_for_retry(
                candidates, failed_validation_ids, retry_counts
            )

            queues = _build_queues(candidates, cooldown)
            if not any(bool(q) for q in queues.values()):
                _worker_state["status"] = "idle"
                _save_state(state_col, "COOLDOWN_ONLY", cooldown, retry_counts)
                # Short sleep when items are in brief retry backoff
                time.sleep(2.0 if _W_RETRY_UNTIL_SUCCESS else _W_POLL_SEC)
                continue

            batch = _round_robin(queues, _W_BATCH_SIZE)
            if not batch:
                time.sleep(_W_POLL_SEC)
                continue

            _worker_state["status"] = "running"
            _worker_state["last_run"] = _utc_now_iso()
            batch_t0 = time.perf_counter()
            updated, failed, batch_failed_ids = _process_batch(
                batch, col, validation_col, cooldown, retry_counts
            )
            took_s = time.perf_counter() - batch_t0
            logger.info(
                "Batch total=%d done=%d failed=%d took=%.2fs pending=%s failures=%s",
                len(batch),
                updated,
                failed,
                took_s,
                _worker_state.get("pending_count"),
                _worker_state.get("validation_failure_count"),
            )

            if failed > 0:
                _save_state_with_batch(
                    state_col=state_col,
                    mode="RUNNING_WITH_FAILURE",
                    cooldown=cooldown,
                    batch_status="failure",
                    batch_total=len(batch),
                    batch_success=updated,
                    batch_failure=failed,
                    last_failed_ids=batch_failed_ids,
                    retry_counts=retry_counts,
                )
            else:
                _save_state_with_batch(
                    state_col=state_col,
                    mode="RUNNING",
                    cooldown=cooldown,
                    batch_status="success",
                    batch_total=len(batch),
                    batch_success=updated,
                    batch_failure=0,
                    retry_counts=retry_counts,
                )

            _update_reconciliation_stats(col, validation_col, cooldown)

            # Retry-until-success: re-poll quickly after failures or successes
            if _W_RETRY_UNTIL_SUCCESS:
                if failed > 0:
                    time.sleep(1.0)
                elif updated > 0:
                    time.sleep(0.2)
                else:
                    time.sleep(_W_POLL_SEC)
            else:
                time.sleep(0.2 if updated > 0 else _W_POLL_SEC)

        except KeyboardInterrupt:
            break
        except Exception:
            _worker_state["status"] = "error"
            logger.exception("Worker loop error")
            try:
                _save_state_with_batch(
                    state_col=state_col,
                    mode="WORKER_ERROR",
                    cooldown=cooldown,
                    batch_status="failure",
                    batch_total=0,
                    batch_success=0,
                    batch_failure=0,
                    retry_counts=retry_counts,
                )
            except Exception:
                logger.exception("Failed to update ocr_state after worker error")
            time.sleep(_W_POLL_SEC)


# ═════════════════════════════════════════════════════════════════════
#  STARTUP — launch worker thread if MONGO_URI is configured
# ═════════════════════════════════════════════════════════════════════

@app.on_event("startup")
def start_worker() -> None:
    if not _W_MONGO_URI:
        logger.info("MONGO_URI not set — OCR Worker is disabled. API-only mode.")
        _worker_state["status"] = "off"
        return

    t = threading.Thread(target=_worker_loop, name="ocr-worker", daemon=True)
    t.start()
    logger.info("OCR Worker thread started.")
