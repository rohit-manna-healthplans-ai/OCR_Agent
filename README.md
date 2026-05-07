# OCR Agent - API Backend

PaddleOCR-powered document and image text extraction. No UI - test via the auto-generated docs or any HTTP client.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| GET | `/docs` | Interactive Swagger UI |
| POST | `/ocr` | Extract text from a single file |
| POST | `/ocr-batch` | Extract text from up to 10 files |

## Test with curl

```bash
# Single file
curl -X POST http://localhost:8000/ocr \
  -F "file=@your_document.pdf" \
  -F "preset=auto" \
  -F "max_pages=10"

# Batch
curl -X POST http://localhost:8000/ocr-batch \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.png"
```

## Query Parameters (`/ocr`)

| Param | Default | Description |
|-------|---------|-------------|
| `preset` | `auto` | `auto` \| `printed_hq` \| `clean_doc` \| `photo` \| `low_light` \| `scan_enhance` |
| `dpi` | `150` | PDF render resolution (72-600) |
| `max_pages` | `5` | Max pages per document (1-200) |
| `return_debug` | `false` | Include per-page OCR debug info |
| `return_layout` | `true` | Include layout zone info |

## Response Shape

```json
{
  "filename": "document.pdf",
  "formatted_text": "<PAGE 1>\n<HEADER>\n  ...\n<BODY>\n  ...",
  "raw_json": {
    "text": "full plain text...",
    "pages": [ { "page_index": 0, "lines": [...], "width": 1240, "height": 1754 } ],
    "meta": { "engine": "paddleocr", "preset_used": "printed_hq" }
  }
}
```
