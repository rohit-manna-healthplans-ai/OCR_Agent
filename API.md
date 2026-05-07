# OCR Agent API – use from other projects

Your OCR Agent is a **REST API**. Other apps can send documents and get extracted text and raw JSON.

**Base URL (Hugging Face Space):**  
`https://YOUR_USERNAME-ocr-agent.hf.space`  
or  
`https://huggingface.co/spaces/YOUR_USERNAME/ocr_agent`

Use the same base URL in the examples below (replace `BASE_URL`).

---

## Endpoints

### 1. Single file: `POST /ocr`

| Item    | Description |
|--------|-------------|
| **URL** | `BASE_URL/ocr` |
| **Method** | POST |
| **Body** | `multipart/form-data` with field **`file`** = your document (PDF, image, DOCX, etc.) |
| **Query (optional)** | `preset=auto`, `dpi=150`, `max_pages=5` |

**Response (JSON):**
```json
{
  "filename": "document.pdf",
  "formatted_text": "<PAGE 1>\n<HEADER>...\n<BODY>...",
  "raw_json": { "meta": {...}, "text": "...", "pages": [...] }
}
```

- **`formatted_text`** – Parser output (layout zones, spell-fixed). Use this for display or downstream logic.
- **`raw_json`** – Full OCR result (pages, lines, words, bboxes, debug, etc.).

---

### 2. Multiple files: `POST /ocr-batch`

| Item    | Description |
|--------|-------------|
| **URL** | `BASE_URL/ocr-batch` |
| **Method** | POST |
| **Body** | `multipart/form-data` with field **`files`** = multiple files (max 10) |
| **Query (optional)** | Same as `/ocr` |

**Response (JSON):**
```json
{
  "count": 2,
  "results": [
    { "filename": "doc1.pdf", "formatted_text": "...", "raw_json": {...} },
    { "filename": "doc2.png", "formatted_text": "...", "raw_json": {...} }
  ]
}
```

---

## Examples

### Python (requests)

```python
import requests

BASE_URL = "https://YOUR_USERNAME-ocr-agent.hf.space"  # or your Space URL

# Single file
with open("document.pdf", "rb") as f:
    r = requests.post(
        f"{BASE_URL}/ocr",
        files={"file": ("document.pdf", f, "application/pdf")},
        params={"max_pages": 5},
    )
data = r.json()
print(data["formatted_text"])   # parsed layout text
print(data["raw_json"]["text"]) # raw OCR text

# Batch (multiple files)
files = [("files", open("doc1.pdf", "rb")), ("files", open("doc2.png", "rb"))]
r = requests.post(f"{BASE_URL}/ocr-batch", files=files)
for item in r.json()["results"]:
    print(item["filename"], "->", item["formatted_text"][:100])
```

### JavaScript (fetch)

```javascript
const BASE_URL = "https://YOUR_USERNAME-ocr-agent.hf.space";

// Single file (e.g. from <input type="file">)
const formData = new FormData();
formData.append("file", fileInput.files[0]);

const res = await fetch(`${BASE_URL}/ocr`, {
  method: "POST",
  body: formData,
});
const data = await res.json();
console.log(data.formatted_text);
console.log(data.raw_json);
```

### cURL

```bash
curl -X POST "https://YOUR_USERNAME-ocr-agent.hf.space/ocr" \
  -F "file=@document.pdf"
```

---

## Allowed file types

PDF, DOC, DOCX, PPT, PPTX, JPG, PNG, WebP, BMP, TIFF, GIF.

---

## Errors

- **400** – Unsupported file type or invalid request (see `detail` in JSON).
- **502 / 503** – Space is starting (cold start) or overloaded; retry after a few seconds.
