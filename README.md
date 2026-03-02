---
title: OCR Agent
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# OCR Agent

Universal OCR: PDF, DOC, DOCX, PPT, PPTX, and images. Smart engine routing (printed → Tesseract, handwritten → PaddleOCR). FastAPI + parser for structured output.

## Local run (Windows)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000

## Hugging Face Space

This Space runs the app in Docker. Upload a document and click **Run OCR**.
