# Gemini OCR Tester

FastAPI backend + single-page frontend. Upload an image or PDF and get text, bounding boxes and key/value fields from Gemini.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.example .env      # then put your GEMINI_API_KEY in .env
uvicorn main:app --reload
```

Open http://127.0.0.1:8000

Without an API key the app runs in **MOCK** mode, which returns fake boxes so you can test the UI.

## API

`POST /api/extract` (multipart `file`) returns:

```json
{ "filename": "...", "model": "...", "seconds": 2.1,
  "pages": [{ "page": 1, "width": 1240, "height": 1754, "image": "data:...",
              "document_type": "invoice",
              "blocks": [{ "text": "...", "label": "heading", "box_2d": [ymin, xmin, ymax, xmax] }],
              "fields": [{ "key": "total", "value": "120.00" }] }] }
```

`box_2d` is normalized to 0–1000. Pixel coordinates are `y * height / 1000` and `x * width / 1000`.

## Pipeline (main.py)
1. `to_page_images`: PDF to page images (150 dpi), resized to a max side of 2000px
2. `call_gemini`: one schema-constrained call per page, temperature 0, one retry
3. `validate`: drop or clamp bad boxes
4. Return JSON so downstream processing can consume it
