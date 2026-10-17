# Gemini OCR Tester

Upload an image or PDF and get text, bounding boxes and key/value fields from Gemini.

- `pipeline.py`: the core OCR pipeline, shared by both frontends
- `app.py`: Streamlit UI, where you enter the API key and model name in the sidebar
- `main.py` + `index.html`: FastAPI backend with a plain HTML test page

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run the Streamlit app

```bash
streamlit run app.py
```

Enter your **Gemini API key** (get one at https://aistudio.google.com/apikey) and pick or type a **model name** in the sidebar. Then upload a file and click **Extract**.

A `.env` file is optional. Copy `.env.example` to `.env` to set a default key and model. Without any key the app runs in **MOCK** mode, which returns fake results.

## Run the FastAPI backend

```bash
uvicorn main:app --reload
```

`POST /api/extract` takes multipart form fields `file` (required) plus optional `api_key` and `model`. It returns:

```json
{ "filename": "...", "model": "...", "seconds": 2.1,
  "pages": [{ "page": 1, "width": 1240, "height": 1754, "image": "data:...",
              "document_type": "invoice",
              "blocks": [{ "text": "...", "label": "heading", "box_2d": [ymin, xmin, ymax, xmax] }],
              "fields": [{ "key": "total", "value": "120.00" }] }] }
```

`box_2d` is normalized to 0–1000. Pixel coordinates are `y * height / 1000` and `x * width / 1000`.

## Pipeline (pipeline.py)
1. `to_page_images`: PDF to page images (150 dpi), resized to a max side of 2000px
2. `call_gemini`: one schema-constrained call per page, temperature 0, one retry
3. `validate`: drop or clamp bad boxes
4. `extract`: runs all steps and returns JSON for downstream processing
