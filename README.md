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

## Receipt line-item extraction + eval harness (plan.md Phase 1)

```
receipt/schema.py       Receipt -> items (qty, unit_price, price, discount, sub_items, box) + totals
receipt/normalize.py    "16,500" / "Rp 16.500" / "@16500" -> 16500, "3X" -> 3, item-name cleanup
receipt/validation.py   math checks: qty*unit = price, items = subtotal, subtotal -> total, cash - total = change
receipt/extractor.py    one Gemini call per receipt, schema-constrained, backoff on 429/503
prompts/receipt_v1.txt  versioned prompt
evals/                  CORD v2 loader, item matching, metrics, runner, cache, HTML report, compare
tests/                  unit tests (no API calls)
```

```bash
python -m pip install -r requirements-dev.txt
python -m pytest                                   # unit tests
python -m evals prepare                            # download CORD v2 -> data/cord/{dev,test}
python -m evals stats --set dev                    # data report
python -m evals run --set dev --mock               # test the harness without an API key
python -m evals run --set dev --model gemini-2.5-flash --prompt v1 --limit 10
python -m evals compare <run_a> <run_b>            # metric deltas + receipts that got better/worse
```

- `dev` (100 receipts) is for iterating; `test` (100) is frozen, for final scores only.
- The dataset is pinned to a fixed Hugging Face revision in `evals/cord.py`.
- Responses are cached in `.cache/responses/` by image + prompt + model + schema version, so re-scoring is free.
- Each run writes `runs/<timestamp>_<model>_<prompt>_<split>/` with `config.json`, `predictions.jsonl`, `report.json` and `report.html` (worst receipts with diffs).
- About 4% of CORD labels fail a math check (rounding lines, label errors), so ~96% is the realistic ceiling for math consistency.
- Costs use the price table in `evals/runner.py`. Check current Gemini pricing.

## Pipeline (pipeline.py)
1. `to_page_images`: PDF to page images (150 dpi), resized to a max side of 2000px
2. `call_gemini`: one schema-constrained call per page, temperature 0, one retry
3. `validate`: drop or clamp bad boxes
4. `extract`: runs all steps and returns JSON for downstream processing
