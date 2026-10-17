"""Simple OCR pipeline: upload image/PDF -> Gemini -> text + bounding boxes + fields (JSON)."""
import base64
import io
import os
import time

import pymupdf
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MOCK = os.getenv("MOCK", "0") == "1" or not API_KEY
MAX_PAGES = int(os.getenv("MAX_PAGES", "10"))
MAX_SIDE = 2000  # px, longest image side sent to the model

client = None if MOCK else genai.Client(api_key=API_KEY)
app = FastAPI(title="Gemini OCR")


# ---------- Schema the model must return ----------
class TextBlock(BaseModel):
    text: str
    label: str  # e.g. heading, paragraph, table_cell, key, value, handwriting
    box_2d: list[int]  # [ymin, xmin, ymax, xmax], normalized 0-1000


class Field(BaseModel):
    key: str
    value: str


class PageResult(BaseModel):
    document_type: str
    blocks: list[TextBlock]
    fields: list[Field]


PROMPT = """You are an OCR engine. Read ALL text in this image.
Return:
- document_type: short guess (invoice, receipt, form, id_card, letter, other...)
- blocks: every text line/block with its exact text, a label
  (heading, paragraph, line, table_cell, key, value, handwriting, other)
  and box_2d as [ymin, xmin, ymax, xmax] normalized to 0-1000.
- fields: important key/value pairs (e.g. invoice_number, date, total, name).
Transcribe text exactly as written. Do not invent text."""


# ---------- Pipeline steps ----------
def to_page_images(data: bytes, filename: str) -> list[Image.Image]:
    """Preprocess: PDF -> page images, image -> RGB, resized."""
    pages = []
    if filename.lower().endswith(".pdf") or data[:4] == b"%PDF":
        doc = pymupdf.open(stream=data, filetype="pdf")
        for page in list(doc)[:MAX_PAGES]:
            pix = page.get_pixmap(dpi=150)
            pages.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
    else:
        pages.append(Image.open(io.BytesIO(data)).convert("RGB"))
    for i, img in enumerate(pages):
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        pages[i] = img
    return pages


def call_gemini(png: bytes) -> PageResult:
    """One structured LLM call per page, one retry on failure."""
    last_err = None
    for _ in range(2):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=[types.Part.from_bytes(data=png, mime_type="image/png"), PROMPT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=PageResult,
                    temperature=0,
                ),
            )
            if resp.parsed:
                return resp.parsed
            return PageResult.model_validate_json(resp.text)
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1)
    raise HTTPException(502, f"Gemini call failed: {last_err}")


def mock_result() -> PageResult:
    return PageResult(
        document_type="mock",
        blocks=[
            TextBlock(text="MOCK MODE - set GEMINI_API_KEY", label="heading", box_2d=[40, 50, 110, 700]),
            TextBlock(text="Sample paragraph text", label="paragraph", box_2d=[150, 50, 220, 600]),
        ],
        fields=[Field(key="mode", value="mock")],
    )


def validate(result: PageResult) -> PageResult:
    """Drop malformed boxes, clamp to 0-1000, fix inverted coordinates."""
    clean = []
    for b in result.blocks:
        if len(b.box_2d) != 4 or not b.text.strip():
            continue
        y0, x0, y1, x1 = (max(0, min(1000, v)) for v in b.box_2d)
        b.box_2d = [min(y0, y1), min(x0, x1), max(y0, y1), max(x0, x1)]
        clean.append(b)
    result.blocks = clean
    return result


# ---------- API ----------
@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


@app.get("/api/health")
def health():
    return {"ok": True, "model": MODEL, "mock": MOCK}


@app.post("/api/extract")
async def extract(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    try:
        images = to_page_images(data, file.filename or "")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not read file as image/PDF: {e}")

    started = time.time()
    pages = []
    for idx, img in enumerate(images):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png = buf.getvalue()
        result = mock_result() if MOCK else call_gemini(png)
        result = validate(result)
        pages.append({
            "page": idx + 1,
            "width": img.width,
            "height": img.height,
            "image": "data:image/png;base64," + base64.b64encode(png).decode(),
            **result.model_dump(),
        })

    return {
        "filename": file.filename,
        "model": "mock" if MOCK else MODEL,
        "seconds": round(time.time() - started, 2),
        "pages": pages,
    }
