"""FastAPI backend for the OCR pipeline (used by index.html or any client)."""
import base64
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

import pipeline

app = FastAPI(title="Gemini OCR")


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "model": pipeline.DEFAULT_MODEL,
        "mock": pipeline.FORCE_MOCK or not pipeline.DEFAULT_API_KEY,
    }


@app.post("/api/extract")
async def extract(
    file: UploadFile = File(...),
    api_key: str = Form(""),
    model: str = Form(""),
):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    try:
        result = pipeline.extract(data, file.filename or "", api_key, model)
    except pipeline.OCRError as e:
        status = 400 if "Could not read" in str(e) else 502
        raise HTTPException(status, str(e))
    for p in result["pages"]:
        p["image"] = "data:image/png;base64," + base64.b64encode(p.pop("png")).decode()
    return result
