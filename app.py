"""Streamlit frontend: enter Gemini API key + model, upload a document, see boxes and extracted data."""
import io
import json

import streamlit as st
from PIL import Image, ImageDraw

import pipeline

st.set_page_config(page_title="Gemini OCR", layout="wide")

LABEL_COLORS = {
    "heading": "#dc2626", "paragraph": "#2563eb", "line": "#2563eb", "table_cell": "#16a34a",
    "key": "#9333ea", "value": "#ea580c", "handwriting": "#db2777",
}

# ---------- Sidebar: settings ----------
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Gemini API key",
        type="password",
        placeholder="Uses GEMINI_API_KEY from .env if empty" if pipeline.DEFAULT_API_KEY else "Paste your key",
        help="Get one at https://aistudio.google.com/apikey. The key is only kept for this browser session.",
    )
    choices = list(dict.fromkeys([pipeline.DEFAULT_MODEL, *pipeline.MODEL_CHOICES]))
    model = st.selectbox(
        "Model",
        choices,
        accept_new_options=True,
        help="Pick one or type any Gemini model name.",
    )
    show_boxes = st.checkbox("Show boxes", value=True)
    show_numbers = st.checkbox("Show box numbers", value=True)

    if not (api_key or pipeline.DEFAULT_API_KEY) or pipeline.FORCE_MOCK:
        st.warning("No API key: running in MOCK mode with fake results.")


def draw_boxes(page: dict) -> Image.Image:
    img = Image.open(io.BytesIO(page["png"])).convert("RGB")
    if not show_boxes:
        return img
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for i, b in enumerate(page["blocks"], start=1):
        y0, x0, y1, x1 = b["box_2d"]
        rect = [x0 * w / 1000, y0 * h / 1000, x1 * w / 1000, y1 * h / 1000]
        color = LABEL_COLORS.get(b["label"], "#6b7280")
        draw.rectangle(rect, outline=color, width=2)
        if show_numbers:
            draw.text((rect[0] + 2, max(0, rect[1] - 12)), str(i), fill=color)
    return img


# ---------- Main ----------
st.title("Gemini OCR Tester")
uploaded = st.file_uploader("Upload an image or PDF", type=["png", "jpg", "jpeg", "webp", "bmp", "tiff", "pdf"])

if uploaded and st.button("Extract", type="primary"):
    with st.spinner(f"Extracting with {model}…"):
        try:
            st.session_state.result = pipeline.extract(uploaded.getvalue(), uploaded.name, api_key, model)
        except pipeline.OCRError as e:
            st.session_state.pop("result", None)
            st.error(str(e))

result = st.session_state.get("result")
if result:
    st.caption(f"{result['filename']} · {len(result['pages'])} page(s) · {result['seconds']}s · {result['model']}")

    export = {**result, "pages": [{k: v for k, v in p.items() if k != "png"} for p in result["pages"]]}
    st.download_button(
        "Download JSON",
        json.dumps(export, indent=2, ensure_ascii=False),
        file_name=f"{result['filename']}.json",
        mime="application/json",
    )

    for page in result["pages"]:
        st.subheader(f"Page {page['page']} · {page['document_type']}")
        left, right = st.columns([1.2, 1])
        with left:
            st.image(draw_boxes(page), width="stretch")
        with right:
            t_blocks, t_fields, t_text, t_json = st.tabs(["Text blocks", "Fields", "Plain text", "JSON"])
            with t_blocks:
                st.dataframe(
                    [{"#": i, "text": b["text"], "label": b["label"], "box_2d": str(b["box_2d"])}
                     for i, b in enumerate(page["blocks"], start=1)],
                    hide_index=True, width="stretch",
                )
            with t_fields:
                if page["fields"]:
                    st.dataframe(page["fields"], hide_index=True, width="stretch")
                else:
                    st.info("No fields found.")
            with t_text:
                st.text("\n".join(b["text"] for b in page["blocks"]))
            with t_json:
                st.json({k: v for k, v in page.items() if k != "png"}, expanded=False)
