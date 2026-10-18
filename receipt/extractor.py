"""Gemini receipt extractor: one schema-constrained call per receipt image."""
import hashlib
import io
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

from receipt.schema import Receipt

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
MAX_SIDE = 2000


def load_prompt(version: str) -> str:
    path = PROMPTS_DIR / f"receipt_{version}.txt"
    if not path.exists():
        raise SystemExit(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def prepare_image(data: bytes) -> bytes:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@dataclass
class Extraction:
    receipt: dict | None
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0  # includes thinking tokens, which are billed as output
    latency_s: float = 0.0
    attempts: int = 0
    raw: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def extract_receipt(image: bytes, *, client: genai.Client, model: str, prompt: str,
                    max_attempts: int = 5) -> Extraction:
    """Call Gemini with exponential backoff on rate limits and server errors."""
    jpeg = prepare_image(image)
    started = time.time()
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=Receipt,
                    temperature=0,
                ),
            )
            usage = resp.usage_metadata
            out = Extraction(
                receipt=None,
                input_tokens=(usage.prompt_token_count or 0) if usage else 0,
                output_tokens=((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0,
                latency_s=round(time.time() - started, 2),
                attempts=attempt,
                raw=resp.text,
            )
            try:
                parsed = resp.parsed if isinstance(resp.parsed, Receipt) else Receipt.model_validate_json(resp.text)
                out.receipt = parsed.model_dump()
            except Exception as e:  # noqa: BLE001 - schema failure is a measured outcome
                out.error = f"schema: {e}"
            return out
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e)
            retryable = any(code in msg for code in ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))
            if not retryable or attempt == max_attempts:
                break
            time.sleep(min(2 ** attempt, 30))
    return Extraction(receipt=None, error=f"api: {last_err}", latency_s=round(time.time() - started, 2),
                      attempts=attempt)
