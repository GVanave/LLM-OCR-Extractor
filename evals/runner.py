"""Eval harness runner (plan Step 5): predict -> score -> aggregate -> report.

Every run writes runs/<timestamp>_<model>_<prompt>/ with:
  config.json        everything needed to reproduce the run
  predictions.jsonl  one line per receipt: prediction, usage, score
  report.json        aggregated metrics
  report.html        summary + the worst receipts with their diffs
"""
import copy
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from evals.cord import DATA_DIR, Sample, dataset_revision, load_split
from evals.metrics import aggregate, score_receipt
from evals.report import write_html
from receipt.extractor import Extraction, extract_receipt, load_prompt, prompt_hash
from receipt.schema import SCHEMA_VERSION

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
CACHE_DIR = ROOT / ".cache" / "responses"

# USD per 1M tokens (input, output). Check current Gemini pricing before trusting cost numbers.
PRICING = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
}


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def cache_key(image: bytes, prompt: str, model: str) -> str:
    h = hashlib.sha256()
    for part in (image, prompt.encode(), model.encode(), SCHEMA_VERSION.encode()):
        h.update(part)
        h.update(b"\x00")
    return h.hexdigest()


# ---------- predictors ----------
class GeminiPredictor:
    def __init__(self, model: str, prompt: str, use_cache: bool = True, cache_dir: Path = CACHE_DIR):
        from google import genai

        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            raise SystemExit("GEMINI_API_KEY is not set (put it in .env), or use --mock to test the harness.")
        self.client = genai.Client(api_key=key)
        self.model, self.prompt, self.use_cache, self.cache_dir = model, prompt, use_cache, cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, sample: Sample) -> tuple[Extraction, bool]:
        image = sample.image_path.read_bytes()
        path = self.cache_dir / f"{cache_key(image, self.prompt, self.model)}.json"
        if self.use_cache and path.exists():
            return Extraction(**json.loads(path.read_text(encoding="utf-8"))), True
        result = extract_receipt(image, client=self.client, model=self.model, prompt=self.prompt)
        if not (result.error or "").startswith("api"):  # never cache transient API failures
            path.write_text(json.dumps(result.to_dict(), ensure_ascii=False), encoding="utf-8")
        return result, False


class MockPredictor:
    """Returns the label with deterministic, realistic mistakes. For testing the harness only."""

    def __call__(self, sample: Sample) -> tuple[Extraction, bool]:
        pred = copy.deepcopy(sample.expected)
        h = int(hashlib.md5(sample.id.encode()).hexdigest(), 16)
        items = pred["items"]
        if h % 4 == 0 and len(items) > 1:
            items.pop()  # missed row
        if h % 5 == 1 and items and items[0]["price"] is not None:
            items[0]["price"] += 1000  # misread digit
        if h % 6 == 2:
            pred["totals"]["tax"] = None  # missed field
        if h % 7 == 3 and items:
            items[0]["name"] = items[0]["name"][: max(3, len(items[0]["name"]) // 2)]  # truncated name
        return Extraction(receipt=pred, input_tokens=1500, output_tokens=300, latency_s=0.01, attempts=1), False


# ---------- run ----------
def run(split: str, model: str, prompt_version: str, *, mock: bool = False, limit: int | None = None,
        workers: int = 2, use_cache: bool = True, data_dir: Path = DATA_DIR, runs_dir: Path = RUNS_DIR,
        cache_dir: Path = CACHE_DIR) -> Path:
    samples = load_split(split, data_dir, limit)
    prompt = load_prompt(prompt_version)
    predictor = MockPredictor() if mock else GeminiPredictor(model, prompt, use_cache, cache_dir)
    model_name = "mock" if mock else model

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = runs_dir / f"{stamp}_{model_name}_{prompt_version}_{split}"
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "split": split, "n": len(samples), "model": model_name, "prompt_version": prompt_version,
        "prompt_sha": prompt_hash(prompt), "schema_version": SCHEMA_VERSION,
        "dataset_revision": dataset_revision(split, data_dir), "git_commit": git_commit(),
        "started_at": datetime.now().isoformat(timespec="seconds"), "cache": use_cache and not mock,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    def one(sample: Sample) -> dict:
        ext, cached = predictor(sample)
        return {
            "id": sample.id, "image": str(sample.image_path), "pred": ext.receipt, "error": ext.error,
            "input_tokens": ext.input_tokens, "output_tokens": ext.output_tokens,
            "latency_s": ext.latency_s, "attempts": ext.attempts, "cached": cached,
            "score": score_receipt(ext.receipt, sample.expected), "gt": sample.expected,
        }

    started = time.time()
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, rec in enumerate(pool.map(one, samples), start=1):
            records.append(rec)
            flag = "cached" if rec["cached"] else f"{rec['latency_s']}s"
            status = rec["error"] or f"items F1 {rec['score']['items']['f1']:.2f}"
            print(f"[{i}/{len(samples)}] {rec['id']} ({flag}) {status}")

    with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report = aggregate(records, None if mock else PRICING.get(model))
    report["wall_time_s"] = round(time.time() - started, 1)
    report["config"] = config
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_html(out, report, records)
    print_summary(report, out)
    return out


def print_summary(report: dict, out: Path) -> None:
    h, it = report["header_fields"], report["items"]
    print("\n=== Summary ===")
    print(f"Receipts:            {report['receipts']}  (schema valid {report['schema_valid_rate']:.1%})")
    print(f"Header fields F1:    {h['f1']:.1%}  (P {h['precision']:.1%} / R {h['recall']:.1%})")
    print(f"Line items F1:       {it['f1']:.1%}  (P {it['precision']:.1%} / R {it['recall']:.1%})")
    acc = report["item_attribute_accuracy"]
    print("Item attributes:     " + ", ".join(f"{k} {v:.1%}" for k, v in acc.items() if v is not None))
    m = report["math_consistency_rate"]
    print(f"Math consistency:    {m:.1%}" if m is not None else "Math consistency:    n/a")
    print(f"Receipt exact match: {report['receipt_exact_match']:.1%}")
    print(f"Latency p50/p95:     {report['latency_s']['p50']}s / {report['latency_s']['p95']}s")
    c = report["cost_usd"]
    if c["total"] is not None:
        print(f"Cost:                ${c['total']} total, ${c['per_receipt']} per receipt")
    if report["errors"]:
        print(f"Errors:              {report['errors']}")
    print(f"\nReport: {out / 'report.html'}")
