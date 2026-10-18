"""CORD v2 dataset: download, convert labels to our schema, load splits (plan Step 1).

Splits are fixed by pinning the Hugging Face dataset revision:
  dev  = CORD 'validation' (100 receipts), used for iterating on prompts
  test = CORD 'test'       (100 receipts), frozen, final scores only
"""
import io
import json
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from receipt.normalize import parse_amount, parse_number, parse_qty
from receipt.validation import check_receipt

REPO = "naver-clova-ix/cord-v2"
REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"
SPLIT_FILES = {
    "dev": "data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet",
    "test": "data/test-00000-of-00001-9c204eb3f4e11791.parquet",
}
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "cord"


@dataclass
class Sample:
    id: str
    image_path: Path
    expected: dict


# ---------- label conversion ----------
def _as_list(v) -> list:
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _text(v) -> str:
    return " ".join(str(x) for x in _as_list(v)).strip()


def convert_gt(gt_parse: dict) -> dict:
    """CORD gt_parse -> our Receipt schema (as a plain dict, numbers parsed)."""
    items = []
    for m in _as_list(gt_parse.get("menu")):
        if not isinstance(m, dict):
            continue
        items.append({
            "name": _text(m.get("nm")),
            "qty": parse_qty(m.get("cnt")),
            "unit_price": parse_amount(m.get("unitprice")),
            "price": parse_number(m.get("price")),
            "discount": parse_amount(m.get("discountprice")),
            "sub_items": [
                {"name": _text(s.get("nm")), "qty": parse_qty(s.get("cnt")), "price": parse_number(s.get("price"))}
                for s in _as_list(m.get("sub")) if isinstance(s, dict)
            ],
            "box_2d": None,
        })

    sub = gt_parse.get("sub_total") or {}
    tot = gt_parse.get("total") or {}
    sub = sub if isinstance(sub, dict) else {}
    tot = tot if isinstance(tot, dict) else {}
    totals = {
        "subtotal": parse_number(sub.get("subtotal_price")),
        "tax": parse_number(sub.get("tax_price")),
        "service": parse_number(sub.get("service_price")),
        "discount": parse_amount(sub.get("discount_price")),
        "total": parse_number(tot.get("total_price")),
        "cash": parse_number(tot.get("cashprice")),
        "change": parse_amount(tot.get("changeprice")),  # sometimes printed as "-4,500"
    }
    return {"items": items, "totals": totals, "currency": "IDR"}


# ---------- download / load ----------
def prepare(splits=("dev", "test"), data_dir: Path = DATA_DIR) -> None:
    """Download the pinned parquet files and write <id>.jpg + <id>.json per receipt."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    for split in splits:
        path = hf_hub_download(REPO, SPLIT_FILES[split], repo_type="dataset", revision=REVISION)
        out = data_dir / split
        out.mkdir(parents=True, exist_ok=True)
        ids = []
        for idx, row in enumerate(pq.read_table(path).to_pylist()):
            sid = f"{split}_{idx:04d}"
            img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            img.save(out / f"{sid}.jpg", quality=95)
            gt = json.loads(row["ground_truth"])
            (out / f"{sid}.json").write_text(json.dumps({
                "id": sid,
                "expected": convert_gt(gt["gt_parse"]),
                "raw_gt_parse": gt["gt_parse"],
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            ids.append(sid)
        (out / "manifest.json").write_text(json.dumps({
            "repo": REPO, "revision": REVISION, "source_file": SPLIT_FILES[split],
            "split": split, "count": len(ids), "ids": ids,
        }, indent=2), encoding="utf-8")
        print(f"{split}: wrote {len(ids)} receipts to {out}")


def load_split(split: str, data_dir: Path = DATA_DIR, limit: int | None = None) -> list[Sample]:
    folder = data_dir / split
    manifest = folder / "manifest.json"
    if not manifest.exists():
        raise SystemExit(f"Split '{split}' not found in {folder}. Run: python -m evals prepare")
    ids = json.loads(manifest.read_text(encoding="utf-8"))["ids"]
    samples = []
    for sid in ids[:limit]:
        label = json.loads((folder / f"{sid}.json").read_text(encoding="utf-8"))
        samples.append(Sample(sid, folder / f"{sid}.jpg", label["expected"]))
    return samples


def dataset_revision(split: str, data_dir: Path = DATA_DIR) -> str:
    return json.loads((data_dir / split / "manifest.json").read_text(encoding="utf-8"))["revision"]


# ---------- data report ----------
def stats(samples: list[Sample]) -> dict:
    """Shape of the data: where the hard cases are, and how clean the labels are."""
    n_items = [len(s.expected["items"]) for s in samples]
    fields = Counter()
    gt_checks = Counter()
    failing = []
    for s in samples:
        e = s.expected
        fields.update(k for k, v in e["totals"].items() if v is not None)
        if any(it["sub_items"] for it in e["items"]):
            fields["<has sub_items>"] += 1
        if any(it["discount"] for it in e["items"]):
            fields["<has item discount>"] += 1
        checks = check_receipt(e)
        if not checks:
            gt_checks["no check applies"] += 1
        elif all(c.passed for c in checks):
            gt_checks["consistent"] += 1
        else:
            gt_checks["inconsistent"] += 1
            failing.append({"id": s.id, "failed": [f"{c.name}: {c.detail}" for c in checks if not c.passed]})
    return {
        "receipts": len(samples),
        "items_per_receipt": {
            "mean": round(statistics.mean(n_items), 2) if n_items else 0,
            "median": statistics.median(n_items) if n_items else 0,
            "max": max(n_items, default=0),
        },
        "receipts_with_field": dict(fields.most_common()),
        "label_math_consistency": dict(gt_checks),
        "inconsistent_labels": failing,
    }
