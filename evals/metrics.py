"""Scoring (plan Step 4): per-receipt scores, then aggregation over a split.

Header fields: each total field counts as TP (correct), FN (missing or wrong),
FP (extra or wrong). A wrong value is both an FP and an FN, the standard rule
for key-information extraction.

Items: an item is correct when it is matched to a ground-truth item, its name
matches, and its price and quantity agree whenever the label has them.
"""
import statistics

from evals.matching import match_items, names_match
from receipt.normalize import numbers_equal
from receipt.schema import TOTAL_FIELDS
from receipt.validation import check_receipt

ITEM_ATTRS = ["name", "qty", "unit_price", "price"]


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}


def score_receipt(pred: dict | None, gt: dict) -> dict:
    pred = pred or {"items": [], "totals": {}}
    pt, gtt = pred.get("totals") or {}, gt["totals"]

    # header fields
    fields = {}
    for f in TOTAL_FIELDS:
        p, g = pt.get(f), gtt.get(f)
        if g is None and p is None:
            continue
        if g is not None and p is not None and numbers_equal(p, g):
            fields[f] = {"tp": 1, "fp": 0, "fn": 0, "pred": p, "gt": g}
        else:
            fields[f] = {"tp": 0, "fp": int(p is not None), "fn": int(g is not None), "pred": p, "gt": g}

    # items
    p_items, g_items = pred.get("items") or [], gt["items"]
    pairs, extra, missed = match_items(p_items, g_items)
    attrs = {a: {"correct": 0, "total": 0} for a in ITEM_ATTRS}
    item_errors = []
    correct_items = 0
    for pair in pairs:
        p, g = p_items[pair.pred_idx], g_items[pair.gt_idx]
        ok = {"name": names_match(p.get("name", ""), g["name"])}
        for a in ("qty", "unit_price", "price"):
            if g.get(a) is not None:
                ok[a] = numbers_equal(p.get(a), g[a])
        for a, v in ok.items():
            attrs[a]["total"] += 1
            attrs[a]["correct"] += int(v)
        if all(ok.values()):
            correct_items += 1
        else:
            item_errors.append({
                "type": "wrong", "gt": g["name"], "pred": p.get("name"),
                "wrong": [a for a, v in ok.items() if not v],
                "detail": {a: {"pred": p.get(a), "gt": g.get(a)} for a, v in ok.items() if not v},
            })
    item_errors += [{"type": "missed", "gt": g_items[j]["name"], "gt_price": g_items[j].get("price")} for j in missed]
    item_errors += [{"type": "extra", "pred": p_items[i].get("name"), "pred_price": p_items[i].get("price")}
                    for i in extra]

    checks = check_receipt(pred) if pred.get("items") or pt else []
    item_prf = _prf(correct_items, len(p_items) - correct_items, len(g_items) - correct_items)
    fields_ok = all(v["tp"] for v in fields.values())
    return {
        "fields": fields,
        "items": {"tp": correct_items, "n_pred": len(p_items), "n_gt": len(g_items), "f1": item_prf["f1"]},
        "item_attrs": attrs,
        "item_errors": item_errors,
        "math": {
            "consistent": (all(c.passed for c in checks) if checks else None),
            "failed": [f"{c.name}: {c.detail}" for c in checks if not c.passed],
        },
        "exact": fields_ok and correct_items == len(g_items) == len(p_items),
    }


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
    return values[k]


def aggregate(records: list[dict], pricing: tuple[float, float] | None) -> dict:
    """records: one dict per receipt with keys score, error, input_tokens, output_tokens, latency_s, cached."""
    n = len(records)
    per_field = {}
    for f in TOTAL_FIELDS:
        tp = sum(r["score"]["fields"].get(f, {}).get("tp", 0) for r in records)
        fp = sum(r["score"]["fields"].get(f, {}).get("fp", 0) for r in records)
        fn = sum(r["score"]["fields"].get(f, {}).get("fn", 0) for r in records)
        if tp + fp + fn:
            per_field[f] = _prf(tp, fp, fn)
    header = _prf(sum(v["tp"] for v in per_field.values()), sum(v["fp"] for v in per_field.values()),
                  sum(v["fn"] for v in per_field.values()))

    tp = sum(r["score"]["items"]["tp"] for r in records)
    n_pred = sum(r["score"]["items"]["n_pred"] for r in records)
    n_gt = sum(r["score"]["items"]["n_gt"] for r in records)
    items = _prf(tp, n_pred - tp, n_gt - tp)

    attr_acc = {}
    for a in ITEM_ATTRS:
        c = sum(r["score"]["item_attrs"][a]["correct"] for r in records)
        t = sum(r["score"]["item_attrs"][a]["total"] for r in records)
        attr_acc[a] = round(c / t, 4) if t else None

    math_known = [r["score"]["math"]["consistent"] for r in records if r["score"]["math"]["consistent"] is not None]
    error_types = {}
    for r in records:
        if r.get("error"):
            kind = r["error"].split(":", 1)[0]
            error_types[kind] = error_types.get(kind, 0) + 1

    fresh = [r for r in records if not r.get("cached")]
    lat = [r["latency_s"] for r in records if not r.get("error")]
    in_tok = sum(r.get("input_tokens", 0) for r in records)
    out_tok = sum(r.get("output_tokens", 0) for r in records)
    cost = None
    if pricing:
        cost = round((in_tok * pricing[0] + out_tok * pricing[1]) / 1_000_000, 4)

    return {
        "receipts": n,
        "header_fields": header,
        "per_field": per_field,
        "items": items,
        "item_attribute_accuracy": attr_acc,
        "math_consistency_rate": round(sum(math_known) / len(math_known), 4) if math_known else None,
        "receipt_exact_match": round(sum(r["score"]["exact"] for r in records) / n, 4) if n else 0,
        "schema_valid_rate": round(sum(1 for r in records if not r.get("error")) / n, 4) if n else 0,
        "errors": error_types,
        "hallucination_rate": None,  # needs the support judge (plan Step 6)
        "latency_s": {"p50": _pct(lat, 0.5), "p95": _pct(lat, 0.95),
                      "mean": round(statistics.mean(lat), 2) if lat else 0},
        "tokens": {"input": in_tok, "output": out_tok,
                   "per_receipt": round((in_tok + out_tok) / n) if n else 0},
        "cost_usd": {"total": cost, "per_receipt": round(cost / n, 6) if cost is not None and n else None},
        "cache": {"hits": n - len(fresh), "misses": len(fresh)},
    }
