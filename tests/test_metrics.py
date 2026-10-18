import copy
import json

from PIL import Image

from evals.matching import match_items, names_match
from evals.metrics import aggregate, score_receipt


def item(name, price, qty=1):
    return {"name": name, "qty": qty, "unit_price": None, "price": price, "discount": None,
            "sub_items": [], "box_2d": None}


def receipt(items, **totals):
    base = dict.fromkeys(["subtotal", "tax", "service", "discount", "total", "cash", "change"])
    base.update(totals)
    return {"items": items, "totals": base, "currency": "IDR"}


GT = receipt([item("REAL GANACHE", 16500), item("EGG TART", 13000), item("PIZZA TOAST", 16000)],
             total=45500, cash=50000, change=4500)


def test_matching_ignores_order():
    pred = list(reversed(GT["items"]))
    pairs, extra, missed = match_items(pred, GT["items"])
    assert len(pairs) == 3 and not extra and not missed
    assert {(p.pred_idx, p.gt_idx) for p in pairs} == {(0, 2), (1, 1), (2, 0)}


def test_names_match_fuzzy():
    assert names_match("Es Teh Manis", "ES TEH MANIS")
    assert names_match("PIZZA TOAST", "PIZA TOAST")
    assert not names_match("EGG TART", "PIZZA TOAST")


def test_perfect_prediction():
    s = score_receipt(copy.deepcopy(GT), GT)
    assert s["exact"] and s["items"]["f1"] == 1.0 and s["math"]["consistent"]


def test_missed_item_and_wrong_total():
    pred = copy.deepcopy(GT)
    pred["items"].pop(1)
    pred["totals"]["total"] = 32500
    s = score_receipt(pred, GT)
    assert s["items"]["tp"] == 2 and s["items"]["n_pred"] == 2 and s["items"]["n_gt"] == 3
    assert s["fields"]["total"] == {"tp": 0, "fp": 1, "fn": 1, "pred": 32500, "gt": 45500}
    assert any(e["type"] == "missed" and e["gt"] == "EGG TART" for e in s["item_errors"])
    assert s["math"]["consistent"] is False  # cash - total != change
    assert not s["exact"]


def test_wrong_price_is_counted_on_matched_item():
    pred = copy.deepcopy(GT)
    pred["items"][0]["price"] = 15500
    s = score_receipt(pred, GT)
    assert s["items"]["tp"] == 2
    assert s["item_attrs"]["price"] == {"correct": 2, "total": 3}
    wrong = [e for e in s["item_errors"] if e["type"] == "wrong"]
    assert wrong[0]["wrong"] == ["price"]


def test_no_prediction():
    s = score_receipt(None, GT)
    assert s["items"]["tp"] == 0 and s["fields"]["total"]["fn"] == 1 and s["math"]["consistent"] is None


def test_aggregate():
    recs = [
        {"score": score_receipt(copy.deepcopy(GT), GT), "error": None, "input_tokens": 1000,
         "output_tokens": 200, "latency_s": 2.0, "cached": False},
        {"score": score_receipt(None, GT), "error": "schema: bad", "input_tokens": 1000,
         "output_tokens": 50, "latency_s": 3.0, "cached": True},
    ]
    a = aggregate(recs, (0.30, 2.50))
    assert a["items"]["recall"] == 0.5 and a["items"]["precision"] == 1.0
    assert a["schema_valid_rate"] == 0.5 and a["errors"] == {"schema": 1}
    assert a["math_consistency_rate"] == 1.0
    assert a["cost_usd"]["total"] == round((2000 * 0.30 + 250 * 2.50) / 1e6, 4)
    assert a["cache"] == {"hits": 1, "misses": 1}


def test_mock_run_end_to_end(tmp_path):
    from evals.runner import run

    split = tmp_path / "data" / "dev"
    split.mkdir(parents=True)
    ids = []
    for i in range(6):
        sid = f"dev_{i:04d}"
        Image.new("RGB", (60, 80), "white").save(split / f"{sid}.jpg")
        (split / f"{sid}.json").write_text(json.dumps({"id": sid, "expected": GT}), encoding="utf-8")
        ids.append(sid)
    (split / "manifest.json").write_text(json.dumps({"revision": "test", "ids": ids}), encoding="utf-8")

    out = run("dev", "gemini-2.5-flash", "v1", mock=True, data_dir=tmp_path / "data",
              runs_dir=tmp_path / "runs", workers=2)
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["receipts"] == 6 and report["config"]["model"] == "mock"
    assert 0 < report["items"]["f1"] <= 1
    assert (out / "report.html").exists()
    assert len((out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 6
