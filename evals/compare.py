"""Compare two eval runs: metric deltas plus newly broken / newly fixed receipts.

Averages hide regressions: a change that fixes 5 receipts and breaks 8 can look
neutral. The per-receipt diff shows it.
"""
import json
from pathlib import Path

from evals.runner import RUNS_DIR


def _resolve(run: str) -> Path:
    p = Path(run)
    if not p.exists():
        p = RUNS_DIR / run
    if not (p / "report.json").exists():
        raise SystemExit(f"Not a run folder: {run}")
    return p


def _load(p: Path) -> tuple[dict, dict]:
    report = json.loads((p / "report.json").read_text(encoding="utf-8"))
    recs = {}
    for line in (p / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        recs[r["id"]] = r
    return report, recs


def compare(run_a: str, run_b: str) -> None:
    a_dir, b_dir = _resolve(run_a), _resolve(run_b)
    (ra, pa), (rb, pb) = _load(a_dir), _load(b_dir)

    metrics = [
        ("Header F1", lambda r: r["header_fields"]["f1"]),
        ("Items F1", lambda r: r["items"]["f1"]),
        ("Items precision", lambda r: r["items"]["precision"]),
        ("Items recall", lambda r: r["items"]["recall"]),
        ("Math consistency", lambda r: r["math_consistency_rate"]),
        ("Exact receipts", lambda r: r["receipt_exact_match"]),
        ("Schema valid", lambda r: r["schema_valid_rate"]),
        ("p95 latency (s)", lambda r: r["latency_s"]["p95"]),
        ("Cost/receipt ($)", lambda r: r["cost_usd"]["per_receipt"]),
    ]
    print(f"A: {a_dir.name}\nB: {b_dir.name}\n")
    print(f"{'metric':<20}{'A':>10}{'B':>10}{'delta':>10}")
    for name, get in metrics:
        va, vb = get(ra), get(rb)
        delta = f"{vb - va:+.4f}" if va is not None and vb is not None else ""
        print(f"{name:<20}{str(va):>10}{str(vb):>10}{delta:>10}")

    common = sorted(set(pa) & set(pb))
    broken, fixed = [], []
    for sid in common:
        fa, fb = pa[sid]["score"]["items"]["f1"], pb[sid]["score"]["items"]["f1"]
        ea, eb = pa[sid]["score"]["exact"], pb[sid]["score"]["exact"]
        if fb < fa or (ea and not eb):
            broken.append(f"{sid}: items F1 {fa:.2f} -> {fb:.2f}")
        elif fb > fa or (eb and not ea):
            fixed.append(f"{sid}: items F1 {fa:.2f} -> {fb:.2f}")
    print(f"\nCompared {len(common)} receipts: {len(fixed)} improved, {len(broken)} got worse")
    for title, rows in (("Got worse", broken), ("Improved", fixed)):
        if rows:
            print(f"\n{title}:")
            for row in rows:
                print("  " + row)
