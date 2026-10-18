"""HTML report: headline metrics plus the worst receipts, which is where the learning happens."""
import html
import json
import os
from pathlib import Path

WORST_N = 20


def _pct(v) -> str:
    return "n/a" if v is None else f"{v:.1%}"


def _badness(rec: dict) -> tuple:
    s = rec["score"]
    field_misses = sum(v["fp"] + v["fn"] for v in s["fields"].values())
    return (bool(rec["error"]), -s["items"]["f1"], field_misses)


def write_html(out: Path, report: dict, records: list[dict]) -> None:
    e = html.escape
    rows_fields = "".join(
        f"<tr><td>{e(f)}</td><td>{_pct(v['f1'])}</td><td>{_pct(v['precision'])}</td>"
        f"<td>{_pct(v['recall'])}</td><td>{v['tp']}/{v['fp']}/{v['fn']}</td></tr>"
        for f, v in report["per_field"].items()
    )
    worst = sorted((r for r in records if r["error"] or not r["score"]["exact"]), key=_badness, reverse=True)
    cards = []
    for r in worst[:WORST_N]:
        s = r["score"]
        img = Path(os.path.relpath(r["image"], out)).as_posix()
        field_errs = "".join(
            f"<li><b>{e(f)}</b>: predicted {e(str(v['pred']))}, expected {e(str(v['gt']))}</li>"
            for f, v in s["fields"].items() if not v["tp"]
        )
        item_errs = "".join(f"<li>{e(json.dumps(x, ensure_ascii=False))}</li>" for x in s["item_errors"])
        math = "".join(f"<li>{e(m)}</li>" for m in s["math"]["failed"])
        cards.append(f"""
<div class="card"><img src="{e(img)}" loading="lazy"><div>
<h3>{e(r['id'])} &middot; items F1 {s['items']['f1']:.2f} ({s['items']['tp']}/{s['items']['n_gt']} correct,
{s['items']['n_pred']} predicted)</h3>
{f'<p class="err">Error: {e(r["error"])}</p>' if r['error'] else ''}
{f'<h4>Header fields</h4><ul>{field_errs}</ul>' if field_errs else ''}
{f'<h4>Line items</h4><ul>{item_errs}</ul>' if item_errs else ''}
{f'<h4>Math checks failed</h4><ul>{math}</ul>' if math else ''}
</div></div>""")

    h, it, cfg = report["header_fields"], report["items"], report["config"]
    acc = report["item_attribute_accuracy"]
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Eval {e(out.name)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;color:#1d2129;background:#fff}}
table{{border-collapse:collapse;margin:8px 0 20px}} td,th{{border:1px solid #ddd;padding:4px 10px;text-align:left}}
.kpi{{display:flex;gap:12px;flex-wrap:wrap}} .kpi div{{border:1px solid #ddd;border-radius:8px;padding:10px 14px}}
.kpi b{{display:block;font-size:22px}} .card{{display:flex;gap:16px;border-top:1px solid #ddd;padding:16px 0}}
.card img{{width:260px;object-fit:contain;align-self:flex-start}} .err{{color:#b91c1c}} li{{font-size:13px}}
</style></head><body>
<h1>Eval run {e(out.name)}</h1>
<p>{e(json.dumps(cfg))}</p>
<div class="kpi">
<div>Header F1<b>{_pct(h['f1'])}</b></div><div>Items F1<b>{_pct(it['f1'])}</b></div>
<div>Math consistency<b>{_pct(report['math_consistency_rate'])}</b></div>
<div>Exact receipts<b>{_pct(report['receipt_exact_match'])}</b></div>
<div>Schema valid<b>{_pct(report['schema_valid_rate'])}</b></div>
<div>p95 latency<b>{report['latency_s']['p95']}s</b></div>
<div>Cost / receipt<b>{'n/a' if report['cost_usd']['per_receipt'] is None else '$' + str(report['cost_usd']['per_receipt'])}</b></div>
</div>
<h2>Item attributes (on matched items)</h2>
<table><tr>{''.join(f'<th>{e(k)}</th>' for k in acc)}</tr><tr>{''.join(f'<td>{_pct(v)}</td>' for v in acc.values())}</tr></table>
<h2>Header fields</h2>
<table><tr><th>Field</th><th>F1</th><th>Precision</th><th>Recall</th><th>TP/FP/FN</th></tr>{rows_fields}</table>
<h2>Worst receipts ({min(len(worst), WORST_N)} of {len(worst)} imperfect)</h2>
{''.join(cards)}
</body></html>"""
    (out / "report.html").write_text(doc, encoding="utf-8")
