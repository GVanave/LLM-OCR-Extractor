# LLM OCR Extractor: Project Plan

---

## 1. Problem statement

### Context
Businesses process large numbers of receipts and invoices for expense reimbursement, bookkeeping, audits and spend analytics. Most still arrive as **photos or scans**. Extracting the header fields (merchant, date, total) is mostly solved. The hard and valuable part is the **line items**: what was bought, how many, at what price, plus sub-items, discounts, tax and service charges. Today this is either typed by hand or extracted by tools that:
- miss or merge rows of the item table,
- misread prices and number formats ("12.000" as 12 instead of 12,000),
- produce **values that are not on the document**,
- give no reliable signal of **which outputs can be trusted**.

### Problem we are solving
> Build an LLM-based pipeline that takes a **photo or scan of a receipt** and returns **validated, structured data that includes nested line items**, with each item marked by a bounding box. The pipeline proves its own correctness through **math consistency checks** and an **automated evaluation harness**, and flags low-confidence results for human review.

### Why this problem (medium complexity)
| Simple (SROIE) | **Our level (CORD)** | Hard (later) |
|---|---|---|
| 4 flat fields | **Nested line items + sub-items + totals** | Multi-page invoices, many languages |
| Exact match is enough | **List matching, fuzzy names, number normalization** | Locale ambiguity, tax regimes |
| No cross-checks | **Math checks: items → subtotal → total → change** | Cross-page table merging |

It is hard enough to teach every real LLM-engineering skill (schema design, eval, judges, validation loops, observability). It is still small enough to finish, using a **free, labeled public dataset**.

### Dataset
- **Primary:** [CORD v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2). About 1,000 real receipt photos with labels for menu items (name, quantity, unit price, price, sub-items), subtotal, tax, service, discount, total, cash and change.
- **Next level:** DocILE (ICDAR 2023, about 6,700 business documents with line items; access may require registration).

### Success criteria
| Metric | Target |
|---|---|
| Header field F1 (total, subtotal, tax...) | ≥ 95% |
| Line-item F1 (name + qty + price matched) | ≥ 85% |
| Math-consistency rate | ≥ 90% |
| Hallucinated-value rate | < 1% |
| p95 latency per receipt | < 8 s |
| Cost per receipt | < $0.002 |

### Out of scope (for now)
ID documents, handwriting, multi-page invoices, multi-language and multi-currency, fraud detection, and Q&A over many documents. These are planned in Phase 5.

---

## 2. Current state
- `pipeline.py`: Gemini extraction with a generic schema (text blocks + boxes + key/value fields), one retry, box validation
- `main.py`: FastAPI `POST /api/extract`
- `app.py`: Streamlit UI (API key + model in the sidebar, boxes drawn, JSON export)
- No dataset, no evaluation, no observability yet

---

## 3. Steps an LLM engineer performs

Each step lists **what** to do and **why** it is needed.

### Phase 1: Foundations (measure before you build)

#### Step 1: Dataset setup
**What**
- Load CORD v2 and convert its labels into our target schema format (`expected.json` per receipt).
- Split the data: **dev** (100) for iterating, **test** (100) frozen for final scores, and the rest held back.
- Write a short data report: number of items per receipt, how many receipts have discounts or sub-items, image quality spread.
- Version the split (a fixed list of IDs in the repo).

**Why**
- Every later decision is judged against this ground truth. Without it, "better" is only an opinion.
- Converting the labels into *our* schema forces us to understand the data deeply before writing prompts.
- A frozen test set prevents tuning the prompt to the examples we keep looking at, which is overfitting.
- The data report shows where the hard cases are, so we know what to look for in failures.

#### Step 2: Target schema design
**What**
```python
class SubItem(BaseModel):  name: str; qty: float | None; price: float | None
class Item(BaseModel):     name: str; qty: float | None; unit_price: float | None
                           price: float; sub_items: list[SubItem]; box_2d: list[int]
class Totals(BaseModel):   subtotal: float | None; tax: float | None; service: float | None
                           discount: float | None; total: float; cash: float | None; change: float | None
class Receipt(BaseModel):  items: list[Item]; totals: Totals; currency: str | None
```
- Use typed numbers, `None` for "not present", and a box per item.
- Keep the schema in its own module with a **version number**.

**Why**
- The schema is the contract between the LLM and all downstream code. Typed fields reject garbage automatically.
- `None` vs `0` matters: "no tax printed" is different from "tax is zero", and confusing them is a common source of errors.
- The box per item powers the UI (click an item to highlight its row) and lets the eval check that the item was actually found on the page.
- Versioning means old results can be interpreted after the schema changes.

#### Step 3: Normalizers
**What**
- Prices: `"12.000"` → `12000`, `"Rp 12,500"` → `12500`, and strip currency symbols.
- Item names: uppercase, collapse whitespace, strip punctuation.
- Quantities: `"2x"` → `2`.
- Unit tests for every normalizer.

**Why**
- The model and the labels often represent the same value differently. Without normalization the eval reports false failures, and the downstream data is inconsistent.
- Number-format mistakes are the most damaging error type in real document extraction, so this code must be tested.

#### Step 4: Evaluation metrics
**What**

| Metric | How |
|---|---|
| Header field accuracy / F1 | Normalized exact match per field |
| **Item matching** | Hungarian matching of predicted to true items on name similarity |
| Item F1 | Match counts as correct when the name matches (judge, Step 6) **and** qty/price are equal |
| Field accuracy inside items | Per attribute: name, qty, unit_price, price |
| Math-consistency rate | Share of receipts passing all math checks (Step 8) |
| Hallucination rate | Predicted values not supported by the page (Step 6) |
| Schema-valid rate, retry rate | From the pipeline |
| Latency, tokens, cost | From traces (Step 10) |

**Why**
- Line items are a **list**, so you cannot compare them position by position. A missing row would shift every later row and count as all wrong. Matching first gives a fair score.
- Separate metrics show *what kind* of error happens (a missed row, a wrong price, a merged row), and that tells us what to fix.
- Cost and latency are quality metrics too. An accurate but slow or expensive system is not usable.

#### Step 5: Eval harness
**What**
- CLI: `python -m eval run --set dev --model gemini-2.5-flash --prompt v1`
- Runs the pipeline over the set and computes all metrics. Writes `runs/<timestamp>/report.json` and an HTML report with the **worst receipts and their diffs**.
- Records for every run: model, prompt version, schema version, judge version, git commit, dataset split.
- **Response cache** keyed by `hash(image) + prompt + model + schema`.
- `python -m eval compare runA runB` shows metric deltas and newly broken or newly fixed receipts.
- **Separate unit tests** (pytest, LLM mocked) for normalizers, matching and math checks.

**Why**
- The harness is the most important tool an LLM engineer has. It turns every change into a measured experiment.
- The failure report is where the learning happens. Reading the 10 worst receipts teaches more than any average.
- The cache makes re-scoring free and instant, so the eval gets run often.
- Comparison shows *regressions*. A change that fixes 5 receipts but breaks 8 looks fine on average, and only a diff reveals it.

### Phase 2: Build and improve the extractor

#### Step 6: Judge for fuzzy matches and hallucinations (JEV first)
**What**
- **Name-match judge:** "Do `ES TEH MNS` and `Es Teh Manis` refer to the same item?" (yes/no)
- **Support judge:** "Is the value `45000` for `total` supported by this page text?" (yes/no)
- Use **JEV** as the primary judge, with a **confidence cascade**: if JEV is confident (p > 0.9 or < 0.1), accept its verdict; otherwise escalate to a reasoning LLM judge (e.g. Gemini Pro).
- **Calibrate** on about 150 human-labeled pairs: measure agreement, then tune the thresholds.
- Math and logic checks always stay in code, never in the judge.

**Why**
- Receipt item names are short, abbreviated and messy, so exact matching under-reports accuracy badly.
- Hallucinated numbers are the most dangerous failure. They need their own detector.
- JEV reportedly matches large judges on decisions that can be read from the text, at a small fraction of the cost. That makes it cheap enough to run on every eval and later on production traffic. The cascade covers the cases where it is unsure.
- An uncalibrated judge is just another unverified model, so calibration is what makes its scores trustworthy.
- *Note:* verify JEV's API, input types and pricing before relying on it.

#### Step 7: Prompt and extraction strategy experiments
**What**
- Baseline: one call per receipt with the schema (`response_schema=Receipt`, temperature 0).
- Experiments, **one change at a time**, each measured on the dev set:
  1. Prompt wording and rules ("one item per printed row", "sub-items are indented lines without a price")
  2. Few-shot examples (1–2 receipts in the prompt)
  3. Image preprocessing: resolution, contrast, deskew
  4. Model choice: Flash-Lite vs Flash vs Pro (accuracy vs cost)
  5. Two-step extraction: transcribe first, then structure, compared with one step
- Store prompts as versioned files: `prompts/receipt_v1.txt`, `v2`...
- Keep an **experiment log** (`experiments.md`) recording each hypothesis, the change, the result and the decision.

**Why**
- Changing several things at once makes it impossible to know what helped.
- Model and preprocessing choices trade accuracy against cost and latency, and only measurement shows the right balance.
- Versioned prompts can be compared and rolled back like code.
- The experiment log is the record of what worked and why. It stops the team repeating failed ideas and is the core artifact of LLM-engineering work.

#### Step 8: Validation and self-correction loop
**What**
- Math checks in code:
  - `sum(item.price) ≈ subtotal`
  - `subtotal + tax + service − discount ≈ total`
  - `cash − total ≈ change`
  - `qty × unit_price ≈ price` for each item
- On failure, make **one targeted re-ask** that shows the model the specific mismatch and asks it to re-read only that part.
- Measure whether the re-ask improves accuracy. Keep it only if the eval says so.

**Why**
- Math checks catch errors that are plausible but impossible: a missed row, a wrong digit, a subtotal taken as the total. No schema can catch these.
- A targeted re-ask is cheaper and more reliable than a blind retry, because the model gets a concrete reason to look again.
- Research shows LLMs often *fail* to self-correct without external feedback. The math check supplies that feedback, but the benefit still has to be measured, not assumed.

#### Step 9: Confidence scoring and human review
**What**
- Confidence per receipt and per item, combining:
  - math checks passed,
  - support-judge verdicts,
  - the item text found inside its box,
  - agreement between 2 runs for key fields (optional, adds cost).
- Routing: **auto-approve** high-confidence receipts; send low-confidence ones to a **review screen** in Streamlit (image with boxes, editable item table, math status).
- Save every correction as a new labeled example.
- Check calibration: a confidence of 0.9 should be correct about 90% of the time.

**Why**
- No model is 100% accurate. Confidence decides what can skip a human and what must not.
- Reviewing only uncertain receipts gives near-perfect output at a small fraction of manual cost.
- Corrections are the best new eval and training data available, because they come from real failures.
- An uncalibrated confidence score is misleading, and calibration makes it usable for routing.

### Phase 3: Observability and reliability

#### Step 10: Observability
**What**
- **Tracing** (OpenTelemetry plus Langfuse or Arize Phoenix): one trace per receipt, with spans for preprocessing, the LLM call, validation, the re-ask and judge calls.
- Each LLM span records the model, prompt version, input/output tokens, latency, retries, finish reason and output.
- **Structured JSON logs** with `request_id` and `receipt_id`. Never log API keys.
- **Metrics dashboard:**
  - Requests, errors by type (schema failure, 429, timeout, math failure)
  - Latency p50/p95, cost per receipt
  - Re-ask rate, review rate, JEV escalation rate
- **Quality monitoring:** confidence distribution over time, plus a weekly human audit of a sample.
- **Alerts:** error, cost or latency spikes, a jump in the review rate, a model version approaching retirement.

**Why**
- When an extraction is wrong, the trace shows exactly which prompt, model and output caused it. Without it, debugging is guesswork.
- Cost grows with usage, and per-receipt cost tracking prevents surprise bills.
- Model providers update models, and incoming data changes. Drift in confidence or review rate is the early warning before users notice.
- Alerts mean the team learns about problems first.

#### Step 11: Pipeline hardening
**What**
- Retry with exponential backoff on 429 and 503 errors. **Fall back to another model** (Flash, then Pro) on repeated failure.
- Idempotency: deduplicate on file hash so the same receipt is never processed or billed twice.
- Input limits: file size, image dimensions, allowed types.
- Batch endpoint plus async jobs for bulk uploads (queue + worker).
- Timeouts and clear error messages in the API and UI.

**Why**
- LLM APIs fail temporarily under load, so without retries users see random errors.
- Duplicates waste money and corrupt downstream data.
- Bulk uploads would time out as synchronous requests. Queues make them reliable.

#### Step 12: CI with eval regression gates
**What**
- On every pull request: lint, unit tests, then **eval on the dev set using the cache**.
- Fail the PR if item F1 drops by more than 1%, math consistency drops, or the hallucination rate rises.
- Post the metric diff as a PR comment.

**Why**
- Prompt and code changes can silently degrade quality. The gate catches this before merge.
- A visible metric diff makes quality part of code review, not an afterthought.

### Phase 4: Productize

#### Step 13: Storage and outputs
**What**
- Postgres: `receipts`, `items`, `sub_items`, `runs` (model, prompt, schema versions), `reviews`.
- Object storage for the original images.
- Exports: CSV/Excel, JSON, and a REST API for downstream systems.

**Why**
- The goal is to **process the data further**, which needs structured, queryable storage.
- Storing versions with each record lets us audit or re-run old extractions after improvements.

#### Step 14: Q&A layer (optional, after extraction is solid)
**What**
- **One receipt:** put its validated JSON in the prompt ("What did I order?").
- **Many receipts:** text-to-SQL or tool calls over Postgres ("Total spent on drinks last month?").
- **RAG** only for free-text parts, if any are added later (notes, terms).
- Evaluate it too: a small question/answer test set with exact answers.

**Why**
- With clean structured data, most questions become database queries, which are exact, cheap and auditable. RAG would only retrieve chunks and guess at sums.
- Building Q&A only after extraction is validated avoids answering questions from wrong data.

#### Step 15: Security and privacy
**What**
- API keys on the server only (in a secret manager). The Streamlit key field is for testing.
- User authentication and rate limits.
- Encryption, a retention policy that auto-deletes images after N days, and audit logs.
- Review the provider's data-usage terms.

**Why**
- A leaked key means someone else spends your money.
- Receipts can contain card digits, names and locations, which are personal data under GDPR and similar laws.

#### Step 16: Deployment
**What**
- Docker images (API, worker, UI), config per environment (dev, staging, prod).
- CI/CD: tests, eval gate, build, staging, production.
- **Shadow testing** for new models or prompts: run them on live traffic in parallel and compare metrics before switching.

**Why**
- Reproducible deploys, and problems are caught in staging rather than production.
- Shadow testing makes model upgrades a data-driven, low-risk decision.

#### Step 17: Cost optimization
**What**
- Use the eval harness to find the **cheapest model and smallest image size** that still meet the targets.
- Route documents: a cheap model first, escalating to Pro only on low confidence or failed math.
- Cache results by file hash. Use JEV instead of a large judge.

**Why**
- Costs scale with volume, so small savings per receipt become large.
- Routing pays premium prices only for the hard minority of receipts.

### Phase 5: Increase difficulty (after targets are met)

| Next problem | New skill learned |
|---|---|
| DocILE business invoices | Varied real-world layouts, more field types |
| Multi-page invoices | Page handling, merging tables across pages, detecting missing pages |
| Multi-language / multi-currency receipts | Locale-aware dates and numbers, currency detection, VAT fields |
| Hybrid e-invoice intake (ZUGFeRD / Factur-X) | Prefer embedded XML when present, map output to the EN 16931 standard |
| Forms with checkboxes | Boolean fields, conditional-field rules |

**Why:** each level adds exactly one new kind of difficulty on top of a working, measured system. That keeps learning focused and lets results be compared fairly.

---

## 4. Implementation order

| Phase | Steps | Outcome |
|---|---|---|
| 1. Foundations | 1–5 | Dataset, schema, normalizers, metrics, eval harness. **We can measure.** |
| 2. Extractor | 6–9 | Judge, prompt experiments, math validation, confidence and review. **We hit the targets.** |
| 3. Reliability | 10–12 | Tracing, hardening, CI gates. **We can trust it over time.** |
| 4. Productize | 13–17 | Storage, Q&A, security, deploy, cost. **Others can use it.** |
| 5. Level up | — | Harder document types. |

**Core principle:** *measure first, change one thing at a time, keep what the eval proves.*
