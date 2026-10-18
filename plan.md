# LLM OCR Extractor: Production Plan

This plan turns the current prototype into a complete application with evaluation, an eval harness, observability and production hardening. The prototype is `pipeline.py` (Gemini extraction), `main.py` (FastAPI) and `app.py` (Streamlit).

Each step lists **what** we build and **why** we need it.

---

## Step 1: Define scope and success criteria

**What**
- Choose the supported document types: invoices, receipts, IDs, forms, handwritten notes.
- List the required fields for each type, such as `invoice_number`, `date` and `total`.
- Set targets, for example:
  - Field accuracy (F1) ≥ 95%
  - Character error rate (CER) < 2%
  - p95 latency < 10 s per page
  - Cost < $0.002 per page
- Replace the generic schema with **typed Pydantic schemas per document type**, such as `Invoice(total: Decimal, date: date, line_items: list[LineItem])`.

**Why**
- Without targets we cannot tell whether a change makes things better or worse.
- Typed schemas catch errors early: a total must be a number and a date must be a valid date. They also make evaluation precise, field by field.
- A fixed scope keeps the prompt focused, and a focused prompt is more accurate than "extract everything".

---

## Step 2: Build a ground-truth dataset

**What**
- Collect 50–200 real documents per type, including the hard cases: blurry, rotated, multi-page, handwritten, low-contrast and mixed-language.
- Label each one with the full text, the expected fields and the bounding boxes for a sample. Use Label Studio or CVAT.
- Folder structure:
  ```
  dataset/
    invoice/
      0001.png
      0001.json   # expected text, fields, boxes
  ```
- Version the dataset with DVC or a versioned bucket, and never edit it silently.
- Split it into a **dev set**, used to tune prompts, and a frozen **test set**, used only for final scores.

**Why**
- Every metric needs correct answers to compare against. The dataset is the foundation of the whole quality process.
- Hard cases are where models fail. A dataset of only clean scans gives false confidence.
- Versioning means scores from different weeks are comparable.
- The separate test set prevents overfitting the prompt to the examples we stare at.

---

## Step 3: Define evaluation metrics

**What**

| Area | Metric | How |
|---|---|---|
| Raw text | CER / WER | Levenshtein distance (`jiwer`) |
| Fields | Precision, recall, F1 per field | Exact match after normalization (dates, numbers, case, whitespace) |
| Fuzzy fields | Semantic match | JEV judge (Step 4) |
| Boxes | IoU, % with IoU > 0.5 | Hungarian matching of predicted to true boxes |
| Structure | Schema-valid rate, retry rate | From pipeline output |
| Hallucination | Unsupported-field rate | JEV support check (Step 4) |
| Operations | p50/p95 latency, tokens, cost per page | From traces (Step 8) |

**Why**
- OCR quality has several dimensions. Text can be perfect while a field is wrong, and fields can be right while the boxes are off. One number hides problems.
- Normalizing before comparison prevents false failures such as `01/02/2026` vs `2026-02-01`.
- Box metrics matter because the app's job is to **mark** text as well as extract it.
- Hallucination is the most dangerous LLM-specific failure: a confident, invented value. It must be measured explicitly.
- Cost and latency are part of quality. A perfect model that is too slow or too expensive is not usable.

---

## Step 4: Judge for fuzzy fields (JEV first, LLM as escalation)

**What**
- Use deterministic code wherever possible: CER, exact match, IoU and business rules.
- Use a **judge only for fuzzy fields**, where wording differs but the meaning may be the same, such as "MG Road, Pune" vs "M.G. Road Pune 411001".
- Use **JEV** as the primary judge. It is a decision-only model that returns label probabilities.
  1. **Binary match question.** Given the field, the predicted value and the expected value, ask whether they mean the same thing (yes/no).
  2. **Support (hallucination) check.** Given the extracted value and the full page text, ask whether the text supports the value (yes/no).
- Use a **confidence cascade**:
  - JEV is confident (p > 0.9 or p < 0.1): accept its verdict.
  - JEV is unsure: escalate to a reasoning LLM judge, such as Gemini Pro.
- **Calibrate** on 100–200 human-labeled field pairs. Measure agreement, tune the thresholds, and re-check whenever the model or data changes.
- Keep **math and logic checks in code**, never in the judge. Examples: line items sum to the total, a due date comes after the invoice date.

**Why**
- Exact match marks correct-but-differently-written values as wrong, so it under-reports accuracy. A judge fixes that.
- JEV reportedly matches large LLM judges on decisions that can be read from the text, at a small fraction of the cost and latency. That makes it cheap enough to run on every eval and even on production traffic.
- JEV is weaker on verdicts that require reasoning. Keeping math in code and escalating uncertain cases covers that gap, and the reported cascade keeps about 99% of the big judge's accuracy.
- Binary questions suit JEV best, since it is weaker on graded 1–10 scores. They are also easier to calibrate.
- Calibration against humans is mandatory. An uncalibrated judge is just another unverified model.
- *Note:* JEV is new. Verify its API, whether it accepts images or only text, and its pricing before you rely on it. This plan assumes text-only input.

---

## Step 5: Build the eval harness

**What**
- A CLI runner:
  ```
  python -m eval run --model gemini-2.5-flash --prompt v3 --set dev
  ```
  It runs `pipeline.extract` over the dataset, computes all the Step 3 metrics with the Step 4 judges, and writes a JSON + HTML report.
- Record per run: the prompt version, model, schema version, judge version, git commit and dataset version.
- **Response cache**, keyed by `hash(image) + prompt + model`, so re-scoring never re-calls the model.
- **Comparison view:** run A vs run B, per-field diffs, the worst documents and newly broken documents.
- **CI regression gate:** every pull request that touches the prompt, schema or pipeline runs the dev set, and fails if field F1 drops by more than 1% or the hallucination rate rises.
- **Keep unit tests separate.** pytest covers `validate`, PDF conversion, box math and normalizers, with Gemini mocked.

**Why**
- The harness turns "looks okay to me" into numbers, so prompt, model and code changes become safe.
- Versioning every input makes results reproducible and explains why a score changed.
- The cache keeps evals fast and cheap, so people actually run them.
- The CI gate stops silent quality regressions before they reach users.
- Unit tests catch code bugs and evals catch model-behavior bugs. They answer different questions, so both are needed.

---

## Step 6: Harden the pipeline

**What**
- **Async jobs** for large PDFs: `POST /jobs` returns a `job_id`, a worker queue (Celery, RQ or Cloud Tasks) processes the pages, and the client polls `GET /jobs/{id}` or receives a webhook.
- **Process pages in parallel**, with a concurrency limit.
- **Retry with exponential backoff** on 429 and 503 errors, and **fall back to another model** (Flash, then Pro).
- **Idempotency** by file hash, so the same document is never processed or billed twice.
- **Better preprocessing:** deskew, denoise, auto-rotate, blank-page detection and DPI suited to the content.
- **Business-rule validation:** totals add up, dates are valid, currency codes are valid. A failed check triggers a targeted re-ask or a review flag.
- **Prompt in a versioned file** (`prompts/v3.txt`), not hardcoded.

**Why**
- A synchronous request on a 50-page PDF times out. Queues make long work reliable.
- Parallel pages cut latency, and the limit keeps us within API rate limits.
- Gemini returns temporary 429 and 503 errors. Without retries and fallback, users see random failures.
- Idempotency prevents double cost and duplicate records.
- Better input images are the cheapest accuracy gain available.
- Rule checks catch mistakes that look plausible but cannot be true, which no schema can catch.
- A versioned prompt can be evaluated, compared and rolled back like code.

---

## Step 7: Confidence scoring and human review

**What**
- A per-field confidence score combining:
  - Rule checks passed
  - The JEV support check (value found in the page text)
  - Agreement between two runs or two models (for important fields only)
  - Extracted text found inside its bounding box
- **Routing:** high-confidence documents are auto-approved. Low-confidence documents go to a **review queue** in the UI, where a person fixes fields and boxes.
- **Store every human correction** as new labeled data.

**Why**
- No model is 100% accurate. Confidence decides which outputs to trust automatically and which a person must check.
- Human review on a small share of documents gives near-perfect output overall, at a fraction of manual cost.
- Corrections are the best training and eval data available. They come from real traffic, so the dataset keeps improving (Step 10).

---

## Step 8: Observability

**What**
- **Tracing:** one trace per document, with spans for upload, preprocessing, each page's LLM call, validation, judge calls and storage. Use OpenTelemetry, plus an LLM-focused tool: Langfuse, Arize Phoenix or LangSmith.
- Each LLM span records the model, prompt version, input/output tokens, latency, retries, finish reason and output.
- **Structured JSON logs** with `request_id`, `doc_id` and `page`. Never log API keys, and redact personal data.
- **Metrics** (Prometheus + Grafana or a cloud equivalent):
  - Requests, errors by type (parse failure, 429, timeout, validation)
  - Latency p50/p95/p99
  - Tokens and **cost per document and per tenant**
  - Schema-failure rate, review-queue rate, JEV escalation rate
- **Production quality monitoring:**
  - Human audit of a 1% sample
  - The confidence-score distribution over time
  - Drift alerts when the distribution shifts
- **Alerts:** error spikes, cost spikes, latency SLO breaches, rising review rate, and a model version approaching retirement.

**Why**
- When a user reports a wrong field, tracing shows the exact prompt, model and output that produced it. Without it, debugging is guesswork.
- LLM costs grow with usage, so per-document cost tracking prevents surprise bills.
- Models and documents change over time, for example a new invoice layout or a provider updating the model. Drift monitoring catches the quality drop before users do.
- Alerts mean problems reach us first, not via complaints.
- Redaction is required because OCR documents often contain personal data.

---

## Step 9: Security and privacy

**What**
- **API keys on the server only.** The Streamlit key input is fine for testing. In production, use a secret manager and real user authentication (OAuth or API tokens).
- **Input validation:** file size, page count, MIME type, and a malware scan if the app is public.
- **Data protection:** encryption at rest and in transit, a retention policy that auto-deletes uploads after N days, and audit logs.
- **Rate limits and quotas** per user.
- Review the **LLM provider's data terms**: training usage, data region, retention.

**Why**
- A leaked API key means someone else spends your money.
- Uploaded documents, such as IDs and invoices, are sensitive. Leaks create legal risk under GDPR and India's DPDP Act.
- Input limits block abuse and runaway cost from oversized files.
- Retention rules reduce how much data is exposed if a breach happens.

---

## Step 10: Feedback loop and continuous improvement

**What**
- Move human corrections (Step 7) and audited samples (Step 8) into the dataset (Step 2) on a regular schedule.
- Re-run the eval harness on new data each month, or whenever a new model is released.
- **Shadow testing:** run a candidate model or prompt on live traffic in parallel without showing its results, compare the metrics, then switch.

**Why**
- Real-world documents keep changing. A dataset frozen at launch slowly stops representing reality.
- New models come out often. A standard eval makes "should we upgrade?" a 10-minute data-driven decision.
- Shadow testing avoids breaking production with an untested change.

---

## Step 11: Storage and downstream processing

**What**
- **Postgres tables:** `documents`, `pages`, `fields` (JSONB), `boxes` and `runs`. Each record keeps its model, prompt and schema versions.
- **Object storage** (S3 or GCS) for the original files and rendered pages.
- **Outputs:** REST API, CSV/Excel export, webhooks, and connectors to an ERP, Google Sheets or other systems.

**Why**
- The goal is to process extracted data further, which needs structured, queryable storage.
- Keeping versions on each record lets us re-run or audit old extractions after a model change.
- Standard outputs let other systems use the data without custom work each time.

---

## Step 12: Deployment and CI/CD

**What**
- **Docker images** for the API, worker and UI.
- **CI pipeline:** lint, unit tests, dev-set eval gate (Step 5), build, deploy to staging, deploy to production.
- **Hosting:** Cloud Run, Render or Kubernetes.
- **Config per environment** (dev, staging, prod) through environment variables and a secret manager.

**Why**
- Containers make "works on my machine" work everywhere.
- Automated CI with the eval gate means every deploy is tested for both code and model quality.
- A staging environment catches problems before real users see them.

---

## Step 13: Cost control

**What**
- **Model routing:** Flash or Flash-Lite by default. Use Pro only for hard documents or low-confidence pages.
- **Cache results by file hash.**
- **Downscale images** to the smallest size that keeps accuracy, which the eval harness can find. Skip blank pages.
- Use **JEV** instead of a large LLM for judging, and escalate only when it is unsure.
- **Budgets and alerts** per user or tenant.

**Why**
- LLM cost scales directly with pages and tokens. Small savings per page become large at volume.
- Most documents are easy. Paying Pro prices for all of them wastes money.
- Measured trade-offs, such as image size against accuracy, avoid guessing.

---

## Implementation order

| Phase | Steps | Why this order |
|---|---|---|
| 1 | 1, 2, 3, 4, 5 | Measurement first. Every later change needs evals to prove it helps. |
| 2 | 8 (basic tracing and cost) | Once real usage starts, we need visibility into behavior and spend. |
| 3 | 6 | Make the pipeline reliable for real workloads. |
| 4 | 7, 10 | Add trust (confidence and review) and the improvement loop. |
| 5 | 9, 11, 12, 13 | Production readiness: security, storage, deployment, cost. |

**Core principle:** build the eval harness before optimizing anything. Once quality is measured, prompts, models, judges and even a later move to agents can all be changed safely.
