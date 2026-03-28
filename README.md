---
title: InvoiceReconcileEnv
emoji: 🧾
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
---

# 🧾 InvoiceReconcileEnv

> An OpenEnv environment for training and evaluating AI agents on real-world Accounts Payable invoice reconciliation workflows.

---

## Motivation

Every company — from a 10-person startup to a Fortune 500 — processes vendor invoices. The standard verification process, called **three-way matching**, cross-checks an invoice against its Purchase Order and Goods Receipt to confirm that what was ordered, received, and billed all align. Done manually, this is slow, error-prone, and expensive. Companies like SAP and Oracle charge millions to automate it.

**InvoiceReconcileEnv** creates a rigorous, reproducible training ground where AI agents learn to do exactly what a real AP clerk does — extract invoice data, verify it against ground-truth records, detect discrepancies and fraud patterns, and make payment decisions — all under efficiency pressure.

---

## Environment Description

The agent operates as an AP clerk inside a simulated Accounts Payable department. It processes invoices one at a time from a batch, using a realistic set of workplace actions to gather information and make decisions.

**Key mechanics:**
- **Partial observability** — agent must explicitly request PO and receipt data, just like a real clerk pulls records from a system
- **OCR noise simulation** — hard mode introduces realistic text errors (`"lndustrial Widget"`, `"G1obal Parts Co"`) that agents must reason through
- **Tolerance bands** — price variance ≤ 2% is acceptable (approve), > 5% must be flagged; grey zone (2–5%) requires judgment
- **Duplicate detection** — same invoice submitted twice with slightly different amounts and the same PO reference
- **Partial shipments** — goods receipt shows fewer units than invoiced; agent must flag and not approve
- **Priority invoices** — early payment discount windows create time pressure, but never override correctness
- **Fraud patterns** — invoice bank account mismatches PO vendor bank account; requires escalation

---

## Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `extract_fields` | `invoice_id` | Extract all structured fields from the current invoice. Always do this first. |
| `retrieve_po` | `invoice_id` | Fetch the matching Purchase Order record for price and quantity comparison. |
| `retrieve_receipt` | `retrieve_receipt` | Fetch the Goods Receipt to verify delivered quantities. |
| `flag_discrepancy` | `invoice_id`, `discrepancy_type`, `reason` | Flag an issue. Types: `price`, `quantity`, `duplicate`, `vendor`, `tax`, `other`. Advances to next invoice. |
| `approve_payment` | `invoice_id`, `amount` | Approve invoice for payment. Advances to next invoice. |
| `reject_invoice` | `invoice_id`, `reason` | Reject the invoice outright. Advances to next invoice. |
| `escalate` | `invoice_id`, `reason` | Escalate to human manager. Required for fraud signals and unknown vendors. Advances to next invoice. |

Every action costs a step. Redundant or repeated actions incur a penalty. Efficient, direct reasoning is rewarded.

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `message` | `string` | Human-readable feedback from the last action, including reward signals and warnings |
| `current_invoice` | `dict \| null` | Invoice being processed: `invoice_id`, `vendor_id`, `vendor_name`, `line_items`, `total`, `po_reference`, `bank_account`, `priority`, `early_payment_discount_pct`, `discount_deadline_steps` |
| `po_data` | `dict \| null` | PO data — only populated after `retrieve_po` |
| `receipt_data` | `dict \| null` | Receipt data — only populated after `retrieve_receipt` |
| `batch_status` | `dict[str, str]` | Status of all invoices: `pending \| approved \| flagged \| rejected \| escalated` |
| `flags` | `list[str]` | Discrepancy types flagged so far this episode |
| `step_count` | `int` | Current step number |
| `task_level` | `string` | `easy \| medium \| hard` |
| `tolerance_soft_pct` | `float` | Price variance at or below this % → approve (default `2.0`) |
| `tolerance_hard_pct` | `float` | Price variance above this % → must flag (default `5.0`) |
| `priority_invoice_active` | `bool` | True when current invoice has an active early-payment discount |
| `discount_deadline_steps` | `int` | Steps remaining before discount expires |
| `early_payment_discount_pct` | `float` | Discount percentage if approved before deadline |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float` | Step-level reward |

---

## Tasks

### 🟢 Easy — Single Invoice Reconciliation
**Invoices:** 1 | **Expected score:** 1.000

A single clean invoice requiring straightforward three-way matching. Agent must extract fields, retrieve PO and receipt, confirm everything matches, and approve. Tests basic comprehension and action sequencing.

**Grader:** Correct final decision (approve/flag/escalate) on the single invoice. Partial credit for information-gathering steps.

---

### 🟡 Medium — Batch Invoice Processing
**Invoices:** 3 | **Expected score:** 1.000

A mixed batch with three conditions:
- **INV-101**: Clean match → approve
- **INV-102**: Price variance of 25% (agreed $20, invoiced $25) → flag as `price`
- **INV-103**: Partial shipment — 80 of 100 units received → flag as `quantity`

Tests multi-step reasoning and pattern recognition across a batch.

**Grader:** Per-invoice decision accuracy averaged across all 3. Partial credit for correct discrepancy detection with wrong type label.

---

### 🔴 Hard — Complex Batch with Fraud Detection
**Invoices:** 5 | **Expected score:** 1.000

A noisy, ambiguous batch with every real-world complication:
- **INV-201**: Clean match with OCR-noisy description → approve
- **INV-202**: Price variance of 1.67% (within soft tolerance) → **must approve**, not flag. Traps naive agents.
- **INV-203**: Duplicate invoice — same PO reference as INV-201, slightly different unit price → flag as `duplicate`
- **INV-204**: Priority invoice with 2% early-payment discount + partial shipment (60 of 100 received) → **flag as `quantity`** regardless of discount temptation
- **INV-205**: Fraud signal — vendor ID is V003 but invoice bank account (`BANK-ACC-FRAUD-999`) doesn't match PO bank account (`BANK-ACC-003`) → escalate

Tests: tolerance band reasoning, duplicate detection, multi-objective decision making under time pressure, and fraud pattern recognition.

**Grader:** Composite score across all 5 invoices. Efficiency penalty if >80% of 40 steps used. Priority bonus if discount captured on valid approval.

---

## Reward Function

| Event | Reward |
|-------|--------|
| Correct field extraction | `+0.15` |
| PO retrieval | `+0.10` |
| Receipt retrieval | `+0.10` |
| Correct flag with correct discrepancy type | `+0.35` |
| Correct flag with wrong discrepancy type | `+0.15` |
| Correct approval | `+0.40` |
| Correct escalation | `+0.35` |
| Correct rejection | `+0.25` |
| False flag (invoice had no discrepancy, within tolerance) | `-0.20` |
| Wrong approval (should have flagged/escalated) | `-0.30` |
| Redundant or repeated action | `-0.05` |
| Efficiency penalty (>80% steps used) | `× 0.85` |
| Early payment discount captured | `+0.10` |
| Episode completion bonus | `+ 0.5 × final_grade` |

Reward range: `[-1.0, 1.5]`. Fully dense — signal provided at every step.

---

## Baseline Scores

Produced by the deterministic rule-based agent in `inference.py`. Fully reproducible with `seed=42`.

| Task | Final Grade | Total Reward |
|------|------------|--------------|
| Easy | **1.000** | ~0.85 |
| Medium | **1.000** | ~2.35 |
| Hard | **1.000** | ~4.10 |
| **Average** | **1.000** | — |

---

## Setup & Usage

### Prerequisites
- Python 3.11+
- pip

### Local Development

```bash
git clone https://huggingface.co/spaces/ShambhaviS08/InvoiceReconcileEnv
cd InvoiceReconcileEnv

pip install git+https://github.com/meta-pytorch/OpenEnv.git#subdirectory=src
pip install fastapi uvicorn pydantic requests openai

# From the OpenEnv root directory
python -m uvicorn InvoiceReconcileEnv.server.app:app --host 0.0.0.0 --port 8001
```

### API Usage

**Reset environment:**
```bash
curl -X POST http://localhost:8001/reset \
  -H "Content-Type: application/json" \
  -d '{"options": {"task_level": "hard", "seed": 42}}'
```

**Execute action:**
```bash
curl -X POST http://localhost:8001/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "extract_fields", "invoice_id": "INV-201"}}'
```

**Get state:**
```bash
curl http://localhost:8001/state
```

**Interactive docs:**
```
http://localhost:8001/docs
```

### Run Baseline Inference

```bash
# Rule-based agent (no API key needed)
python InvoiceReconcileEnv/inference.py

# OpenAI-powered agent
export OPENAI_API_KEY=your_key_here
python InvoiceReconcileEnv/inference.py
```

### Docker

```bash
# Build
docker build -f InvoiceReconcileEnv/server/Dockerfile -t invoicereconcileenv .

# Run
docker run -p 7860:7860 invoicereconcileenv

# With OpenAI key
docker run -p 7860:7860 -e OPENAI_API_KEY=your_key invoicereconcileenv
```

---

## Project Structure

```
InvoiceReconcileEnv/
├── server/
│   ├── app.py                          # FastAPI application
│   ├── InvoiceReconcileEnv_environment.py  # Core environment logic
│   ├── Dockerfile                      # Container definition
│   └── requirements.txt               # Python dependencies
├── models.py                           # Pydantic action/observation models
├── inference.py                        # Baseline inference script
├── openenv.yaml                        # OpenEnv spec metadata
└── README.md                           # This file
```

---

## Design Decisions

**Why Accounts Payable?** It's a billion-dollar automation problem with clear ground truth, natural partial observability, and deterministic grading. Every discrepancy is either present or not — no subjective scoring.

**Why OCR noise?** Real AP systems process scanned documents. Agents that only handle clean text fail in production. OCR noise is seeded and reproducible so grading remains deterministic.

**Why tolerance bands instead of exact matching?** Real AP policies allow small price variances (typically 1–3%) to avoid holding up payment for rounding errors. Flagging a 1.67% variance is a mistake in practice — this tests whether the agent learned policy, not just comparison.

**Why the priority invoice trap?** A 2% discount sounds compelling, but never approve a partial shipment regardless of incentives. This tests whether the agent prioritizes correctness over optimization.

---

## Author

**ShambhaviS08** — OpenEnv AI Hackathon 2026

---

*Built with [OpenEnv](https://github.com/meta-pytorch/OpenEnv) by Meta PyTorch.*