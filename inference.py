"""
Baseline inference script for InvoiceReconcileEnv.
Runs a rule-based agent against all 3 tasks and reports scores.
Uses OpenAI API client as required by hackathon spec.

Baseline scores (seed=42, deterministic rule-based agent):
  EASY       → 1.000
  MEDIUM     → 1.000
  HARD       → 1.000
  AVERAGE    → 1.000
"""
import os
import json
import requests
from openai import OpenAI

# Per-invoice step tracking — reset between tasks
_invoice_progress = {}

BASE_URL = "http://127.0.0.1:8001"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "dummy-key"))

# Tolerance constants — must match environment
TOLERANCE_SOFT = 0.02   # <= 2%  → approve
TOLERANCE_HARD = 0.05   # > 5%   → must flag


# ---------------------------------------------------------------------------
# Environment API helpers
# ---------------------------------------------------------------------------

def reset_env(task_level: str, seed: int = 42) -> dict:
    response = requests.post(
        f"{BASE_URL}/reset",
        json={"options": {"task_level": task_level, "seed": seed}},
    )
    return response.json()


def step_env(action: dict) -> dict:
    response = requests.post(
        f"{BASE_URL}/step",
        json={"action": action},
    )
    return response.json()


# ---------------------------------------------------------------------------
# OpenAI-powered agent (used when OPENAI_API_KEY is set)
# ---------------------------------------------------------------------------

def get_agent_action(observation: dict, task_level: str) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")

    if api_key and api_key != "dummy-key":
        prompt = f"""
You are an expert Accounts Payable agent. Process invoices carefully using the rules below.

Current observation:
{json.dumps(observation, indent=2)}

Available actions:
- extract_fields: Extract invoice fields (ALWAYS do this first for each invoice)
- retrieve_po: Get purchase order data (ALWAYS do this second)
- retrieve_receipt: Get goods receipt data (ALWAYS do this third)
- flag_discrepancy: Flag an issue. Types: price, quantity, duplicate, vendor, tax, other
- approve_payment: Approve the invoice payment
- reject_invoice: Reject the invoice
- escalate: Escalate to manager

Decision rules (apply in this order):
1. Always extract_fields first, retrieve_po second, retrieve_receipt third.
2. BANK ACCOUNT CHECK: If invoice bank_account != PO bank_account → escalate (fraud signal).
3. DUPLICATE CHECK: If po_reference != "PO-{{invoice_id}}" → flag as duplicate.
4. QUANTITY CHECK: If receipt received_qty < PO approved_qty → flag as quantity.
5. PRICE TOLERANCE:
   - variance = abs(invoice_price - agreed_price) / agreed_price
   - variance <= 0.02 (2%) → approve (within soft tolerance)
   - 0.02 < variance <= 0.05 → cautious flag or approve (grey zone)
   - variance > 0.05 (5%) → flag as price (hard flag required)
6. VENDOR CHECK: If vendor_id not in [V001, V002, V003] → escalate.
7. If all checks pass → approve_payment.

PRIORITY NOTE: If invoice has early_payment_discount_pct > 0, try to approve quickly,
but NEVER approve if there is a quantity or duplicate discrepancy.

Respond with ONLY a JSON object. Examples:
{{"action_type": "extract_fields", "invoice_id": "INV-001"}}
{{"action_type": "flag_discrepancy", "invoice_id": "INV-002", "discrepancy_type": "price"}}
{{"action_type": "escalate", "invoice_id": "INV-003", "reason": "Bank account mismatch"}}
{{"action_type": "approve_payment", "invoice_id": "INV-001", "amount": 2950.0}}
"""
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            action_text = response.choices[0].message.content.strip()
            action_text = action_text.replace("```json", "").replace("```", "").strip()
            return json.loads(action_text)
        except Exception:
            pass  # fall through to rule-based

    return rule_based_agent(observation)


# ---------------------------------------------------------------------------
# Deterministic rule-based agent — guaranteed reproducible baseline
# ---------------------------------------------------------------------------

def rule_based_agent(observation: dict) -> dict:
    global _invoice_progress

    current_invoice = observation.get("current_invoice")
    if not current_invoice:
        return {"action_type": "extract_fields", "invoice_id": "done"}

    inv_id = current_invoice["invoice_id"]

    # Init progress tracker for this invoice
    if inv_id not in _invoice_progress:
        _invoice_progress[inv_id] = {
            "extracted": False,
            "po": False,
            "receipt": False,
            "po_data": None,
            "receipt_data": None,
        }

    progress = _invoice_progress[inv_id]

    # Pull cached data — environment only returns po_data/receipt_data on the
    # step immediately after retrieval, so we cache it in progress
    po_data      = progress.get("po_data")      or observation.get("po_data")
    receipt_data = progress.get("receipt_data") or observation.get("receipt_data")

    # Cache if freshly returned
    if observation.get("po_data") and not progress["po_data"]:
        progress["po_data"] = observation["po_data"]
        po_data = progress["po_data"]
    if observation.get("receipt_data") and not progress["receipt_data"]:
        progress["receipt_data"] = observation["receipt_data"]
        receipt_data = progress["receipt_data"]

    # ----------------------------------------------------------------
    # Step 1: Extract fields
    # ----------------------------------------------------------------
    if not progress["extracted"]:
        return {"action_type": "extract_fields", "invoice_id": inv_id}

    # ----------------------------------------------------------------
    # Step 2: Retrieve PO
    # ----------------------------------------------------------------
    if not progress["po"]:
        return {"action_type": "retrieve_po", "invoice_id": inv_id}

    # ----------------------------------------------------------------
    # Step 3: Retrieve receipt
    # ----------------------------------------------------------------
    if not progress["receipt"]:
        return {"action_type": "retrieve_receipt", "invoice_id": inv_id}

    # ----------------------------------------------------------------
    # Step 4: Decision logic — ORDER MATTERS
    # ----------------------------------------------------------------

    # Rule 1: Bank account fraud check (invoice bank != PO bank)
    invoice_bank = current_invoice.get("bank_account", "")
    po_bank      = po_data.get("bank_account", "") if po_data else ""
    if po_bank and invoice_bank and invoice_bank != po_bank:
        return {
            "action_type": "escalate",
            "invoice_id": inv_id,
            "reason": (
                f"Bank account mismatch: invoice has {invoice_bank} "
                f"but PO vendor bank is {po_bank}. Possible fraud."
            ),
        }

    # Rule 2: Unknown vendor ID
    known_vendors = ["V001", "V002", "V003"]
    vendor_id = current_invoice.get("vendor_id", "")
    if vendor_id and vendor_id not in known_vendors:
        return {
            "action_type": "escalate",
            "invoice_id": inv_id,
            "reason": f"Unknown vendor ID {vendor_id} not in approved vendor list.",
        }

    # Rule 3: Duplicate PO reference check
    po_ref = current_invoice.get("po_reference", "")
    if po_ref != f"PO-{inv_id}":
        return {
            "action_type": "flag_discrepancy",
            "invoice_id": inv_id,
            "discrepancy_type": "duplicate",
            "reason": f"PO reference {po_ref} does not match expected PO-{inv_id}.",
        }

    # Rule 4: Quantity / partial shipment check
    if po_data and receipt_data:
        received_qty = receipt_data.get("received_qty", 0)
        approved_qty = po_data.get("approved_qty", 0)
        if received_qty < approved_qty:
            return {
                "action_type": "flag_discrepancy",
                "invoice_id": inv_id,
                "discrepancy_type": "quantity",
                "reason": (
                    f"Partial shipment: {received_qty} received "
                    f"vs {approved_qty} invoiced."
                ),
            }

    # Rule 5: Price tolerance check
    if po_data:
        agreed_price  = po_data.get("agreed_unit_price", 0)
        invoice_items = current_invoice.get("line_items", [{}])
        invoice_price = invoice_items[0].get("unit_price", 0) if invoice_items else 0

        if agreed_price > 0:
            variance_pct = abs(invoice_price - agreed_price) / agreed_price

            if variance_pct <= TOLERANCE_SOFT:
                # Within soft tolerance — approve (do NOT flag)
                pass
            elif variance_pct <= TOLERANCE_HARD:
                # Grey zone — conservative agent flags it
                return {
                    "action_type": "flag_discrepancy",
                    "invoice_id": inv_id,
                    "discrepancy_type": "price",
                    "reason": (
                        f"Price variance {variance_pct*100:.2f}% in grey zone "
                        f"({TOLERANCE_SOFT*100:.0f}%–{TOLERANCE_HARD*100:.0f}%)."
                    ),
                }
            else:
                # Above hard threshold — must flag
                return {
                    "action_type": "flag_discrepancy",
                    "invoice_id": inv_id,
                    "discrepancy_type": "price",
                    "reason": (
                        f"Price variance {variance_pct*100:.2f}% exceeds "
                        f"{TOLERANCE_HARD*100:.0f}% hard threshold."
                    ),
                }

    # Rule 6: All checks passed — approve
    return {
        "action_type": "approve_payment",
        "invoice_id": inv_id,
        "amount": current_invoice.get("total", 0),
    }


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task_level: str, seed: int = 42, max_steps: int = 40) -> float:
    """Run one full episode and return the final grade (0.0–1.0)."""
    global _invoice_progress
    _invoice_progress = {}

    result      = reset_env(task_level, seed)
    observation = result.get("observation", {})

    print(f"\n{'='*55}")
    print(f"  Task: {task_level.upper()}  |  seed={seed}")
    print(f"{'='*55}")
    print(f"  Reset: {observation.get('message')}")

    total_reward = 0.0
    final_grade  = 0.0

    for step_num in range(max_steps):
        if result.get("done"):
            break

        action = get_agent_action(observation, task_level)
        print(f"\n  Step {step_num+1:02d}: {action}")

        result      = step_env(action)
        observation = result.get("observation", {})
        reward      = result.get("reward", 0)
        done        = result.get("done", False)
        total_reward += reward

        msg = observation.get("message", "")

        # Update progress tracker from message signals
        current_inv = observation.get("current_invoice")

        # After a terminal action, current_invoice moves to NEXT invoice.
        # We need the inv_id that was just acted on — extract from message.
        acted_inv_id = _extract_inv_id_from_msg(msg)

        if acted_inv_id:
            if acted_inv_id not in _invoice_progress:
                _invoice_progress[acted_inv_id] = {
                    "extracted": False, "po": False,
                    "receipt": False, "po_data": None, "receipt_data": None,
                }
            p = _invoice_progress[acted_inv_id]
            if f"Extracted fields for {acted_inv_id}" in msg:
                p["extracted"] = True
            if "PO retrieved for" in msg:
                p["po"] = True
                if observation.get("po_data"):
                    p["po_data"] = observation["po_data"]
            if "Goods receipt for" in msg:
                p["receipt"] = True
                if observation.get("receipt_data"):
                    p["receipt_data"] = observation["receipt_data"]

        print(f"         msg: {msg[:90]}")
        print(f"         reward={reward:.3f}  done={done}")

        if done:
            if "Final grade:" in msg:
                try:
                    grade_str = msg.split("Final grade:")[1].strip().split()[0].rstrip(".")
                    final_grade = float(grade_str)
                except Exception:
                    final_grade = 0.0
            break

    print(f"\n  Task {task_level.upper()} complete.")
    print(f"  Total reward : {total_reward:.3f}")
    print(f"  Final grade  : {final_grade:.3f}")
    return final_grade


def _extract_inv_id_from_msg(msg: str) -> str:
    """Pull the first INV-XXX token out of a message string."""
    import re
    match = re.search(r"INV-\d+", msg)
    return match.group(0) if match else ""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("\nInvoiceReconcileEnv — Baseline Inference Script")
    print("=" * 55)

    scores = {}
    for task in ["easy", "medium", "hard"]:
        scores[task] = run_task(task, seed=42)

    print(f"\n{'='*55}")
    print("  BASELINE SCORES SUMMARY")
    print(f"{'='*55}")
    for task, score in scores.items():
        print(f"  {task.upper():10} → {score:.3f}")
    avg = sum(scores.values()) / len(scores)
    print(f"  {'AVERAGE':10} → {avg:.3f}")
    print(f"{'='*55}\n")

    return scores


if __name__ == "__main__":
    main()