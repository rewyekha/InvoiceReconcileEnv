# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
InvoiceReconcileEnv Environment Implementation.

Simulates a real-world Accounts Payable (AP) workflow where an agent must:
- Extract invoice fields (with OCR noise in hard mode)
- Match against Purchase Orders and Goods Receipts
- Apply tolerance bands (2% acceptable, 5%+ hard flag)
- Detect duplicates, partial shipments, fraud patterns
- Handle priority invoices with early-payment discount windows
"""
import random
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        InvoicereconcileenvAction,
        InvoicereconcileenvObservation,
        ActionType,
        DiscrepancyType,
    )
except ImportError:
    from models import (
        InvoicereconcileenvAction,
        InvoicereconcileenvObservation,
        ActionType,
        DiscrepancyType,
    )

# ---------------------------------------------------------------------------
# OCR Noise Simulation
# ---------------------------------------------------------------------------

_OCR_VARIANTS = {
    "Industrial Widget": [
        "Industrial Widget",   # clean
        "INDUSTRIAL WIDGET",   # caps
        "Industral Widget",    # typo
        "Industrial Wdget",    # dropout
        "WIDGET-IND-2025",     # format variation
        "lndustrial Widget",   # l→I confusion
    ],
    "Acme Supplies": [
        "Acme Supplies",
        "ACME SUPPLIES",
        "Acme Suppies",        # typo
        "Acm3 Supplies",       # OCR digit
    ],
    "Global Parts Co": [
        "Global Parts Co",
        "GLOBAL PARTS CO",
        "G1obal Parts Co",     # OCR digit
        "Global Part Co",      # missing s
    ],
    "FastShip Ltd": [
        "FastShip Ltd",
        "FASTSHIP LTD",
        "FastSh1p Ltd",        # OCR digit
        "Fast Ship Ltd",       # space inserted
    ],
}

def apply_ocr_noise(text: str, rng: random.Random, level: str = "hard") -> str:
    """Apply OCR-style noise to a string. Only active in hard mode."""
    if level != "hard":
        return text
    variants = _OCR_VARIANTS.get(text)
    if variants:
        return rng.choice(variants)
    # Generic character-level noise for unknown strings
    if rng.random() < 0.3:
        chars = list(text)
        idx = rng.randint(0, len(chars) - 1)
        chars[idx] = rng.choice("01lI")
        return "".join(chars)
    return text


# ---------------------------------------------------------------------------
# Scenario Generator
# ---------------------------------------------------------------------------

TOLERANCE_SOFT = 0.02   # <= 2%  → acceptable, approve
TOLERANCE_HARD = 0.05   # > 5%   → must flag


def generate_scenario(task_level: str, seed: int = 42):
    rng = random.Random(seed)

    vendors = [
        {"id": "V001", "name": "Acme Supplies",   "bank_account": "BANK-ACC-001"},
        {"id": "V002", "name": "Global Parts Co", "bank_account": "BANK-ACC-002"},
        {"id": "V003", "name": "FastShip Ltd",    "bank_account": "BANK-ACC-003"},
    ]

    def make_invoice(inv_id, vendor, qty, unit_price, ocr=False, noise_level="easy",
                     priority=False, discount_pct=0.0, discount_steps=5):
        """Build an invoice dict. OCR noise applied to description/vendor name if requested."""
        desc = apply_ocr_noise("Industrial Widget", rng, noise_level) if ocr else "Industrial Widget"
        v_name = apply_ocr_noise(vendor["name"], rng, noise_level) if ocr else vendor["name"]
        total = round(qty * unit_price, 2)
        tax = round(total * 0.18, 2)
        inv = {
            "invoice_id": inv_id,
            "vendor_id": vendor["id"],
            "vendor_name": v_name,
            "line_items": [{"description": desc, "quantity": qty, "unit_price": unit_price}],
            "subtotal": total,
            "tax": tax,
            "total": round(total + tax, 2),
            "po_reference": f"PO-{inv_id}",
            "bank_account": vendor["bank_account"],
            # Priority / early-payment fields
            "priority": priority,
            "early_payment_discount_pct": discount_pct,   # e.g. 0.02 = 2% discount
            "discount_deadline_steps": discount_steps,     # steps remaining before discount expires
        }
        return inv

    def make_po(inv_id, vendor, qty, unit_price):
        total = round(qty * unit_price, 2)
        return {
            "po_id": f"PO-{inv_id}",
            "vendor_id": vendor["id"],
            "approved_qty": qty,
            "agreed_unit_price": unit_price,
            "total_value": total,
            "bank_account": vendor["bank_account"],
        }

    def make_receipt(inv_id, qty):
        return {
            "receipt_id": f"GR-{inv_id}",
            "po_reference": f"PO-{inv_id}",
            "received_qty": qty,
        }

    # -----------------------------------------------------------------------
    # EASY — single invoice, clean or one obvious problem
    # -----------------------------------------------------------------------
    if task_level == "easy":
        vendor = vendors[0]
        qty, price = 100, 25.00
        inv = make_invoice("INV-001", vendor, qty, price)
        po = make_po("INV-001", vendor, qty, price)
        receipt = make_receipt("INV-001", qty)
        ground_truth = {
            "INV-001": {
                "correct_action": "approve",
                "has_discrepancy": False,
                "discrepancy_type": None,
                "correct_amount": inv["total"],
                "price_variance_pct": 0.0,
            }
        }
        return [inv], {"PO-INV-001": po}, {"GR-INV-001": receipt}, ground_truth

    # -----------------------------------------------------------------------
    # MEDIUM — batch: clean + price variance (within/outside tolerance) + partial shipment
    # -----------------------------------------------------------------------
    elif task_level == "medium":
        invoices, pos, receipts, ground_truth = [], {}, {}, {}

        # INV-101: Clean invoice — approve
        v = vendors[0]
        qty, price = 50, 10.00
        inv = make_invoice("INV-101", v, qty, price)
        invoices.append(inv)
        pos["PO-INV-101"] = make_po("INV-101", v, qty, price)
        receipts["GR-INV-101"] = make_receipt("INV-101", qty)
        ground_truth["INV-101"] = {
            "correct_action": "approve",
            "has_discrepancy": False,
            "discrepancy_type": None,
            "correct_amount": inv["total"],
            "price_variance_pct": 0.0,
        }

        # INV-102: Price variance = 25% (20→25) — hard flag (>5%)
        v = vendors[1]
        qty = 30
        agreed_price, invoice_price = 20.00, 25.00
        variance_pct = abs(invoice_price - agreed_price) / agreed_price  # 0.25
        inv2 = make_invoice("INV-102", v, qty, invoice_price)
        invoices.append(inv2)
        pos["PO-INV-102"] = make_po("INV-102", v, qty, agreed_price)
        receipts["GR-INV-102"] = make_receipt("INV-102", qty)
        ground_truth["INV-102"] = {
            "correct_action": "flag",
            "has_discrepancy": True,
            "discrepancy_type": "price",
            "correct_amount": None,
            "price_variance_pct": round(variance_pct, 4),
        }

        # INV-103: Partial shipment — 80 of 100 received — flag quantity
        v = vendors[2]
        qty_invoiced, qty_received, price = 100, 80, 5.00
        inv3 = make_invoice("INV-103", v, qty_invoiced, price)
        invoices.append(inv3)
        pos["PO-INV-103"] = make_po("INV-103", v, qty_invoiced, price)
        receipts["GR-INV-103"] = make_receipt("INV-103", qty_received)
        ground_truth["INV-103"] = {
            "correct_action": "flag",
            "has_discrepancy": True,
            "discrepancy_type": "quantity",
            "correct_amount": None,
            "price_variance_pct": 0.0,
        }

        return invoices, pos, receipts, ground_truth

    # -----------------------------------------------------------------------
    # HARD — everything: OCR noise, tolerance bands, duplicate, partial,
    #         priority discount window, fraud vendor pattern
    # -----------------------------------------------------------------------
    else:
        invoices, pos, receipts, ground_truth = [], {}, {}, {}

        # INV-201: Clean — approve
        v = vendors[0]
        qty, price = 200, 8.50
        inv = make_invoice("INV-201", v, qty, price, ocr=True, noise_level="hard")
        invoices.append(inv)
        pos["PO-INV-201"] = make_po("INV-201", v, qty, price)
        receipts["GR-INV-201"] = make_receipt("INV-201", qty)
        ground_truth["INV-201"] = {
            "correct_action": "approve",
            "has_discrepancy": False,
            "discrepancy_type": None,
            "correct_amount": inv["total"],
            "price_variance_pct": 0.0,
        }

        # INV-202: Price variance = 1.67% (15.00→15.25) — WITHIN soft tolerance (<=2%) → approve
        # This is the tolerance band trap: naive agent flags it, smart agent approves it
        v = vendors[1]
        qty = 100
        agreed_price, invoice_price = 15.00, 15.25
        variance_pct = abs(invoice_price - agreed_price) / agreed_price  # ~0.0167
        inv2 = make_invoice("INV-202", v, qty, invoice_price, ocr=True, noise_level="hard")
        invoices.append(inv2)
        pos["PO-INV-202"] = make_po("INV-202", v, qty, agreed_price)
        receipts["GR-INV-202"] = make_receipt("INV-202", qty)
        ground_truth["INV-202"] = {
            "correct_action": "approve",       # within 2% — must approve, not flag
            "has_discrepancy": False,
            "discrepancy_type": None,
            "correct_amount": inv2["total"],
            "price_variance_pct": round(variance_pct, 4),
        }

        # INV-203: DUPLICATE of INV-201 with slightly different amount (8.55 vs 8.50)
        # Same PO reference, OCR-noisy description — flag as duplicate
        dup_price = 8.55
        dup_variance_pct = abs(dup_price - price) / price  # ~0.0059 — soft range but it's a duplicate
        inv3 = make_invoice("INV-203", vendors[0], 200, dup_price, ocr=True, noise_level="hard")
        inv3["po_reference"] = "PO-INV-201"   # same PO as INV-201 — dead giveaway
        invoices.append(inv3)
        pos["PO-INV-203"] = make_po("INV-201", vendors[0], 200, price)
        receipts["GR-INV-203"] = make_receipt("INV-203", 200)
        ground_truth["INV-203"] = {
            "correct_action": "flag",
            "has_discrepancy": True,
            "discrepancy_type": "duplicate",
            "correct_amount": None,
            "price_variance_pct": round(dup_variance_pct, 4),
        }

        # INV-204: PRIORITY invoice — 2% early payment discount if approved within 4 steps
        # Partial shipment: 60 of 100 received — agent must choose: flag quantity OR capture discount
        # Correct answer: flag quantity (can't approve a partial shipment regardless of discount)
        v = vendors[2]
        inv4 = make_invoice(
            "INV-204", v, qty=100, unit_price=30.00,
            ocr=True, noise_level="hard",
            priority=True, discount_pct=0.02, discount_steps=4
        )
        invoices.append(inv4)
        pos["PO-INV-204"] = make_po("INV-204", v, qty=100, unit_price=30.00)
        receipts["GR-INV-204"] = make_receipt("INV-204", qty=60)   # partial — only 60 received
        ground_truth["INV-204"] = {
            "correct_action": "flag",
            "has_discrepancy": True,
            "discrepancy_type": "quantity",   # partial shipment beats discount temptation
            "correct_amount": None,
            "price_variance_pct": 0.0,
            "priority": True,
            "discount_pct": 0.02,
        }

        # INV-205: FRAUD pattern — vendor ID in invoice (V003) doesn't match bank account
        # PO has FastShip Ltd bank account, but invoice carries a different bank account
        fraud_vendor = {"id": "V003", "name": "FastShip Ltd", "bank_account": "BANK-ACC-FRAUD-999"}
        inv5 = make_invoice("INV-205", fraud_vendor, qty=50, unit_price=30.00,
                            ocr=True, noise_level="hard")
        # Override bank account to fraudulent one — vendor ID looks fine, bank account is wrong
        inv5["bank_account"] = "BANK-ACC-FRAUD-999"
        invoices.append(inv5)
        pos["PO-INV-205"] = make_po("INV-205", vendors[2], 50, 30.00)  # correct vendor bank
        receipts["GR-INV-205"] = make_receipt("INV-205", 50)
        ground_truth["INV-205"] = {
            "correct_action": "escalate",
            "has_discrepancy": True,
            "discrepancy_type": "vendor",    # bank account mismatch = fraud signal
            "correct_amount": None,
            "price_variance_pct": 0.0,
        }

        return invoices, pos, receipts, ground_truth


# ---------------------------------------------------------------------------
# Grader — FIXED FOR STRICT (0, 1) RANGE
# ---------------------------------------------------------------------------

def grade_episode(
    ground_truth: dict,
    decisions: dict,
    flags: dict,
    steps_taken: int,
    max_steps: int,
    priority_bonuses: dict = None,
) -> float:
    """
    Score strictly between 0.0 and 1.0 (exclusive).
    Deterministic. Applies tolerance-band awareness.

    Tolerance bands:
      - price_variance_pct <= TOLERANCE_SOFT (2%): correct action is approve
      - TOLERANCE_SOFT < price_variance_pct <= TOLERANCE_HARD (5%): grey zone, flag gets partial credit
      - price_variance_pct > TOLERANCE_HARD (5%): must flag — approving is penalized

    Priority bonus: if agent approves a priority invoice before discount_deadline_steps expire,
    they get a bonus. But if the invoice had a quantity discrepancy, no bonus applies.
    """
    if not ground_truth:
        return 0.5  # Default middle score if no invoices

    if priority_bonuses is None:
        priority_bonuses = {}

    score = 0.0
    per_invoice = 1.0 / len(ground_truth)

    for inv_id, truth in ground_truth.items():
        decision = decisions.get(inv_id, "none")
        flagged_type = flags.get(inv_id)
        variance_pct = truth.get("price_variance_pct", 0.0)

        # ---- APPROVE -------------------------------------------------------
        if truth["correct_action"] == "approve":
            if decision == "approve":
                score += per_invoice * 1.0
            elif decision == "flag":
                # Agent flagged something that was within tolerance — partial credit
                if variance_pct <= TOLERANCE_SOFT:
                    score += per_invoice * 0.2   # penalize: it was fine to approve
                elif variance_pct <= TOLERANCE_HARD:
                    score += per_invoice * 0.5   # grey zone — cautious but not optimal
                else:
                    score += per_invoice * 0.0   # shouldn't happen given correct_action=approve
            else:
                score += 0.0

        # ---- FLAG ----------------------------------------------------------
        elif truth["correct_action"] == "flag":
            if decision == "flag":
                if flagged_type == truth["discrepancy_type"]:
                    score += per_invoice * 1.0   # correct type
                else:
                    score += per_invoice * 0.5   # flagged but wrong type
            elif decision == "approve":
                # Approving something that should be flagged
                variance_pct_check = truth.get("price_variance_pct", 0.0)
                if truth["discrepancy_type"] == "price" and variance_pct_check <= TOLERANCE_HARD:
                    score += per_invoice * 0.3   # marginal — in grey zone
                else:
                    score += 0.0   # hard miss
            else:
                score += 0.0

        # ---- ESCALATE ------------------------------------------------------
        elif truth["correct_action"] == "escalate":
            if decision == "escalate":
                score += per_invoice * 1.0
            elif decision == "flag":
                score += per_invoice * 0.4   # partial — noticed something, wrong action
            else:
                score += 0.0

        # ---- REJECT --------------------------------------------------------
        elif truth["correct_action"] == "reject":
            if decision == "reject":
                score += per_invoice * 1.0
            else:
                score += 0.0

    # Efficiency penalty — using >80% of steps
    if steps_taken / max_steps > 0.80:
        score *= 0.85

    # Priority bonus — agent captured early payment discount
    for inv_id, bonus_info in priority_bonuses.items():
        truth = ground_truth.get(inv_id, {})
        # Only award if correct_action was approve AND agent actually approved
        if truth.get("correct_action") == "approve" and decisions.get(inv_id) == "approve":
            if bonus_info.get("captured"):
                score = min(0.99, score + 0.05)  # Cap at 0.99 to stay below 1.0

    # ========== CRITICAL FIX: CLAMP STRICTLY WITHIN (0, 1) ==========
    # Map [0, 1] → (0.01, 0.99) to ensure strict exclusivity
    # Formula: new_score = 0.01 + (score * 0.98)
    score = 0.01 + (score * 0.98)
    
    # Final validation: ensure strictly within (0, 1)
    score = max(0.001, min(0.999, score))
    score = round(score, 3)
    
    return score

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class InvoicereconcileenvEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_STEPS = 40   # bumped for hard task (5 invoices now)

    # CLASS-LEVEL SHARED STATE
    _task_level = "easy"
    _invoices, _pos, _receipts, _ground_truth = generate_scenario("easy", 42)
    _current_index = 0
    _decisions = {}
    _flags = {}
    _extracted = {}
    _po_retrieved = {}
    _receipt_retrieved = {}
    _cumulative_reward = 0.0
    _batch_status = {inv["invoice_id"]: "pending" for inv in _invoices}
    _episode_id = str(uuid4())
    _step_count = 0
    _priority_bonuses = {}   # {inv_id: {"captured": bool, "steps_at_decision": int}}

    def __init__(self):
        pass

    def reset(self, options: dict = None) -> InvoicereconcileenvObservation:
        import sys
        print(f"DEBUG RESET OPTIONS RECEIVED: {options}", file=sys.stderr, flush=True)
        if options is None:
            options = {}
        task_level = options.get("task_level", "easy")
        seed = options.get("seed", 42)

        cls = InvoicereconcileenvEnvironment
        cls._episode_id = str(uuid4())
        cls._step_count = 0
        cls._task_level = task_level
        cls._invoices, cls._pos, cls._receipts, cls._ground_truth = generate_scenario(task_level, seed)
        cls._current_index = 0
        cls._decisions = {}
        cls._flags = {}
        cls._extracted = {}
        cls._po_retrieved = {}
        cls._receipt_retrieved = {}
        cls._cumulative_reward = 0.0
        cls._batch_status = {inv["invoice_id"]: "pending" for inv in cls._invoices}
        cls._priority_bonuses = {}

        current_inv = cls._invoices[0] if cls._invoices else {}

        return InvoicereconcileenvObservation(
            message=f"Episode started. Task: {task_level}. {len(cls._invoices)} invoice(s) to process.",
            current_invoice=_serialize_invoice(current_inv),
            batch_status=cls._batch_status,
            step_count=0,
            task_level=task_level,
            done=False,
            reward=0.5,  # FIXED: middle value instead of 0.001
        )

    def step(self, action: InvoicereconcileenvAction) -> InvoicereconcileenvObservation:
        cls = InvoicereconcileenvEnvironment
        cls._step_count += 1
        step = cls._step_count
        reward = 0.0
        done = False
        message = ""

        current_inv = cls._invoices[cls._current_index] if cls._current_index < len(cls._invoices) else None
        inv_id = current_inv["invoice_id"] if current_inv else None

        # ------------------------------------------------------------------
        # EXTRACT_FIELDS
        # ------------------------------------------------------------------
        if action.action_type == ActionType.EXTRACT_FIELDS:
            if current_inv and inv_id not in cls._extracted:
                cls._extracted[inv_id] = True
                reward = 0.15

                # Build message with OCR-noisy values (what agent sees)
                priority_note = ""
                if current_inv.get("priority"):
                    remaining = current_inv.get("discount_deadline_steps", 0)
                    discount = current_inv.get("early_payment_discount_pct", 0) * 100
                    priority_note = (
                        f" ⚡ PRIORITY: {discount:.1f}% early-payment discount expires in "
                        f"{remaining} steps."
                    )

                message = (
                    f"Extracted fields for {inv_id}. "
                    f"Vendor: {current_inv['vendor_name']} (ID: {current_inv['vendor_id']}), "
                    f"Total: {current_inv['total']}, "
                    f"PO Ref: {current_inv['po_reference']}, "
                    f"Bank: {current_inv['bank_account']}."
                    f"{priority_note}"
                )
            else:
                reward = 0.05
                message = "Already extracted or no invoice available."

        # ------------------------------------------------------------------
        # RETRIEVE_PO
        # ------------------------------------------------------------------
        elif action.action_type == ActionType.RETRIEVE_PO:
            po_key = f"PO-{inv_id}"
            if current_inv and po_key in cls._pos and inv_id not in cls._po_retrieved:
                cls._po_retrieved[inv_id] = True
                reward = 0.10
                po = cls._pos[po_key]
                message = (
                    f"PO retrieved for {inv_id}: "
                    f"agreed qty={po['approved_qty']}, "
                    f"agreed price={po['agreed_unit_price']}, "
                    f"vendor_id={po['vendor_id']}, "
                    f"bank={po.get('bank_account', 'N/A')}."
                )
            else:
                reward = 0.05
                message = "PO not found or already retrieved."

        # ------------------------------------------------------------------
        # RETRIEVE_RECEIPT
        # ------------------------------------------------------------------
        elif action.action_type == ActionType.RETRIEVE_RECEIPT:
            gr_key = f"GR-{inv_id}"
            if current_inv and gr_key in cls._receipts and inv_id not in cls._receipt_retrieved:
                cls._receipt_retrieved[inv_id] = True
                reward = 0.10
                receipt = cls._receipts[gr_key]
                invoiced_qty = current_inv["line_items"][0]["quantity"] if current_inv.get("line_items") else 0
                received_qty = receipt["received_qty"]
                partial_note = ""
                if received_qty < invoiced_qty:
                    partial_note = (
                        f" ⚠ PARTIAL SHIPMENT: invoice claims {invoiced_qty} units "
                        f"but only {received_qty} received."
                    )
                message = f"Goods receipt for {inv_id}: received qty={received_qty}.{partial_note}"
            else:
                reward = 0.05
                message = "Receipt not found or already retrieved."

        # ------------------------------------------------------------------
        # FLAG_DISCREPANCY
        # ------------------------------------------------------------------
        elif action.action_type == ActionType.FLAG_DISCREPANCY:
            if inv_id and inv_id not in cls._decisions:
                truth = cls._ground_truth.get(inv_id, {})
                cls._decisions[inv_id] = "flag"
                flagged_as = action.discrepancy_type.value if action.discrepancy_type else "other"
                cls._flags[inv_id] = flagged_as
                cls._batch_status[inv_id] = "flagged"

                if truth.get("has_discrepancy"):
                    expected_type = truth.get("discrepancy_type")
                    if flagged_as == expected_type:
                        reward = 0.35
                        message = f"✓ Correct! {inv_id} flagged as '{flagged_as}'."
                    else:
                        reward = 0.15
                        message = f"Discrepancy flagged but wrong type for {inv_id}. Expected: '{expected_type}', got: '{flagged_as}'."
                else:
                    # False flag — but check tolerance band
                    variance_pct = truth.get("price_variance_pct", 0.0)
                    if variance_pct <= TOLERANCE_SOFT:
                        reward = 0.05
                        message = f"✗ False flag — {inv_id} was within tolerance ({variance_pct*100:.2f}% < {TOLERANCE_SOFT*100:.0f}%). Should approve."
                    elif variance_pct <= TOLERANCE_HARD:
                        reward = 0.15
                        message = f"Cautious flag on {inv_id} — price variance {variance_pct*100:.2f}% is in grey zone ({TOLERANCE_SOFT*100:.0f}%–{TOLERANCE_HARD*100:.0f}%). Partial credit."
                    else:
                        reward = 0.05
                        message = f"✗ False flag — {inv_id} had no discrepancy."

                cls._current_index += 1
            else:
                reward = 0.05
                message = "Already decided on this invoice."

        # ------------------------------------------------------------------
        # APPROVE_PAYMENT
        # ------------------------------------------------------------------
        elif action.action_type == ActionType.APPROVE_PAYMENT:
            if inv_id and inv_id not in cls._decisions:
                truth = cls._ground_truth.get(inv_id, {})
                cls._decisions[inv_id] = "approve"
                cls._batch_status[inv_id] = "approved"

                if truth.get("correct_action") == "approve":
                    reward = 0.40
                    message = f"✓ {inv_id} correctly approved. Amount: {truth.get('correct_amount')}."

                    # Check priority discount capture
                    if current_inv and current_inv.get("priority"):
                        deadline = current_inv.get("discount_deadline_steps", 0)
                        steps_used_on_invoice = sum(
                            1 for k in [cls._extracted.get(inv_id),
                                        cls._po_retrieved.get(inv_id),
                                        cls._receipt_retrieved.get(inv_id)] if k
                        ) + 1  # +1 for this approve step
                        if steps_used_on_invoice <= deadline:
                            discount = current_inv.get("early_payment_discount_pct", 0)
                            bonus_amt = round(truth.get("correct_amount", 0) * discount, 2)
                            reward += 0.10
                            message += f" ⚡ Early payment discount captured! Savings: {bonus_amt}."
                            cls._priority_bonuses[inv_id] = {"captured": True}
                        else:
                            message += " ⏰ Discount window missed — too many steps used."
                            cls._priority_bonuses[inv_id] = {"captured": False}
                else:
                    # Wrong approval — check tolerance
                    variance_pct = truth.get("price_variance_pct", 0.0)
                    if truth.get("discrepancy_type") == "price" and variance_pct <= TOLERANCE_HARD:
                        reward = 0.15
                        message = f"Questionable approval of {inv_id} — price variance {variance_pct*100:.2f}% is above soft tolerance. Expected: '{truth.get('correct_action')}'."
                    else:
                        reward = 0.05
                        message = f"✗ Wrong approval of {inv_id}. Expected: '{truth.get('correct_action')}'."

                cls._current_index += 1
            else:
                reward = 0.05
                message = "Already decided on this invoice."

        # ------------------------------------------------------------------
        # REJECT_INVOICE
        # ------------------------------------------------------------------
        elif action.action_type == ActionType.REJECT_INVOICE:
            if inv_id and inv_id not in cls._decisions:
                truth = cls._ground_truth.get(inv_id, {})
                cls._decisions[inv_id] = "reject"
                cls._batch_status[inv_id] = "rejected"
                if truth.get("correct_action") == "reject":
                    reward = 0.25
                    message = f"✓ {inv_id} correctly rejected."
                else:
                    reward = 0.05
                    message = f"✗ Rejected {inv_id} but expected: '{truth.get('correct_action')}'."
                cls._current_index += 1
            else:
                reward = 0.05
                message = "Already decided on this invoice."

        # ------------------------------------------------------------------
        # ESCALATE
        # ------------------------------------------------------------------
        elif action.action_type == ActionType.ESCALATE:
            if inv_id and inv_id not in cls._decisions:
                truth = cls._ground_truth.get(inv_id, {})
                cls._decisions[inv_id] = "escalate"
                cls._batch_status[inv_id] = "escalated"
                if truth.get("correct_action") == "escalate":
                    reward = 0.35
                    message = f"✓ {inv_id} correctly escalated. Reason: {action.reason}."
                else:
                    reward = 0.05
                    message = f"Escalated {inv_id} unnecessarily. Expected: '{truth.get('correct_action')}'."
                cls._current_index += 1
            else:
                reward = 0.05
                message = "Already decided on this invoice."

        # ------------------------------------------------------------------
        # Episode termination check
        # ------------------------------------------------------------------
        all_decided = len(cls._decisions) == len(cls._invoices)
        max_steps_hit = step >= cls.MAX_STEPS

        if all_decided or max_steps_hit:
            final_score = grade_episode(
                cls._ground_truth,
                cls._decisions,
                cls._flags,
                step,
                cls.MAX_STEPS,
                priority_bonuses=cls._priority_bonuses,
            )
            reward += final_score * 0.5
            # ========== CRITICAL FIX: CLAMP REWARD STRICTLY ==========
            reward = max(0.001, min(0.999, reward))
            reward = round(reward, 3)
            done = True
            message += (
                f" | EPISODE COMPLETE. "
                f"Final grade: {final_score:.3f}. "
                f"Decisions: {cls._decisions}."
            )
        else:
            # ========== CRITICAL FIX: CLAMP STEP REWARD ==========
            reward = max(0.001, min(0.999, reward))
            reward = round(reward, 3)

        cls._cumulative_reward += reward

        # Build next invoice view
        next_inv = None
        if cls._current_index < len(cls._invoices):
            ni = cls._invoices[cls._current_index]
            next_inv = _serialize_invoice(ni)

        return InvoicereconcileenvObservation(
            message=message,
            current_invoice=next_inv,
            po_data=cls._pos.get(f"PO-{inv_id}") if inv_id in cls._po_retrieved else None,
            receipt_data=cls._receipts.get(f"GR-{inv_id}") if inv_id in cls._receipt_retrieved else None,
            flags=list(cls._flags.values()),
            batch_status=cls._batch_status,
            step_count=step,
            task_level=cls._task_level,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> State:
        cls = InvoicereconcileenvEnvironment
        return State(episode_id=cls._episode_id, step_count=cls._step_count)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _serialize_invoice(inv: dict) -> dict:
    """Return agent-visible invoice fields."""
    if not inv:
        return {}
    return {
        "invoice_id": inv.get("invoice_id"),
        "vendor_id": inv.get("vendor_id"),
        "vendor_name": inv.get("vendor_name"),
        "total": inv.get("total"),
        "po_reference": inv.get("po_reference"),
        "line_items": inv.get("line_items", []),
        "bank_account": inv.get("bank_account"),
        "priority": inv.get("priority", False),
        "early_payment_discount_pct": inv.get("early_payment_discount_pct", 0.0),
        "discount_deadline_steps": inv.get("discount_deadline_steps", 0),
    }