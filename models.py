# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the InvoiceReconcileEnv Environment.

Typed Pydantic models for Action, Observation per OpenEnv spec.
"""
from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Optional, List, Dict, Any
from enum import Enum


class ActionType(str, Enum):
    EXTRACT_FIELDS   = "extract_fields"
    RETRIEVE_PO      = "retrieve_po"
    RETRIEVE_RECEIPT = "retrieve_receipt"
    FLAG_DISCREPANCY = "flag_discrepancy"
    APPROVE_PAYMENT  = "approve_payment"
    REJECT_INVOICE   = "reject_invoice"
    ESCALATE         = "escalate"


class DiscrepancyType(str, Enum):
    PRICE     = "price"
    QUANTITY  = "quantity"
    DUPLICATE = "duplicate"
    VENDOR    = "vendor"
    TAX       = "tax"
    OTHER     = "other"


class InvoicereconcileenvAction(Action):
    action_type:      ActionType            = Field(...,  description="Type of action to perform")
    invoice_id:       Optional[str]         = Field(None, description="Target invoice ID")
    discrepancy_type: Optional[DiscrepancyType] = Field(None, description="Type of discrepancy if flagging")
    reason:           Optional[str]         = Field(None, description="Reason for rejection/escalation/flagging")
    amount:           Optional[float]       = Field(None, description="Amount for payment approval")


class InvoicereconcileenvObservation(Observation):
    message:          str                   = Field(default="",               description="Status/feedback message from the environment")
    current_invoice:  Optional[Dict[str, Any]] = Field(None,                  description="Current invoice being processed (may contain OCR noise in hard mode)")
    po_data:          Optional[Dict[str, Any]] = Field(None,                  description="Purchase order data — only populated after retrieve_po action")
    receipt_data:     Optional[Dict[str, Any]] = Field(None,                  description="Goods receipt data — only populated after retrieve_receipt action")
    flags:            List[str]             = Field(default_factory=list,     description="Discrepancy types flagged so far in this episode")
    batch_status:     Dict[str, str]        = Field(default_factory=dict,     description="Status of all invoices: pending/approved/flagged/rejected/escalated")
    step_count:       int                   = Field(default=0,                description="Current step number in the episode")
    task_level:       str                   = Field(default="easy",           description="Task difficulty: easy | medium | hard")
    done:             bool                  = Field(default=False,            description="Whether the episode has ended")
    reward:           float                 = Field(default=0.0,              description="Step reward")

    # --- Tolerance band hints (surface to agent for learning) ---
    tolerance_soft_pct: float               = Field(default=2.0,              description="Price variance <= this % is acceptable (approve)")
    tolerance_hard_pct: float               = Field(default=5.0,              description="Price variance > this % must be flagged")

    # --- Priority invoice fields ---
    priority_invoice_active: bool           = Field(default=False,            description="True when current invoice has an early-payment discount")
    discount_deadline_steps: int            = Field(default=0,                description="Steps remaining before early-payment discount expires")
    early_payment_discount_pct: float       = Field(default=0.0,              description="Discount percentage available if approved before deadline")