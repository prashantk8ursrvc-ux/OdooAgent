"""
Agent Surface — the set of Odoo 19 models an MCP agent is actually allowed to drive
===================================================================================

Why this module exists
----------------------
The previous generator walked ``models[i % len(models)]`` over all 2,266 extracted
Odoo models and paired them with ``methods[i % len(methods)]``.  That produced
training rows such as ``action_post()`` on ``decimal.precision`` and
``action_register_payment()`` on ``crm.tag`` — 1,417 tool calls invoking methods
that do not exist on the target model.

An MCP agent for Odoo does not operate on EDI XML serializers, report handlers,
abstract mixins, or ``ir.*`` plumbing.  It operates on the ~40 business documents
a human clerk, accountant, or manager works with every day.  This module is the
curated allowlist of those documents, together with:

  * the methods that are genuinely callable over RPC on each one,
  * the state each method expects and the state it produces,
  * the fields a create call must realistically supply,
  * the fields a human actually filters and reads,
  * realistic value synthesis (no ``Sample Name #1`` placeholders).

Every entry is verified against the extracted knowledge graph at load time via
:func:`verify_against_kg`; anything the AST scan cannot confirm is dropped and
logged rather than silently trusted.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# The dataset is authored as if "now" is this date.  Fixed so regeneration is
# reproducible and so relative dates in prompts stay internally consistent.
TODAY = date(2026, 3, 17)


# ──────────────────────────────────────────────────────────────────────────────
# Method specification
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MethodSpec:
    """A single RPC-callable business operation on an Odoo model."""

    name: str
    intent: str                      # how a human describes it, lowercase, verb-first
    from_state: Optional[str]        # state the record must be in, None if stateless
    to_state: Optional[str]          # state after a successful call, None if stateless
    kwargs: Dict[str, Any] = field(default_factory=dict)
    # Errors this call realistically raises in production, as (exception, message).
    failures: Tuple[Tuple[str, str], ...] = ()
    # True when the method opens a wizard/action dict rather than mutating state.
    returns_action: bool = False


@dataclass(frozen=True)
class ModelSpec:
    """An Odoo model an MCP agent is allowed to drive, with its full surface."""

    model: str
    label: str                       # what a human calls it
    domain: str                      # sales | purchase | inventory | ...
    personas: Tuple[str, ...]        # job titles that touch this document
    methods: Tuple[MethodSpec, ...]
    create_fields: Dict[str, str]    # field -> value-generator key
    search_fields: Tuple[str, ...]   # fields a human filters/reads
    state_field: Optional[str] = "state"
    doc_prefix: Optional[str] = None # sequence prefix, None for non-sequenced models
    # Weight for sampling: 3 = core anchor, 2 = high frequency, 1 = specialty.
    weight: int = 2


# ──────────────────────────────────────────────────────────────────────────────
# The allowlist
# ──────────────────────────────────────────────────────────────────────────────

_SALE_ORDER = ModelSpec(
    model="sale.order",
    label="sales order",
    domain="sales",
    personas=("sales representative", "sales manager", "order desk clerk", "customer service agent"),
    methods=(
        MethodSpec("action_confirm", "confirm a quotation into a sales order", "draft", "sale",
                   failures=(("UserError", "You cannot confirm a sales order without any order lines."),
                             ("UserError", "The customer has reached their credit limit."),
                             ("ValidationError", "Some products are not available in the requested quantity."))),
        MethodSpec("action_cancel", "cancel a sales order", "sale", "cancel",
                   failures=(("UserError", "You cannot cancel a sales order that has already been invoiced."),)),
        MethodSpec("action_draft", "reset a cancelled order back to quotation", "cancel", "draft"),
        MethodSpec("action_lock", "lock a confirmed order to prevent further edits", "sale", "sale"),
        MethodSpec("action_unlock", "unlock a locked order for editing", "sale", "sale"),
        MethodSpec("action_quotation_send", "email the quotation to the customer", "draft", "sent",
                   returns_action=True,
                   failures=(("UserError", "No email address set on the customer record."),)),
        MethodSpec("action_open_delivery_wizard", "open the delivery-method selection wizard", None, None,
                   returns_action=True),
    ),
    create_fields={"partner_id": "partner_ref", "date_order": "datetime_recent",
                   "payment_term_id": "small_id", "order_line": "sale_lines"},
    search_fields=("name", "partner_id", "date_order", "amount_total", "state", "invoice_status"),
    doc_prefix="S",
    weight=3,
)

_ACCOUNT_MOVE = ModelSpec(
    model="account.move",
    label="invoice",
    domain="accounting",
    personas=("accounts receivable clerk", "accounts payable clerk", "financial controller", "bookkeeper"),
    methods=(
        MethodSpec("action_post", "post a draft invoice or vendor bill to the ledger", "draft", "posted",
                   failures=(("UserError", "You cannot post an entry with no lines."),
                             ("UserError", "This journal entry is not balanced: debit 4,200.00 != credit 4,180.00."),
                             ("AccessError", "You do not have the rights to post journal entries."),
                             ("UserError", "The accounting period for 2026-01 is locked."))),
        MethodSpec("button_draft", "reset a posted entry back to draft", "posted", "draft",
                   failures=(("UserError", "You cannot reset to draft an entry that is part of a hashed sequence."),)),
        MethodSpec("button_cancel", "cancel a journal entry", "draft", "cancel"),
        MethodSpec("action_reverse", "issue a credit note reversing an invoice", "posted", "posted",
                   returns_action=True,
                   failures=(("UserError", "You cannot reverse an entry that is fully reconciled."),)),
        MethodSpec("action_register_payment", "register a payment against an invoice", "posted", "posted",
                   returns_action=True,
                   failures=(("UserError", "You can only register payments for posted entries."),)),
        MethodSpec("action_send_and_print", "email the invoice and generate the PDF", "posted", "posted",
                   returns_action=True,
                   failures=(("UserError", "No recipient email address found for this customer."),)),
        MethodSpec("action_debit_note", "raise a debit note against an invoice", "posted", "posted",
                   returns_action=True),
        MethodSpec("action_duplicate", "duplicate a journal entry", None, None, returns_action=True),
    ),
    create_fields={"move_type": "move_type", "partner_id": "partner_ref",
                   "invoice_date": "date_recent", "invoice_date_due": "date_future",
                   "journal_id": "small_id", "invoice_line_ids": "invoice_lines"},
    search_fields=("name", "partner_id", "invoice_date", "invoice_date_due",
                   "amount_total", "amount_residual", "state", "payment_state", "move_type"),
    doc_prefix="INV",
    weight=3,
)

_PURCHASE_ORDER = ModelSpec(
    model="purchase.order",
    label="purchase order",
    domain="purchase",
    personas=("purchasing officer", "procurement manager", "buyer", "supply chain planner"),
    methods=(
        MethodSpec("button_confirm", "confirm a request for quotation into a purchase order", "draft", "purchase",
                   failures=(("UserError", "You cannot confirm a purchase order with no lines."),
                             ("UserError", "This purchase order exceeds your approval limit of 5,000.00 and requires manager approval."))),
        MethodSpec("button_approve", "approve a purchase order that is waiting for sign-off", "to approve", "purchase",
                   failures=(("AccessError", "Only Purchase Managers can approve orders above the double-validation threshold."),)),
        MethodSpec("button_cancel", "cancel a purchase order", "purchase", "cancel",
                   failures=(("UserError", "Unable to cancel this purchase order: some receipts have already been processed."),)),
        MethodSpec("button_draft", "reset a cancelled purchase order to draft", "cancel", "draft"),
        MethodSpec("button_lock", "lock a purchase order against further changes", "purchase", "purchase"),
        MethodSpec("action_create_invoice", "create the vendor bill from a confirmed purchase order", "purchase", "purchase",
                   returns_action=True,
                   failures=(("UserError", "There is no invoiceable line on this purchase order."),)),
        MethodSpec("action_rfq_send", "email the request for quotation to the vendor", "draft", "sent",
                   returns_action=True),
    ),
    create_fields={"partner_id": "vendor_ref", "date_order": "datetime_recent",
                   "currency_id": "small_id", "order_line": "purchase_lines"},
    search_fields=("name", "partner_id", "date_order", "amount_total", "state", "invoice_status"),
    doc_prefix="P",
    weight=3,
)

_STOCK_PICKING = ModelSpec(
    model="stock.picking",
    label="transfer",
    domain="inventory",
    personas=("warehouse operator", "inventory manager", "shipping clerk", "receiving clerk"),
    methods=(
        MethodSpec("action_confirm", "mark a draft transfer as to-do", "draft", "confirmed"),
        MethodSpec("action_assign", "check availability and reserve stock for a transfer", "confirmed", "assigned",
                   failures=(("UserError", "No stock could be reserved: quantity on hand is 0 in WH/Stock."),)),
        MethodSpec("button_validate", "validate a delivery, receipt, or internal transfer", "assigned", "done",
                   failures=(("UserError", "You cannot validate a transfer with no quantities done."),
                             ("UserError", "You need to supply a lot/serial number for product 'Industrial Pressure Sensor V2'."),
                             ("ValidationError", "The quantity done exceeds the demanded quantity and backorder handling is disabled."))),
        MethodSpec("action_cancel", "cancel a warehouse transfer", "assigned", "cancel"),
        MethodSpec("do_unreserve", "release the stock reserved by a transfer", "assigned", "confirmed"),
        MethodSpec("action_create_return_picking", "create a return for a completed transfer", "done", "done",
                   returns_action=True),
        MethodSpec("button_scrap", "scrap damaged goods from a transfer", None, None, returns_action=True),
    ),
    create_fields={"partner_id": "partner_ref", "picking_type_id": "small_id",
                   "scheduled_date": "datetime_future", "location_id": "small_id",
                   "location_dest_id": "small_id", "move_ids_without_package": "stock_moves"},
    search_fields=("name", "partner_id", "scheduled_date", "date_done", "state", "picking_type_id", "origin"),
    doc_prefix="WH/OUT",
    weight=3,
)

_MRP_PRODUCTION = ModelSpec(
    model="mrp.production",
    label="manufacturing order",
    domain="manufacturing",
    personas=("production planner", "shop floor supervisor", "manufacturing manager", "quality inspector"),
    methods=(
        MethodSpec("action_confirm", "confirm a draft manufacturing order", "draft", "confirmed"),
        MethodSpec("button_plan", "schedule the work orders for a manufacturing order", "confirmed", "progress",
                   failures=(("UserError", "No work center is available in the requested time window."),)),
        MethodSpec("action_assign", "reserve the components for a manufacturing order", "confirmed", "confirmed",
                   failures=(("UserError", "Components are not available: 12 units of 'Custom PCB Mainboard' are missing."),)),
        MethodSpec("button_mark_done", "close a manufacturing order as produced", "progress", "done",
                   failures=(("UserError", "You must set a serial number on the finished product before closing."),
                             ("UserError", "Some components have not been consumed. Confirm the consumption warning to proceed."))),
        MethodSpec("action_cancel", "cancel a manufacturing order", "confirmed", "cancel"),
        MethodSpec("button_unbuild", "unbuild a finished manufacturing order", "done", "done", returns_action=True),
        MethodSpec("button_scrap", "scrap defective components on the shop floor", None, None, returns_action=True),
        MethodSpec("action_generate_serial", "generate the serial number for the finished good", None, None),
    ),
    create_fields={"product_id": "product_ref", "product_qty": "quantity",
                   "bom_id": "small_id", "date_start": "datetime_future", "picking_type_id": "small_id"},
    search_fields=("name", "product_id", "product_qty", "date_start", "state", "bom_id"),
    doc_prefix="WH/MO",
    weight=3,
)

_ACCOUNT_PAYMENT = ModelSpec(
    model="account.payment",
    label="payment",
    domain="accounting",
    personas=("accounts receivable clerk", "treasury analyst", "bookkeeper", "financial controller"),
    methods=(
        MethodSpec("action_post", "post a draft payment", "draft", "posted",
                   failures=(("UserError", "The payment amount must be strictly positive."),
                             ("UserError", "No outstanding receipts account is configured on journal 'Bank'."))),
        MethodSpec("action_draft", "reset a posted payment to draft", "posted", "draft",
                   failures=(("UserError", "You cannot reset a reconciled payment to draft."),)),
        MethodSpec("action_cancel", "cancel a payment", "draft", "cancel"),
        MethodSpec("action_validate", "validate a payment awaiting confirmation", "draft", "posted"),
    ),
    create_fields={"payment_type": "payment_type", "partner_id": "partner_ref",
                   "amount": "amount", "journal_id": "small_id", "date": "date_recent"},
    search_fields=("name", "partner_id", "amount", "date", "state", "payment_type", "journal_id"),
    doc_prefix="BNK1",
    weight=2,
)

_RES_PARTNER = ModelSpec(
    model="res.partner",
    label="contact",
    domain="crm",
    personas=("sales representative", "customer service agent", "master data steward", "accounts receivable clerk"),
    methods=(
        MethodSpec("action_open_business_doc", "open the related business document", None, None, returns_action=True),
        MethodSpec("action_open_overdue_entries", "list the overdue journal entries for a customer", None, None,
                   returns_action=True),
    ),
    create_fields={"name": "company_name", "is_company": "bool_true", "email": "email",
                   "phone": "phone", "street": "street", "city": "city",
                   "country_id": "small_id", "vat": "vat"},
    search_fields=("name", "email", "phone", "city", "country_id", "customer_rank", "supplier_rank"),
    state_field=None,
    doc_prefix=None,
    weight=3,
)

_CRM_LEAD = ModelSpec(
    model="crm.lead",
    label="opportunity",
    domain="crm",
    personas=("sales representative", "sales manager", "pre-sales consultant", "SDR"),
    methods=(
        MethodSpec("action_set_won", "mark an opportunity as won", None, None),
        MethodSpec("action_set_lost", "mark an opportunity as lost", None, None,
                   kwargs={"lost_reason_id": 3},
                   failures=(("UserError", "A lost reason is required when marking an opportunity as lost."),)),
        MethodSpec("action_new_quotation", "create a quotation from an opportunity", None, None, returns_action=True),
        MethodSpec("action_schedule_meeting", "schedule a meeting with the prospect", None, None, returns_action=True),
        MethodSpec("action_assign_partner", "link the opportunity to a partner record", None, None),
        MethodSpec("action_restore", "restore an archived opportunity", None, None),
    ),
    create_fields={"name": "opportunity_name", "partner_id": "partner_ref",
                   "expected_revenue": "amount", "probability": "probability",
                   "team_id": "small_id", "date_deadline": "date_future"},
    search_fields=("name", "partner_id", "expected_revenue", "probability", "stage_id", "date_deadline", "user_id"),
    state_field=None,
    doc_prefix=None,
    weight=2,
)

_STOCK_QUANT = ModelSpec(
    model="stock.quant",
    label="stock level",
    domain="inventory",
    personas=("inventory manager", "warehouse operator", "cycle count auditor"),
    methods=(
        MethodSpec("action_apply_inventory", "apply a counted inventory adjustment", None, None,
                   failures=(("UserError", "You cannot apply an inventory adjustment on a location of type 'view'."),)),
        MethodSpec("action_set_inventory_quantity", "set the counted quantity on a stock line", None, None),
        MethodSpec("action_clear_inventory_quantity", "clear the counted quantity", None, None),
        MethodSpec("action_stock_quant_relocate", "move stock to a different location", None, None,
                   returns_action=True),
    ),
    create_fields={"product_id": "product_ref", "location_id": "small_id", "inventory_quantity": "quantity"},
    search_fields=("product_id", "location_id", "quantity", "reserved_quantity",
                   "inventory_quantity", "inventory_date"),
    state_field=None,
    doc_prefix=None,
    weight=2,
)

_HR_LEAVE = ModelSpec(
    model="hr.leave",
    label="time off request",
    domain="hr",
    personas=("HR officer", "team manager", "payroll administrator"),
    methods=(
        MethodSpec("action_approve", "approve a time off request", "confirm", "validate",
                   failures=(("AccessError", "Only a Time Off Officer can approve this request."),
                             ("UserError", "The employee does not have enough allocation days remaining."))),
        MethodSpec("action_refuse", "refuse a time off request", "confirm", "refuse"),
        MethodSpec("action_cancel", "cancel an approved time off request", "validate", "cancel",
                   returns_action=True),
        MethodSpec("action_reset_confirm", "send a refused request back for approval", "refuse", "confirm"),
    ),
    create_fields={"employee_id": "small_id", "holiday_status_id": "small_id",
                   "request_date_from": "date_future", "request_date_to": "date_future_later"},
    search_fields=("employee_id", "holiday_status_id", "request_date_from",
                   "request_date_to", "number_of_days", "state"),
    doc_prefix=None,
    weight=2,
)

_PROJECT_TASK = ModelSpec(
    model="project.task",
    label="task",
    domain="project",
    personas=("project manager", "consultant", "team lead", "field service technician"),
    methods=(
        MethodSpec("action_archive", "archive a completed task", None, None),
        MethodSpec("action_unschedule_task", "remove a task from the schedule", None, None),
        MethodSpec("action_create_invoice", "invoice the timesheets logged on a task", None, None,
                   returns_action=True,
                   failures=(("UserError", "There is nothing to invoice: no billable timesheet entries were found."),)),
        MethodSpec("action_convert_to_subtask", "convert a task into a subtask", None, None, returns_action=True),
        MethodSpec("action_fsm_validate", "close a field service task as done", None, None),
    ),
    create_fields={"name": "task_name", "project_id": "small_id", "user_ids": "user_list",
                   "date_deadline": "date_future", "partner_id": "partner_ref"},
    search_fields=("name", "project_id", "stage_id", "user_ids", "date_deadline",
                   "allocated_hours", "effective_hours"),
    state_field=None,
    doc_prefix=None,
    weight=2,
)

_HR_EXPENSE = ModelSpec(
    model="hr.expense",
    label="expense",
    domain="hr",
    personas=("employee", "team manager", "accounts payable clerk", "HR officer"),
    methods=(
        MethodSpec("action_submit", "submit an expense for approval", "draft", "submitted"),
        MethodSpec("action_approve", "approve a submitted expense", "submitted", "approved",
                   failures=(("AccessError", "Only an Expense Approver can validate this expense."),
                             ("UserError", "You cannot approve your own expense."))),
        MethodSpec("action_refuse", "refuse an expense claim", "submitted", "refused", returns_action=True),
        MethodSpec("action_post", "post the accounting entry for an approved expense", "approved", "posted",
                   failures=(("UserError", "No expense journal is configured on the company."),)),
        MethodSpec("action_reset", "reset an expense back to draft", "submitted", "draft"),
        MethodSpec("action_pay", "reimburse an approved expense", "posted", "done"),
    ),
    create_fields={"name": "expense_name", "employee_id": "small_id",
                   "total_amount": "amount", "date": "date_recent", "product_id": "small_id"},
    search_fields=("name", "employee_id", "total_amount", "state", "date"),
    doc_prefix=None,
    weight=1,
)

_STOCK_LOT = ModelSpec(
    model="stock.lot",
    label="lot / serial number",
    domain="inventory",
    personas=("quality inspector", "warehouse operator", "traceability analyst"),
    methods=(
        MethodSpec("action_lot_open_quants", "show the stock on hand for a lot", None, None, returns_action=True),
        MethodSpec("action_lot_open_transfers", "show every transfer that moved a lot", None, None,
                   returns_action=True),
    ),
    create_fields={"name": "lot_name", "product_id": "product_ref", "company_id": "small_id"},
    search_fields=("name", "product_id", "create_date", "product_qty"),
    state_field=None,
    doc_prefix=None,
    weight=1,
)

_POS_ORDER = ModelSpec(
    model="pos.order",
    label="point of sale order",
    domain="sales",
    personas=("store manager", "cashier", "retail operations analyst"),
    methods=(
        MethodSpec("action_pos_order_invoice", "generate the invoice for a POS order", "paid", "invoiced",
                   returns_action=True,
                   failures=(("UserError", "You cannot invoice a POS order without a customer."),)),
        MethodSpec("action_pos_order_cancel", "cancel a point of sale order", "draft", "cancel"),
        MethodSpec("action_send_receipt", "email the receipt to the customer", None, None),
    ),
    create_fields={"partner_id": "partner_ref", "session_id": "small_id", "amount_total": "amount"},
    search_fields=("name", "partner_id", "date_order", "amount_total", "state", "session_id"),
    doc_prefix="POS",
    weight=1,
)

_ACCOUNT_BANK_STATEMENT_LINE = ModelSpec(
    model="account.bank.statement.line",
    label="bank statement line",
    domain="accounting",
    personas=("bookkeeper", "treasury analyst", "financial controller"),
    methods=(
        MethodSpec("action_undo_reconciliation", "undo the reconciliation of a bank line", None, None),
        MethodSpec("action_unreconcile_entry", "unreconcile a single matched entry", None, None),
        MethodSpec("action_button_draft", "reset a bank statement line to draft", None, None),
        MethodSpec("action_save_close", "save and close the reconciliation widget", None, None, returns_action=True),
    ),
    create_fields={"payment_ref": "payment_ref", "amount": "amount",
                   "journal_id": "small_id", "partner_id": "partner_ref"},
    search_fields=("payment_ref", "partner_id", "amount", "is_reconciled"),
    state_field=None,
    doc_prefix=None,
    weight=1,
)

_HELPDESK_TICKET = ModelSpec(
    model="helpdesk.ticket",
    label="helpdesk ticket",
    domain="services",
    personas=("support agent", "support team lead", "customer success manager"),
    methods=(
        MethodSpec("action_convert_to_task", "convert a ticket into a project task", None, None,
                   returns_action=True),
        MethodSpec("action_convert_ticket_to_lead_or_opportunity",
                   "convert a ticket into a sales opportunity", None, None, returns_action=True),
        MethodSpec("action_generate_fsm_task", "dispatch a field service task from a ticket", None, None,
                   returns_action=True),
        MethodSpec("action_create_replacement", "raise a replacement delivery for a ticket", None, None,
                   returns_action=True),
        MethodSpec("action_timer_start", "start the billable timer on a ticket", None, None),
        MethodSpec("action_timer_stop", "stop the billable timer on a ticket", None, None,
                   returns_action=True),
    ),
    create_fields={"name": "ticket_name", "partner_id": "partner_ref",
                   "team_id": "small_id", "priority": "priority"},
    search_fields=("name", "partner_id", "team_id", "stage_id", "priority", "create_date", "user_id"),
    state_field=None,
    doc_prefix=None,
    weight=1,
)

_PRODUCT_TEMPLATE = ModelSpec(
    model="product.template",
    label="product",
    domain="sales",
    personas=("product manager", "master data steward", "purchasing officer", "e-commerce manager"),
    methods=(
        MethodSpec("action_archive", "archive a discontinued product", None, None),
        MethodSpec("action_open_label_layout", "print product labels", None, None, returns_action=True),
        MethodSpec("action_view_orderpoints", "review the reordering rules for a product", None, None,
                   returns_action=True),
        MethodSpec("action_product_tmpl_forecast_report", "open the forecasted stock report", None, None,
                   returns_action=True),
    ),
    create_fields={"name": "product_name", "type": "product_type", "list_price": "amount",
                   "standard_price": "cost", "categ_id": "small_id", "uom_id": "small_id"},
    search_fields=("name", "default_code", "list_price", "standard_price", "categ_id", "type", "active"),
    state_field=None,
    doc_prefix=None,
    weight=2,
)

_ACCOUNT_MOVE_LINE = ModelSpec(
    model="account.move.line",
    label="journal item",
    domain="accounting",
    personas=("bookkeeper", "financial controller", "auditor"),
    methods=(
        MethodSpec("action_reconcile", "reconcile a set of journal items", None, None,
                   failures=(("UserError", "You can only reconcile journal items on the same account."),
                             ("UserError", "Entries are not balanced: the reconciliation would leave 12.50 open."))),
        MethodSpec("action_register_payment", "register a payment from selected journal items", None, None,
                   returns_action=True),
        MethodSpec("action_automatic_entry", "create an automatic adjustment entry", None, None, returns_action=True),
    ),
    create_fields={},
    search_fields=("move_id", "account_id", "partner_id", "debit", "credit",
                   "balance", "date", "reconciled", "full_reconcile_id"),
    state_field=None,
    doc_prefix=None,
    weight=2,
)

_STOCK_WAREHOUSE_ORDERPOINT = ModelSpec(
    model="stock.warehouse.orderpoint",
    label="reordering rule",
    domain="inventory",
    personas=("supply chain planner", "inventory manager", "purchasing officer"),
    methods=(
        MethodSpec("action_replenish", "trigger replenishment for a reordering rule", None, None,
                   failures=(("UserError", "No supplier is defined for product 'High-Torque Electric Motor'."),)),
        MethodSpec("action_open_orderpoints", "open the replenishment dashboard", None, None, returns_action=True),
    ),
    create_fields={"product_id": "product_ref", "location_id": "small_id",
                   "product_min_qty": "quantity", "product_max_qty": "quantity_large"},
    search_fields=("product_id", "location_id", "product_min_qty", "product_max_qty", "qty_to_order"),
    state_field=None,
    doc_prefix=None,
    weight=1,
)

_REPAIR_ORDER = ModelSpec(
    model="repair.order",
    label="repair order",
    domain="operations",
    personas=("repair technician", "after-sales manager", "warehouse operator"),
    methods=(
        MethodSpec("action_repair_start", "start work on a repair order", "confirmed", "under_repair"),
        MethodSpec("action_repair_end", "finish a repair and record the outcome", "under_repair", "done",
                   failures=(("UserError", "You cannot end a repair with unconsumed parts still reserved."),)),
        MethodSpec("action_repair_cancel", "cancel a repair order", "confirmed", "cancel"),
        MethodSpec("action_create_sale_order", "bill the repair to the customer", "done", "done",
                   returns_action=True),
        MethodSpec("action_assign", "reserve the spare parts for a repair", "confirmed", "confirmed"),
    ),
    create_fields={"product_id": "product_ref", "partner_id": "partner_ref",
                   "picking_type_id": "small_id", "schedule_date": "datetime_future"},
    search_fields=("name", "product_id", "partner_id", "state", "schedule_date"),
    doc_prefix="RO",
    weight=1,
)

ALL_SPECS: Tuple[ModelSpec, ...] = (
    _SALE_ORDER, _ACCOUNT_MOVE, _PURCHASE_ORDER, _STOCK_PICKING, _MRP_PRODUCTION,
    _ACCOUNT_PAYMENT, _RES_PARTNER, _CRM_LEAD, _STOCK_QUANT, _HR_LEAVE,
    _PROJECT_TASK, _HR_EXPENSE, _STOCK_LOT, _POS_ORDER,
    _ACCOUNT_BANK_STATEMENT_LINE, _PRODUCT_TEMPLATE, _ACCOUNT_MOVE_LINE,
    _STOCK_WAREHOUSE_ORDERPOINT, _HELPDESK_TICKET, _REPAIR_ORDER,
)

SPEC_BY_MODEL: Dict[str, ModelSpec] = {s.model: s for s in ALL_SPECS}


# ──────────────────────────────────────────────────────────────────────────────
# Tier B — the wider surface, discovered from the codebase
# ──────────────────────────────────────────────────────────────────────────────
#
# The specs above are Tier A: hand-written state machines with real failure
# messages, which is what the workflow, recovery, verification and refusal
# families need and what cannot be derived automatically.
#
# Tier B widens coverage across the rest of the Odoo 19 Community + Enterprise
# tree. The split is deliberate about what can and cannot be inferred:
#
#   methods       AUTO — read from the AST scan and filtered, so they are
#                        verified by construction, same as Tier A
#   fields        AUTO — read from the extracted field table
#   state machine  NO  — the `state` field's selection values are absent from the
#                        extraction for most models, and guessing which method
#                        moves a record from which state to which other state is
#                        exactly the kind of invention that produced v1's
#                        `action_post()` on `decimal.precision`. Tier B methods
#                        therefore carry no from/to state and the simulator does
#                        not claim one.
#   failures       NO  — a plausible-sounding but fabricated Odoo exception is
#                        worse than no failure sample at all.
#   label         HAND — `models.name` is just the technical name for most models
#                        ("mrp.workorder" -> "mrp.workorder"). Feeding that to the
#                        persona would make them say the technical name, which is
#                        the one thing phase 1 forbids. So the label, the domain
#                        and the personas are written out below; everything else
#                        about these models comes from the codebase.

#: model -> (human label, domain, extra personas beyond the domain default)
TIER_B_MODELS: Dict[str, Tuple[str, str]] = {
    # -- sales & commerce ------------------------------------------------------
    "sale.order.line":            ("order line", "sales"),
    "product.product":            ("product variant", "sales"),
    "product.pricelist":          ("price list", "sales"),
    "product.pricelist.item":     ("price list rule", "sales"),
    "product.supplierinfo":       ("vendor price", "purchase"),
    "pos.session":                ("point of sale session", "sales"),
    "payment.transaction":        ("online payment", "accounting"),
    "sale.commission.plan":       ("commission plan", "sales"),
    "utm.campaign":               ("marketing campaign", "sales"),
    "mailing.mailing":            ("email campaign", "sales"),
    "event.event":                ("event", "services"),
    "event.registration":         ("event registration", "services"),

    # -- purchasing ------------------------------------------------------------
    "purchase.order.line":        ("purchase order line", "purchase"),
    "purchase.requisition":       ("purchase agreement", "purchase"),

    # -- inventory & logistics -------------------------------------------------
    "stock.move":                 ("stock move", "inventory"),
    "stock.move.line":            ("stock move line", "inventory"),
    "stock.picking.batch":        ("picking batch", "inventory"),
    "stock.scrap":                ("scrap order", "inventory"),
    "stock.warehouse":            ("warehouse", "inventory"),
    "stock.location":             ("stock location", "inventory"),
    "stock.putaway.rule":         ("put-away rule", "inventory"),
    "stock.valuation.layer":      ("stock valuation entry", "inventory"),
    "delivery.carrier":           ("delivery method", "inventory"),

    # -- manufacturing ---------------------------------------------------------
    "mrp.workorder":              ("work order", "manufacturing"),
    "mrp.bom":                    ("bill of materials", "manufacturing"),
    "mrp.workcenter":             ("work centre", "manufacturing"),
    "mrp.unbuild":                ("unbuild order", "manufacturing"),
    "quality.check":              ("quality check", "manufacturing"),
    "quality.alert":              ("quality alert", "manufacturing"),

    # -- accounting ------------------------------------------------------------
    "account.journal":            ("journal", "accounting"),
    "account.account":            ("general ledger account", "accounting"),
    "account.tax":                ("tax", "accounting"),
    "account.bank.statement":     ("bank statement", "accounting"),
    "account.analytic.account":   ("analytic account", "accounting"),
    "account.analytic.line":      ("analytic entry", "accounting"),
    "account.payment.term":       ("payment term", "accounting"),
    "account.fiscal.position":    ("fiscal position", "accounting"),
    "account.asset":              ("fixed asset", "accounting"),
    "account.reconcile.model":    ("reconciliation model", "accounting"),
    "account.return":             ("tax return", "accounting"),
    "sdd.mandate":                ("direct debit mandate", "accounting"),

    # -- hr --------------------------------------------------------------------
    "hr.employee":                ("employee record", "hr"),
    "hr.contract":                ("employment contract", "hr"),
    "hr.attendance":              ("attendance entry", "hr"),
    "hr.applicant":               ("job applicant", "hr"),
    "hr.payslip":                 ("payslip", "hr"),
    "hr.payslip.run":             ("payslip batch", "hr"),
    "hr.leave.allocation":        ("time off allocation", "hr"),
    "hr.appraisal":               ("appraisal", "hr"),
    "hr.work.entry":              ("work entry", "hr"),
    "planning.slot":              ("shift", "hr"),

    # -- projects & services ---------------------------------------------------
    "project.project":            ("project", "project"),
    "project.milestone":          ("project milestone", "project"),
    "account.analytic.plan":      ("analytic plan", "project"),
    "helpdesk.team":              ("support team", "services"),
    "maintenance.request":        ("maintenance request", "operations"),
    "maintenance.equipment":      ("equipment record", "operations"),
    "fleet.vehicle":              ("vehicle", "operations"),

    # -- CRM & communication ---------------------------------------------------
    "crm.team":                   ("sales team", "crm"),
    "mail.activity":              ("scheduled activity", "crm"),
    "calendar.event":             ("meeting", "crm"),
    "res.partner.bank":           ("bank account", "crm"),
    "res.users":                  ("user account", "crm"),
}

#: Method names that open a view or a wizard rather than doing business work.
#: These are legitimate Odoo methods, but a sample whose whole content is
#: "I opened a list view for you" teaches nothing.
_UI_ONLY_METHOD = __import__("re").compile(
    r"^action_(?:view|open|see|show|print|preview|redirect|goto|get|toggle)_"
    r"|^action_(?:view|open)$"
    r"|^action_l10n_|^button_l10n_|^action_.*_l10n_"
    r"|_wizard$|^action_.*dashboard|graph|kanban|_report$|^action_help",
    __import__("re").IGNORECASE,
)

#: The verbs that denote a real business operation, most central first. Used to
#: rank a model's methods so the discovered surface keeps the ones a user would
#: actually ask for.
_CORE_VERBS = (
    "confirm", "validate", "approve", "post", "done", "cancel", "submit",
    "refuse", "reject", "start", "close", "pay", "send", "assign", "plan",
    "apply", "reset", "draft", "archive", "publish", "generate", "compute",
    "split", "merge", "duplicate", "replan", "unreserve", "scrap",
    "pass", "fail", "register", "reconcile", "invoice", "receive",
)

#: Country and localisation markers. A method like
#: `action_absence_swiss_employee_from_payslip` is a real method, but a sample
#: built on it teaches a jurisdiction-specific edge case rather than Odoo.
_LOCALISATION_HINT = __import__("re").compile(
    r"_(?:swiss|belgian|french|german|italian|spanish|dutch|indian|brazil|"
    r"mexic|austral|kenya|malays|philipp|romania|turkish|vietnam|saudi|uae|"
    r"peppol|nemhandel|fatturapa|facturae|cfdi|ubl|oioubl|edi|einvoice|"
    r"gstin|gstr|ewaybill|sepa|sdd)\w*",
    __import__("re").IGNORECASE,
)


def _method_score(name: str) -> int:
    """Ranks a method by how likely a user is to ask for it.

    Without this the surface was truncated alphabetically, which handed
    ``hr.payslip`` methods like ``action_absence_swiss_employee_from_payslip``
    while dropping ``action_payslip_done``.
    """
    score = 0
    stem = name
    for prefix in ("action_", "button_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    words = stem.split("_")

    for i, verb in enumerate(_CORE_VERBS):
        if verb in words:
            score += 100 - i          # earlier verbs in the list rank higher
            break

    if _LOCALISATION_HINT.search(name):
        score -= 200
    # Short names are the primary operations; long ones are edge cases.
    score -= 6 * max(0, len(words) - 2)
    if name.startswith("button_"):
        score += 10                   # button_* are the on-form primary actions
    return score


#: Fields worth reading back, in preference order, when auto-deriving a spec.
_PREFERRED_SEARCH_FIELDS = (
    "name", "display_name", "reference", "code", "default_code",
    "partner_id", "employee_id", "product_id", "user_id", "project_id",
    "date", "date_order", "date_start", "date_deadline", "create_date",
    "amount_total", "amount", "total_amount", "quantity", "product_qty",
    "state", "stage_id", "company_id", "active", "priority",
)

#: Field name -> value-generator key, for auto-derived create calls.
_AUTO_FIELD_GENERATORS = {
    "name": "company_name", "partner_id": "partner_ref", "product_id": "product_ref",
    "employee_id": "small_id", "user_id": "small_id", "company_id": "small_id",
    "project_id": "small_id", "journal_id": "small_id", "location_id": "small_id",
    "picking_type_id": "small_id", "team_id": "small_id", "currency_id": "small_id",
    "date": "date_recent", "date_order": "datetime_recent", "date_start": "datetime_future",
    "date_deadline": "date_future", "amount": "amount", "amount_total": "amount",
    "total_amount": "amount", "quantity": "quantity", "product_qty": "quantity",
    "product_uom_qty": "quantity", "priority": "priority", "code": "lot_name",
}

#: Domain -> personas, so an auto-derived model still gets a plausible speaker.
_DOMAIN_PERSONAS: Dict[str, Tuple[str, ...]] = {
    "sales": ("sales representative", "sales manager", "store manager", "office manager"),
    "purchase": ("purchasing officer", "procurement manager", "inventory manager"),
    "inventory": ("warehouse operator", "inventory manager", "shop floor supervisor"),
    "manufacturing": ("production planner", "shop floor supervisor", "quality inspector"),
    "accounting": ("bookkeeper", "financial controller", "accounts receivable clerk",
                   "finance director"),
    "hr": ("HR officer", "team manager", "payroll administrator"),
    "project": ("project manager", "consultant", "team lead"),
    "services": ("support agent", "customer success manager", "support team lead"),
    "operations": ("maintenance technician", "plant manager", "shop floor supervisor"),
    "crm": ("sales representative", "customer success manager", "office manager"),
}


def discover_tier_b(kg, exclude: Optional[Dict[str, ModelSpec]] = None,
                    min_methods: int = 1) -> Tuple[Dict[str, ModelSpec], List[str]]:
    """Builds specs for the wider model set from the extracted codebase.

    Methods and fields come from the AST scan, so they are verified the same way
    Tier A is. States and failure modes are deliberately left empty — see the
    note above ``TIER_B_MODELS``.
    """
    exclude = exclude or {}
    discovered: Dict[str, ModelSpec] = {}
    notes: List[str] = []

    for model, (label, domain) in TIER_B_MODELS.items():
        if model in exclude:
            continue
        try:
            fields = kg.get_model_fields(model)
        except Exception as exc:                                      # pragma: no cover
            notes.append(f"{model}: field lookup failed ({exc})")
            continue
        if not fields:
            notes.append(f"{model}: skipped — not in the knowledge graph")
            continue
        field_names = {f["name"] for f in fields}

        try:
            raw = {m["name"] for m in kg.get_model_methods(model, public_only=True,
                                                           actions_only=True)}
        except Exception as exc:                                      # pragma: no cover
            notes.append(f"{model}: method lookup failed ({exc})")
            continue

        usable = [m for m in raw if not _UI_ONLY_METHOD.search(m)]
        # Rank before truncating: alphabetical order buries the primary
        # operations under whatever happens to start with "a".
        usable.sort(key=lambda m: (-_method_score(m), m))
        usable = [m for m in usable if _method_score(m) > -100][:10]

        if len(usable) < min_methods:
            notes.append(f"{model}: skipped — {len(raw)} methods, none survived "
                         f"the UI-only and localisation filters")
            continue

        methods = tuple(
            MethodSpec(name=m, intent=_intent_for(m, label),
                       from_state=None, to_state=None)
            for m in usable
        )

        search = tuple(f for f in _PREFERRED_SEARCH_FIELDS if f in field_names)[:7]
        if not search:
            search = ("id", "display_name")

        required = {f["name"] for f in fields if f.get("required")}
        create = {name: key for name, key in _AUTO_FIELD_GENERATORS.items()
                  if name in field_names and (name in required or name in
                                              ("name", "partner_id", "product_id", "date"))}

        discovered[model] = ModelSpec(
            model=model, label=label, domain=domain,
            personas=_DOMAIN_PERSONAS.get(domain, ("office manager",)),
            methods=methods, create_fields=create, search_fields=search,
            state_field="state" if "state" in field_names else None,
            doc_prefix=None,
            weight=1,
        )

    return discovered, notes


def _intent_for(method: str, label: str) -> str:
    """A human phrasing of what a method does, from its name.

    Only used to steer phase 1 towards the right subject; the trained model never
    sees it. Deliberately hedged ("deal with") when the verb is unrecognised,
    rather than asserting a state change that may not happen.
    """
    verb = method
    for prefix in ("action_", "button_"):
        if verb.startswith(prefix):
            verb = verb[len(prefix):]
            break
    words = verb.replace("_", " ").strip()

    known = {
        "confirm": f"confirm the {label}", "cancel": f"cancel the {label}",
        "validate": f"validate the {label}", "approve": f"approve the {label}",
        "refuse": f"refuse the {label}", "done": f"complete the {label}",
        "post": f"post the {label}", "draft": f"put the {label} back to draft",
        "start": f"start the {label}", "assign": f"assign the {label}",
        "archive": f"archive the {label}", "submit": f"submit the {label}",
        "pay": f"pay the {label}", "send": f"send the {label}",
        "reset": f"reset the {label}", "apply": f"apply the {label}",
        "plan": f"schedule the {label}", "close": f"close the {label}",
    }
    if words in known:
        return known[words]
    for key, phrase in known.items():
        if words.startswith(key + " ") or words.endswith(" " + key):
            return f"{phrase} ({words})"
    return f"deal with the {label} — specifically, {words}"


# ──────────────────────────────────────────────────────────────────────────────
# Verification against the extracted knowledge graph
# ──────────────────────────────────────────────────────────────────────────────

def verify_against_kg(kg, strict: bool = False) -> Tuple[Dict[str, ModelSpec], List[str]]:
    """Drop any curated model or method the AST extraction cannot confirm.

    Returns the verified spec map and a list of human-readable warnings.  This is
    what stops the dataset from ever again containing a call to a method that does
    not exist on the target model: the allowlist is curated by hand, then
    intersected with what was actually parsed out of the Odoo 19 source tree.
    """
    verified: Dict[str, ModelSpec] = {}
    warnings: List[str] = []

    for spec in ALL_SPECS:
        try:
            fields = {f["name"] for f in kg.get_model_fields(spec.model)}
        except Exception as exc:                                      # pragma: no cover
            warnings.append(f"{spec.model}: field lookup failed ({exc})")
            fields = set()

        if not fields:
            warnings.append(f"{spec.model}: DROPPED — model not present in knowledge graph")
            continue

        try:
            known = {m["name"] for m in kg.get_model_methods(spec.model, public_only=True)}
        except Exception as exc:                                      # pragma: no cover
            warnings.append(f"{spec.model}: method lookup failed ({exc})")
            known = set()

        kept, dropped = [], []
        for meth in spec.methods:
            if meth.name in known:
                kept.append(meth)
            else:
                dropped.append(meth.name)

        if dropped:
            warnings.append(f"{spec.model}: dropped unverified methods {sorted(dropped)}")
        if not kept:
            warnings.append(f"{spec.model}: DROPPED — no method survived KG verification")
            continue

        bad_search = [f for f in spec.search_fields if f not in fields]
        search = tuple(f for f in spec.search_fields if f in fields) or ("id", "display_name")
        if bad_search:
            warnings.append(f"{spec.model}: dropped unverified search fields {bad_search}")

        bad_create = [f for f in spec.create_fields if f not in fields]
        create = {k: v for k, v in spec.create_fields.items() if k in fields}
        if bad_create:
            warnings.append(f"{spec.model}: dropped unverified create fields {bad_create}")

        state_field = spec.state_field if (spec.state_field and spec.state_field in fields) else None

        verified[spec.model] = ModelSpec(
            model=spec.model, label=spec.label, domain=spec.domain, personas=spec.personas,
            methods=tuple(kept), create_fields=create, search_fields=search,
            state_field=state_field, doc_prefix=spec.doc_prefix, weight=spec.weight,
        )

    if strict and warnings:
        raise RuntimeError("Agent surface failed strict KG verification:\n  " + "\n  ".join(warnings))

    return verified, warnings


# ──────────────────────────────────────────────────────────────────────────────
# Realistic value synthesis
# ──────────────────────────────────────────────────────────────────────────────

COMPANIES: Tuple[str, ...] = (
    "Northwind Traders BV", "Kessler Präzisionstechnik GmbH", "Aurora Medical Supplies",
    "Marchetti Componenti S.r.l.", "Baltic Freight Solutions AS", "Hollis & Vane Ltd",
    "Cedarline Furniture Co.", "Verde Agro Cooperative", "Sanderson Dental Group",
    "Ridgeway Plumbing Supplies", "Okonkwo Textiles Ltd", "Solaris Roofing Systems",
    "Fenwick Marine Services", "Delacroix Patisserie SARL", "Halvorsen Fisheries AS",
    "Meridian Diagnostics Inc.", "Batra Electricals Pvt Ltd", "Groupe Lemoine SA",
    "Applewood Veterinary Clinic", "Torres Hermanos Distribución", "Kirkland Safety Wear",
    "Nordvik Timber AB", "Chen Precision Tooling", "Ferrara Ceramics S.p.A.",
    "Prairie Grain Handlers", "Vandenberg Horticulture BV", "Ashworth Legal Services",
    "Coastal Yacht Refit Ltd", "Zubair Trading LLC", "Lindqvist Bygg AB",
)

VENDORS: Tuple[str, ...] = (
    "Rhein Stahlhandel GmbH", "Pacific Component Imports", "Bergström Råvaror AB",
    "Duval Emballages SAS", "Kowalski Narzędzia Sp. z o.o.", "Anand Polymers Ltd",
    "Trentino Metalli S.r.l.", "Fairbairn Fasteners Ltd", "Yamashita Bearings K.K.",
    "Costa Verde Embalagens Lda", "Novak Chemicals d.o.o.", "Ellingsen Elektro AS",
)

PRODUCTS: Tuple[str, ...] = (
    "Industrial Pressure Sensor V2", "Ergonomic Office Desk 160x80", "Custom PCB Mainboard Rev.C",
    "High-Torque Electric Motor 3kW", "Acoustic Wall Panel 60x60", "Stainless Steel Flange DN50",
    "Hydraulic Hose Assembly 2m", "LED Panel Light 40W 4000K", "Conveyor Belt Section 1.2m",
    "Precision Ball Bearing 6205", "Aluminium Extrusion Profile 40x40", "Safety Harness Class A",
    "Powder Coating Primer 20L", "CNC Milling Insert TNMG160408", "Thermal Insulation Roll 10m²",
)

CITIES: Tuple[str, ...] = (
    "Rotterdam", "Stuttgart", "Lyon", "Bologna", "Gothenburg", "Manchester", "Bilbao",
    "Kraków", "Antwerp", "Porto", "Aarhus", "Pune", "Guadalajara", "Christchurch", "Ljubljana",
)

STREETS: Tuple[str, ...] = (
    "Havenstraat 42", "Industriestraße 118", "12 Rue des Ateliers", "Via Emilia 305",
    "Verkstadsgatan 7", "Unit 14, Pollard Industrial Estate", "Polígono Industrial Norte, Nave 22",
    "ul. Fabryczna 63", "Noorderlaan 91", "Rua da Alfândega 210",
)

TASK_NAMES: Tuple[str, ...] = (
    "Migrate legacy pricelists to the new structure", "On-site commissioning of line 3",
    "Draft the acceptance test protocol", "Rework the packaging artwork for EU labelling",
    "Second-round supplier audit", "Configure the EDI mapping for the new customer",
    "Close out the punch list from the Q1 installation", "Train the warehouse team on the new scanner flow",
)

OPPORTUNITY_NAMES: Tuple[str, ...] = (
    "Replace ageing conveyor line", "Annual maintenance contract renewal",
    "Fit-out for the new Rotterdam depot", "Pilot order for the coated variant",
    "Framework agreement — 3 year supply", "Retrofit of the packaging cell",
    "Spare parts consignment stock", "Upgrade to the automated inspection rig",
)

EXPENSE_NAMES: Tuple[str, ...] = (
    "Client visit — Stuttgart, week 11", "Trade fair travel and accommodation",
    "Monthly mileage claim", "Team offsite catering", "Certification course fees",
)

TICKET_NAMES: Tuple[str, ...] = (
    "Scanner disconnects during picking", "Invoice PDF shows the wrong tax label",
    "Cannot log in after the password reset", "Delivery address not syncing from the webshop",
    "Reordering rule triggered twice for the same product",
)

MAINTENANCE_NAMES: Tuple[str, ...] = (
    "Quarterly service — compressor A2", "Coolant leak on the CNC lathe",
    "Replace worn drive belt on line 2", "Calibration of the torque wrenches",
)


class ValueFactory:
    """Generates realistic, internally consistent field values for tool arguments.

    Deterministic: seeded per sample index so a regenerated dataset is reproducible
    and so the same record is described consistently across the turns of one sample.
    """

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    # -- individual generators -------------------------------------------------
    def company_name(self) -> str:
        return self.rng.choice(COMPANIES)

    def vendor_name(self) -> str:
        return self.rng.choice(VENDORS)

    def product_name(self) -> str:
        return self.rng.choice(PRODUCTS)

    def _gen(self, key: str) -> Any:
        r = self.rng
        if key == "partner_ref":
            return r.randint(120, 9800)
        if key == "vendor_ref":
            return r.randint(120, 9800)
        if key == "product_ref":
            return r.randint(50, 4200)
        if key == "small_id":
            return r.randint(1, 40)
        if key == "user_list":
            return [[6, 0, [r.randint(2, 45)]]]
        if key == "amount":
            return round(r.uniform(180, 48000), 2)
        if key == "cost":
            return round(r.uniform(40, 12000), 2)
        if key == "quantity":
            return float(r.randint(1, 250))
        if key == "quantity_large":
            return float(r.randint(300, 2000))
        if key == "probability":
            return float(r.choice((10, 25, 40, 55, 70, 85)))
        if key == "priority":
            return r.choice(("0", "1", "2", "3"))
        if key == "bool_true":
            return True
        if key == "move_type":
            return r.choice(("out_invoice", "in_invoice", "out_refund", "in_refund"))
        if key == "payment_type":
            return r.choice(("inbound", "outbound"))
        if key == "product_type":
            return r.choice(("consu", "service", "combo"))
        if key == "maintenance_type":
            return r.choice(("corrective", "preventive"))
        if key == "date_recent":
            return (TODAY - timedelta(days=r.randint(1, 45))).isoformat()
        if key == "date_future":
            return (TODAY + timedelta(days=r.randint(3, 60))).isoformat()
        if key == "date_future_later":
            return (TODAY + timedelta(days=r.randint(61, 90))).isoformat()
        if key == "datetime_recent":
            d = TODAY - timedelta(days=r.randint(1, 30))
            return f"{d.isoformat()} {r.randint(7, 17):02d}:{r.choice((0, 15, 30, 45)):02d}:00"
        if key == "datetime_future":
            d = TODAY + timedelta(days=r.randint(2, 40))
            return f"{d.isoformat()} {r.randint(7, 17):02d}:{r.choice((0, 15, 30, 45)):02d}:00"
        if key == "company_name":
            return self.company_name()
        if key == "product_name":
            return self.product_name()
        if key == "task_name":
            return r.choice(TASK_NAMES)
        if key == "opportunity_name":
            return r.choice(OPPORTUNITY_NAMES)
        if key == "expense_name":
            return r.choice(EXPENSE_NAMES)
        if key == "ticket_name":
            return r.choice(TICKET_NAMES)
        if key == "maintenance_name":
            return r.choice(MAINTENANCE_NAMES)
        if key == "lot_name":
            return f"LOT-{TODAY.year}-{r.randint(10000, 99999)}"
        if key == "payment_ref":
            return f"SEPA CT {r.choice(COMPANIES).split()[0].upper()} {r.randint(100000, 999999)}"
        if key == "email":
            return f"{r.choice(('info', 'accounts', 'purchasing', 'orders'))}@" \
                   f"{r.choice(('northwind', 'kessler', 'aurora-med', 'marchetti', 'hollisvane'))}." \
                   f"{r.choice(('com', 'nl', 'de', 'it', 'co.uk'))}"
        if key == "phone":
            return f"+{r.choice((31, 49, 33, 39, 46, 44))} {r.randint(10, 99)} {r.randint(100, 999)} {r.randint(1000, 9999)}"
        if key == "street":
            return r.choice(STREETS)
        if key == "city":
            return r.choice(CITIES)
        if key == "vat":
            return f"{r.choice(('NL', 'DE', 'FR', 'IT', 'SE', 'BE'))}{r.randint(100000000, 999999999)}B{r.randint(10, 99)}"
        if key == "sale_lines":
            return [[0, 0, {"product_id": r.randint(50, 4200),
                            "product_uom_qty": float(r.randint(1, 40)),
                            "price_unit": round(r.uniform(15, 1800), 2)}]
                    for _ in range(r.randint(1, 3))]
        if key == "purchase_lines":
            return [[0, 0, {"product_id": r.randint(50, 4200),
                            "product_qty": float(r.randint(5, 500)),
                            "price_unit": round(r.uniform(4, 900), 2),
                            "date_planned": (TODAY + timedelta(days=r.randint(7, 45))).isoformat() + " 08:00:00"}]
                    for _ in range(r.randint(1, 3))]
        if key == "invoice_lines":
            return [[0, 0, {"product_id": r.randint(50, 4200),
                            "quantity": float(r.randint(1, 30)),
                            "price_unit": round(r.uniform(20, 2400), 2)}]
                    for _ in range(r.randint(1, 3))]
        if key == "stock_moves":
            return [[0, 0, {"product_id": r.randint(50, 4200),
                            "product_uom_qty": float(r.randint(1, 120))}]
                    for _ in range(r.randint(1, 2))]
        if key == "expense_lines":
            return [[0, 0, {"name": r.choice(("Taxi to client site", "Hotel — 2 nights", "Rail ticket", "Client lunch")),
                            "total_amount": round(r.uniform(18, 640), 2),
                            "product_id": r.randint(50, 400)}]
                    for _ in range(r.randint(1, 4))]
        raise KeyError(f"Unknown value generator key: {key!r}")

    # -- public API ------------------------------------------------------------
    def create_values(self, spec: ModelSpec) -> Dict[str, Any]:
        """Builds a realistic ``values`` dict for an ``odoo_create`` call."""
        return {fname: self._gen(key) for fname, key in spec.create_fields.items()}

    def doc_ref(self, spec: ModelSpec, seq: int) -> str:
        """Renders the reference a human would use for this record.

        Models without a sequence are named by their content, not by a synthetic
        counter: an opportunity is "Replace ageing conveyor line", not
        "Opportunity #3446". The label-plus-number form reads as generated data
        and is exactly the tell that made the previous dataset feel synthetic.
        """
        r = self.rng
        if spec.model == "res.partner":
            return self.company_name()
        if spec.model == "product.template":
            return self.product_name()
        if spec.model == "crm.lead":
            return r.choice(OPPORTUNITY_NAMES)
        if spec.model == "project.task":
            return r.choice(TASK_NAMES)
        if spec.model == "helpdesk.ticket":
            return r.choice(TICKET_NAMES)
        if spec.model == "hr.expense":
            return r.choice(EXPENSE_NAMES)
        if spec.model == "hr.leave":
            kind = r.choice(("Paid Time Off", "Sick Leave", "Unpaid Leave", "Parental Leave"))
            start = TODAY + timedelta(days=r.randint(5, 50))
            return f"{kind} — {start.strftime('%d %b')}"
        if spec.model == "stock.lot":
            return f"LOT-{TODAY.year}-{r.randint(10000, 99999)}"
        if spec.model == "stock.quant":
            return f"{self.product_name()} @ WH/Stock/Shelf {r.randint(1, 24)}"
        if spec.model == "stock.warehouse.orderpoint":
            return f"reordering rule for {self.product_name()}"
        if spec.model == "account.move.line":
            return f"journal item on INV/{TODAY.year}/{r.randint(1, 9999):05d}"
        if spec.model == "account.bank.statement.line":
            return self._gen("payment_ref")
        if not spec.doc_prefix:
            return f"{spec.label} {seq}"
        if spec.model == "account.move":
            prefix = self.rng.choice(("INV", "BILL", "RINV", "RBILL"))
            return f"{prefix}/{TODAY.year}/{seq:05d}"
        if spec.model == "stock.picking":
            prefix = self.rng.choice(("WH/OUT", "WH/IN", "WH/INT", "CHIC/OUT"))
            return f"{prefix}/{seq:05d}"
        if spec.model in ("sale.order", "purchase.order"):
            return f"{spec.doc_prefix}{seq:05d}"
        if spec.model == "account.payment":
            return f"BNK1/{TODAY.year}/{seq:05d}"
        if spec.model == "mrp.production":
            return f"WH/MO/{seq:05d}"
        return f"{spec.doc_prefix}/{seq:05d}"


# ──────────────────────────────────────────────────────────────────────────────
# Weighted, non-repeating sampling
# ──────────────────────────────────────────────────────────────────────────────

def build_sampling_pool(verified: Dict[str, ModelSpec]) -> List[ModelSpec]:
    """Expands the verified specs into a weighted pool for sampling.

    Unlike the old ``models[i % len(models)]``, callers draw from this pool with a
    seeded RNG, so the model/method/persona/phrasing combination for sample *i* is
    an independent draw rather than a lockstep cycle that pairs index 7 of the
    model list with index 7 of the method list forever.
    """
    pool: List[ModelSpec] = []
    for spec in verified.values():
        pool.extend([spec] * spec.weight)
    return pool


def pick_method(spec: ModelSpec, rng: random.Random,
                mutating_only: bool = False) -> Optional[MethodSpec]:
    """Chooses a callable method, optionally restricted to state-mutating ones."""
    candidates: Sequence[MethodSpec] = spec.methods
    if mutating_only:
        candidates = [m for m in spec.methods if not m.returns_action and m.to_state]
    if not candidates:
        return None
    return rng.choice(list(candidates))
