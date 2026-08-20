"""
Business Situations, Personas, and Task Shapes
==============================================

Why this module exists
----------------------
In the previous pipeline the *user turn* — the single most important token span in
an SFT sample, because it is the only thing the trained model will ever condition
on — was a Python f-string::

    f"Execute standard ERP database operation on {title} ({model})."
    f"Execute full multi-step business workflow: {base} Search existing records, "
    f"create new entry, execute '{method}()', and verify state."

8,470 of 14,285 rows used phrasing like that.  No warehouse operator has ever
typed those words.  A model trained on them learns to expect a machine-generated
instruction that names the technical model and the method, which is exactly the
information a real user does *not* supply.

This module replaces templating with **situation grounding**.  A sample starts
from a concrete business circumstance ("the customer's dock is closed Friday so
the delivery has to be split"), a persona with a distinct communication style,
and a grounded document reference.  The teacher LLM is then asked to write what
that person would actually type.  The technical detail lives in the *plan*, not
in the request — the model has to infer it, which is the skill being trained.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Personas
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Persona:
    """Who is talking to the agent, and how they write."""

    role: str
    style: str          # instruction to the teacher LLM about voice
    verbosity: str      # terse | normal | rambling
    channel: str        # chat | email | voice-transcript | ticket


PERSONAS: Tuple[Persona, ...] = (
    Persona("warehouse operator", "Blunt, hurried, lowercase, uses warehouse slang "
            "('the transfer', 'that pick', 'WH out'). Often skips punctuation. "
            "Types on a scanner terminal so keeps it short.", "terse", "chat"),
    Persona("accounts receivable clerk", "Precise about numbers and dates, mildly formal, "
            "refers to documents by their reference. Mentions the customer by name.",
            "normal", "chat"),
    Persona("sales representative", "Conversational, in a hurry between calls, gives context "
            "about what the customer said rather than what the system needs.", "normal", "chat"),
    Persona("purchasing officer", "Methodical, cites vendor names and PO numbers, "
            "cares about lead times and prices.", "normal", "email"),
    Persona("financial controller", "Formal, cautious, asks for confirmation before "
            "anything is posted, references periods and journals.", "rambling", "email"),
    Persona("production planner", "Practical shop-floor language, talks about lines, "
            "batches and shifts rather than records.", "normal", "chat"),
    Persona("office manager", "Non-technical. Describes the outcome they want in plain "
            "business language and has no idea what an Odoo model is.", "normal", "chat"),
    Persona("support agent", "Copies fragments of what the customer wrote, "
            "sometimes pastes a ticket excerpt, asks for a quick fix.", "normal", "ticket"),
    Persona("HR officer", "Careful about people's names and dates, slightly formal, "
            "conscious of policy and approval rules.", "normal", "email"),
    Persona("business owner", "Impatient, big-picture, asks compound questions, "
            "occasionally types a fragment and expects the agent to work it out.",
            "terse", "chat"),
    Persona("ERP consultant", "Knows Odoo well, uses technical vocabulary correctly, "
            "may name the model or method directly, expects precision.", "normal", "chat"),
    Persona("store manager", "Retail vocabulary — tills, shifts, footfall. "
            "Writes on a phone, so short sentences and the odd typo.", "terse", "chat"),
    # Personas multiply against situations, so widening this list widens the
    # distinct-sample ceiling as directly as adding scenarios does.
    Persona("inventory manager", "Thinks in locations, bins and counts. Precise about "
            "quantities, impatient with anything that blocks a pick.", "normal", "chat"),
    Persona("bookkeeper", "Careful and literal. Quotes reference numbers exactly and "
            "worries about periods and reconciliation.", "normal", "email"),
    Persona("procurement manager", "Cost-conscious and slightly adversarial about "
            "suppliers. Mentions lead times and contract terms unprompted.", "normal", "email"),
    Persona("project manager", "Talks in deliverables, budgets and client expectations "
            "rather than records. Chases things politely but persistently.", "normal", "chat"),
    Persona("shop floor supervisor", "Direct, uses production shorthand (line, batch, "
            "shift, run). Writes between jobs so keeps it very short.", "terse", "chat"),
    Persona("customer success manager", "Frames everything around the customer "
            "relationship and what was promised. Diplomatic.", "rambling", "email"),
    Persona("finance director", "Reads everything as a control question — who approved "
            "it, is it auditable, what is the exposure.", "normal", "email"),
    Persona("operations intern", "New, unsure of the right terminology, describes things "
            "by what they saw on screen. Apologises for asking.", "rambling", "chat"),
)


# ──────────────────────────────────────────────────────────────────────────────
# Situations
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Situation:
    """A concrete business circumstance the user is reacting to.

    ``text`` is *background for the teacher LLM*, never shown to the trained
    model and never copied verbatim into the user turn.  It exists so the
    generated request is about something real rather than about the database.
    """

    domain: str
    text: str
    # Models this situation can plausibly be grounded on.
    models: Tuple[str, ...] = ()


SITUATIONS: Tuple[Situation, ...] = (
    # -- sales -----------------------------------------------------------------
    Situation("sales", "The customer phoned to say the quote is approved and they want it "
              "processed today because their own project starts Monday.", ("sale.order",)),
    Situation("sales", "A quotation has been sitting unconfirmed for three weeks and the rep "
              "wants to know whether to chase it or close it out.", ("sale.order", "crm.lead")),
    Situation("sales", "The customer changed their mind after confirming and now wants the "
              "whole order pulled back.", ("sale.order",)),
    Situation("sales", "Finance flagged that an order was edited after confirmation, so the "
              "sales manager wants it locked down.", ("sale.order",)),
    Situation("sales", "A prospect went quiet for two months and the rep is cleaning up the "
              "pipeline before the quarterly review.", ("crm.lead",)),
    Situation("sales", "The deal finally closed after a long negotiation and the rep wants it "
              "recorded properly before the commission cut-off.", ("crm.lead",)),
    Situation("sales", "A walk-in customer at the shop asked for a VAT receipt after paying "
              "at the till.", ("pos.order",)),
    Situation("sales", "Marketing wants a discontinued line hidden from the webshop without "
              "deleting its history.", ("product.template",)),

    # -- accounting ------------------------------------------------------------
    Situation("accounting", "Month-end is tomorrow and a batch of draft invoices still has to "
              "hit the ledger before the period closes.", ("account.move",)),
    Situation("accounting", "The customer says they never received the invoice and is refusing "
              "to pay until they get a copy.", ("account.move",)),
    Situation("accounting", "A customer was billed twice for the same delivery and is asking "
              "for the duplicate to be cancelled.", ("account.move",)),
    Situation("accounting", "A payment landed in the bank account this morning and needs to be "
              "matched to the right invoice.", ("account.payment", "account.bank.statement.line")),
    Situation("accounting", "The auditor asked why two journal items on the same account were "
              "never matched off against each other.", ("account.move.line",)),
    Situation("accounting", "A supplier is chasing payment on a bill that was approved weeks "
              "ago but apparently never posted.", ("account.move",)),
    Situation("accounting", "The controller noticed an entry was posted into a period that has "
              "since been locked and needs it reversed cleanly.", ("account.move",)),
    Situation("accounting", "A bank line was matched against the wrong customer and the "
              "reconciliation has to be undone.", ("account.bank.statement.line",)),

    # -- purchase --------------------------------------------------------------
    Situation("purchase", "The vendor came back with a revised price and the buyer wants the "
              "RFQ turned into a firm order before the quote expires.", ("purchase.order",)),
    Situation("purchase", "A purchase order is stuck waiting for manager approval and the "
              "delivery date is slipping.", ("purchase.order",)),
    Situation("purchase", "The goods arrived and the buyer needs the vendor bill raised so "
              "accounts payable can schedule payment.", ("purchase.order",)),
    Situation("purchase", "A supplier went into administration and every open order with them "
              "has to be stopped.", ("purchase.order",)),
    Situation("purchase", "Stock of a fast-moving part dropped below the safety level over the "
              "weekend and nobody reordered.", ("stock.warehouse.orderpoint",)),

    # -- inventory -------------------------------------------------------------
    Situation("inventory", "The truck is at the gate and the delivery paperwork still has not "
              "been validated in the system.", ("stock.picking",)),
    Situation("inventory", "A pick could not be completed because the shelf was empty even "
              "though the system showed stock.", ("stock.picking", "stock.quant")),
    Situation("inventory", "The customer refused half the shipment at the door and it is coming "
              "back on the same truck.", ("stock.picking",)),
    Situation("inventory", "A cycle count found fewer units on the shelf than the system says "
              "and the difference has to be booked.", ("stock.quant",)),
    Situation("inventory", "Quality flagged a batch and the traceability team needs to know "
              "everywhere that lot has been.", ("stock.lot",)),
    Situation("inventory", "Stock was reserved for an order that has since been cancelled and "
              "is now blocking other picks.", ("stock.picking",)),

    # -- manufacturing ---------------------------------------------------------
    Situation("manufacturing", "The line finished the batch early and the supervisor wants the "
              "order closed off before the shift changes.", ("mrp.production",)),
    Situation("manufacturing", "Components for tomorrow's build are not reserved and the "
              "planner is trying to work out whether the run can go ahead.", ("mrp.production",)),
    Situation("manufacturing", "A machine broke down mid-run and the planned order has to be "
              "pulled from the schedule.", ("mrp.production",)),
    Situation("manufacturing", "A finished unit failed final inspection and has to be taken "
              "apart to recover the good components.", ("mrp.production",)),

    # -- hr --------------------------------------------------------------------
    Situation("hr", "An employee's holiday starts next week and their request is still sitting "
              "unapproved in the manager's queue.", ("hr.leave",)),
    Situation("hr", "Someone booked time off over a shutdown period that was already covered "
              "and the request needs turning down.", ("hr.leave",)),
    Situation("hr", "An expense claim from a client trip has been waiting since last month and "
              "the employee is asking when they will be paid.", ("hr.expense",)),
    Situation("hr", "A claim was submitted with the wrong receipt attached and needs sending "
              "back to the employee.", ("hr.expense",)),

    # -- services / project ----------------------------------------------------
    Situation("services", "A consultant finished on-site work and the client should be billed "
              "for the hours logged.", ("project.task",)),
    Situation("services", "A ticket has turned into a proper piece of work and belongs in the "
              "project plan rather than the support queue.", ("helpdesk.ticket",)),
    Situation("services", "A customer's equipment failed under warranty and a replacement has "
              "to go out today.", ("helpdesk.ticket", "repair.order")),
    Situation("services", "A repair has been on the bench for two days and the customer is "
              "asking for an update before they will pay.", ("repair.order",)),
    Situation("services", "The technician finished the repair and it needs invoicing along with "
              "the parts used.", ("repair.order",)),

    # -- master data -----------------------------------------------------------
    Situation("crm", "A new customer signed today and their details need to exist in the system "
              "before the first order can be raised.", ("res.partner",)),
    Situation("crm", "A long-standing customer moved premises and their invoices are going to "
              "the old address.", ("res.partner",)),
    Situation("crm", "Credit control wants to see everything a customer currently owes before "
              "releasing the next order.", ("res.partner", "account.move")),

    # ── Second wave ───────────────────────────────────────────────────────────
    # The situation catalogue is the main multiplier on dataset size: distinct
    # samples are bounded by (model x method x persona x situation), so at 1,500
    # per family the first wave alone gave error_recovery only ~370 distinct
    # groundings — roughly four samples per scenario, which the near-duplicate
    # check (it normalises digits away) then starts rejecting. These roughly
    # double the space.

    # -- sales -----------------------------------------------------------------
    Situation("sales", "A repeat customer wants the same order as last quarter, but two of "
              "the products have been discontinued since.", ("sale.order", "product.template")),
    Situation("sales", "The customer is disputing the price on a confirmed order because "
              "they were quoted differently over the phone.", ("sale.order",)),
    Situation("sales", "A rush order came in and the rep wants to know whether it can ship "
              "before the end of the week.", ("sale.order", "stock.picking")),
    Situation("sales", "Sales and finance disagree about whether an order was ever invoiced, "
              "and the customer is chasing.", ("sale.order", "account.move")),
    Situation("sales", "The team is preparing for the quarterly review and wants to know "
              "where the pipeline actually stands.", ("crm.lead",)),
    Situation("sales", "A large opportunity has slipped its close date twice and the manager "
              "wants it either progressed or written off.", ("crm.lead",)),
    Situation("sales", "The webshop is showing a product as available when the warehouse says "
              "there is none.", ("product.template", "stock.quant")),
    Situation("sales", "A price list change was applied and the team wants to check what it "
              "did to margins.", ("product.template",)),
    Situation("sales", "The till was short at close of business and the manager is trying to "
              "find the mismatched transaction.", ("pos.order",)),

    # -- accounting ------------------------------------------------------------
    Situation("accounting", "The VAT return is due and there are still unposted entries in "
              "the period.", ("account.move",)),
    Situation("accounting", "A customer paid a round sum covering several invoices and it has "
              "to be allocated correctly.", ("account.payment", "account.move.line")),
    Situation("accounting", "An invoice was raised against the wrong company entity and has to "
              "be undone before month end.", ("account.move",)),
    Situation("accounting", "The bank feed imported the same transaction twice and one has to "
              "be backed out.", ("account.bank.statement.line",)),
    Situation("accounting", "A vendor changed their bank details by email and the finance team "
              "is suspicious about paying it.", ("account.payment", "res.partner")),
    Situation("accounting", "Credit control wants the list of who is more than sixty days "
              "overdue before the weekly call.", ("account.move",)),
    Situation("accounting", "A customer went into liquidation and the outstanding balance has "
              "to be dealt with.", ("account.move", "res.partner")),
    Situation("accounting", "An expense was reimbursed twice and the duplicate has to be "
              "reversed.", ("hr.expense", "account.move")),
    Situation("accounting", "The controller is checking that every payment last month actually "
              "matched an invoice.", ("account.payment", "account.move.line")),

    # -- purchase --------------------------------------------------------------
    Situation("purchase", "The vendor delivered short and the buyer needs to decide whether to "
              "chase the balance or close the order.", ("purchase.order", "stock.picking")),
    Situation("purchase", "Prices from a key supplier went up mid-contract and procurement "
              "wants to see the exposure.", ("purchase.order",)),
    Situation("purchase", "An order was raised against the wrong vendor and nothing has "
              "shipped yet.", ("purchase.order",)),
    Situation("purchase", "Two buyers raised separate orders for the same parts in the same "
              "week.", ("purchase.order",)),
    Situation("purchase", "A supplier is asking why their invoice has not been paid when the "
              "goods went in weeks ago.", ("purchase.order", "account.move")),

    # -- inventory -------------------------------------------------------------
    Situation("inventory", "Goods arrived without paperwork and the receiving clerk needs to "
              "match them to an expected delivery.", ("stock.picking",)),
    Situation("inventory", "A pallet was put away in the wrong location and nobody can find "
              "it.", ("stock.quant",)),
    Situation("inventory", "The annual stock count is next week and the manager wants to know "
              "which lines look wrong beforehand.", ("stock.quant",)),
    Situation("inventory", "A customer is asking which batch their order shipped from because "
              "of a quality complaint.", ("stock.lot", "stock.picking")),
    Situation("inventory", "Stock has been sitting reserved against a cancelled order for "
              "three weeks.", ("stock.picking",)),
    Situation("inventory", "A fast-moving line keeps running out even though there is a "
              "reordering rule on it.", ("stock.warehouse.orderpoint",)),
    Situation("inventory", "The warehouse is full and the manager wants to know what has not "
              "moved in six months.", ("stock.quant", "product.template")),

    # -- manufacturing ---------------------------------------------------------
    Situation("manufacturing", "A rush order needs to jump the production queue and the "
              "planner is working out what it displaces.", ("mrp.production",)),
    Situation("manufacturing", "The bill of materials was updated after an order was already "
              "released to the floor.", ("mrp.production",)),
    Situation("manufacturing", "Yield on the last run was lower than expected and the "
              "supervisor wants the scrap recorded properly.", ("mrp.production",)),
    Situation("manufacturing", "A subcontractor returned finished goods and they need booking "
              "in against the order.", ("mrp.production", "stock.picking")),

    # -- hr --------------------------------------------------------------------
    Situation("hr", "Two people on the same team have booked the same week off and only one "
              "can go.", ("hr.leave",)),
    Situation("hr", "An employee is leaving and their remaining holiday has to be settled.",
              ("hr.leave",)),
    Situation("hr", "Someone submitted expenses from a trip four months ago, past the claim "
              "window.", ("hr.expense",)),
    Situation("hr", "A manager wants to see what their team has claimed this quarter before "
              "signing anything off.", ("hr.expense",)),

    # -- services --------------------------------------------------------------
    Situation("services", "A support ticket has bounced between two teams for a week without "
              "anyone owning it.", ("helpdesk.ticket",)),
    Situation("services", "A customer on a support contract is raising far more tickets than "
              "the contract assumed.", ("helpdesk.ticket", "res.partner")),
    Situation("services", "A project has overrun its budgeted hours and the manager wants to "
              "know what is still unbilled.", ("project.task",)),
    Situation("services", "A repair came back a second time with the same fault.",
              ("repair.order",)),
    Situation("services", "Warranty has expired but the customer insists the fault predates "
              "it.", ("repair.order", "helpdesk.ticket")),

    # -- master data -----------------------------------------------------------
    Situation("crm", "The same customer exists three times in the system with slightly "
              "different names.", ("res.partner",)),
    Situation("crm", "A customer's VAT number was rejected on an invoice submission.",
              ("res.partner", "account.move")),
    Situation("crm", "A company was acquired and their invoices now have to go to the parent "
              "group.", ("res.partner",)),
)

SITUATIONS_BY_MODEL: Dict[str, List[Situation]] = {}
for _s in SITUATIONS:
    for _m in _s.models:
        SITUATIONS_BY_MODEL.setdefault(_m, []).append(_s)


# ──────────────────────────────────────────────────────────────────────────────
# Ambiguity and under-specification, for the clarification family
# ──────────────────────────────────────────────────────────────────────────────

#: Things a real user leaves out, and what the agent must therefore ask for.
UNDER_SPECIFICATIONS: Tuple[Tuple[str, str], ...] = (
    ("no document identified",
     "The user refers to 'the order' / 'that invoice' without giving a reference, "
     "and more than one candidate matches."),
    ("ambiguous customer",
     "The user names a customer whose name matches several partner records "
     "(e.g. a parent company and two of its subsidiaries)."),
    ("missing mandatory field",
     "The requested create call is missing a field Odoo will reject without."),
    ("ambiguous date",
     "The user says 'next Friday' or 'end of the month' without a concrete date, "
     "and the value affects a posting period."),
    ("destructive without confirmation",
     "The request would cancel, reverse, or delete a posted document; the agent "
     "should state the consequence and confirm before acting."),
    ("quantity unstated",
     "The user asks to move, receive, or scrap stock without saying how much."),
    ("wrong lifecycle stage",
     "What the user is asking for is only possible from a different state; the "
     "agent must explain the prerequisite rather than guessing."),
)


# ──────────────────────────────────────────────────────────────────────────────
# Task shapes — what the sample is teaching
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TaskShape:
    """The behavioural skill a family is training."""

    key: str
    description: str


TASK_SHAPES: Dict[str, TaskShape] = {
    "single_call": TaskShape(
        "single_call",
        "One unambiguous request that resolves to exactly one tool call."),
    "lookup_then_act": TaskShape(
        "lookup_then_act",
        "The user names a document in human terms; the agent must search for its "
        "id first, then act on it."),
    "multi_step": TaskShape(
        "multi_step",
        "A business outcome that requires a sequence of tool calls, where each "
        "call's arguments depend on the previous result."),
    "clarify": TaskShape(
        "clarify",
        "The request is under-specified; the correct behaviour is to ask rather "
        "than to guess."),
    "recover": TaskShape(
        "recover",
        "The first attempt fails with a real Odoo exception; the agent must "
        "diagnose it and either fix and retry or explain what the human must do."),
    "verify": TaskShape(
        "verify",
        "After mutating, the agent re-reads the record to confirm the change "
        "actually committed before reporting success."),
    "refuse_or_warn": TaskShape(
        "refuse_or_warn",
        "The request is possible but consequential; the agent explains the "
        "impact and asks for confirmation instead of executing."),
    "coreference": TaskShape(
        "coreference",
        "A follow-up turn refers to an earlier record as 'it' / 'that one' / "
        "'the same customer'; the agent must resolve the reference from history."),
    "analysis": TaskShape(
        "analysis",
        "The user asks a question about the data; the agent queries, aggregates, "
        "and answers in business language rather than dumping rows."),
    "explain": TaskShape(
        "explain",
        "The user asks how something works in Odoo; no tool call, just a correct "
        "and specific explanation grounded in the real schema."),
}


# ──────────────────────────────────────────────────────────────────────────────
# Sampling helpers
# ──────────────────────────────────────────────────────────────────────────────

def pick_persona(rng: random.Random, spec_personas: Sequence[str] = ()) -> Persona:
    """Prefers a persona whose role matches the model's own persona list."""
    if spec_personas:
        matches = [p for p in PERSONAS if p.role in spec_personas]
        if matches and rng.random() < 0.75:
            return rng.choice(matches)
    return rng.choice(PERSONAS)


def pick_situation(rng: random.Random, model: str, domain: str) -> Optional[Situation]:
    """Prefers a situation bound to this exact model, then to its domain."""
    exact = SITUATIONS_BY_MODEL.get(model)
    if exact:
        return rng.choice(exact)
    same_domain = [s for s in SITUATIONS if s.domain == domain]
    if same_domain:
        return rng.choice(same_domain)
    return None
