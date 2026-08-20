"""
Odoo Call Simulator
===================

Builds tool calls and the results they would actually return.

The previous pipeline returned ``{"status": "success", "step": "..."}`` for every
call in every trajectory — 19,168 tool calls, none of which ever failed and none
of which carried data the next step could depend on.  A model trained on that
learns two false things: that ERP operations always succeed, and that a tool
result is not worth reading.

Here each result is shaped like the real RPC response for that primitive, is
populated from the model's own verified field list, and fails at a configurable
rate with an exception the method genuinely raises (taken from
``MethodSpec.failures``).  Crucially the results are generated *before* the
teacher writes the agent's turns, so the reasoning is conditioned on what the
calls actually returned rather than on a plan nobody executed.
"""

from __future__ import annotations

import random
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

from odoo_agent_forge.agent_surface import TODAY, MethodSpec, ModelSpec, ValueFactory

#: The MCP primitives the target agent is trained to emit.
TOOL_NAMES = (
    "odoo_search_read",
    "odoo_read_group",
    "odoo_create",
    "odoo_write",
    "odoo_execute_method",
    "odoo_unlink",
)


#: Failures that are true of any Odoo model, whatever its business logic —
#: access control, record lifetime, and concurrency are enforced by the ORM
#: itself. Used only when a method has no hand-written failure of its own, so a
#: model-specific exception is never fabricated.
_GENERIC_FAILURES: List[Tuple[str, str]] = [
    ("AccessError",
     "You are not allowed to modify '{label}' ({model}) records. "
     "This operation is allowed for the following groups: Administration/Settings."),
    ("AccessError",
     "Sorry, you are not allowed to access this document. "
     "Records: {ref} ({model}). Contact your administrator to request access."),
    ("MissingError",
     "Record does not exist or has been deleted. ({model}: {ref})"),
    ("UserError",
     "The record {ref} was modified by another user while you were working on it. "
     "Reload it and try again."),
    ("ValidationError",
     "A mandatory field is not set on {ref}. Fill it in before continuing."),
]

#: Corrections the *tool layer* returns, as opposed to exceptions the ORM raises.
#:
#: These exist because the serving side was built after the first dataset, and the
#: model had never seen them. The Odoo module now answers a wrong model or field
#: name with the right one instead of a bare error — but a model trained without
#: these in the data does not use the correction. Asked to list categories it sent
#: a field that does not exist, was told the real field names, and replied by
#: advising the user to add a field to the model.
#:
#: A correction is only worth returning if the model knows to read it and call
#: again, so the behaviour has to be trained, not just implemented. Keeping this
#: list in step with the error messages your MCP server or Odoo addon returns
#: is what makes the two halves fit.
_TOOL_CORRECTIONS: List[Tuple[str, str]] = [
    ("ToolError",
     "Unknown model '{model}s'. Did you mean: {model}?"),
    ("ToolError",
     "{model} has no field 'code'. Available fields include: name, display_name, "
     "state, active, create_date."),
    ("ToolError",
     "Refusing to write values that reference records which do not exist:\n"
     "  - 'categ_id' = 2 does not exist in product.category\n\n"
     "Do not guess ids. Use odoo_search_read on the related model to find the "
     "right record, or odoo_fields_get to see what a field expects. If the user "
     "has not told you which one they want, ask."),
    ("ToolError",
     "{model} has no method '{bad_method}'. "
     "Did you mean: action_confirm, action_cancel, action_draft?"),
    # The two-phase create. Not an error at all — the tool did exactly what it
    # should — but it arrives on the same channel and demands the same thing: read
    # the response, do not treat it as done, and take the next step. Here the next
    # step is to show the user what would be created and wait, which is the one
    # behaviour that stops an invented product reaching the database.
    ("ToolResult",
     "NOTHING WAS CREATED. This is a preview of the 1 {model} record(s) you are "
     "proposing. Show these exact values to the user and ask them to confirm or "
     "correct them. Any value the user did not give you is a guess - say so. When "
     "they approve, call odoo_create again with the same values plus "
     "\"confirm\": true."),
]


class OdooSimulator:
    """Constructs grounded tool calls and their plausible responses."""

    def __init__(self, verified: Dict[str, ModelSpec]) -> None:
        self.surface = verified

    # ──────────────────────────────────────────────────────────────────────
    # Call construction
    # ──────────────────────────────────────────────────────────────────────

    def build_method_call(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        meth: MethodSpec = ctx["method"]
        return {
            "name": "odoo_execute_method",
            "arguments": {
                "model": ctx["spec"].model,
                "method": meth.name,
                "res_ids": [ctx["res_id"]],
                "kwargs": dict(meth.kwargs),
            },
        }

    def build_methods_get_call(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """The introspection call an agent makes when it does not know the method."""
        return {
            "name": "odoo_methods_get",
            "arguments": {"model": ctx["spec"].model},
        }

    def methods_get_result(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """The MCP server's reply to odoo_methods_get: the model's whole surface.

        Deliberately not just the drawn method. A listing containing only the
        right answer would teach "call methods_get, then echo the one entry"
        instead of "choose the entry that matches what was asked", and the second
        is the behaviour that generalises to methods absent from training.

        Built here rather than in execute() because execute() models a *mutation*
        - it injects failures, advances state and consumes ctx["method"]. An
        introspection read has none of that; routing it through the same path
        would let a simulated OperationalError land on a call that cannot fail.
        """
        spec: ModelSpec = ctx["spec"]
        methods = []
        for m in spec.methods:
            entry: Dict[str, Any] = {"name": m.name, "does": m.intent}
            if m.from_state and m.to_state:
                entry["state"] = f"{m.from_state} -> {m.to_state}"
            if m.returns_action:
                entry["returns"] = "action"
            methods.append(entry)
        return {
            "ok": True,
            "payload": {"model": spec.model, "methods": methods},
            "narration": f"listed {len(methods)} callable methods on {spec.model}",
        }

    def build_create_call(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        spec: ModelSpec = ctx["spec"]
        vf: ValueFactory = ctx["vf"]
        return {
            "name": "odoo_create",
            "arguments": {"model": spec.model, "values": vf.create_values(spec)},
        }

    def build_write_call(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        spec: ModelSpec = ctx["spec"]
        vf: ValueFactory = ctx["vf"]
        values = vf.create_values(spec)
        # A write touches a couple of fields, not the whole record.
        keys = list(values)[:2] or ["name"]
        return {
            "name": "odoo_write",
            "arguments": {
                "model": spec.model,
                "res_ids": [ctx["res_id"]],
                "values": {k: values[k] for k in keys if k in values},
            },
        }

    def build_search_call(self, ctx: Dict[str, Any],
                          by_reference: bool = False) -> Dict[str, Any]:
        """A search either pinpointing one document or filtering a working set."""
        spec: ModelSpec = ctx["spec"]
        rng: random.Random = ctx["rng"]

        if by_reference:
            name_field = "name" if "name" in spec.search_fields else spec.search_fields[0]
            domain = [[name_field, "=", ctx["doc_ref"]]]
            limit = 1
        else:
            domain = self._business_domain(ctx)
            limit = rng.choice((10, 20, 25, 50))

        return {
            "name": "odoo_search_read",
            "arguments": {
                "model": spec.model,
                "domain": domain,
                "fields": list(spec.search_fields[:6]),
                "limit": limit,
            },
        }

    def build_read_call(self, ctx: Dict[str, Any],
                        state: Optional[str] = None) -> Dict[str, Any]:
        spec: ModelSpec = ctx["spec"]
        fields = list(spec.search_fields[:5])
        if spec.state_field and spec.state_field not in fields:
            fields.append(spec.state_field)
        return {
            "name": "odoo_search_read",
            "arguments": {
                "model": spec.model,
                "domain": [["id", "=", ctx["res_id"]]],
                "fields": fields,
                "limit": 1,
            },
            "_expect_state": state,
        }

    def build_read_group_call(self, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """An aggregation over a field that can meaningfully be grouped."""
        spec: ModelSpec = ctx["spec"]
        rng: random.Random = ctx["rng"]

        groupable = [f for f in spec.search_fields
                     if f.endswith("_id") or f in ("state", "payment_state", "move_type",
                                                   "priority", "picking_type_id")]
        measurable = [f for f in spec.search_fields
                      if f in ("amount_total", "amount_residual", "total_amount",
                               "expected_revenue", "product_qty", "quantity",
                               "list_price", "standard_price", "balance", "debit", "credit",
                               "number_of_days", "allocated_hours", "effective_hours")]
        if not groupable:
            return None

        group_by = rng.choice(groupable)
        aggregates = [f"{m}:sum" for m in measurable[:2]] or ["__count"]

        # Grouping by a field the domain already pins to a single value would
        # return exactly one bucket, which is not an aggregation anyone asks for.
        domain = [c for c in self._business_domain(ctx)
                  if not (isinstance(c, list) and c and c[0] == group_by)]

        return {
            "name": "odoo_read_group",
            "arguments": {
                "model": spec.model,
                "domain": domain or [["id", ">", 0]],
                "groupby": [group_by],
                "aggregates": aggregates,
            },
            "_group_by": group_by,
            "_aggregates": aggregates,
        }

    def _business_domain(self, ctx: Dict[str, Any]) -> List[Any]:
        """A filter a human would actually apply, using only verified fields."""
        spec: ModelSpec = ctx["spec"]
        rng: random.Random = ctx["rng"]
        fields = set(spec.search_fields)
        domain: List[Any] = []

        if spec.state_field and spec.state_field in fields:
            state = ctx["method"].from_state or ctx["method"].to_state
            if state:
                domain.append([spec.state_field, "=", state])

        date_field = next((f for f in ("invoice_date", "date_order", "scheduled_date",
                                       "date", "date_start", "request_date_from",
                                       "date_deadline", "create_date")
                           if f in fields), None)
        if date_field:
            since = (TODAY - timedelta(days=rng.choice((30, 60, 90)))).isoformat()
            domain.append([date_field, ">=", since])

        if "amount_residual" in fields and rng.random() < 0.5:
            domain.append(["amount_residual", ">", 0])
        elif "partner_id" in fields and rng.random() < 0.35:
            domain.append(["partner_id", "ilike", ctx["partner_name"].split()[0]])

        return domain or [["id", ">", 0]]

    def describe_query(self, call: Dict[str, Any], ctx: Dict[str, Any]) -> str:
        """Plain-English summary of what a query returns.

        Fed to phase 1 so the persona asks a question this data can answer.
        Without it the question and the query were drawn independently and
        disagreed most of the time.
        """
        spec: ModelSpec = ctx["spec"]
        args = call["arguments"]
        parts: List[str] = []

        # partner_id is the vendor on buy-side documents and the customer on
        # sell-side ones. Calling a supplier "the customer" in every purchasing
        # sample is the kind of small wrongness a domain expert notices first.
        readable = dict(_READABLE_FIELD)
        if spec.domain == "purchase" or spec.model.startswith("purchase."):
            readable["partner_id"] = "vendor"
        elif spec.domain == "hr":
            readable["partner_id"] = "contact"

        if call["name"] == "odoo_read_group":
            group_by = call.get("_group_by", "")
            measures = [a.split(":")[0] for a in call.get("_aggregates", [])
                        if a != "__count"]
            grouped = readable.get(group_by,
                                   group_by.replace("_id", "").replace("_", " "))
            if measures:
                measure_text = " and ".join(readable.get(m, m.replace("_", " "))
                                            for m in measures)
                parts.append(f"your {spec.label} records totalled up by {grouped}, "
                             f"showing {measure_text} and how many there are in each")
            else:
                parts.append(f"a count of your {spec.label} records by {grouped}")
        else:
            fields = [f for f in (args.get("fields") or []) if f != "id"][:5]
            field_text = ", ".join(readable.get(f, f.replace("_", " "))
                                   for f in fields)
            parts.append(f"a list of {spec.label} records showing {field_text}")

        for cond in (args.get("domain") or []):
            if not isinstance(cond, list) or len(cond) != 3:
                continue
            field, op, val = cond
            if field == "id":
                continue
            name = readable.get(field, field.replace("_id", "").replace("_", " "))
            if op == "=":
                parts.append(f"only those where {name} is {val}")
            elif op == ">=" and isinstance(val, str):
                parts.append(f"only those dated {val} or later")
            elif op == ">":
                parts.append(f"only those with {name} above {val}")
            elif op == "ilike":
                parts.append(f"only those matching '{val}'")

        return "; ".join(parts) + "."

    def plausible_wrong_tools(self, ctx: Dict[str, Any]) -> str:
        """Primitives that would superficially fit but are wrong here."""
        wrong = []
        if ctx["method"].to_state:
            wrong.append(
                f"`odoo_write` setting state to '{ctx['method'].to_state}' directly "
                f"(bypasses the business logic the method runs)")
        wrong.append("`odoo_unlink` (deletes the record rather than transitioning it)")
        return "; ".join(wrong)

    # ──────────────────────────────────────────────────────────────────────
    # Execution
    # ──────────────────────────────────────────────────────────────────────

    def execute(
        self,
        call: Dict[str, Any],
        ctx: Dict[str, Any],
        force_success: bool = False,
        force_failure: bool = False,
        failure_rate: float = 0.0,
    ) -> Dict[str, Any]:
        """Returns ``{"ok", "payload", "narration"}`` for one call."""
        rng: random.Random = ctx["rng"]
        meth: MethodSpec = ctx["method"]

        # Only 29 of 380 methods carry a hand-written failure mode, and those all
        # live in Tier A. Trajectories draw from the whole surface, so requiring a
        # curated failure meant the configured 22% failure rate delivered 1.7% in
        # practice — a dataset that still taught "nothing ever goes wrong".
        #
        # The fallbacks below are ORM-level errors that are true of *every* Odoo
        # model regardless of its business logic, so using them is not the same as
        # inventing a model-specific exception. A method's own curated failures are
        # always preferred; these only fill the gap.
        available = list(meth.failures) or _GENERIC_FAILURES
        # One failure in four is a tool-layer correction rather than an ORM
        # exception. The two teach different recoveries and both are needed: an
        # AccessError means stop and tell the user, a correction means fix the
        # argument and call again. A dataset with only the first teaches the model
        # to give up when it should retry.
        if rng.random() < 0.25:
            available = available + _TOOL_CORRECTIONS
        should_fail = force_failure or (
            not force_success and available and rng.random() < failure_rate
        )
        if should_fail and available:
            exc, message = rng.choice(available)
            message = message.format(model=ctx["spec"].model,
                                     label=ctx["spec"].label,
                                     ref=ctx["doc_ref"],
                                     bad_method=f"do_{meth.name}")
            return {
                "ok": False,
                "error": f"{exc}: {message}",
                "payload": {"error": {"name": f"odoo.exceptions.{exc}", "message": message}},
                "narration": f"Calling `{meth.name}`.",
            }

        name = call["name"]
        if name == "odoo_execute_method":
            return self._exec_method_result(call, ctx)
        if name == "odoo_create":
            return self._create_result(call, ctx)
        if name == "odoo_write":
            return {"ok": True, "payload": True,
                    "narration": f"Updating {ctx['doc_ref']}."}
        if name == "odoo_search_read":
            return self._search_result(call, ctx)
        if name == "odoo_read_group":
            return self._read_group_result(call, ctx)
        if name == "odoo_unlink":
            return {"ok": True, "payload": True, "narration": "Deleting the record."}
        return {"ok": True, "payload": True, "narration": "Running the call."}

    # -- per-primitive results -------------------------------------------------

    def _exec_method_result(self, call: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        meth: MethodSpec = ctx["method"]
        if meth.returns_action:
            payload = {
                "type": "ir.actions.act_window",
                "res_model": self._wizard_model(ctx),
                "view_mode": "form",
                "target": "new",
                "context": {"active_model": ctx["spec"].model, "active_ids": [ctx["res_id"]]},
            }
        elif meth.to_state:
            payload = {"result": True, "id": ctx["res_id"], "state": meth.to_state}
        else:
            payload = {"result": True, "id": ctx["res_id"]}
        return {"ok": True, "payload": payload,
                "narration": f"Calling `{meth.name}` on {ctx['doc_ref']}."}

    @staticmethod
    def _wizard_model(ctx: Dict[str, Any]) -> str:
        name = ctx["method"].name
        table = {
            "action_register_payment": "account.payment.register",
            "action_reverse": "account.move.reversal",
            "action_send_and_print": "account.move.send.wizard",
            "action_quotation_send": "mail.compose.message",
            "action_rfq_send": "mail.compose.message",
            "action_create_invoice": "sale.advance.payment.inv",
            "button_scrap": "stock.scrap",
            "button_unbuild": "mrp.unbuild",
            "action_create_return_picking": "stock.return.picking",
        }
        return table.get(name, f"{ctx['spec'].model}.wizard")

    def _create_result(self, call: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ok": True,
            "payload": {"id": ctx["res_id"], "display_name": ctx["doc_ref"]},
            "narration": f"Creating the {ctx['spec'].label}.",
        }

    def _search_result(self, call: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        args = call["arguments"]
        fields = args.get("fields") or ["id", "display_name"]
        limit = args.get("limit", 10)
        rng: random.Random = ctx["rng"]

        pinpoint = limit == 1
        count = 1 if pinpoint else rng.randint(2, min(limit, 7))
        rows = [self._row(ctx, fields, primary=(i == 0),
                          state=call.get("_expect_state"))
                for i in range(count)]
        return {
            "ok": True,
            "payload": rows,
            "narration": (f"Looking up {ctx['doc_ref']}." if pinpoint
                          else f"Searching {ctx['spec'].label} records."),
        }

    def _row(self, ctx: Dict[str, Any], fields: List[str],
             primary: bool, state: Optional[str] = None) -> Dict[str, Any]:
        """One record shaped by the model's own verified field names and types.

        Values for the record the sample is *about* are memoised per sample, so
        reading it twice returns the same data. Without this, a verify-after-write
        transcript showed quantity 33 on the search and 212 on the re-read of the
        same id — teaching the model that re-reading a record is meaningless.
        Only ``state`` is allowed to differ between reads, because that is exactly
        what the intervening method call changed.
        """
        spec: ModelSpec = ctx["spec"]
        rng: random.Random = ctx["rng"]
        vf: ValueFactory = ctx["vf"]

        if not primary:
            row: Dict[str, Any] = {"id": rng.randint(101, 98999)}
            for f in fields:
                if f != "id":
                    row[f] = self._field_value(f, ctx, spec, rng, vf, False, state)
            return row

        memo: Dict[str, Any] = ctx.setdefault("_record", {"id": ctx["res_id"]})
        for f in fields:
            if f == "id" or f in memo:
                continue
            memo[f] = self._field_value(f, ctx, spec, rng, vf, True, state)

        row = {f: memo[f] for f in (["id"] + [x for x in fields if x != "id"]) if f in memo}
        # The state field is the one value a later read is *supposed* to change.
        if state and spec.state_field and spec.state_field in row:
            row[spec.state_field] = state
            memo[spec.state_field] = state
        return row

    @staticmethod
    def _field_value(f: str, ctx, spec: ModelSpec, rng: random.Random,
                     vf: ValueFactory, primary: bool, state: Optional[str]) -> Any:
        if f in ("name", "display_name", "payment_ref"):
            return ctx["doc_ref"] if primary else vf.doc_ref(spec, rng.randint(1, 9999))
        if f == "state":
            if primary and state:
                return state
            return state or ctx["method"].from_state or "draft"
        if f.endswith("_ids"):
            return [rng.randint(1, 400) for _ in range(rng.randint(1, 3))]
        if f == "partner_id":
            return [rng.randint(120, 9800), ctx["partner_name"] if primary else vf.company_name()]
        if f == "product_id":
            return [rng.randint(50, 4200), ctx["product_name"] if primary else vf.product_name()]
        if f.endswith("_id"):
            return [rng.randint(1, 60), _LABELS.get(f, f.replace("_id", "").replace("_", " ").title())]
        if f in ("amount_total", "total_amount", "expected_revenue", "list_price",
                 "standard_price", "amount", "balance", "debit", "credit"):
            return round(rng.uniform(120, 42000), 2)
        if f == "amount_residual":
            return round(rng.uniform(0, 18000), 2)
        if f in ("quantity", "product_qty", "product_uom_qty", "reserved_quantity",
                 "inventory_quantity", "product_min_qty", "product_max_qty", "qty_to_order"):
            return float(rng.randint(0, 480))
        if f in ("number_of_days", "allocated_hours", "effective_hours"):
            return float(rng.randint(1, 40))
        if f in ("probability",):
            return float(rng.choice((10, 25, 40, 60, 80)))
        if f in _PAST_DATE_FIELDS or f in _FUTURE_DATE_FIELDS:
            # Dates must respect what the field means. A completed POS order
            # dated after today, or an invoice booked next month, reads as
            # fabricated data and is exactly the incoherence that makes a
            # synthetic dataset obvious.
            if f in _PAST_DATE_FIELDS:
                offset = -rng.randint(1, 75)
            else:
                offset = rng.randint(2, 60)
            d = (TODAY + timedelta(days=offset)).isoformat()
            if f in _DATETIME_FIELDS:
                return f"{d} {rng.randint(7, 18):02d}:{rng.choice((0, 15, 30, 45)):02d}:00"
            return d
        if f == "payment_state":
            return rng.choice(("not_paid", "partial", "in_payment", "paid"))
        if f == "invoice_status":
            return rng.choice(("no", "to invoice", "invoiced", "upselling"))
        if f == "move_type":
            return rng.choice(("out_invoice", "in_invoice", "out_refund"))
        if f == "priority":
            return rng.choice(("0", "1", "2"))
        if f in ("is_reconciled", "reconciled", "active"):
            return rng.random() < 0.6
        if f in ("default_code",):
            return f"[{rng.choice('ABCDEFGH')}{rng.randint(1000, 9999)}]"
        if f == "type":
            return rng.choice(("consu", "service"))
        if f == "origin":
            return f"S{rng.randint(1000, 9999):05d}"
        return None

    def _read_group_result(self, call: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        rng: random.Random = ctx["rng"]
        group_by = call.get("_group_by", "state")
        aggregates = call.get("_aggregates", ["__count"])

        buckets = self._buckets(group_by, ctx, rng)
        rows = []
        for label in buckets:
            row: Dict[str, Any] = {group_by: label, "__count": rng.randint(1, 90)}
            for agg in aggregates:
                if agg == "__count":
                    continue
                row[agg] = round(rng.uniform(400, 260000), 2)
            rows.append(row)
        return {"ok": True, "payload": rows,
                "narration": f"Grouping {ctx['spec'].label} records by {group_by}."}

    @staticmethod
    def _buckets(group_by: str, ctx: Dict[str, Any], rng: random.Random) -> List[Any]:
        spec: ModelSpec = ctx["spec"]
        if group_by == "state":
            states = {m.from_state for m in spec.methods if m.from_state}
            states |= {m.to_state for m in spec.methods if m.to_state}
            return sorted(s for s in states if s) or ["draft", "posted"]
        if group_by == "payment_state":
            return ["not_paid", "partial", "paid"]
        if group_by == "move_type":
            return ["out_invoice", "in_invoice", "out_refund"]
        if group_by == "priority":
            return ["0", "1", "2"]
        vf: ValueFactory = ctx["vf"]
        # Aggregation buckets are the most-read part of an analysis sample, so
        # they get real names. "Product A / Product B / Product C" reads as
        # placeholder data and invites the model to answer in placeholders too.
        if group_by == "partner_id":
            return [[rng.randint(120, 9800), vf.company_name()]
                    for _ in range(rng.randint(3, 6))]
        if group_by == "product_id":
            return [[rng.randint(50, 4200), vf.product_name()]
                    for _ in range(rng.randint(3, 6))]
        if group_by in _BUCKET_NAMES:
            pool = list(_BUCKET_NAMES[group_by])
            rng.shuffle(pool)
            chosen = pool[:rng.randint(3, min(5, len(pool)))]
            return [[rng.randint(1, 60), name] for name in chosen]
        label = _LABELS.get(group_by, group_by.replace("_id", "").replace("_", " ").title())
        return [[rng.randint(1, 60), f"{label} {chr(65 + i)}"] for i in range(rng.randint(3, 5))]

    # ──────────────────────────────────────────────────────────────────────
    # Trajectories
    # ──────────────────────────────────────────────────────────────────────

    def build_trajectory(
        self, ctx: Dict[str, Any], failure_rate: float = 0.22,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """A realistic multi-step plan and the results it produced.

        Stops at the first failure, exactly as a real agent loop would: the
        remaining steps never run, so the transcript never shows an agent
        blithely continuing past an exception.
        """
        spec: ModelSpec = ctx["spec"]
        rng: random.Random = ctx["rng"]

        plan: List[Dict[str, Any]] = []
        if spec.model != "res.partner" and rng.random() < 0.7:
            plan.append(self.build_search_call(ctx, by_reference=False))
        if spec.create_fields:
            plan.append(self.build_create_call(ctx))
        plan.append(self.build_method_call(ctx))
        plan.append(self.build_read_call(ctx, state=ctx["method"].to_state))

        results: List[Dict[str, Any]] = []
        executed: List[Dict[str, Any]] = []
        for call in plan:
            is_method = call["name"] == "odoo_execute_method"
            res = self.execute(
                call, ctx,
                force_success=not is_method,
                failure_rate=failure_rate if is_method else 0.0,
            )
            executed.append(call)
            results.append(res)
            if not res["ok"]:
                break

        return executed, results


#: Field names rendered the way a person would say them, for query descriptions.
_READABLE_FIELD: Dict[str, str] = {
    "partner_id": "customer", "product_id": "product", "user_id": "salesperson",
    "employee_id": "employee", "team_id": "team", "journal_id": "journal",
    "picking_type_id": "operation type", "location_id": "location",
    "stage_id": "stage", "categ_id": "product category", "project_id": "project",
    "company_id": "company", "state": "status", "payment_state": "payment status",
    "invoice_status": "invoicing status", "move_type": "document type",
    "priority": "priority", "amount_total": "total amount",
    "amount_residual": "amount still outstanding", "total_amount": "total amount",
    "expected_revenue": "expected revenue", "product_qty": "quantity",
    "quantity": "quantity", "list_price": "sales price",
    "standard_price": "cost price", "number_of_days": "number of days",
    "allocated_hours": "planned hours", "effective_hours": "hours logged",
    "invoice_date": "invoice date", "date_order": "order date",
    "date_deadline": "deadline", "scheduled_date": "scheduled date",
    "balance": "balance", "debit": "debit", "credit": "credit",
}

#: Fields recording something that has already happened. Always in the past.
_PAST_DATE_FIELDS = frozenset({
    "invoice_date", "date_order", "date", "date_done", "create_date",
    "accounting_date", "inventory_date", "write_date",
})

#: Fields recording something planned or owed. Always ahead of today.
_FUTURE_DATE_FIELDS = frozenset({
    "invoice_date_due", "scheduled_date", "date_deadline", "date_start",
    "date_planned", "request_date_from", "request_date_to", "commitment_date",
})

#: Of those, the ones Odoo stores as Datetime rather than Date.
_DATETIME_FIELDS = frozenset({
    "date_order", "scheduled_date", "date_done", "date_start",
    "create_date", "write_date", "date_planned",
})

#: Realistic value pools for grouped aggregations, so a breakdown reads like a
#: real company's data rather than "Category A, Category B, Category C".
_BUCKET_NAMES: Dict[str, Tuple[str, ...]] = {
    "journal_id": ("Customer Invoices", "Vendor Bills", "Bank", "Cash",
                   "Miscellaneous Operations", "Exchange Difference"),
    "team_id": ("Benelux Sales", "DACH Sales", "Key Accounts", "Inside Sales",
                "E-commerce"),
    "picking_type_id": ("Delivery Orders", "Receipts", "Internal Transfers",
                        "Returns", "Manufacturing"),
    "location_id": ("WH/Stock", "WH/Stock/Shelf 1", "WH/Input", "WH/Output",
                    "WH/Quality Control", "Partners/Customers"),
    "stage_id": ("New", "Qualified", "Proposition", "Negotiation", "Won",
                 "In Progress", "Done"),
    "categ_id": ("All / Saleable", "All / Saleable / Components",
                 "All / Saleable / Finished Goods", "All / Consumable",
                 "All / Services"),
    "user_id": ("Marta Rinaldi", "Tomasz Wójcik", "Ines Duarte", "Karel Jansen",
                "Sofia Bergström", "Ahmed Chalabi"),
    "employee_id": ("Tomasz Wójcik", "Ines Duarte", "Karel Jansen",
                    "Sofia Bergström", "Ahmed Chalabi", "Marta Rinaldi"),
    "company_id": ("Northwind Group NV", "Northwind Group — Belgium",
                   "Northwind Group — Germany", "Northwind Retail BV"),
    "project_id": ("Depot Fit-out", "Line 3 Upgrade", "ERP Rollout",
                   "Warehouse Automation", "Annual Maintenance"),
    "account_id": ("400000 Product Sales", "600000 Purchases of Goods",
                   "110000 Trade Receivables", "440000 Trade Payables",
                   "700000 Services"),
    "holiday_status_id": ("Paid Time Off", "Sick Leave", "Unpaid Leave",
                          "Parental Leave", "Compensatory Days"),
    "workcenter_id": ("Assembly Line 1", "CNC Cell", "Paint Booth",
                      "Packing Station", "Quality Bench"),
    "payment_term_id": ("Immediate Payment", "15 Days", "30 Days",
                        "45 Days end of month", "60 Days"),
    "currency_id": ("EUR", "USD", "GBP", "SEK", "PLN"),
    "country_id": ("Netherlands", "Germany", "France", "Italy", "Sweden", "Poland"),
}

#: Human labels for common relational fields, so simulated rows read like data
#: rather than like a field dump.
_LABELS: Dict[str, str] = {
    "journal_id": "Bank",
    "team_id": "Sales Team",
    "picking_type_id": "Delivery Orders",
    "location_id": "WH/Stock",
    "location_dest_id": "Partners/Customers",
    "stage_id": "In Progress",
    "categ_id": "All / Saleable",
    "user_id": "Marta Rinaldi",
    "user_ids": "Marta Rinaldi",
    "employee_id": "Tomasz Wójcik",
    "holiday_status_id": "Paid Time Off",
    "project_id": "Depot Fit-out",
    "bom_id": "BOM / Assembly",
    "account_id": "400000 Product Sales",
    "move_id": "INV/2026/00187",
    "session_id": "POS/00042",
    "uom_id": "Units",
    "payment_term_id": "30 Days",
    "currency_id": "EUR",
    "country_id": "Netherlands",
    "equipment_id": "Compressor A2",
}
