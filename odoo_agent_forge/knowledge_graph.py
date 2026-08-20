"""
Stage 2: Odoo Knowledge Graph Engine
===================================
Constructs a unified graph representation of Odoo entities, fields, relationships,
state machines, and business workflow chains. Persists state to SQLite and exports
subgraphs for synthetic generation.
"""

import json
import logging
import sqlite3
from typing import Dict, List, Optional, Set, Tuple, Any
from odoo_agent_forge.extractor import ModuleMeta, ModelMeta

logger = logging.getLogger(__name__)


class OdooKnowledgeGraph:
    """In-memory and SQLite-backed Knowledge Graph for Odoo."""

    def __init__(self, db_path: str = "./forge_knowledge.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Table for Modules
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS modules (
                    technical_name TEXT PRIMARY KEY,
                    name TEXT,
                    category TEXT,
                    description TEXT,
                    depends TEXT
                )
            """)
            # Table for Models
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    technical_name TEXT PRIMARY KEY,
                    name TEXT,
                    module_name TEXT,
                    is_transient INTEGER,
                    inherits TEXT,
                    description TEXT
                )
            """)
            # Table for Fields
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fields (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT,
                    field_name TEXT,
                    field_type TEXT,
                    comodel_name TEXT,
                    required INTEGER,
                    selection TEXT,
                    string TEXT
                )
            """)
            # Table for Relationships & Edges
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS relationship_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_model TEXT,
                    target_model TEXT,
                    rel_type TEXT, -- Many2one, One2many, Many2many, INHERITS, TRANSITION
                    field_name TEXT,
                    extra_info TEXT
                )
            """)
            # Table for Methods (public XML-RPC callable business methods per model)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS methods (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT,
                    method_name TEXT,
                    docstring TEXT,
                    decorators TEXT,
                    is_action INTEGER DEFAULT 0,
                    is_public INTEGER DEFAULT 1,
                    signature TEXT
                )
            """)
            conn.commit()

    def build_graph_from_modules(self, modules: Dict[str, ModuleMeta]):
        logger.info("Building Knowledge Graph from extracted modules...")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM modules")
            cursor.execute("DELETE FROM models")
            cursor.execute("DELETE FROM fields")
            cursor.execute("DELETE FROM relationship_edges")
            cursor.execute("DELETE FROM methods")
            
            for tech_name, mod in modules.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO modules (technical_name, name, category, description, depends)
                    VALUES (?, ?, ?, ?, ?)
                """, (mod.technical_name, mod.name, mod.category, mod.description, json.dumps(mod.depends)))

                for model_tech_name, model in mod.models.items():
                    cursor.execute("""
                        INSERT OR REPLACE INTO models (technical_name, name, module_name, is_transient, inherits, description)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        model.technical_name,
                        model.name,
                        mod.technical_name,
                        1 if model.is_transient else 0,
                        json.dumps(model.inherits),
                        model.description
                    ))

                    # Insert fields
                    for fname, fmeta in model.fields.items():
                        cursor.execute("""
                            INSERT INTO fields (model_name, field_name, field_type, comodel_name, required, selection, string)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            model.technical_name,
                            fname,
                            fmeta.field_type,
                            fmeta.comodel_name,
                            1 if fmeta.required else 0,
                            json.dumps(fmeta.selection) if fmeta.selection else None,
                            fmeta.string
                        ))

                        if fmeta.comodel_name:
                            cursor.execute("""
                                INSERT INTO relationship_edges (source_model, target_model, rel_type, field_name)
                                VALUES (?, ?, ?, ?)
                            """, (model.technical_name, fmeta.comodel_name, fmeta.field_type, fname))

                    # Insert inheritance edges
                    for parent in model.inherits:
                        cursor.execute("""
                            INSERT INTO relationship_edges (source_model, target_model, rel_type, field_name)
                            VALUES (?, ?, ?, ?)
                        """, (model.technical_name, parent, "INHERITS", "_inherit"))

                    # Insert methods (public, XML-RPC callable)
                    for mname, mmeta in model.methods.items():
                        is_public = int(not mname.startswith("_"))
                        is_action = int(mname.startswith(("action_", "button_", "do_", "write_", "post_", "validate_", "confirm_", "cancel_", "reset_")))
                        cursor.execute("""
                            INSERT INTO methods (model_name, method_name, docstring, decorators, is_action, is_public, signature)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            model.technical_name,
                            mname,
                            mmeta.docstring,
                            json.dumps(mmeta.decorators) if mmeta.decorators else None,
                            is_action,
                            is_public,
                            mmeta.signature
                        ))

                    # Insert state transitions
                    for trans in model.state_transitions:
                        cursor.execute("""
                            INSERT INTO relationship_edges (source_model, target_model, rel_type, field_name, extra_info)
                            VALUES (?, ?, ?, ?, ?)
                        """, (model.technical_name, model.technical_name, "STATE_TRANSITION", "state", json.dumps(trans)))

            conn.commit()
        logger.info("Knowledge Graph construction complete.")

    def get_model_fields(self, model_name: str) -> List[Dict[str, Any]]:
        """Retrieves all extracted fields for a model from SQLite."""
        fields = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT field_name, field_type, comodel_name, required, selection, string
                FROM fields WHERE model_name = ?
            """, (model_name,))
            for row in cursor.fetchall():
                fields.append({
                    "name": row[0],
                    "field_type": row[1],
                    "comodel_name": row[2],
                    "required": bool(row[3]),
                    "selection": json.loads(row[4]) if row[4] else None,
                    "string": row[5]
                })
        return fields

    def get_related_cluster(self, model_name: str, depth: int = 2) -> Dict[str, Any]:
        """Retrieves neighboring models and relations up to specified depth."""
        cluster = {"root": model_name, "models": [model_name], "edges": []}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT source_model, target_model, rel_type, field_name FROM relationship_edges
                WHERE source_model = ? OR target_model = ?
            """, (model_name, model_name))
            rows = cursor.fetchall()
            for r in rows:
                cluster["edges"].append({"source": r[0], "target": r[1], "type": r[2], "field": r[3]})
                if r[0] not in cluster["models"]:
                    cluster["models"].append(r[0])
                if r[1] not in cluster["models"]:
                    cluster["models"].append(r[1])
        return cluster

    def get_business_process_chains(self) -> List[List[str]]:
        """Returns standard multi-model business process chains."""
        return [
            ["crm.lead", "sale.order", "stock.picking", "account.move", "account.payment"],
            ["purchase.order", "stock.picking", "account.move", "account.payment"],
            ["mrp.production", "stock.picking", "account.move"],
            ["helpdesk.ticket", "project.task", "account.move"],
            ["hr.employee", "hr.contract", "hr.payslip", "account.move"],
        ]

    def get_model_methods(self, model_name: str, public_only: bool = True, actions_only: bool = False) -> List[Dict[str, Any]]:
        """Retrieves extracted methods for a model. Optionally filters to public or action-prefixed methods only."""
        methods = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                query = "SELECT method_name, docstring, decorators, is_action, is_public, signature FROM methods WHERE model_name = ?"
                params = [model_name]
                if public_only:
                    query += " AND is_public = 1"
                if actions_only:
                    query += " AND is_action = 1"
                cursor.execute(query, params)
                for row in cursor.fetchall():
                    methods.append({
                        "name": row[0],
                        "docstring": row[1] or "",
                        "decorators": json.loads(row[2]) if row[2] else [],
                        "is_action": bool(row[3]),
                        "is_public": bool(row[4]),
                        "signature": row[5] or f"def {row[0]}(self, *args, **kwargs)"
                    })
        except Exception as e:
            logger.warning(f"Could not load methods for {model_name}: {e}")
        return methods

    def get_model_state_transitions(self, model_name: str) -> List[Dict[str, Any]]:
        """Returns all extracted state machine states for a model."""
        transitions = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT extra_info FROM relationship_edges
                    WHERE source_model = ? AND rel_type = 'STATE_TRANSITION'
                """, (model_name,))
                for row in cursor.fetchall():
                    if row[0]:
                        transitions.append(json.loads(row[0]))
        except Exception as e:
            logger.warning(f"Could not load state transitions for {model_name}: {e}")
        return transitions

    def get_all_models_with_methods(self) -> List[Dict[str, Any]]:
        """Returns all models that have at least one extracted public method (XML-RPC candidates)."""
        models = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT m.technical_name, m.name, m.module_name
                    FROM models m
                    INNER JOIN methods mt ON mt.model_name = m.technical_name
                    WHERE mt.is_public = 1
                    ORDER BY m.technical_name
                """)
                for row in cursor.fetchall():
                    models.append({
                        "technical_name": row[0],
                        "name": row[1] or row[0],
                        "module_name": row[2] or ""
                    })
        except Exception as e:
            logger.warning(f"Could not load models with methods: {e}")
        return models

    def get_all_models_prioritized(self, core_weight_ratio: float = 0.8) -> List[Dict[str, Any]]:
        """
        Returns all models sorted and weighted according to the 4-Tier Odoo AI SFT Hierarchy:
          - Tier 1 (Must be Rock-Solid: 1-40 FK Anchors): 50% Weight
          - Tier 2 (High-Frequency Operational: 41-110): 30% Weight
          - Tier 3 (Supporting/Child Tables: 111-200): 15% Weight
          - Tier 4 (Specialty Apps & Remaining Models: 201+): 5% Weight
        """
        TIER_1_MODELS = {
            'res.partner', 'res.users', 'res.company', 'res.currency', 'res.country',
            'res.country.state', 'product.template', 'product.product', 'product.category',
            'uom.uom', 'sale.order', 'sale.order.line', 'purchase.order', 'purchase.order.line',
            'account.move', 'account.move.line', 'account.account', 'account.journal',
            'account.tax', 'account.payment', 'stock.move', 'stock.move.line', 'stock.picking',
            'stock.quant', 'stock.location', 'stock.warehouse', 'hr.employee', 'hr.department',
            'crm.lead', 'mail.message', 'mail.activity', 'calendar.event', 'project.project',
            'project.task', 'res.groups', 'ir.model', 'ir.model.fields', 'res.partner.bank',
            'account.payment.term', 'account.fiscal.position'
        }

        TIER_2_MODELS = {
            'sale.order.template', 'crm.stage', 'crm.team', 'purchase.requisition', 'stock.rule',
            'stock.picking.type', 'stock.lot', 'stock.orderpoint', 'delivery.carrier',
            'account.bank.statement', 'account.bank.statement.line', 'account.analytic.account',
            'account.analytic.line', 'account.reconcile.model', 'account.asset', 'account.fiscal.year',
            'hr.contract', 'hr.leave', 'hr.leave.type', 'hr.attendance', 'hr.applicant',
            'hr.expense', 'hr.expense.sheet', 'hr.payslip', 'mrp.production', 'mrp.bom',
            'mrp.bom.line', 'mrp.workorder', 'mrp.workcenter', 'pos.order', 'pos.order.line',
            'pos.session', 'pos.config', 'pos.payment', 'project.milestone', 'project.task.type',
            'mail.template', 'mail.activity.type', 'discuss.channel', 'resource.calendar',
            'resource.resource', 'product.attribute', 'product.attribute.value', 'product.pricelist',
            'product.pricelist.item', 'product.supplierinfo', 'product.packaging', 'payment.transaction',
            'payment.provider', 'website', 'website.page', 'helpdesk.ticket', 'helpdesk.team',
            'maintenance.request', 'maintenance.equipment', 'fleet.vehicle', 'survey.survey',
            'survey.user_input', 'event.event', 'event.registration', 'mailing.mailing',
            'mailing.list', 'mailing.contact', 'sign.request', 'ir.attachment', 'ir.sequence',
            'ir.cron', 'ir.rule', 'ir.actions.act_window', 'ir.ui.view'
        }

        TIER_3_MODELS = {
            'account.tax.group', 'account.tax.repartition.line', 'account.payment.method',
            'account.fiscal.position.tax', 'account.move.reversal', 'account.full.reconcile',
            'account.partial.reconcile', 'account.analytic.plan', 'account.analytic.distribution.model',
            'account.asset.category', 'account.account.type', 'account.account.tag', 'account.incoterms',
            'stock.quant.package', 'stock.package.type', 'stock.package.level', 'stock.putaway.rule',
            'stock.storage.category', 'stock.location.route', 'stock.replenish.mixin', 'stock.inventory',
            'stock.scrap', 'stock.landed.cost', 'stock.valuation.layer', 'sale.order.option',
            'sale.order.template.line', 'sale.commitment', 'purchase.requisition.line', 'purchase.requisition.type',
            'crm.team.member', 'crm.tag', 'crm.lost.reason', 'crm.recurring.plan', 'utm.source',
            'utm.medium', 'utm.campaign', 'hr.contract.type', 'hr.job', 'hr.work.location',
            'hr.leave.allocation', 'hr.attendance.overtime', 'hr.recruitment.stage', 'hr.recruitment.source',
            'hr.candidate', 'hr.expense.split', 'hr.salary.rule', 'hr.salary.rule.category',
            'hr.payslip.run', 'hr.payslip.line', 'hr.appraisal', 'hr.appraisal.goal', 'hr.plan',
            'hr.plan.activity.type', 'mrp.bom.byproduct', 'mrp.routing.workcenter', 'mrp.workcenter.productivity',
            'mrp.unbuild', 'mrp.document', 'mrp.consumption.warning', 'quality.check', 'quality.point',
            'quality.alert', 'pos.category', 'pos.payment.method', 'pos.combo', 'pos.combo.line',
            'pos.bill', 'pos.note', 'project.tags', 'project.collaborator', 'project.update',
            'mail.alias', 'mail.thread', 'mail.mail', 'mail.notification', 'mail.followers',
            'discuss.channel.member', 'calendar.alarm', 'calendar.attendee', 'delivery.price.rule',
            'delivery.zip.prefix', 'res.partner.title', 'res.partner.industry', 'res.partner.category',
            'res.bank', 'res.currency.rate', 'res.lang', 'res.country.group', 'decimal.precision'
        }

        all_models = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT technical_name, name, module_name FROM models ORDER BY technical_name")
                for row in cursor.fetchall():
                    all_models.append({
                        "technical_name": row[0],
                        "name": row[1] or row[0],
                        "module_name": row[2] or ""
                    })
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
            return all_models

        t1_list = [m for m in all_models if m["technical_name"] in TIER_1_MODELS]
        t2_list = [m for m in all_models if m["technical_name"] in TIER_2_MODELS]
        t3_list = [m for m in all_models if m["technical_name"] in TIER_3_MODELS]
        t4_list = [m for m in all_models if m["technical_name"] not in TIER_1_MODELS and m["technical_name"] not in TIER_2_MODELS and m["technical_name"] not in TIER_3_MODELS]

        if not t1_list:
            return all_models

        # Construct mathematically weighted 50/30/15/5 distribution
        prioritized = []
        i1, i2, i3, i4 = 0, 0, 0, 0
        total_slots = len(all_models) * 3

        for slot in range(total_slots):
            rem = slot % 100
            if rem < 50: # 50% Tier 1 (Rock-solid Anchors)
                prioritized.append(t1_list[i1 % len(t1_list)])
                i1 += 1
            elif rem < 80: # 30% Tier 2 (High-Frequency Operational)
                if t2_list:
                    prioritized.append(t2_list[i2 % len(t2_list)])
                    i2 += 1
                else:
                    prioritized.append(t1_list[i1 % len(t1_list)])
                    i1 += 1
            elif rem < 95: # 15% Tier 3 (Supporting/Child Tables)
                if t3_list:
                    prioritized.append(t3_list[i3 % len(t3_list)])
                    i3 += 1
                else:
                    prioritized.append(t1_list[i1 % len(t1_list)])
                    i1 += 1
            else: # 5% Tier 4 & Specialty Apps
                if t4_list:
                    prioritized.append(t4_list[i4 % len(t4_list)])
                    i4 += 1
                else:
                    prioritized.append(t1_list[i1 % len(t1_list)])
                    i1 += 1

        return prioritized
