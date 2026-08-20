"""
Stage 1: Odoo Codebase Knowledge Extractor
==========================================
Scans Odoo 19 Python files, XML views, security CSVs, and manifests to extract
structured entity definitions, field maps, method signatures, state machines, and security rules.
"""

import ast
import csv
import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

logger = logging.getLogger(__name__)


@dataclass
class FieldMeta:
    name: str
    field_type: str
    comodel_name: Optional[str] = None
    string: Optional[str] = None
    required: bool = False
    readonly: bool = False
    compute: Optional[str] = None
    related: Optional[str] = None
    domain: Optional[str] = None
    selection: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class MethodMeta:
    name: str
    docstring: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    signature: str = ""
    is_action: bool = False


@dataclass
class ModelMeta:
    technical_name: str
    name: str
    inherits: List[str] = field(default_factory=list)
    description: Optional[str] = None
    is_transient: bool = False  # True for Wizards
    fields: Dict[str, FieldMeta] = field(default_factory=dict)
    methods: Dict[str, MethodMeta] = field(default_factory=dict)
    state_transitions: List[Dict[str, Any]] = field(default_factory=list)
    file_path: Optional[str] = None


@dataclass
class ModuleMeta:
    name: str
    technical_name: str
    category: str = "Uncategorized"
    description: str = ""
    depends: List[str] = field(default_factory=list)
    path: Path = field(default_factory=Path)
    models: Dict[str, ModelMeta] = field(default_factory=dict)
    security_rules: List[Dict[str, Any]] = field(default_factory=list)
    views: List[Dict[str, Any]] = field(default_factory=list)


class OdooASTVisitor(ast.NodeVisitor):
    """AST Visitor to extract Odoo models, fields, decorators, and business methods."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.extracted_models: Dict[str, ModelMeta] = {}

    def visit_ClassDef(self, node: ast.ClassDef):
        model_name = None
        inherits = []
        description = None
        is_transient = False
        
        # Check base classes for models.Model, models.TransientModel
        for base in node.bases:
            if isinstance(base, ast.Attribute) and base.attr in ("TransientModel", "AbstractModel"):
                if base.attr == "TransientModel":
                    is_transient = True
        
        # Inspect class assignments (_name, _inherit, _description)
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        if target.id == "_name" and isinstance(stmt.value, ast.Constant):
                            model_name = str(stmt.value.value)
                        elif target.id == "_inherit":
                            if isinstance(stmt.value, ast.Constant):
                                inherits.append(str(stmt.value.value))
                            elif isinstance(stmt.value, (ast.List, ast.Tuple)):
                                for elt in stmt.value.elts:
                                    if isinstance(elt, ast.Constant):
                                        inherits.append(str(elt.value))
                        elif target.id == "_description" and isinstance(stmt.value, ast.Constant):
                            description = str(stmt.value.value)

        if not model_name and inherits:
            model_name = inherits[0]

        if model_name:
            model_meta = ModelMeta(
                technical_name=model_name,
                name=description or model_name,
                inherits=inherits,
                description=description,
                is_transient=is_transient,
                file_path=self.file_path
            )

            # Process fields and methods inside class body
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    field_meta = self._parse_field_assign(stmt)
                    if field_meta:
                        model_meta.fields[field_meta.name] = field_meta
                elif isinstance(stmt, ast.FunctionDef):
                    method_meta = self._parse_method_def(stmt)
                    if method_meta:
                        model_meta.methods[method_meta.name] = method_meta
                        if method_meta.name.startswith(("action_", "button_")):
                            method_meta.is_action = True

            # Extract state machine transitions if 'state' field exists
            if "state" in model_meta.fields:
                state_field = model_meta.fields["state"]
                if state_field.selection:
                    for val, label in state_field.selection:
                        model_meta.state_transitions.append({"state": val, "label": label})

            self.extracted_models[model_name] = model_meta
            
        self.generic_visit(node)

    def _parse_field_assign(self, stmt: Any) -> Optional[FieldMeta]:
        field_name = None
        if isinstance(stmt, ast.Assign) and stmt.targets and isinstance(stmt.targets[0], ast.Name):
            field_name = stmt.targets[0].id
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            field_name = stmt.target.id
        
        if not field_name or field_name.startswith("_"):
            return None

        val = stmt.value
        if not val:
            return None
        
        # Check for fields.Type(...)
        field_type = None
        if isinstance(val, ast.Call):
            if isinstance(val.func, ast.Attribute) and isinstance(val.func.value, ast.Name) and val.func.value.id == "fields":
                field_type = val.func.attr
            elif isinstance(val.func, ast.Name):
                field_type = val.func.id

        if not field_type:
            return None

        comodel = None
        string_val = None
        required = False
        readonly = False
        compute = None
        related = None
        selection_vals = []

        # Parse positional and keyword arguments
        if val.args:
            if isinstance(val.args[0], ast.Constant):
                if field_type in ("Many2one", "One2many", "Many2many"):
                    comodel = str(val.args[0].value)
                else:
                    string_val = str(val.args[0].value)

        for kw in val.keywords:
            if kw.arg == "string" and isinstance(kw.value, ast.Constant):
                string_val = str(kw.value.value)
            elif kw.arg == "required" and isinstance(kw.value, ast.Constant):
                required = bool(kw.value.value)
            elif kw.arg == "readonly" and isinstance(kw.value, ast.Constant):
                readonly = bool(kw.value.value)
            elif kw.arg == "compute" and isinstance(kw.value, ast.Constant):
                compute = str(kw.value.value)
            elif kw.arg == "related" and isinstance(kw.value, ast.Constant):
                related = str(kw.value.value)
            elif kw.arg == "comodel_name" and isinstance(kw.value, ast.Constant):
                comodel = str(kw.value.value)
            elif kw.arg == "selection" and isinstance(kw.value, (ast.List, ast.Tuple)):
                for elt in kw.value.elts:
                    if isinstance(elt, (ast.List, ast.Tuple)) and len(elt.elts) >= 2:
                        k = elt.elts[0].value if isinstance(elt.elts[0], ast.Constant) else ""
                        v = elt.elts[1].value if isinstance(elt.elts[1], ast.Constant) else ""
                        selection_vals.append((str(k), str(v)))

        return FieldMeta(
            name=field_name,
            field_type=field_type,
            comodel_name=comodel,
            string=string_val,
            required=required,
            readonly=readonly,
            compute=compute,
            related=related,
            selection=selection_vals
        )

    def _parse_method_def(self, stmt: ast.FunctionDef) -> MethodMeta:
        docstring = ast.get_docstring(stmt)
        decorators = []
        for dec in stmt.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(f"{dec.value.id if isinstance(dec.value, ast.Name) else ''}.{dec.attr}")
            elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                decorators.append(f"{dec.func.value.id if isinstance(dec.func.value, ast.Name) else ''}.{dec.func.attr}")

        return MethodMeta(
            name=stmt.name,
            docstring=docstring,
            decorators=decorators,
            signature=f"def {stmt.name}(...)"
        )


class OdooCodebaseExtractor:
    """Main Knowledge Extractor for the full Odoo codebase."""

    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.modules: Dict[str, ModuleMeta] = {}

    def discover_and_extract_all(self) -> Dict[str, ModuleMeta]:
        logger.info(f"Scanning Odoo codebase at {self.root_path}...")
        for root, dirs, files in os.walk(self.root_path):
            if "__manifest__.py" in files:
                module_path = Path(root)
                tech_name = module_path.name
                module_meta = self.extract_module(module_path)
                if module_meta:
                    self.modules[tech_name] = module_meta
        logger.info(f"Successfully extracted {len(self.modules)} modules.")
        return self.modules

    def extract_module(self, module_path: Path) -> Optional[ModuleMeta]:
        mf_path = module_path / "__manifest__.py"
        if not mf_path.exists():
            return None

        try:
            with open(mf_path, "r", encoding="utf-8", errors="ignore") as fh:
                manifest_data = eval(fh.read(), {"__builtins__": {}})
        except Exception:
            manifest_data = {}

        module_meta = ModuleMeta(
            name=manifest_data.get("name", module_path.name),
            technical_name=module_path.name,
            category=manifest_data.get("category", "Uncategorized"),
            description=manifest_data.get("description", manifest_data.get("summary", "")),
            depends=manifest_data.get("depends", []),
            path=module_path
        )

        # 1. Parse all Python files
        for root, _, files in os.walk(module_path):
            for file in files:
                if file.endswith(".py") and not file.startswith("__manifest__"):
                    py_path = Path(root) / file
                    try:
                        with open(py_path, "r", encoding="utf-8", errors="ignore") as fh:
                            tree = ast.parse(fh.read(), filename=str(py_path))
                        visitor = OdooASTVisitor(str(py_path))
                        visitor.visit(tree)
                        for mname, mmeta in visitor.extracted_models.items():
                            if mname in module_meta.models:
                                module_meta.models[mname].fields.update(mmeta.fields)
                                module_meta.models[mname].methods.update(mmeta.methods)
                            else:
                                module_meta.models[mname] = mmeta
                    except Exception as e:
                        logger.debug(f"Failed to parse AST for {py_path}: {e}")

        # 2. Parse Security CSVs (ir.model.access.csv)
        security_file = module_path / "security" / "ir.model.access.csv"
        if security_file.exists():
            try:
                with open(security_file, "r", encoding="utf-8", errors="ignore") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        module_meta.security_rules.append(dict(row))
            except Exception as e:
                logger.debug(f"Failed to parse security file {security_file}: {e}")

        return module_meta
