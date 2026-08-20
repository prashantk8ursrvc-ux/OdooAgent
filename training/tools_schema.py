"""
The MCP tool schema, shared by training and inference.

This file is the contract. The tool definitions rendered into the prompt at
training time must match what the MCP server advertises at inference time —
if they drift, the model has been trained against a prompt it will never see
again, and tool calling degrades in ways that look like a quality problem
rather than a formatting one.

Keep this in sync with ``AGENT_SYSTEM_PROMPT`` in
``odoo_agent_forge/prompts.py``; that is the human-readable statement of the
same six primitives.
"""

from __future__ import annotations

from typing import Any, Dict, List

#: OpenAI/JSON-Schema tool definitions for the six Odoo MCP primitives.
#: Passed to ``tokenizer.apply_chat_template(..., tools=ODOO_TOOLS)`` so the
#: model is trained with the tool block in context, exactly as it will be served.
ODOO_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "odoo_search_read",
            "description": (
                "Search records on an Odoo model and read fields from the matches. "
                "Use this to resolve a document a user named in human terms, or to "
                "answer a question about a set of records."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string",
                              "description": "Technical model name, e.g. 'sale.order'."},
                    "domain": {"type": "array",
                               "description": "Odoo domain, e.g. [[\"state\",\"=\",\"draft\"]].",
                               "items": {}},
                    "fields": {"type": "array", "items": {"type": "string"},
                               "description": "Field names to return."},
                    "limit": {"type": "integer", "description": "Maximum rows."},
                },
                "required": ["model", "domain", "fields"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "odoo_read_group",
            "description": (
                "Aggregate records on an Odoo model, grouped by one or more fields. "
                "Use this for totals, counts and breakdowns rather than reading rows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string"},
                    "domain": {"type": "array", "items": {}},
                    "groupby": {"type": "array", "items": {"type": "string"}},
                    "aggregates": {"type": "array", "items": {"type": "string"},
                                   "description": "e.g. [\"amount_total:sum\"]."},
                },
                "required": ["model", "domain", "groupby"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "odoo_create",
            "description": "Create one record on an Odoo model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string"},
                    "values": {"type": "object",
                               "description": "Field values for the new record."},
                    "confirm": {
                        "type": "boolean",
                        "description": "Omit on the first call: the tool returns a preview and creates nothing. Show the preview to the user, then call again with confirm=true once they approve."
                    }
                },
                "required": ["model", "values"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "odoo_write",
            "description": (
                "Update fields on existing records. Prefer a business method when one "
                "exists — writing a state field directly skips the logic Odoo runs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string"},
                    "res_ids": {"type": "array", "items": {"type": "integer"}},
                    "values": {"type": "object"},
                },
                "required": ["model", "res_ids", "values"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "odoo_execute_method",
            "description": (
                "Call a business method on records, e.g. action_confirm on sale.order "
                "or button_validate on stock.picking. This is how a document moves "
                "through its lifecycle."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string"},
                    "method": {"type": "string",
                               "description": "Public method name on that model."},
                    "res_ids": {"type": "array", "items": {"type": "integer"}},
                    "kwargs": {"type": "object", "description": "Keyword arguments."},
                },
                "required": ["model", "method", "res_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "odoo_unlink",
            "description": (
                "Delete records permanently. Rarely correct — most business documents "
                "should be cancelled or archived instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string"},
                    "res_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["model", "res_ids"],
            },
        },
    },
]

#: Optional introspection tools.
#:
#: Strongly recommended. A model of this size cannot reliably recall
#: the long tail of Odoo's method and field names — an audit of the training data
#: found 190 model+method pairs appearing fewer than ten times, which a 4B will
#: not retain. Given these, it can look up instead of guessing; without them, its
#: only options are recall or invention.
#:
#: Only enable them if your MCP server actually implements them. Training the
#: model to call a tool that does not exist is worse than not having it.
INTROSPECTION_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "odoo_fields_get",
            "description": (
                "List the fields of an Odoo model with their types and whether they "
                "are required. Use this before building a domain or a create call on "
                "a model you are not certain about."
            ),
            "parameters": {
                "type": "object",
                "properties": {"model": {"type": "string"}},
                "required": ["model"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "odoo_methods_get",
            "description": (
                "List the callable business methods of an Odoo model. Use this when "
                "you are unsure which method performs an operation, instead of "
                "guessing a method name."
            ),
            "parameters": {
                "type": "object",
                "properties": {"model": {"type": "string"}},
                "required": ["model"],
            },
        },
    },
]


def get_tools(with_introspection: bool = False) -> List[Dict[str, Any]]:
    return ODOO_TOOLS + (INTROSPECTION_TOOLS if with_introspection else [])
