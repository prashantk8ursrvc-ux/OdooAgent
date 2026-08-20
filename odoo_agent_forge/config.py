"""
Configuration Settings for Odoo Agent Forge
============================================
Loads ALL settings from the .env file that lives in the SAME directory as this
file (odoo_agent_forge/.env), so the package is fully self-contained.

No matter where you launch a script from (workspace root, another folder, CI),
pydantic-settings will always find the correct .env.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field as PydanticField, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute path to odoo_agent_forge/.env — always correct regardless of CWD.
_HERE = Path(__file__).resolve().parent
_ENV_FILE = _HERE / ".env"


def export_api_keys(env_file: Path = _ENV_FILE) -> int:
    """Copies every ``*API_KEY*`` in the env file into ``os.environ``.

    pydantic-settings only reads fields that are declared on the Settings class,
    so extra keys — ``NVIDIA_API_KEY2``, ``OPENROUTER_API_KEY1`` and the rest —
    are parsed and then discarded. The LLM pool discovers keys by scanning the
    process environment, so without this it would see one key and run at a
    sixth of the available throughput.

    Existing environment values win, so a key exported in the shell overrides
    the file. Returns how many were exported.
    """
    import os

    if not env_file.exists():
        return 0

    exported = 0
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip().strip('"').strip("'")
        if "API_KEY" not in name.upper() or not value:
            continue
        if os.environ.get(name):
            continue
        os.environ[name] = value
        exported += 1
    return exported


class Settings(BaseSettings):
    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    # Primary Odoo codebase for AST extraction (can be overridden via env)
    odoo_codebase_path: Path = PydanticField(
        default=Path(os.environ.get("ODOO_SOURCE", "./odoo"))
    )

    # Output artifacts live one level up (workspace root), not inside the package
    output_dir: Path = PydanticField(default=_HERE.parent / "forge_outputs")
    db_file: Path = PydanticField(default=_HERE.parent / "forge_knowledge.db")

    # ------------------------------------------------------------------
    # LLM Backend
    # ------------------------------------------------------------------
    llm_backend: str = PydanticField(default="nvidia")  # nvidia | ollama | openai | vllm

    # NVIDIA NIM — key MUST come from odoo_agent_forge/.env or the environment.
    # It is NEVER hardcoded in source. Missing key → clear startup error.
    nvidia_api_key: str = PydanticField(default="")
    nvidia_base_url: str = PydanticField(default="https://integrate.api.nvidia.com/v1")

    @field_validator("nvidia_api_key")
    @classmethod
    def _require_nvidia_api_key(cls, v: str) -> str:
        if not v:
            raise ValueError(
                "NVIDIA_API_KEY is required. "
                f"Add it to {_ENV_FILE} or set the environment variable before running."
            )
        return v

    # ------------------------------------------------------------------
    # Teacher Model Configuration
    # ------------------------------------------------------------------
    nvidia_model: str = PydanticField(default="nvidia/nemotron-3-ultra-550b-a55b")
    fallback_model: str = PydanticField(default="meta/llama-3.3-70b-instruct")
    enable_thinking: bool = PydanticField(default=True)
    # reasoning_budget MUST stay below max_tokens: on these endpoints the think
    # trace and the answer share one allowance, and equal values let reasoning
    # consume it all, truncating the answer mid-sentence.
    reasoning_budget: int = PydanticField(default=4096)
    max_tokens: int = PydanticField(default=6144)
    # 1.0 drifted off the injected schema; 0.7 keeps variety without inventing fields.
    temperature: float = PydanticField(default=0.7)
    top_p: float = PydanticField(default=0.95)

    # ------------------------------------------------------------------
    # Optional fallback backends
    # ------------------------------------------------------------------
    ollama_host: str = PydanticField(default="http://localhost:11434")
    openai_api_key: Optional[str] = PydanticField(default=None)
    openai_base_url: Optional[str] = PydanticField(default=None)

    # ------------------------------------------------------------------
    # Concurrency
    # ------------------------------------------------------------------
    max_concurrency: int = PydanticField(default=4)

    # ------------------------------------------------------------------
    # Dataset Composition Targets
    # ------------------------------------------------------------------
    target_extracted_ratio: float = 0.70
    target_synthetic_ratio: float = 0.15
    target_agentic_mcp_ratio: float = 0.15

    # Always load from odoo_agent_forge/.env — never from CWD
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )
