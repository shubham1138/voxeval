"""Run configuration (YAML -> validated pydantic model)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .profiles import OPTIN, PROFILES, SKIP
from .types import VoxevalError

OPS = ("<=", ">=", "<", ">", "==", "!=")


class SuiteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = None  # built-in suite: en_core, es_core, ... or base name "core"
    path: str | None = None  # user suite: .yaml / .jsonl / .csv
    languages: list[str] | None = None
    categories: list[str] | None = None
    limit: int | None = None
    per_category: int | None = None


class VoicesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dir: str  # <dir>/<lang>/*.wav or <dir>/*.wav


class GateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    op: str
    value: float

    @field_validator("op")
    @classmethod
    def _op(cls, v: str) -> str:
        if v not in OPS:
            raise ValueError(f"op must be one of {OPS}")
        return v


class CompositeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    weights: dict[str, float] | None = None
    anchors: dict[str, list] | None = None  # field -> [higher|lower, lo, hi]


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "run"
    profile: str = "laptop"
    adapter: str
    adapter_args: dict[str, Any] = Field(default_factory=dict)
    sweep: dict[str, list] | None = None
    variants: list[dict] | None = None
    suite: SuiteConfig
    voices: VoicesConfig | None = None
    metrics: list[str] | dict[str, dict | None] | None = None
    gates: list[GateConfig] = Field(default_factory=list)
    composite: CompositeConfig = Field(default_factory=CompositeConfig)
    output_dir: str = "runs"
    seed: int = 0
    device: str = "auto"

    @field_validator("profile")
    @classmethod
    def _profile(cls, v: str) -> str:
        if v not in PROFILES:
            raise ValueError(f"profile must be one of {sorted(PROFILES)}")
        return v


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> RunConfig:
    p = Path(path)
    if not p.exists():
        raise VoxevalError(f"config not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    data.update({k: v for k, v in (overrides or {}).items() if v is not None})
    try:
        return RunConfig(**data)
    except ValidationError as e:
        raise VoxevalError(f"invalid config {p}:\n{e}") from e


def resolve_metrics(cfg: RunConfig) -> tuple[dict[str, dict], dict[str, str]]:
    """Merge profile defaults under user args. Returns (enabled name->args, skipped name->reason)."""
    table = PROFILES[cfg.profile]
    if cfg.metrics is None:
        wanted: dict[str, dict] = {k: {} for k in table}
    elif isinstance(cfg.metrics, list):
        wanted = {k: {} for k in cfg.metrics}
    else:
        wanted = {k: dict(v or {}) for k, v in cfg.metrics.items()}
    enabled: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    for name, user_args in wanted.items():
        base = dict(table.get(name, {}))
        listed = cfg.metrics is not None
        if SKIP in base and not user_args:
            skipped[name] = base[SKIP]
            continue
        if OPTIN in base and not listed:
            skipped[name] = base[OPTIN]
            continue
        base.pop(SKIP, None)
        base.pop(OPTIN, None)
        enabled[name] = {**base, **user_args}
    return enabled, skipped
