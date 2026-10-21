"""Expand a parameter grid into named variants and validate against adapter capabilities."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Any

from .types import Capabilities, VoxevalError


@dataclass
class Variant:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=,-]+", "_", s)


def expand_variants(sweep: dict[str, list] | None, variants: list[dict] | None) -> list[Variant]:
    if sweep and variants:
        raise VoxevalError("use either `sweep` (grid) or `variants` (explicit), not both")
    if sweep:
        keys = list(sweep)
        for k in keys:
            if not isinstance(sweep[k], list) or not sweep[k]:
                raise VoxevalError(f"sweep['{k}'] must be a non-empty list")
        out = []
        for combo in itertools.product(*(sweep[k] for k in keys)):
            params = dict(zip(keys, combo, strict=True))
            out.append(Variant(_slug(",".join(f"{k}={v}" for k, v in params.items())), params))
        return out
    if variants:
        out = [
            Variant(_slug(str(v.get("name") or f"v{i}")), dict(v.get("params") or {}))
            for i, v in enumerate(variants)
        ]
        names = [v.name for v in out]
        if len(names) != len(set(names)):
            raise VoxevalError("variant names must be unique")
        return out
    return [Variant("default", {})]


def validate_variants(variants: list[Variant], caps: Capabilities) -> None:
    """Fail early: a param the adapter does not declare cannot be swept."""
    for v in variants:
        for k, val in v.params.items():
            if k not in caps.params:
                declared = ", ".join(caps.params) or "none"
                raise VoxevalError(
                    f"param '{k}' is not declared by the adapter (declared: {declared})"
                )
            try:
                caps.params[k].validate(k, val)
            except ValueError as e:
                raise VoxevalError(str(e)) from e
