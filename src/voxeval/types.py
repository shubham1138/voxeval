"""Plain data types shared by adapters, metrics and the runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ParamSpec:
    """Declares one tunable provider parameter so sweeps can be validated."""

    type: str = "float"  # float | int | str | bool
    min: float | None = None
    max: float | None = None
    choices: tuple | None = None
    default: Any = None
    description: str = ""

    def validate(self, name: str, value: Any) -> None:
        if self.choices is not None and value not in self.choices:
            raise ValueError(f"param '{name}'={value!r} not in choices {list(self.choices)}")
        if self.type in ("float", "int"):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"param '{name}'={value!r} must be numeric")
            if self.min is not None and value < self.min:
                raise ValueError(f"param '{name}'={value} below min {self.min}")
            if self.max is not None and value > self.max:
                raise ValueError(f"param '{name}'={value} above max {self.max}")


@dataclass
class Capabilities:
    """What an adapter can do. The runner and metrics switch features on/off from this."""

    voice_cloning: bool = False
    languages: tuple[str, ...] | None = None  # None = any language
    params: dict[str, ParamSpec] = field(default_factory=dict)
    style_control: bool = False
    sample_rate: int | None = None


@dataclass
class Prompt:
    id: str
    text: str
    language: str = "en"
    category: str = "general"
    style: str | None = None
    asr_reference: str | None = None  # what ASR should hear, if different from `text`
    ref_audio: Path | None = None  # per-prompt reference voice (overrides `voices`)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesisRequest:
    prompt_id: str
    text: str
    language: str = "en"
    ref_audio: Path | None = None
    params: dict[str, Any] = field(default_factory=dict)
    style: str | None = None
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioResult:
    audio: np.ndarray  # float32 mono, [-1, 1]
    sample_rate: int
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Item:
    """One generated clip handed to metrics."""

    sample_id: str
    variant: str
    prompt: Prompt
    wav_path: Path
    ref_audio: Path | None
    gen: dict[str, Any]
    results: dict[str, Any] = field(default_factory=dict)  # fields from earlier metrics

    @property
    def text(self) -> str:
        return self.prompt.text

    @property
    def language(self) -> str:
        return self.prompt.language


class VoxevalError(Exception):
    """User-facing configuration or usage error."""
