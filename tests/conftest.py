"""Test doubles: a tone-generating adapter and model-free metrics (no downloads, no network)."""

from __future__ import annotations

import numpy as np
import pytest

from voxeval.adapters.base import TTSAdapter
from voxeval.metrics.base import Metric
from voxeval.registry import register_adapter, register_metric
from voxeval.types import AudioResult, Capabilities, Item, ParamSpec, SynthesisRequest

SR = 16000


def tone(seconds: float, hz: float = 220.0, amp: float = 0.3, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


@register_adapter("stub_tone")
class StubTone(TTSAdapter):
    """Duration scales with text length; `pitch` param shifts the tone. Counts calls."""

    calls = 0

    def __init__(self, cloning: bool = False, langs: list[str] | None = None, fail_on: str = ""):
        self.fail_on = fail_on
        self.capabilities = Capabilities(
            voice_cloning=cloning,
            languages=tuple(langs) if langs else None,
            params={"pitch": ParamSpec(type="float", min=100, max=400, default=220)},
        )
        self.seen_ref: list = []

    def synthesize(self, req: SynthesisRequest) -> AudioResult:
        StubTone.calls += 1
        self.seen_ref.append(req.ref_audio)
        if self.fail_on and self.fail_on in req.text:
            raise RuntimeError("boom")
        return AudioResult(tone(0.05 * len(req.text) + 0.5, req.params.get("pitch", 220)), SR)


@register_metric("stub_len")
class StubLen(Metric):
    fields = {"n_chars": None}
    license = "n/a"

    def score(self, item: Item) -> dict:
        return {"n_chars": len(item.text)}


@register_metric("stub_dep")
class StubDep(Metric):
    fields = {"double": None}
    depends_on = ("stub_len",)
    requires = frozenset()

    def score(self, item: Item) -> dict:
        return {"double": item.results["n_chars"] * 2}


@register_metric("stub_ref")
class StubRef(Metric):
    fields = {"has_ref": None}
    requires = frozenset({"ref_audio"})

    def score(self, item: Item) -> dict:
        return {"has_ref": 1.0}


@register_metric("stub_broken")
class StubBroken(Metric):
    fields = {"never": None}

    def setup(self, device: str) -> None:
        raise RuntimeError("model weights unavailable")

    def score(self, item: Item) -> dict:
        return {"never": 1}


@pytest.fixture(autouse=True)
def _reset_calls():
    StubTone.calls = 0


@pytest.fixture
def suite_file(tmp_path):
    p = tmp_path / "suite.yaml"
    p.write_text(
        "prompts:\n"
        "  - {id: a1, text: 'Hello there', category: short, language: en}\n"
        "  - {id: a2, text: 'A much longer sentence about nothing', category: long, language: en}\n"
        "  - {id: a3, text: 'Hola amigo', category: short, language: es}\n"
    )
    return p
