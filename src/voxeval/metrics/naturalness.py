"""Naturalness: reference-free MOS prediction (UTMOS22)."""

from __future__ import annotations

from ..audio import load_audio
from ..registry import register_metric
from ..types import Item
from .base import Metric, lazy_import


@register_metric("naturalness")
class Naturalness(Metric):
    description = "Predicted MOS (1-5) from UTMOS22-strong; no reference needed."
    license = "SpeechMOS code MIT; UTMOS22 weights: check upstream terms before commercial use"
    extra = "mos"
    fields = {"mos": True}

    def setup(self, device: str) -> None:
        self.torch = lazy_import("torch", "mos")
        self.device = device
        self.model = (
            self.torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
            .to(device)
            .eval()
        )

    def score(self, item: Item) -> dict:
        y, _ = load_audio(item.wav_path, sr=16000)
        wav = self.torch.from_numpy(y).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            mos = self.model(wav, 16000)
        return {"mos": float(mos.reshape(-1)[0])}

    def teardown(self) -> None:
        self.model = None
