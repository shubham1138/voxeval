"""Speaker similarity between the reference voice and the synthesized clip."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..audio import load_audio
from ..registry import register_metric
from ..types import Item
from .base import Metric, lazy_import


@register_metric("speaker_similarity")
class SpeakerSimilarity(Metric):
    description = "Cosine similarity of ECAPA-TDNN speaker embeddings (needs reference audio)."
    license = "SpeechBrain ECAPA: Apache-2.0"
    extra = "sim"
    requires = frozenset({"ref_audio"})
    fields = {"sim": True}

    def setup(self, device: str) -> None:
        sb = lazy_import("speechbrain.inference.speaker", "sim")
        self.torch = lazy_import("torch", "sim")
        source = self.args.get("model", "speechbrain/spkrec-ecapa-voxceleb")
        savedir = Path.home() / ".cache" / "voxeval" / source.replace("/", "_")
        self.enc = sb.EncoderClassifier.from_hparams(
            source=source, savedir=str(savedir), run_opts={"device": device}
        )
        self.device = device
        self._ref_cache: dict[str, np.ndarray] = {}

    def _embed(self, path) -> np.ndarray:
        y, _ = load_audio(path, sr=16000)
        wav = self.torch.from_numpy(y).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            emb = self.enc.encode_batch(wav).squeeze().cpu().numpy()
        return emb / (np.linalg.norm(emb) + 1e-9)

    def score(self, item: Item) -> dict:
        key = str(item.ref_audio)
        if key not in self._ref_cache:
            self._ref_cache[key] = self._embed(item.ref_audio)
        return {"sim": float(np.dot(self._ref_cache[key], self._embed(item.wav_path)))}

    def teardown(self) -> None:
        self.enc = None
