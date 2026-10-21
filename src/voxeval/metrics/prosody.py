"""Prosody descriptors: pitch, speaking rate and pauses (+ pitch offset vs. the reference)."""

from __future__ import annotations

import numpy as np

from ..audio import load_audio
from ..registry import register_metric
from ..types import Item
from .base import Metric, lazy_import
from .signal import frame_rms_db

SR = 16000
HOP = 320  # 20 ms


@register_metric("prosody")
class Prosody(Metric):
    description = (
        "F0 level/variation (semitones), speaking rate, pauses; pitch offset vs. reference."
    )
    license = "librosa ISC"
    extra = "prosody"
    fields = {
        "f0_mean_hz": None,
        "f0_std_st": None,
        "f0_range_st": None,
        "voiced_ratio": None,
        "speaking_rate_wps": None,
        "pause_count": False,
        "pause_mean_s": False,
        "f0_mean_delta_st": None,
    }

    def setup(self, device: str) -> None:
        self.librosa = lazy_import("librosa", "prosody")
        self._ref: dict[str, float | None] = {}

    def _f0(self, y: np.ndarray) -> np.ndarray:
        f0, _, _ = self.librosa.pyin(y, fmin=65, fmax=450, sr=SR, frame_length=1024, hop_length=HOP)
        return f0[~np.isnan(f0)]

    def _ref_mean(self, path) -> float | None:
        key = str(path)
        if key not in self._ref:
            f0 = self._f0(load_audio(path, sr=SR)[0])
            self._ref[key] = float(np.mean(f0)) if len(f0) else None
        return self._ref[key]

    def score(self, item: Item) -> dict:
        y, _ = load_audio(item.wav_path, sr=SR)
        f0 = self._f0(y)
        n_frames = 1 + max(len(y) - 1024, 0) // HOP
        out: dict = {"voiced_ratio": len(f0) / max(n_frames, 1)}
        if len(f0):
            st = 12 * np.log2(f0)
            out.update(
                f0_mean_hz=float(f0.mean()),
                f0_std_st=float(st.std()),
                f0_range_st=float(np.percentile(st, 95) - np.percentile(st, 5)),
            )
        db = frame_rms_db(y, SR, win_s=0.020, hop_s=0.020)
        active = db > (db.max() - 40.0)
        if active.any():
            first, last = int(np.argmax(active)), len(active) - 1 - int(np.argmax(active[::-1]))
            speech_s = (last - first + 1) * 0.020
            out["speaking_rate_wps"] = len(item.text.split()) / max(speech_s, 1e-3)
            runs, run = [], 0
            for a in active[first : last + 1]:
                if not a:
                    run += 1
                elif run:
                    runs.append(run * 0.020)
                    run = 0
            pauses = [r for r in runs if r >= 0.25]
            out["pause_count"] = len(pauses)
            out["pause_mean_s"] = float(np.mean(pauses)) if pauses else 0.0
        if item.ref_audio is not None and len(f0):
            ref_mean = self._ref_mean(item.ref_audio)
            if ref_mean:
                out["f0_mean_delta_st"] = float(12 * np.log2(f0.mean() / ref_mean))
        return out

    def teardown(self) -> None:
        self.librosa = None
