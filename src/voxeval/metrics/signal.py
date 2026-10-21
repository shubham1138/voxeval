"""Signal sanity checks: cheap, model-free, ideal as hard gates."""

from __future__ import annotations

import numpy as np

from ..audio import load_audio
from ..registry import register_metric
from ..types import Item
from .base import Metric


def frame_rms_db(y: np.ndarray, sr: int, win_s: float = 0.025, hop_s: float = 0.010) -> np.ndarray:
    win, hop = max(int(win_s * sr), 1), max(int(hop_s * sr), 1)
    if len(y) < win:
        y = np.pad(y, (0, win - len(y)))
    n = 1 + (len(y) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(y[idx] ** 2, axis=1))
    return 20 * np.log10(rms + 1e-10)


@register_metric("signal")
class Signal(Metric):
    description = "Clipping, silence, loudness and empty/NaN output checks."
    license = "n/a (no model)"
    fields = {
        "duration_s": None,
        "clipping_ratio": False,
        "silence_ratio": False,
        "lead_silence_s": False,
        "trail_silence_s": False,
        "lufs": None,
        "is_empty": False,
    }

    def score(self, item: Item) -> dict:
        y, sr = load_audio(item.wav_path)
        dur = len(y) / sr
        bad = (not np.isfinite(y).all()) or dur < 0.2 or float(np.max(np.abs(y), initial=0)) < 1e-4
        if bad:
            return {
                "duration_s": dur,
                "clipping_ratio": None,
                "silence_ratio": None,
                "lead_silence_s": None,
                "trail_silence_s": None,
                "lufs": None,
                "is_empty": True,
            }
        db = frame_rms_db(y, sr)
        active = db > (db.max() - 40.0)  # speech = within 40 dB of the loudest frame
        hop = 0.010
        first = int(np.argmax(active))
        last = len(active) - 1 - int(np.argmax(active[::-1]))
        lufs = None
        try:
            import pyloudnorm as pyln

            lufs = float(pyln.Meter(sr).integrated_loudness(y.astype(np.float64)))
            lufs = lufs if np.isfinite(lufs) else None
        except Exception:
            pass
        return {
            "duration_s": dur,
            "clipping_ratio": float(np.mean(np.abs(y) >= 0.999)),
            "silence_ratio": float(1.0 - active.mean()),
            "lead_silence_s": first * hop,
            "trail_silence_s": (len(active) - 1 - last) * hop,
            "lufs": lufs,
            "is_empty": False,
        }
