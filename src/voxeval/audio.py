"""Audio I/O helpers. Kept dependency-light: numpy, scipy, soundfile only."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio(path: str | Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load as float32 mono, optionally resampled to `sr`."""
    audio, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr is not None and sr != file_sr:
        audio = resample(audio, file_sr, sr)
        file_sr = sr
    return audio, file_sr


def resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return audio
    g = gcd(src, dst)
    return resample_poly(audio, dst // g, src // g).astype(np.float32)


def save_audio(path: str | Path, audio: np.ndarray, sr: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sr, subtype="PCM_16")
