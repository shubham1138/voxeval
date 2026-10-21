"""Built-in metrics. Heavy dependencies are imported lazily inside `setup`."""

from . import (  # noqa: F401
    efficiency,
    emotion,
    intelligibility,
    judge,
    naturalness,
    prosody,
    robustness,
    signal,
    speaker_similarity,
)
from .base import Metric

__all__ = ["Metric"]
