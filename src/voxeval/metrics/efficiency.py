"""Generation cost, read from the records the runner took while calling the adapter."""

from __future__ import annotations

from ..registry import register_metric
from ..types import Item
from .base import Metric


@register_metric("efficiency")
class Efficiency(Metric):
    description = "Latency, real-time factor (RTF) and peak GPU memory of synthesis."
    license = "n/a (no model)"
    fields = {"latency_s": False, "rtf": False, "peak_vram_mb": False}

    def score(self, item: Item) -> dict:
        lat, dur = item.gen.get("latency_s"), item.gen.get("duration_s")
        return {
            "latency_s": lat,
            "rtf": (lat / dur) if lat is not None and dur else None,  # < 1 = faster than real time
            "peak_vram_mb": item.gen.get("peak_vram_mb"),
        }
