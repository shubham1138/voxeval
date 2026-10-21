"""Failure-mode flags for hard cases, derived from the intelligibility output.

Per-category WER (numbers, URLs, tongue twisters...) comes for free from aggregation;
this metric adds the failures a mean WER hides: truncation, hallucination, repetition.
"""

from __future__ import annotations

from ..registry import register_metric
from ..types import Item
from .base import Metric


def repetition(text: str, n: int = 3) -> float:
    """Fraction of word n-grams that are repeats (0 = no repetition)."""
    w = text.split()
    grams = [tuple(w[i : i + n]) for i in range(len(w) - n + 1)]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


@register_metric("robustness")
class Robustness(Metric):
    description = "Truncation / hallucination / repetition flags and duration sanity."
    license = "n/a (no model)"
    requires = frozenset({"transcript"})
    depends_on = ("intelligibility",)
    fields = {
        "duration_ratio": None,
        "repetition_score": False,
        "truncated": False,
        "hallucinated": False,
        "robust_fail": False,
    }

    def score(self, item: Item) -> dict:
        r = item.results
        cps = float(self.args.get("chars_per_second", 14.0))
        max_del = float(self.args.get("max_deletion_rate", 0.3))
        max_ins = float(self.args.get("max_insertion_rate", 0.3))
        n_ref = max(int(r.get("ref_words") or 1), 1)
        ref_text = item.prompt.asr_reference or item.text
        ratio = item.gen["duration_s"] / max(len(ref_text) / cps, 1e-3)
        rep = max(repetition(r["transcript"]) - repetition(ref_text), 0.0)
        truncated = (r["wer_del"] / n_ref) >= max_del or ratio < 0.4
        hallucinated = (r["wer_ins"] / n_ref) >= max_ins or ratio > 2.5 or rep >= 0.3
        return {
            "duration_ratio": ratio,
            "repetition_score": rep,
            "truncated": truncated,
            "hallucinated": hallucinated,
            "robust_fail": truncated or hallucinated,
        }
