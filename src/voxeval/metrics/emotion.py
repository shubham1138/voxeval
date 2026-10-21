"""Emotion: what does the clip sound like, and (if the prompt asks for a style) did it land?"""

from __future__ import annotations

from ..registry import register_metric
from ..types import Item
from .base import Metric, lazy_import

# free-text style -> emotion2vec label
STYLE_TO_LABEL = {
    "neutral": "neutral",
    "calm": "neutral",
    "happy": "happy",
    "joyful": "happy",
    "excited": "happy",
    "cheerful": "happy",
    "sad": "sad",
    "upset": "sad",
    "angry": "angry",
    "furious": "angry",
    "fearful": "fearful",
    "scared": "fearful",
    "anxious": "fearful",
    "surprised": "surprised",
    "disgusted": "disgusted",
}


@register_metric("emotion")
class Emotion(Metric):
    description = "emotion2vec utterance-level emotion; style_match when a prompt sets `style`."
    license = "emotion2vec model weights: see model card (custom licence); opt-in"
    extra = "emotion"
    fields = {"emotion_confidence": None, "style_match": True}
    info = ("emotion_label",)

    def setup(self, device: str) -> None:
        funasr = lazy_import("funasr", "emotion")
        model = self.args.get("model", "iic/emotion2vec_plus_base")
        hub = self.args.get("hub", "ms" if model.startswith("iic/") else "hf")
        self.model = funasr.AutoModel(model=model, hub=hub, device=device, disable_update=True)

    def score(self, item: Item) -> dict:
        res = self.model.generate(
            str(item.wav_path), granularity="utterance", extract_embedding=False
        )[0]
        scores = res["scores"]
        best = max(range(len(scores)), key=scores.__getitem__)
        label = res["labels"][best].split("/")[-1].strip().lower()
        out = {"emotion_label": label, "emotion_confidence": float(scores[best])}
        if item.prompt.style:
            target = STYLE_TO_LABEL.get(item.prompt.style.lower())
            if target:
                out["style_match"] = label == target
        return out

    def teardown(self) -> None:
        self.model = None
