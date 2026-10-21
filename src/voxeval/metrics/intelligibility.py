"""Intelligibility: transcribe with Whisper, compare to the text (WER / CER)."""

from __future__ import annotations

from typing import Any

from ..audio import load_audio
from ..registry import register_metric
from ..types import Item
from .base import Metric, lazy_import


def make_normalizer(language: str):
    """Whisper's normalizers (English-specific for 'en', language-agnostic otherwise).

    Output whitespace is collapsed: the raw normalizers pad with spaces, which would
    otherwise be counted as characters by CER.
    """
    if language == "en":
        norm = lazy_import("whisper_normalizer.english", "asr").EnglishTextNormalizer()
    else:
        norm = lazy_import("whisper_normalizer.basic", "asr").BasicTextNormalizer()
    return lambda text: " ".join(norm(text).split())


def word_error_stats(ref: str, hyp: str) -> dict[str, Any]:
    jiwer = lazy_import("jiwer", "asr")
    ref_words = ref.split()
    if not ref_words:
        raise ValueError("reference text is empty after normalization")
    if not hyp.strip():  # silence / nothing recognised
        return {
            "wer": 1.0,
            "cer": 1.0,
            "wer_sub": 0,
            "wer_del": len(ref_words),
            "wer_ins": 0,
            "ref_words": len(ref_words),
        }
    w = jiwer.process_words(ref, hyp)
    return {
        "wer": w.wer,
        "cer": jiwer.cer(ref, hyp),
        "wer_sub": w.substitutions,
        "wer_del": w.deletions,
        "wer_ins": w.insertions,
        "ref_words": len(ref_words),
    }


@register_metric("intelligibility")
class Intelligibility(Metric):
    description = "ASR round-trip: Whisper transcript vs. input text -> WER/CER."
    license = "faster-whisper MIT; Whisper weights MIT"
    extra = "asr"
    fields = {
        "wer": False,
        "cer": False,
        "wer_sub": False,
        "wer_del": False,
        "wer_ins": False,
        "ref_words": None,
    }
    info = ("transcript",)

    def setup(self, device: str) -> None:
        fw = lazy_import("faster_whisper", "asr")
        lazy_import("jiwer", "asr")
        model = self.args.get("model", "large-v3-turbo")
        compute = self.args.get("compute_type", "int8" if device == "cpu" else "float16")
        if device == "cpu" and compute in ("float16", "bfloat16"):
            compute = "int8"  # fp16 is not available on CPU
        self.beam = int(self.args.get("beam_size", 1))
        self.model = fw.WhisperModel(model, device=device, compute_type=compute)
        self._norms: dict[str, Any] = {}

    def _norm(self, language: str):
        key = "en" if language == "en" else "other"
        if key not in self._norms:
            self._norms[key] = make_normalizer(language)
        return self._norms[key]

    def score(self, item: Item) -> dict:
        audio, _ = load_audio(item.wav_path, sr=16000)
        segments, _ = self.model.transcribe(
            audio,
            language=item.language,
            beam_size=self.beam,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        transcript = " ".join(s.text.strip() for s in segments).strip()
        norm = self._norm(item.language)
        ref = norm(item.prompt.asr_reference or item.text)
        return {"transcript": transcript, **word_error_stats(ref, norm(transcript))}

    def teardown(self) -> None:
        self.model = None
