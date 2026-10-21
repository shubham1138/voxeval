"""LALM-as-judge: an open audio-language model rates each clip on a fixed rubric (1-5).

Backends are pluggable (`backend:` arg). The default is Qwen2.5-Omni run locally, which
needs a ~16 GB GPU, so it is only enabled in the `gpu` profile. Judges are noisy and
biased: use scores to compare variants under the same judge, not as absolute truth.
"""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar

from ..audio import load_audio
from ..registry import register_metric
from ..types import Item, VoxevalError
from .base import Metric, lazy_import

RUBRIC = """You are an expert evaluator of synthesized speech. Listen to the audio.
The intended text was: "{text}"
{style_line}
Rate each criterion as an integer from 1 (very poor) to 5 (excellent):
- naturalness: does it sound like a real human speaker (no robotic, glitchy or muffled artifacts)?
- prosody: appropriate rhythm, stress, intonation and pausing for this text
- pronunciation: words, names and numbers pronounced correctly
{style_criterion}Be strict; reserve 5 for near-human quality. Reply with ONLY a JSON object:
{{"naturalness": n, "prosody": n, "pronunciation": n, {style_key}"overall": n, "reason": "<one short sentence>"}}"""


def build_prompt(text: str, style: str | None) -> str:
    return RUBRIC.format(
        text=text,
        style_line=f'The requested speaking style was: "{style}".' if style else "",
        style_criterion=(
            "- style_adherence: how well does the delivery match the requested style?\n"
            if style
            else ""
        ),
        style_key='"style_adherence": n, ' if style else "",
    )


def parse_judgement(reply: str, with_style: bool) -> dict[str, Any]:
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        raise ValueError(f"no JSON in judge reply: {reply[:120]!r}")
    data = json.loads(m.group(0))
    keys = ["naturalness", "prosody", "pronunciation", "overall"] + (
        ["style_adherence"] if with_style else []
    )
    out: dict[str, Any] = {}
    for k in keys:
        v = float(data[k])
        if not 1 <= v <= 5:
            raise ValueError(f"{k}={v} outside 1-5")
        out[f"judge_{k}"] = v
    out["judge_reason"] = str(data.get("reason", ""))[:300]
    return out


class JudgeBackend:
    def setup(self, device: str, args: dict) -> None: ...
    def generate(self, wav_path, prompt: str, sample: bool) -> str: ...
    def teardown(self) -> None: ...


class QwenOmniBackend(JudgeBackend):
    def setup(self, device: str, args: dict) -> None:
        torch = lazy_import("torch", "judge")
        tf = lazy_import("transformers", "judge")
        name = args.get("model", "Qwen/Qwen2.5-Omni-7B")
        self.model = tf.Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            name, torch_dtype=torch.bfloat16, device_map="auto" if device == "cuda" else None
        )
        self.processor = tf.Qwen2_5OmniProcessor.from_pretrained(name)
        self.max_new_tokens = int(args.get("max_new_tokens", 200))

    def generate(self, wav_path, prompt: str, sample: bool) -> str:
        audio, _ = load_audio(wav_path, sr=16000)
        conv = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(wav_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        try:
            inputs = self.processor(text=text, audio=[audio], return_tensors="pt", padding=True)
        except TypeError:  # older transformers used `audios=`
            inputs = self.processor(text=text, audios=[audio], return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        kw = {"do_sample": True, "temperature": 0.3} if sample else {"do_sample": False}
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, **kw)
        return self.processor.batch_decode(
            out[:, inputs["input_ids"].size(1) :], skip_special_tokens=True
        )[0]

    def teardown(self) -> None:
        self.model = self.processor = None


BACKENDS: dict[str, type[JudgeBackend]] = {"qwen2_5_omni": QwenOmniBackend}


@register_metric("judge")
class Judge(Metric):
    description = (
        "Open audio-LLM judge: naturalness / prosody / pronunciation / overall (1-5) + reason."
    )
    license = "Qwen2.5-Omni: Apache-2.0"
    extra = "judge"
    fields: ClassVar[dict[str, bool | None]] = {
        "judge_naturalness": True,
        "judge_prosody": True,
        "judge_pronunciation": True,
        "judge_style_adherence": True,
        "judge_overall": True,
    }
    info = ("judge_reason",)

    def setup(self, device: str) -> None:
        backend = self.args.get("backend", "qwen2_5_omni")
        if backend not in BACKENDS:
            raise VoxevalError(f"unknown judge backend '{backend}'. Available: {list(BACKENDS)}")
        self.backend = BACKENDS[backend]()
        self.backend.setup(device, self.args)

    def score(self, item: Item) -> dict:
        style = item.prompt.style
        prompt = build_prompt(item.text, style)
        last: Exception | None = None
        for attempt in range(int(self.args.get("retries", 2))):
            try:
                reply = self.backend.generate(item.wav_path, prompt, sample=attempt > 0)
                return parse_judgement(reply, with_style=bool(style))
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                last = e  # malformed reply: retry with sampling, then give up
        raise ValueError(f"judge reply could not be parsed: {last}")

    def teardown(self) -> None:
        self.backend.teardown()
