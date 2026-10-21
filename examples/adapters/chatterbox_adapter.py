"""Reference adapter: Chatterbox (Resemble AI, MIT) running locally.

This lives in examples/ on purpose: voxeval's core knows no specific TTS model. Copy this
file as a starting point for your own model. It shows the three things an adapter does:

  1. declare `capabilities` (cloning? which languages? which tunable params + ranges?)
  2. load the model in `setup`
  3. turn a SynthesisRequest into audio in `synthesize`

Use it from a config with:   adapter: examples/adapters/chatterbox_adapter.py:ChatterboxAdapter
Install its deps in their own venv (it pins torch):   pip install -r examples/adapters/requirements.txt
"""

from __future__ import annotations

import numpy as np

from voxeval.adapters.base import TTSAdapter
from voxeval.types import AudioResult, Capabilities, ParamSpec, SynthesisRequest

LANGUAGES = ("en", "es", "fr", "de")  # the multilingual model supports 23; these ship prompts


class ChatterboxAdapter(TTSAdapter):
    name = "chatterbox"

    def __init__(self, model: str = "multilingual", device: str = "auto"):
        if model not in ("multilingual", "english"):
            raise ValueError("model must be 'multilingual' or 'english'")
        self.model_kind, self.device_arg = model, device
        self.capabilities = Capabilities(
            voice_cloning=True,  # optional per request: no ref audio -> built-in voice
            languages=LANGUAGES if model == "multilingual" else ("en",),
            params={
                "exaggeration": ParamSpec(type="float", min=0.25, max=2.0, default=0.5),
                "cfg_weight": ParamSpec(type="float", min=0.0, max=1.0, default=0.5),
                "temperature": ParamSpec(type="float", min=0.05, max=1.5, default=0.8),
            },
        )

    def setup(self) -> None:
        import torch

        device = self.device_arg
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        if self.model_kind == "multilingual":
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            # pass the device as a *string*: upstream compares `device in ['cpu', 'mps']` to pick
            # map_location, so a torch.device object would try to restore CUDA tensors on CPU
            self.model = ChatterboxMultilingualTTS.from_pretrained(device)
        else:
            from chatterbox.tts import ChatterboxTTS

            self.model = ChatterboxTTS.from_pretrained(device)

    def synthesize(self, req: SynthesisRequest) -> AudioResult:
        if req.seed is not None:
            self.torch.manual_seed(req.seed)
        kwargs = dict(req.params)
        if req.ref_audio is not None:
            kwargs["audio_prompt_path"] = str(req.ref_audio)
        if self.model_kind == "multilingual":
            wav = self.model.generate(req.text, language_id=req.language, **kwargs)
        else:
            wav = self.model.generate(req.text, **kwargs)
        return AudioResult(np.asarray(wav.squeeze().cpu(), dtype=np.float32), self.model.sr)

    def teardown(self) -> None:
        self.model = None
