# Writing an adapter

An adapter is one class. voxeval never needs to know your model.

```python
from voxeval.adapters.base import TTSAdapter
from voxeval.types import AudioResult, Capabilities, ParamSpec, SynthesisRequest


class MyTTS(TTSAdapter):
    capabilities = Capabilities(
        voice_cloning=False,  # True only if you can use req.ref_audio
        languages=("en",),  # None = any
        params={"speed": ParamSpec(type="float", min=0.5, max=2.0, default=1.0)},
    )

    def setup(self): ...  # load model / open session
    def synthesize(self, req: SynthesisRequest) -> AudioResult:
        audio = my_model(req.text, speed=req.params.get("speed", 1.0))  # float32 mono
        return AudioResult(audio, sample_rate=24000)

    def teardown(self): ...
```

Point a config at it: `adapter: path/to/my_tts.py:MyTTS` (or `package.module:MyTTS`).
voxeval then only sends what you declared: reference audio only if `voice_cloning`,
sweepable params only from `params`, languages only from `languages`.

`chatterbox_adapter.py` is a complete worked example (local model, cloning, three sweepable dials).
For a model behind an HTTP API you usually need no code at all: use the built-in `http` adapter.
For audio you already generated: use the built-in `folder` adapter.
