"""Generic HTTP adapter for any TTS endpoint: your own server, a cloud API, a gateway.

Everything is config: URL, headers, body template and how to read the audio back.
Secrets are never stored in config: write `${ENV_VAR}` and it is expanded at call time.
No key is required by voxeval itself. Only your own endpoint may need one.
"""

from __future__ import annotations

import base64
import io
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
import soundfile as sf

from ..registry import register_adapter
from ..types import AudioResult, Capabilities, ParamSpec, SynthesisRequest, VoxevalError
from .base import TTSAdapter

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value: str) -> str:
    def sub(m: re.Match) -> str:
        if m.group(1) not in os.environ:
            raise VoxevalError(f"environment variable {m.group(1)} is not set")
        return os.environ[m.group(1)]

    return _ENV.sub(sub, value)


def _fill(obj: Any, values: dict[str, Any]) -> Any:
    """Recursively substitute {placeholders} in a JSON-like body template."""
    if isinstance(obj, str):
        if obj.startswith("{") and obj.endswith("}") and obj[1:-1] in values:
            return values[obj[1:-1]]  # whole-string placeholder keeps its type
        for k, v in values.items():
            obj = obj.replace("{" + k + "}", str(v))
        return obj
    if isinstance(obj, dict):
        return {k: _fill(v, values) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fill(v, values) for v in obj]
    return obj


@register_adapter("http")
class HTTPAdapter(TTSAdapter):
    """
    adapter_args:
      url, method (POST), headers {k: v | "${ENV}"}, timeout,
      body: JSON template using {text} {language} {style} {ref_audio_base64} and any param name,
      response: {type: audio | json_base64, path: "audio_base64"},
      voice_cloning, languages, params: {name: {min, max, default, type}}
    """

    def __init__(
        self,
        url: str,
        body: dict | None = None,
        headers: dict[str, str] | None = None,
        method: str = "POST",
        timeout: float = 120.0,
        response: dict | None = None,
        voice_cloning: bool = False,
        languages: list[str] | None = None,
        params: dict[str, dict] | None = None,
        style_control: bool = False,
    ):
        self.url, self.method, self.timeout = url, method.upper(), timeout
        self.body = body or {"text": "{text}"}
        self.headers = headers or {}
        self.response = response or {"type": "audio"}
        self.capabilities = Capabilities(
            voice_cloning=voice_cloning,
            languages=tuple(languages) if languages else None,
            params={k: ParamSpec(**v) for k, v in (params or {}).items()},
            style_control=style_control,
        )

    def synthesize(self, req: SynthesisRequest) -> AudioResult:
        values: dict[str, Any] = {
            "text": req.text,
            "language": req.language,
            "style": req.style or "",
            **req.params,
        }
        if req.ref_audio is not None:
            values["ref_audio_base64"] = base64.b64encode(Path(req.ref_audio).read_bytes()).decode()
        headers = {k: _expand_env(v) for k, v in self.headers.items()}
        t0 = time.perf_counter()
        resp = requests.request(
            self.method,
            _expand_env(self.url),
            json=_fill(self.body, values),
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        if self.response.get("type") == "json_base64":
            data: Any = resp.json()
            for key in str(self.response.get("path", "audio_base64")).split("."):
                data = data[key]
            raw = base64.b64decode(data)
        else:
            raw = resp.content
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        return AudioResult(audio.mean(axis=1), sr, {"http_latency_s": time.perf_counter() - t0})
