"""Bring-your-own-audio: score WAVs that already exist on disk.

Works for any TTS, closed or local: generate the clips however you like, then point
voxeval at them. Files are found, in order, from: the prompt's `audio` field,
`<root>/<variant dir>/<prompt id>.wav`, or `<root>/<prompt id>.wav`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..audio import load_audio
from ..registry import register_adapter
from ..types import AudioResult, Capabilities, SynthesisRequest, VoxevalError
from .base import TTSAdapter


@register_adapter("folder")
class FolderAdapter(TTSAdapter):
    def __init__(
        self,
        root: str,
        variant_dirs: bool = False,
        voice_cloning: bool = False,
        languages: list[str] | None = None,
    ):
        self.root = Path(root).expanduser()
        self.variant_dirs = variant_dirs
        self.capabilities = Capabilities(
            voice_cloning=voice_cloning, languages=tuple(languages) if languages else None
        )

    def setup(self) -> None:
        if not self.root.is_dir():
            raise VoxevalError(f"folder adapter: root does not exist: {self.root}")

    def synthesize(self, req: SynthesisRequest) -> AudioResult:
        candidates: list[Path] = []
        if req.extra.get("audio"):
            p = Path(req.extra["audio"])
            candidates.append(p if p.is_absolute() else self.root / p)
        if self.variant_dirs and req.extra.get("variant"):
            candidates.append(self.root / req.extra["variant"] / f"{req.prompt_id}.wav")
        candidates.append(self.root / f"{req.prompt_id}.wav")
        for path in candidates:
            if path.exists():
                audio, sr = load_audio(path)
                meta: dict[str, Any] = {"source": str(path), "precomputed": True}
                return AudioResult(audio, sr, meta)
        raise FileNotFoundError(f"no audio for prompt '{req.prompt_id}' (tried {candidates})")
