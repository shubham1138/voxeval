"""The one interface every TTS system plugs into."""

from __future__ import annotations

from typing import ClassVar

from ..types import AudioResult, Capabilities, SynthesisRequest


class TTSAdapter:
    """Subclass this, declare `capabilities`, implement `synthesize`.

    The runner only sends what `capabilities` declares: no ref audio unless
    `voice_cloning`, no params that are not in `params`, no style unless
    `style_control`, and no languages outside `languages`.
    """

    name: ClassVar[str] = "adapter"
    capabilities: Capabilities = Capabilities()

    def setup(self) -> None:
        """Load models / open connections. Called once before generation."""

    def synthesize(self, req: SynthesisRequest) -> AudioResult:
        raise NotImplementedError

    def teardown(self) -> None:
        """Release resources. Called once after generation."""
