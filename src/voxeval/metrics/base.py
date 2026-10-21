"""The Metric interface. A metric scores one generated clip and returns named fields."""

from __future__ import annotations

import importlib
from typing import Any, ClassVar

from ..types import Item, VoxevalError


class Metric:
    """Subclass, set the class vars, implement `score`.

    fields:     numeric outputs -> higher_is_better (True / False / None if neutral).
                Booleans are allowed and aggregate as rates.
    info:       non-numeric outputs (transcript, label, reason) kept per sample only.
    requires:   subset of {"ref_audio", "transcript"}; items missing one are skipped, not failed.
    depends_on: metrics that must be enabled and run first (hard dependency).
    after:      soft ordering only (use their outputs when present).
    """

    name: ClassVar[str] = "metric"
    fields: ClassVar[dict[str, bool | None]] = {}
    info: ClassVar[tuple[str, ...]] = ()
    requires: ClassVar[frozenset[str]] = frozenset()
    depends_on: ClassVar[tuple[str, ...]] = ()
    after: ClassVar[tuple[str, ...]] = ()
    license: ClassVar[str] = "unknown"
    extra: ClassVar[str] = ""  # pip extra that provides the dependencies
    description: ClassVar[str] = ""

    def __init__(self, **args: Any):
        self.args = args

    def setup(self, device: str) -> None:
        """Load models. Called once, only if there is something to score."""

    def score(self, item: Item) -> dict[str, Any]:
        raise NotImplementedError

    def teardown(self) -> None:
        """Free memory so the next metric's model can load."""


def lazy_import(module: str, extra: str):
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise VoxevalError(
            f"missing dependency '{module}'. Install with: pip install 'voxeval[{extra}]'"
        ) from e
