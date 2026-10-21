from ..registry import ADAPTERS, load_object
from ..types import VoxevalError
from . import folder, http  # noqa: F401  (register built-ins)
from .base import TTSAdapter


def resolve_adapter(spec: str) -> type[TTSAdapter]:
    """A registered name ('folder') or an import path ('pkg.mod:Class' / 'file.py:Class')."""
    if ":" in spec:
        cls = load_object(spec)
        if not (isinstance(cls, type) and issubclass(cls, TTSAdapter)):
            raise VoxevalError(f"{spec} is not a TTSAdapter subclass")
        return cls
    return ADAPTERS.get(spec)


__all__ = ["TTSAdapter", "resolve_adapter"]
