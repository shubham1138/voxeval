"""Small name -> object registry with entry-point discovery for third-party plugins."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from .types import VoxevalError


class Registry:
    def __init__(self, kind: str, entry_point_group: str):
        self.kind = kind
        self.group = entry_point_group
        self._items: dict[str, Any] = {}
        self._discovered = False

    def register(self, name: str, obj: Any, override: bool = False) -> Any:
        if name in self._items and not override and self._items[name] is not obj:
            raise VoxevalError(f"{self.kind} '{name}' is already registered")
        self._items[name] = obj
        return obj

    def _discover(self) -> None:
        if self._discovered:
            return
        self._discovered = True
        for ep in metadata.entry_points(group=self.group):
            try:
                self._items.setdefault(ep.name, ep.load())
            except Exception as e:  # a broken plugin must not break the core
                print(f"[voxeval] could not load plugin {ep.name}: {e}", file=sys.stderr)

    def get(self, name: str) -> Any:
        if name not in self._items:
            self._discover()
        if name not in self._items:
            raise VoxevalError(
                f"unknown {self.kind} '{name}'. Available: {', '.join(self.names()) or '(none)'}"
            )
        return self._items[name]

    def names(self) -> list[str]:
        self._discover()
        return sorted(self._items)


ADAPTERS = Registry("adapter", "voxeval.adapters")
METRICS = Registry("metric", "voxeval.metrics")


def register_adapter(name: str):
    def deco(cls):
        cls.name = name
        ADAPTERS.register(name, cls)
        return cls

    return deco


def register_metric(name: str):
    def deco(cls):
        cls.name = name
        METRICS.register(name, cls)
        return cls

    return deco


def load_object(spec: str) -> Any:
    """Resolve 'pkg.module:Attr' or '/path/to/file.py:Attr' to a Python object."""
    if ":" not in spec:
        raise VoxevalError(f"import path '{spec}' must look like 'module:Class' or 'file.py:Class'")
    mod_part, attr = spec.rsplit(":", 1)
    if mod_part.endswith(".py") or "/" in mod_part:
        path = Path(mod_part).expanduser().resolve()
        if not path.exists():
            raise VoxevalError(f"adapter file not found: {path}")
        mod_name = f"_voxeval_ext_{path.stem}"
        spec_obj = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec_obj)
        sys.modules[mod_name] = module
        spec_obj.loader.exec_module(module)
    else:
        if "" not in sys.path and str(Path.cwd()) not in sys.path:
            sys.path.insert(0, str(Path.cwd()))
        module = importlib.import_module(mod_part)
    try:
        return getattr(module, attr)
    except AttributeError as e:
        raise VoxevalError(f"'{mod_part}' has no attribute '{attr}'") from e
