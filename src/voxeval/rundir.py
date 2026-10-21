"""On-disk layout of a run, JSONL helpers and environment capture."""

from __future__ import annotations

import json
import platform
import re
import sys
from pathlib import Path
from typing import Any


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=,-]+", "_", s)


class RunDir:
    def __init__(self, root: Path):
        self.root = Path(root)

    @property
    def config(self) -> Path:
        return self.root / "config.resolved.yaml"

    @property
    def env(self) -> Path:
        return self.root / "env.json"

    @property
    def meta(self) -> Path:
        return self.root / "run_meta.json"

    @property
    def generation(self) -> Path:
        return self.root / "generation.jsonl"

    @property
    def samples_csv(self) -> Path:
        return self.root / "samples.csv"

    @property
    def summary_csv(self) -> Path:
        return self.root / "summary.csv"

    @property
    def summary_json(self) -> Path:
        return self.root / "summary.json"

    def scores(self, metric: str) -> Path:
        return self.root / "scores" / f"{slug(metric)}.jsonl"

    def audio(self, variant: str, prompt_id: str) -> Path:
        return self.root / "audio" / slug(variant) / f"{slug(prompt_id)}.wav"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn last line from an interrupted run
    return out


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def capture_env(profile: str, device: str) -> dict[str, Any]:
    env: dict[str, Any] = {
        "profile": profile,
        "device": device,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": __import__("os").cpu_count(),
    }
    try:
        from importlib.metadata import version

        env["voxeval"] = version("voxeval")
    except Exception:
        pass
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except Exception:
        env["torch"] = None
    return env


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
