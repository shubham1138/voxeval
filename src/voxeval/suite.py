"""Prompt suites: built-in (by name) or user files (YAML / JSONL / CSV)."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import yaml

from .types import Prompt, VoxevalError

BUILTIN_DIR = Path(__file__).parent / "suites"
_KNOWN = {"id", "text", "language", "category", "style", "asr_reference", "ref_audio"}


def builtin_suites() -> list[str]:
    return sorted(p.stem for p in BUILTIN_DIR.glob("*.yaml"))


def _rows(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
        return data["prompts"] if isinstance(data, dict) else data
    if suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if suffix == ".csv":
        return list(csv.DictReader(text.splitlines()))
    raise VoxevalError(f"unsupported suite format: {path}")


def _to_prompt(row: dict, default_lang: str, i: int, base: Path) -> Prompt:
    if not row.get("text"):
        raise VoxevalError(f"suite row {i} has no 'text': {row}")
    ref = row.get("ref_audio")
    if ref:
        ref = Path(ref)
        ref = ref if ref.is_absolute() else base / ref
    return Prompt(
        id=str(row.get("id") or f"p{i:04d}"),
        text=row["text"],
        language=row.get("language") or default_lang,
        category=row.get("category") or "general",
        style=row.get("style") or None,
        asr_reference=row.get("asr_reference") or None,
        ref_audio=ref or None,
        extra={k: v for k, v in row.items() if k not in _KNOWN and v not in (None, "")},
    )


def load_suite(
    name: str | None = None,
    path: str | None = None,
    languages: list[str] | None = None,
    categories: list[str] | None = None,
    limit: int | None = None,
    per_category: int | None = None,
    seed: int = 0,
) -> list[Prompt]:
    if bool(name) == bool(path):
        raise VoxevalError("suite needs exactly one of `name` (built-in) or `path`")
    prompts: list[Prompt] = []
    if name:
        # a built-in name may be a base name ('core') or one file ('en_core')
        files = [BUILTIN_DIR / f"{name}.yaml"]
        if not files[0].exists():
            files = sorted(BUILTIN_DIR.glob(f"*_{name}.yaml"))
        if not files:
            raise VoxevalError(f"unknown built-in suite '{name}'. Available: {builtin_suites()}")
    else:
        files = [Path(path).expanduser()]
        if not files[0].exists():
            raise VoxevalError(f"suite file not found: {files[0]}")
    for f in files:
        default_lang = (
            f.stem.split("_")[0] if f.stem.split("_")[0] in {"en", "es", "fr", "de"} else "en"
        )
        for i, row in enumerate(_rows(f)):
            prompts.append(_to_prompt(row, default_lang, i, f.parent))

    ids = [p.id for p in prompts]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise VoxevalError(f"duplicate prompt ids in suite: {dupes[:5]}")
    if languages:
        prompts = [p for p in prompts if p.language in languages]
    if categories:
        prompts = [p for p in prompts if p.category in categories]
    if per_category:
        seen: dict[tuple[str, str], int] = {}
        kept = []
        for p in prompts:
            key = (p.language, p.category)
            if seen.get(key, 0) < per_category:
                kept.append(p)
                seen[key] = seen.get(key, 0) + 1
        prompts = kept
    if limit and len(prompts) > limit:
        rng = random.Random(seed)
        prompts = sorted(rng.sample(prompts, limit), key=lambda p: ids.index(p.id))
    if not prompts:
        raise VoxevalError("suite is empty after applying filters")
    return prompts
