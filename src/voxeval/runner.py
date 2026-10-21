"""Orchestrates a run: GENERATE (adapter x variants) -> SCORE (metric by metric) -> AGGREGATE.

Generation and scoring are separate, cached stages, so an interrupted run resumes and
adding a metric later never re-synthesizes audio.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

import voxeval.metrics  # noqa: F401,E402  (registers built-in metrics)

from .adapters import resolve_adapter
from .aggregate import aggregate
from .audio import save_audio
from .config import RunConfig, resolve_metrics
from .metrics.base import Metric  # noqa: E402
from .registry import METRICS, load_object
from .rundir import RunDir, append_jsonl, capture_env, read_jsonl, resolve_device
from .suite import load_suite
from .sweep import Variant, expand_variants, validate_variants
from .types import Capabilities, Item, Prompt, SynthesisRequest, VoxevalError


def _log(msg: str) -> None:
    print(f"[voxeval] {msg}", file=sys.stderr, flush=True)


def _free_memory() -> None:
    gc.collect()
    if "torch" in sys.modules:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def resolve_metric_class(name: str) -> type[Metric]:
    if ":" in name:
        cls = load_object(name)
        if not (isinstance(cls, type) and issubclass(cls, Metric)):
            raise VoxevalError(f"{name} is not a Metric subclass")
        return cls
    return METRICS.get(name)


def order_metrics(classes: dict[str, type[Metric]]) -> list[str]:
    """Topological order honouring `depends_on` (hard, must be enabled) and `after` (soft)."""
    for name, cls in classes.items():
        for dep in cls.depends_on:
            if dep not in classes:
                raise VoxevalError(f"metric '{name}' needs '{dep}'; add it to `metrics`")
    owners: dict[str, str] = {}
    for name, cls in classes.items():
        for f in [*cls.fields, *cls.info]:
            if f in owners:
                raise VoxevalError(f"metrics '{owners[f]}' and '{name}' both emit field '{f}'")
            owners[f] = name
    ordered: list[str] = []
    visiting: set[str] = set()

    def visit(n: str) -> None:
        if n in ordered:
            return
        if n in visiting:
            raise VoxevalError(f"metric dependency cycle at '{n}'")
        visiting.add(n)
        for d in (*classes[n].depends_on, *(a for a in classes[n].after if a in classes)):
            visit(d)
        visiting.discard(n)
        ordered.append(n)

    for n in classes:
        visit(n)
    return ordered


def assign_voices(
    prompts: list[Prompt], voices_dir: str | None, caps: Capabilities
) -> dict[str, Path]:
    """Deterministically pick a reference clip per prompt (only if the adapter can clone)."""
    if not voices_dir or not caps.voice_cloning:
        return {p.id: p.ref_audio for p in prompts if p.ref_audio and caps.voice_cloning}
    root = Path(voices_dir).expanduser()
    if not root.is_dir():
        raise VoxevalError(f"voices dir not found: {root}  (try `voxeval fetch-voices`)")
    flat = sorted(root.glob("*.wav"))
    out: dict[str, Path] = {}
    counters: dict[str, int] = {}
    for p in prompts:
        if p.ref_audio:
            out[p.id] = p.ref_audio
            continue
        pool = sorted((root / p.language).glob("*.wav")) or flat
        if not pool:
            continue
        i = counters.get(p.language, 0)
        counters[p.language] = i + 1
        out[p.id] = pool[i % len(pool)]
    return out


def _peak_vram_reset() -> bool:
    if "torch" not in sys.modules:
        return False
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        return True
    return False


def generate(
    run: RunDir,
    cfg: RunConfig,
    adapter,
    variants: list[Variant],
    prompts: list[Prompt],
    voices: dict[str, Path],
) -> None:
    caps = adapter.capabilities
    done = {
        r["sample_id"]
        for r in read_jsonl(run.generation)
        if r.get("status") == "ok" and (run.root / r["wav"]).exists()
    }
    todo = [(v, p) for v in variants for p in prompts if f"{v.name}/{p.id}" not in done]
    if not todo:
        _log("generation: everything cached")
        return
    adapter.setup()
    try:
        for v, p in tqdm(todo, desc="generate", unit="clip", file=sys.stderr):
            sid = f"{v.name}/{p.id}"
            row: dict[str, Any] = {
                "sample_id": sid,
                "variant": v.name,
                "prompt_id": p.id,
                "language": p.language,
                "category": p.category,
                "style": p.style,
                "params": v.params,
                "ref_audio": str(voices[p.id]) if p.id in voices else None,
            }
            if caps.languages is not None and p.language not in caps.languages:
                append_jsonl(
                    run.generation,
                    {
                        **row,
                        "status": "skipped",
                        "reason": f"language '{p.language}' not supported",
                    },
                )
                continue
            req = SynthesisRequest(
                prompt_id=p.id,
                text=p.text,
                language=p.language,
                ref_audio=voices.get(p.id) if caps.voice_cloning else None,
                params=dict(v.params),
                style=p.style if caps.style_control else None,
                seed=cfg.seed,
                extra={**p.extra, "variant": v.name},
            )
            tracked = _peak_vram_reset()
            t0 = time.perf_counter()
            try:
                res = adapter.synthesize(req)
                latency = time.perf_counter() - t0
                wav = run.audio(v.name, p.id)
                save_audio(wav, res.audio, res.sample_rate)
                peak = None
                if tracked:
                    import torch

                    peak = round(torch.cuda.max_memory_allocated() / 2**20, 1)
                append_jsonl(
                    run.generation,
                    {
                        **row,
                        "status": "ok",
                        "wav": str(wav.relative_to(run.root)),
                        "sample_rate": res.sample_rate,
                        "duration_s": round(len(res.audio) / res.sample_rate, 4),
                        "latency_s": None if res.meta.get("precomputed") else round(latency, 4),
                        "peak_vram_mb": peak,
                    },
                )
            except Exception as e:  # one bad sample must never stop a run
                _log(f"generation failed for {sid}: {type(e).__name__}: {e}")
                append_jsonl(
                    run.generation, {**row, "status": "error", "error": f"{type(e).__name__}: {e}"}
                )
    finally:
        adapter.teardown()


def build_items(run: RunDir, prompts: list[Prompt]) -> list[Item]:
    by_id = {p.id: p for p in prompts}
    latest: dict[str, dict] = {}
    for g in read_jsonl(run.generation):
        latest[g["sample_id"]] = g  # last write wins (a retry supersedes an earlier error)
    items = []
    for g in latest.values():
        if g.get("status") != "ok":
            continue
        items.append(
            Item(
                sample_id=g["sample_id"],
                variant=g["variant"],
                prompt=by_id[g["prompt_id"]],
                wav_path=run.root / g["wav"],
                ref_audio=Path(g["ref_audio"]) if g.get("ref_audio") else None,
                gen=g,
            )
        )
    return items


def score(
    run: RunDir,
    cfg: RunConfig,
    items: list[Item],
    enabled: dict[str, dict],
    skipped: dict[str, str],
    device: str,
) -> dict[str, Any]:
    classes = {n: resolve_metric_class(n) for n in enabled}
    order = order_metrics(classes)
    status: dict[str, Any] = {n: {"status": "skipped", "detail": why} for n, why in skipped.items()}
    for name in order:
        cls = classes[name]
        info: dict[str, Any] = {
            "status": "ok",
            "license": cls.license,
            "fields": dict(cls.fields),
            "info": list(cls.info),
            "args": enabled[name],
            "description": cls.description,
        }
        cache = {r["sample_id"]: r for r in read_jsonl(run.scores(name))}
        pending = [it for it in items if it.sample_id not in cache]
        if pending:
            metric = cls(**enabled[name])
            t0 = time.perf_counter()
            try:
                metric.setup(device)
            except Exception as e:
                _log(f"metric '{name}' unavailable: {e}")
                info.update(status="failed", detail=str(e))
                status[name] = info
                continue
            n_err = 0
            try:
                for it in tqdm(pending, desc=name, unit="clip", file=sys.stderr):
                    missing = [
                        r
                        for r in cls.requires
                        if (r == "ref_audio" and it.ref_audio is None)
                        or (r == "transcript" and "transcript" not in it.results)
                    ]
                    if missing:
                        row = {
                            "sample_id": it.sample_id,
                            "skipped": f"missing {', '.join(missing)}",
                        }
                    else:
                        try:
                            row = {"sample_id": it.sample_id, "fields": metric.score(it)}
                        except Exception as e:
                            n_err += 1
                            row = {"sample_id": it.sample_id, "error": f"{type(e).__name__}: {e}"}
                            _log(f"{name} failed on {it.sample_id}: {row['error']}")
                    append_jsonl(run.scores(name), row)
                    cache[it.sample_id] = row
            finally:
                metric.teardown()
                del metric
                _free_memory()
            info["seconds"] = round(time.perf_counter() - t0, 2)
            info["n_error"] = n_err
        for it in items:
            row = cache.get(it.sample_id, {})
            it.results.update(row.get("fields", {}))
        info["n_scored"] = sum(1 for r in cache.values() if "fields" in r)
        info["n_skipped"] = sum(1 for r in cache.values() if "skipped" in r)
        status[name] = info
    return status


def run(cfg: RunConfig, run_dir: str | Path | None = None) -> Path:
    """Execute (or resume) a run. Returns the run directory."""
    prompts = load_suite(**cfg.suite.model_dump(), seed=cfg.seed)
    adapter_cls = resolve_adapter(cfg.adapter)
    try:
        adapter = adapter_cls(**cfg.adapter_args)
    except TypeError as e:
        raise VoxevalError(f"bad adapter_args for '{cfg.adapter}': {e}") from e
    variants = expand_variants(cfg.sweep, cfg.variants)
    validate_variants(variants, adapter.capabilities)
    enabled, skipped = resolve_metrics(cfg)
    order_metrics({n: resolve_metric_class(n) for n in enabled})  # fail fast on bad deps

    if run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(cfg.output_dir) / f"{cfg.name}-{stamp}"
        n = 1
        while run_dir.exists():  # never silently reuse another run's cache
            n += 1
            run_dir = Path(cfg.output_dir) / f"{cfg.name}-{stamp}-{n}"
    run = RunDir(Path(run_dir))
    run.root.mkdir(parents=True, exist_ok=True)
    run.config.write_text(yaml.safe_dump(json.loads(cfg.model_dump_json()), sort_keys=False))
    device = resolve_device(cfg.device)
    run.env.write_text(json.dumps(capture_env(cfg.profile, device), indent=2))
    _log(
        f"run dir: {run.root}  | profile={cfg.profile} device={device} "
        f"| {len(prompts)} prompts x {len(variants)} variants"
    )

    voices = assign_voices(prompts, cfg.voices.dir if cfg.voices else None, adapter.capabilities)
    if cfg.voices and not adapter.capabilities.voice_cloning:
        _log("adapter has no voice cloning: reference voices ignored, speaker_similarity will skip")

    generate(run, cfg, adapter, variants, prompts, voices)
    items = build_items(run, prompts)
    if not items:
        raise VoxevalError(f"no clips were generated; see {run.generation}")
    status = score(run, cfg, items, enabled, skipped, device)
    run.meta.write_text(
        json.dumps(
            {
                "name": cfg.name,
                "profile": cfg.profile,
                "metrics": status,
                "n_prompts": len(prompts),
                "variants": [v.name for v in variants],
            },
            indent=2,
        )
    )
    aggregate(run.root)
    return run.root
