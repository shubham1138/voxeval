"""Stage 3: per-sample table, bootstrap CIs, pass/fail gates and the optional composite."""

from __future__ import annotations

import json
import operator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .config import RunConfig
from .rundir import RunDir, read_jsonl

# field -> (direction, lo, hi). score = clip((x - lo) / (hi - lo), 0, 1), inverted if lower-better.
DEFAULT_ANCHORS: dict[str, list] = {
    "wer": ["lower", 0.0, 0.2],
    "cer": ["lower", 0.0, 0.2],
    "mos": ["higher", 1.0, 5.0],
    "sim": ["higher", 0.2, 0.8],
    "judge_overall": ["higher", 1.0, 5.0],
}
DEFAULT_WEIGHTS = {"wer": 0.3, "mos": 0.3, "sim": 0.2, "judge_overall": 0.2}
_OPS = {
    "<=": operator.le,
    ">=": operator.ge,
    "<": operator.lt,
    ">": operator.gt,
    "==": operator.eq,
    "!=": operator.ne,
}
N_BOOT = 1000


def bootstrap_ci(
    values: np.ndarray, n_boot: int = N_BOOT, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and 95% percentile-bootstrap CI over samples."""
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(v.mean())
    if len(v) < 2:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return mean, float(lo), float(hi)


def composite_score(row: pd.Series, weights: dict[str, float], anchors: dict[str, list]) -> float:
    num = den = 0.0
    for field, w in weights.items():
        x = row.get(field)
        if field not in anchors or x is None or pd.isna(x):
            continue
        direction, lo, hi = anchors[field]
        s = float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))
        num += w * (1.0 - s if direction == "lower" else s)
        den += w
    return 100.0 * num / den if den else float("nan")


def build_samples(
    run: RunDir, meta: dict, cfg: RunConfig
) -> tuple[pd.DataFrame, dict[str, bool | None]]:
    gen = read_jsonl(run.generation)
    rows: dict[str, dict[str, Any]] = {}
    for g in gen:
        r = {
            k: g.get(k)
            for k in (
                "sample_id",
                "variant",
                "prompt_id",
                "language",
                "category",
                "style",
                "ref_audio",
                "status",
                "error",
                "reason",
                "wav",
            )
        }
        r["params"] = json.dumps(g.get("params") or {}, sort_keys=True)
        rows[g["sample_id"]] = r
    higher: dict[str, bool | None] = {}
    for name, m in (meta.get("metrics") or {}).items():
        higher.update(m.get("fields", {}))
        for s in read_jsonl(run.scores(name)):
            if s["sample_id"] in rows and "fields" in s:
                rows[s["sample_id"]].update(s["fields"])
    df = pd.DataFrame(list(rows.values()))
    for f in higher:
        if f in df:
            df[f] = pd.to_numeric(
                df[f].map(lambda x: float(x) if isinstance(x, bool) else x), errors="coerce"
            )
    if cfg.composite.enabled and not df.empty:
        weights = cfg.composite.weights or {k: v for k, v in DEFAULT_WEIGHTS.items() if k in df}
        anchors = {**DEFAULT_ANCHORS, **(cfg.composite.anchors or {})}
        if weights:
            df["composite"] = df.apply(composite_score, axis=1, weights=weights, anchors=anchors)
            higher["composite"] = True
    for i, gate in enumerate(cfg.gates):
        col = f"gate:{gate.field}{gate.op}{gate.value:g}"
        if gate.field in df:
            ok = _OPS[gate.op](df[gate.field], gate.value)
            df[col] = ok.where(df[gate.field].notna(), other=np.nan)
    return df, higher


def aggregate(run_dir: str | Path) -> dict[str, Any]:
    run = RunDir(Path(run_dir))
    cfg = RunConfig(**yaml.safe_load(run.config.read_text()))
    meta = json.loads(run.meta.read_text()) if run.meta.exists() else {}
    df, higher = build_samples(run, meta, cfg)
    df.to_csv(run.samples_csv, index=False)
    if df.empty:
        summary: dict[str, Any] = {"variants": {}}
        run.summary_json.write_text(json.dumps(summary, indent=2))
        pd.DataFrame().to_csv(run.summary_csv, index=False)
        return summary

    fields = [f for f in higher if f in df]
    gate_cols = [c for c in df if c.startswith("gate:")]
    long_rows: list[dict[str, Any]] = []
    summary = {"variants": {}, "higher_is_better": higher}

    def stats(sub: pd.DataFrame) -> dict[str, dict[str, float]]:
        out = {}
        for f in fields:
            vals = sub[f].to_numpy(dtype=float)
            if np.isnan(vals).all():
                continue
            mean, lo, hi = bootstrap_ci(vals)
            out[f] = {"mean": mean, "ci_low": lo, "ci_high": hi, "n": int((~np.isnan(vals)).sum())}
        return out

    for variant, vdf in df.groupby("variant", sort=False):
        ok = vdf[vdf["status"] == "ok"]
        entry: dict[str, Any] = {
            "params": json.loads(vdf["params"].iloc[0]),
            "generation": {
                "total": int(len(vdf)),
                "ok": int(len(ok)),
                "error": int((vdf["status"] == "error").sum()),
                "skipped": int((vdf["status"] == "skipped").sum()),
            },
            "overall": stats(ok),
            "gates": {
                c[5:]: {"pass_rate": float(ok[c].mean()), "n": int(ok[c].notna().sum())}
                for c in gate_cols
                if ok[c].notna().any()
            },
            "by_category": {},
            "by_language": {},
        }
        for level, col in (("category", "category"), ("language", "language")):
            for group, gdf in ok.groupby(col):
                s = stats(gdf)
                entry[f"by_{level}"][str(group)] = s
                for f, v in s.items():
                    long_rows.append(
                        {"variant": variant, "level": level, "group": group, "field": f, **v}
                    )
        for f, v in entry["overall"].items():
            long_rows.append(
                {"variant": variant, "level": "overall", "group": "all", "field": f, **v}
            )
        summary["variants"][str(variant)] = entry

    pd.DataFrame(long_rows).to_csv(run.summary_csv, index=False)
    run.summary_json.write_text(json.dumps(summary, indent=2, default=float))
    return summary
