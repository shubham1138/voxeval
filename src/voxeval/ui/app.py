"""Streamlit viewer for voxeval runs:  voxeval ui runs/"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

parser = argparse.ArgumentParser()
parser.add_argument("--runs-dir", default="runs")
ARGS, _ = parser.parse_known_args(sys.argv[1:])

st.set_page_config(page_title="voxeval", layout="wide")


@st.cache_data
def load_run(path: str) -> dict:
    p = Path(path)
    out = {"path": path, "meta": {}, "summary": {}, "samples": pd.DataFrame(), "env": {}}
    for key, fname in (("meta", "run_meta.json"), ("summary", "summary.json"), ("env", "env.json")):
        if (p / fname).exists():
            out[key] = json.loads((p / fname).read_text())
    if (p / "samples.csv").exists():
        out["samples"] = pd.read_csv(p / "samples.csv")
    return out


runs_root = Path(st.sidebar.text_input("Runs folder", ARGS.runs_dir))
run_dirs = sorted([d for d in runs_root.glob("*") if (d / "summary.json").exists()], reverse=True)
st.title("voxeval")
if not run_dirs:
    st.info(f"No finished runs in `{runs_root}`. Create one with `voxeval run <config.yaml>`.")
    st.stop()

chosen = st.sidebar.multiselect("Runs", [d.name for d in run_dirs], default=[run_dirs[0].name])
if not chosen:
    st.stop()
runs = [load_run(str(runs_root / n)) for n in chosen]
profiles = {r["meta"].get("profile") for r in runs}
if len(profiles) > 1:
    st.warning(
        f"These runs used different profiles ({', '.join(map(str, profiles))}): "
        "ASR/MOS models differ, so absolute numbers are not directly comparable."
    )

# ---- build a combined view: variant label = run::variant (run prefix only when comparing runs)
rows, samples, higher = [], [], {}
for name, r in zip(chosen, runs, strict=True):
    higher.update(r["summary"].get("higher_is_better", {}))
    for variant, v in r["summary"]["variants"].items():
        label = f"{name}::{variant}" if len(runs) > 1 else variant
        for f, s in v["overall"].items():
            rows.append({"variant": label, "field": f, **s})
    df = r["samples"].copy()
    if not df.empty:
        df["variant"] = df["variant"].map(lambda x, n=name: f"{n}::{x}" if len(runs) > 1 else x)
        df["run"] = name
        samples.append(df)
overall = pd.DataFrame(rows)
samples = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()

tab_over, tab_gates, tab_break, tab_samples, tab_run = st.tabs(
    ["Compare variants", "Gates", "Breakdown", "Listen / worst samples", "Run info"]
)

with tab_over:
    fields = sorted(overall["field"].unique()) if not overall.empty else []
    default = [f for f in ("composite", "wer", "mos", "sim") if f in fields]
    pick = st.selectbox("Metric", fields, index=fields.index(default[0]) if default else 0)
    d = overall[overall["field"] == pick]
    base = alt.Chart(d).encode(y=alt.Y("variant:N", sort=None, title=None))
    chart = base.mark_bar().encode(x=alt.X("mean:Q", title=pick)) + base.mark_errorbar().encode(
        x=alt.X("ci_low:Q"), x2="ci_high:Q"
    )
    st.altair_chart(chart, width="stretch")
    hb = higher.get(pick)
    st.caption(
        "Bars show the mean with a 95% bootstrap CI over samples. "
        + ("Higher is better." if hb else "Lower is better." if hb is False else "")
    )
    wide = overall.assign(
        txt=lambda x: x.apply(
            lambda r: f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]", axis=1
        )
    ).pivot(index="variant", columns="field", values="txt")
    st.dataframe(wide, width="stretch")

with tab_gates:
    g = []
    for name, r in zip(chosen, runs, strict=True):
        for variant, v in r["summary"]["variants"].items():
            for gate, s in v.get("gates", {}).items():
                g.append(
                    {
                        "variant": f"{name}::{variant}" if len(runs) > 1 else variant,
                        "gate": gate,
                        "pass_rate": s["pass_rate"],
                        "n": s["n"],
                    }
                )
    if g:
        gd = pd.DataFrame(g)
        st.dataframe(
            gd.pivot(index="variant", columns="gate", values="pass_rate").style.format("{:.0%}"),
            width="stretch",
        )
    else:
        st.info(
            "No gates configured. Add `gates:` to the run config, then `voxeval aggregate <run>`."
        )

with tab_break:
    by = st.radio("Group by", ["category", "language"], horizontal=True)
    field = st.selectbox("Metric ", sorted(higher), key="bk")
    if not samples.empty and field in samples:
        ok = samples[samples["status"] == "ok"]
        st.dataframe(
            ok.pivot_table(index=by, columns="variant", values=field, aggfunc="mean").round(3),
            width="stretch",
        )
        st.caption(
            "Means per group. Groups are small in quick runs: check n before drawing conclusions."
        )
        st.dataframe(ok.groupby([by, "variant"]).size().unstack(), width="stretch")

with tab_samples:
    if samples.empty:
        st.info("No samples.")
    else:
        ok = samples[samples["status"] == "ok"]
        c1, c2, c3, c4 = st.columns(4)
        field = c1.selectbox("Sort by", sorted(higher))
        order = c2.radio("Show", ["worst first", "best first"], horizontal=True)
        variant = c3.selectbox("Variant", ["(all)"] + sorted(ok["variant"].unique()))
        n = c4.slider("How many", 3, 30, 8)
        view = ok if variant == "(all)" else ok[ok["variant"] == variant]
        bad_is_high = higher.get(field) is False
        asc = (order == "worst first") != bad_is_high
        view = view.dropna(subset=[field]).sort_values(field, ascending=asc).head(n)
        for _, row in view.iterrows():
            st.markdown(
                f"**{row['sample_id']}** · {row['category']} · {row['language']} · "
                f"`{field}={row[field]:.3f}`"
            )
            st.audio(str(runs_root / row["run"] / row["wav"]))
            extra = {
                k: row[k]
                for k in ("transcript", "judge_reason", "emotion_label")
                if k in row and pd.notna(row[k])
            }
            if extra:
                st.json(extra, expanded=False)

with tab_run:
    for name, r in zip(chosen, runs, strict=True):
        st.subheader(name)
        st.json(r["env"], expanded=False)
        mt = pd.DataFrame(r["meta"].get("metrics", {})).T
        keep = [
            c
            for c in ("status", "detail", "license", "seconds", "n_scored", "n_skipped", "n_error")
            if c in mt
        ]
        st.dataframe(mt[keep] if keep else mt, width="stretch")
        skipped = mt[mt["status"] == "skipped"] if "status" in mt else pd.DataFrame()
        if len(skipped):
            st.caption("Skipped metrics are listed with the reason in `detail`.")
