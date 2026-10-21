import json

import pandas as pd
import pytest
import yaml
from conftest import SR, StubTone, tone

from voxeval.aggregate import aggregate
from voxeval.audio import save_audio
from voxeval.config import RunConfig
from voxeval.runner import run
from voxeval.types import VoxevalError


def _cfg(tmp_path, suite_file, **kw):
    base = dict(
        name="t",
        adapter="stub_tone",
        suite={"path": str(suite_file)},
        metrics={"signal": {}, "efficiency": {}, "stub_len": {}, "stub_dep": {}},
        output_dir=str(tmp_path / "runs"),
    )
    base.update(kw)
    return RunConfig(**base)


def test_end_to_end_sweep_with_gates_and_composite(tmp_path, suite_file):
    cfg = _cfg(
        tmp_path,
        suite_file,
        sweep={"pitch": [150, 300]},
        gates=[
            {"field": "duration_s", "op": ">=", "value": 1.0},
            {"field": "is_empty", "op": "==", "value": 0},
        ],
    )
    out = run(cfg)
    gen = [json.loads(line) for line in (out / "generation.jsonl").read_text().splitlines()]
    assert len(gen) == 6 and all(g["status"] == "ok" for g in gen)  # 3 prompts x 2 variants
    df = pd.read_csv(out / "samples.csv")
    assert set(df["variant"]) == {"pitch=150", "pitch=300"}
    assert (df["double"] == df["n_chars"] * 2).all()  # dependency ran first and was visible
    summary = json.loads((out / "summary.json").read_text())
    v = summary["variants"]["pitch=150"]
    assert v["generation"] == {"total": 3, "ok": 3, "error": 0, "skipped": 0}
    stat = v["overall"]["duration_s"]
    assert stat["ci_low"] <= stat["mean"] <= stat["ci_high"] and stat["n"] == 3
    assert v["gates"]["is_empty==0"]["pass_rate"] == 1.0
    assert "pitch" in v["params"] and set(v["by_category"]) == {"short", "long"}
    assert (out / "env.json").exists() and (out / "config.resolved.yaml").exists()
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["metrics"]["signal"]["n_scored"] == 6


def test_resume_does_not_regenerate_or_rescore(tmp_path, suite_file):
    cfg = _cfg(tmp_path, suite_file)
    out = run(cfg)
    first = StubTone.calls
    assert first == 3
    run(cfg, run_dir=out)
    assert StubTone.calls == first  # everything cached
    # adding a metric later scores new things without regenerating audio
    cfg2 = _cfg(tmp_path, suite_file, metrics={"signal": {}, "stub_len": {}, "stub_ref": {}})
    run(cfg2, run_dir=out)
    assert StubTone.calls == first
    assert (out / "scores" / "stub_ref.jsonl").exists()


def test_capability_gating_no_cloning_means_no_ref_and_skipped_metric(tmp_path, suite_file):
    voices = tmp_path / "voices"
    voices.mkdir()
    save_audio(voices / "v1.wav", tone(1.0), SR)
    cfg = _cfg(
        tmp_path, suite_file, voices={"dir": str(voices)}, metrics={"signal": {}, "stub_ref": {}}
    )
    out = run(cfg)
    meta = json.loads((out / "run_meta.json").read_text())
    assert (
        meta["metrics"]["stub_ref"]["n_skipped"] == 3
        and meta["metrics"]["stub_ref"]["n_scored"] == 0
    )
    cloned = _cfg(
        tmp_path,
        suite_file,
        adapter_args={"cloning": True},
        voices={"dir": str(voices)},
        metrics={"signal": {}, "stub_ref": {}},
    )
    out2 = run(cloned)
    meta2 = json.loads((out2 / "run_meta.json").read_text())
    assert meta2["metrics"]["stub_ref"]["n_scored"] == 3


def test_unsupported_language_is_skipped_not_crashed(tmp_path, suite_file):
    out = run(_cfg(tmp_path, suite_file, adapter_args={"langs": ["en"]}))
    summary = json.loads((out / "summary.json").read_text())
    assert summary["variants"]["default"]["generation"]["skipped"] == 1
    assert summary["variants"]["default"]["generation"]["ok"] == 2


def test_one_failing_sample_does_not_stop_the_run(tmp_path, suite_file):
    out = run(_cfg(tmp_path, suite_file, adapter_args={"fail_on": "longer"}))
    gen = (out / "generation.jsonl").read_text()
    assert gen.count('"status": "error"') == 1
    assert len(pd.read_csv(out / "samples.csv")) == 3


def test_failed_metric_setup_is_reported_not_fatal(tmp_path, suite_file):
    out = run(_cfg(tmp_path, suite_file, metrics={"signal": {}, "stub_broken": {}}))
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["metrics"]["signal"]["status"] == "ok"
    assert meta["metrics"]["stub_broken"]["status"] == "failed"
    assert "unavailable" in meta["metrics"]["stub_broken"]["detail"]
    assert (out / "samples.csv").exists()


def test_undeclared_sweep_param_fails_before_any_generation(tmp_path, suite_file):
    with pytest.raises(VoxevalError, match="not declared"):
        run(_cfg(tmp_path, suite_file, sweep={"nope": [1]}))
    assert StubTone.calls == 0


def test_reaggregate_after_changing_gates(tmp_path, suite_file):
    out = run(_cfg(tmp_path, suite_file))
    conf = yaml.safe_load((out / "config.resolved.yaml").read_text())
    conf["gates"] = [{"field": "duration_s", "op": "<=", "value": 0.1}]
    (out / "config.resolved.yaml").write_text(yaml.safe_dump(conf))
    s = aggregate(out)
    assert s["variants"]["default"]["gates"]["duration_s<=0.1"]["pass_rate"] == 0.0


def test_folder_adapter_scores_existing_audio(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    save_audio(audio / "x1.wav", tone(1.5), SR)
    save_audio(audio / "x2.wav", tone(0.5, amp=1.5), SR)  # clipped
    suite = tmp_path / "s.jsonl"
    suite.write_text('{"id": "x1", "text": "first"}\n{"id": "x2", "text": "second"}\n')
    cfg = RunConfig(
        name="byo",
        adapter="folder",
        adapter_args={"root": str(audio)},
        suite={"path": str(suite)},
        metrics=["signal", "efficiency"],
        output_dir=str(tmp_path / "runs"),
    )
    out = run(cfg)
    df = pd.read_csv(out / "samples.csv").set_index("prompt_id")
    assert df.loc["x2", "clipping_ratio"] > 0.1 and df.loc["x1", "clipping_ratio"] == 0
    assert df["latency_s"].isna().all()  # no fake latency for pre-generated audio


def test_runs_started_in_the_same_second_get_separate_dirs(tmp_path, suite_file):
    a, b = run(_cfg(tmp_path, suite_file)), run(_cfg(tmp_path, suite_file))
    assert a != b and StubTone.calls == 6
