import numpy as np
import pandas as pd
import pytest

from voxeval.aggregate import DEFAULT_ANCHORS, bootstrap_ci, composite_score
from voxeval.config import RunConfig, resolve_metrics
from voxeval.registry import load_object
from voxeval.runner import order_metrics, resolve_metric_class
from voxeval.suite import load_suite
from voxeval.sweep import expand_variants, validate_variants
from voxeval.types import Capabilities, ParamSpec, VoxevalError


def test_sweep_grid_expands_and_names_variants():
    vs = expand_variants({"a": [1, 2], "b": [0.1, 0.5]}, None)
    assert len(vs) == 4
    assert {v.name for v in vs} == {"a=1,b=0.1", "a=1,b=0.5", "a=2,b=0.1", "a=2,b=0.5"}
    assert expand_variants(None, None)[0].name == "default"


def test_sweep_and_variants_are_exclusive():
    with pytest.raises(VoxevalError):
        expand_variants({"a": [1]}, [{"name": "x"}])


def test_sweep_rejects_undeclared_and_out_of_range_params():
    caps = Capabilities(params={"temp": ParamSpec(type="float", min=0, max=1)})
    validate_variants(expand_variants({"temp": [0.2, 1.0]}, None), caps)
    with pytest.raises(VoxevalError, match="not declared"):
        validate_variants(expand_variants({"top_k": [1]}, None), caps)
    with pytest.raises(VoxevalError, match="above max"):
        validate_variants(expand_variants({"temp": [1.5]}, None), caps)


def test_suite_filters(suite_file):
    assert len(load_suite(path=str(suite_file))) == 3
    assert [p.id for p in load_suite(path=str(suite_file), languages=["es"])] == ["a3"]
    assert len(load_suite(path=str(suite_file), per_category=1)) == 3
    assert len(load_suite(path=str(suite_file), limit=2)) == 2
    with pytest.raises(VoxevalError):
        load_suite(path=str(suite_file), categories=["nope"])


def test_builtin_suites_have_unique_ids_and_cover_languages():
    prompts = load_suite(name="core")
    assert {p.language for p in prompts} == {"en", "es", "fr", "de"}
    assert len({p.id for p in prompts}) == len(prompts)


def test_bootstrap_ci_brackets_the_mean_and_handles_edges():
    rng = np.random.default_rng(1)
    v = rng.normal(3.0, 0.5, 200)
    mean, lo, hi = bootstrap_ci(v)
    assert lo < mean < hi and hi - lo < 0.3
    assert bootstrap_ci(np.array([2.0])) == (2.0, 2.0, 2.0)
    assert all(np.isnan(bootstrap_ci(np.array([np.nan]))))


def test_composite_renormalises_over_present_fields():
    w = {"wer": 0.5, "mos": 0.5}
    full = composite_score(pd.Series({"wer": 0.0, "mos": 5.0}), w, DEFAULT_ANCHORS)
    assert full == pytest.approx(100.0)
    worst = composite_score(pd.Series({"wer": 0.2, "mos": 1.0}), w, DEFAULT_ANCHORS)
    assert worst == pytest.approx(0.0)
    only_mos = composite_score(pd.Series({"wer": np.nan, "mos": 3.0}), w, DEFAULT_ANCHORS)
    assert only_mos == pytest.approx(50.0)  # (3-1)/4 = 0.5, weight renormalised
    assert np.isnan(composite_score(pd.Series({"x": 1}), w, DEFAULT_ANCHORS))


def test_metric_ordering_and_errors():
    classes = {n: resolve_metric_class(n) for n in ["stub_dep", "stub_len"]}
    assert order_metrics(classes) == ["stub_len", "stub_dep"]
    with pytest.raises(VoxevalError, match="needs 'stub_len'"):
        order_metrics({"stub_dep": resolve_metric_class("stub_dep")})


def test_profiles_skip_judge_and_optin_emotion_on_laptop():
    cfg = RunConfig(adapter="folder", suite={"name": "core"})
    enabled, skipped = resolve_metrics(cfg)
    assert "judge" in skipped and "emotion" in skipped and "GPU" in skipped["judge"]
    assert enabled["intelligibility"]["model"] == "large-v3-turbo"
    gpu, gskipped = resolve_metrics(
        RunConfig(adapter="folder", suite={"name": "core"}, profile="gpu")
    )
    assert "judge" in gpu and gpu["intelligibility"]["model"] == "large-v3"
    listed, _ = resolve_metrics(
        RunConfig(adapter="folder", suite={"name": "core"}, metrics=["emotion", "signal"])
    )
    assert set(listed) == {"emotion", "signal"}


def test_user_args_override_profile_defaults():
    cfg = RunConfig(
        adapter="folder", suite={"name": "core"}, metrics={"intelligibility": {"model": "small"}}
    )
    assert resolve_metrics(cfg)[0]["intelligibility"]["model"] == "small"
    assert resolve_metrics(cfg)[0]["intelligibility"]["compute_type"] == "int8"


def test_import_path_loading(tmp_path):
    f = tmp_path / "myadapter.py"
    f.write_text("class Foo:\n    pass\n")
    assert load_object(f"{f}:Foo").__name__ == "Foo"
    with pytest.raises(VoxevalError):
        load_object("no_colon_here")
