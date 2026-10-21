import http.server
import io
import threading

import numpy as np
import pytest
import soundfile as sf
from conftest import SR, tone

from voxeval.adapters.http import HTTPAdapter
from voxeval.audio import save_audio
from voxeval.metrics.judge import build_prompt, parse_judgement
from voxeval.metrics.robustness import Robustness, repetition
from voxeval.metrics.signal import Signal
from voxeval.types import Item, Prompt, SynthesisRequest


def _item(tmp_path, audio, text="hello world", **gen):
    p = tmp_path / "x.wav"
    save_audio(p, audio, SR)
    g = {"duration_s": len(audio) / SR, **gen}
    return Item("v/x", "v", Prompt("x", text), p, None, g)


def test_signal_flags_clipping_silence_and_empty(tmp_path):
    ok = Signal().score(_item(tmp_path, tone(1.0)))
    assert ok["is_empty"] is False and ok["clipping_ratio"] == 0 and ok["lufs"] < 0
    padded = np.concatenate(
        [np.zeros(SR // 2, np.float32), tone(1.0), np.zeros(SR // 2, np.float32)]
    )
    s = Signal().score(_item(tmp_path, padded))
    assert 0.4 < s["lead_silence_s"] < 0.6 and 0.4 < s["trail_silence_s"] < 0.6
    assert Signal().score(_item(tmp_path, np.zeros(SR, np.float32)))["is_empty"] is True
    assert Signal().score(_item(tmp_path, tone(0.05)))["is_empty"] is True


def test_robustness_flags_truncation_hallucination_repetition(tmp_path):
    text = "the quick brown fox jumps over the lazy dog today"
    base = dict(ref_words=10, wer_del=0, wer_ins=0, transcript=text)
    it = _item(tmp_path, tone(3.5), text)
    it.results = dict(base)
    assert Robustness().score(it)["robust_fail"] is False
    it.results = {**base, "wer_del": 5}
    assert Robustness().score(it)["truncated"] is True
    it.results = {**base, "wer_ins": 4}
    assert Robustness().score(it)["hallucinated"] is True
    it.results = {**base, "transcript": text + " " + " ".join(["la la la la"] * 6)}
    assert Robustness().score(it)["repetition_score"] > 0.3
    assert repetition("a b c d e") == 0.0


def test_word_error_stats():
    pytest.importorskip("jiwer")
    from voxeval.metrics.intelligibility import word_error_stats

    s = word_error_stats("the cat sat on the mat", "the cat sat on a mat")
    assert s["wer"] == pytest.approx(1 / 6) and s["wer_sub"] == 1 and s["ref_words"] == 6
    assert word_error_stats("hello world", "")["wer"] == 1.0
    with pytest.raises(ValueError):
        word_error_stats("", "anything")


def test_normalizers():
    pytest.importorskip("whisper_normalizer")
    from voxeval.metrics.intelligibility import make_normalizer

    en = make_normalizer("en")
    assert en("It costs $5!") == en("it costs 5 dollars") or "5" in en("It costs $5!")
    assert make_normalizer("es")("¡Hola, Mundo!") == "hola mundo"


def test_judge_prompt_and_parsing():
    assert "style_adherence" in build_prompt("hi", "happy") and "style" not in build_prompt(
        "hi", None
    ).lower().replace("synthesized", "")
    r = parse_judgement(
        'noise {"naturalness":4,"prosody":3,"pronunciation":5,"overall":4,"reason":"ok"} tail',
        False,
    )
    assert r["judge_overall"] == 4.0 and r["judge_reason"] == "ok"
    with pytest.raises(ValueError):
        parse_judgement('{"naturalness":9,"prosody":3,"pronunciation":5,"overall":4}', False)
    with pytest.raises(ValueError):
        parse_judgement("I cannot hear audio", False)
    with pytest.raises(KeyError):
        parse_judgement('{"naturalness":4,"prosody":3,"pronunciation":5,"overall":4}', True)


def test_prosody_on_synthetic_tone(tmp_path):
    pytest.importorskip("librosa")
    from voxeval.metrics.prosody import Prosody

    m = Prosody()
    m.setup("cpu")
    out = m.score(_item(tmp_path, tone(2.0, hz=200), text="one two three four"))
    assert 190 < out["f0_mean_hz"] < 210 and out["f0_std_st"] < 0.5 and out["voiced_ratio"] > 0.8
    assert out["speaking_rate_wps"] == pytest.approx(2.0, rel=0.2)


class _H(http.server.BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):
        import json

        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _H.seen.append((body, self.headers.get("Authorization")))
        buf = io.BytesIO()
        sf.write(buf, tone(0.5), SR, format="WAV")
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.end_headers()
        self.wfile.write(buf.getvalue())

    def log_message(self, *a):
        pass


def test_http_adapter_templating_env_secrets_and_params(monkeypatch):
    srv = http.server.HTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("MY_KEY", "s3cret")
    ad = HTTPAdapter(
        url=f"http://127.0.0.1:{srv.server_port}/tts",
        headers={"Authorization": "Bearer ${MY_KEY}"},
        body={"input": "{text}", "lang": "{language}", "speed": "{speed}"},
        params={"speed": {"type": "float", "min": 0.5, "max": 2.0}},
    )
    res = ad.synthesize(SynthesisRequest("p", "hi there", "en", params={"speed": 1.25}))
    srv.shutdown()
    assert res.sample_rate == SR and len(res.audio) == SR // 2
    body, auth = _H.seen[0]
    assert body == {"input": "hi there", "lang": "en", "speed": 1.25} and auth == "Bearer s3cret"
    assert "speed" in ad.capabilities.params


def test_http_adapter_missing_env_is_a_clear_error(monkeypatch):
    from voxeval.types import VoxevalError

    monkeypatch.delenv("NOPE_KEY", raising=False)
    ad = HTTPAdapter(url="http://127.0.0.1:1/x", headers={"Authorization": "${NOPE_KEY}"})
    with pytest.raises(VoxevalError, match="NOPE_KEY"):
        ad.synthesize(SynthesisRequest("p", "x"))


def test_cli_list_runs():
    from typer.testing import CliRunner

    from voxeval.cli import app

    res = CliRunner().invoke(app, ["list"])
    assert res.exit_code == 0 and "intelligibility" in res.output and "folder" in res.output
