# voxeval

**Modular, fully automated, open-source evaluation for text-to-speech.**
Bring any TTS (a local model, an HTTP API, or a folder of WAVs) and get intelligibility, speaker
similarity, naturalness, robustness, prosody, signal-sanity, efficiency and (optionally) audio-LLM-judge
scores, with confidence intervals, pass/fail gates and a viewer. **No API keys. No humans in the loop.
Runs on a laptop.**

> "It sounds fine" is not an evaluation. voxeval is a small, hackable v0.1 that turns the metrics
> industry benchmarks use (Whisper WER, speaker-embedding similarity, MOS predictors, hard-case prompt
> categories, LALM judges) into something a team can point at their own model in minutes.

## Why this exists

Most TTS evaluation code is a one-off script tied to one model. Papers and benchmarks
([EmergentTTS-Eval](https://arxiv.org/abs/2505.23009), Seed-TTS-eval, TTSDS, VERSA) are great but are
either benchmark-specific, need paid judge APIs, or are heavyweight. voxeval keeps the useful core:

- **One adapter interface.** The core knows no TTS model. Capabilities (cloning, tunable params, languages) are
  *declared*, and metrics/sweeps switch on and off accordingly. Voice cloning is optional.
- **Open weights only, no keys.** Every default metric downloads open weights. The optional judge is a local open model.
- **Parameter sweeps built in.** A grid in YAML becomes variants that are compared side by side.
- **Honest statistics.** Per-variant, per-category, per-language means with bootstrap 95% CIs. Gates for "good enough".
- **Resumable and inspectable.** Generation and scoring are separate cached stages; every run is a self-contained folder.

## Quickstart (CPU laptop)

```bash
git clone <this repo> && cd voxeval
uv venv && source .venv/bin/activate
uv pip install -e ".[asr,sim,mos,prosody,ui]"
voxeval list                                   # adapters, metrics, suites, licences
```

**1. You already have audio from some TTS** (closed API, another team's service...):

```bash
# put <prompt id>.wav files in ./my_audio, describe prompts in my_prompts.jsonl
cp configs/folder_eval.yaml my.yaml && voxeval run my.yaml
voxeval ui runs/
```

**2. You want voxeval to call your TTS.** Write an adapter (about 20 lines, see
[examples/adapters/](examples/adapters/README.md)) or use the built-in `http` adapter
([configs/http_endpoint.yaml](configs/http_endpoint.yaml)).

**3. Try the reference example** (Chatterbox, local, with cloning and a parameter sweep):

```bash
uv venv .venv-cb && source .venv-cb/bin/activate          # separate env: chatterbox pins torch
uv pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install -r examples/adapters/requirements.txt && uv pip install -e ".[asr,sim,mos,prosody,voices,ui]"
voxeval fetch-voices --lang en --n 2                      # CC-BY reference voices (a few minutes)
voxeval run examples/configs/chatterbox_quick_cpu.yaml
voxeval ui runs/
```

## What it measures

| Metric | Question | Default model | Licence |
|---|---|---|---|
| `intelligibility` | Right words? WER/CER | faster-whisper (large-v3-turbo int8 / large-v3) | MIT |
| `speaker_similarity` | Same voice as the reference? | SpeechBrain ECAPA-TDNN | Apache-2.0 |
| `naturalness` | Human-sounding? predicted MOS | UTMOS22 | see [docs/metrics.md](docs/metrics.md) |
| `robustness` | Truncation / hallucination / repetition on hard cases | derived from ASR | n/a |
| `signal` | Clipping, silence, loudness, empty output | none | n/a |
| `efficiency` | Latency, real-time factor, VRAM | none | n/a |
| `prosody` | Pitch, speaking rate, pauses | librosa pYIN | ISC |
| `emotion` *(opt-in)* | Emotion / style match | emotion2vec+ | non-standard |
| `judge` *(gpu profile)* | Rubric rating 1-5 + reason | Qwen2.5-Omni-7B | Apache-2.0 |

Built-in prompt suites (en: 90, es/fr/de: 24 each) cover short, long-form, questions, numbers, abbreviations,
URLs/emails/phones, tongue twisters, foreign words, syntactic complexity and expressive styles. Bring your own as
YAML / JSONL / CSV. Details: [docs/metrics.md](docs/metrics.md), [docs/scoring.md](docs/scoring.md).

## Profiles

| | `laptop` (default) | `gpu` |
|---|---|---|
| ASR | Whisper large-v3-turbo, int8, CPU | Whisper large-v3, fp16 |
| Judge | off (needs ~16 GB GPU) | Qwen2.5-Omni-7B |
| Use for | iterating, CI, comparing variants | numbers you want to publish |

Numbers are only comparable between runs with the same profile (recorded in `env.json`; the viewer warns).

**Measured on an 8-thread Intel i5-9300H laptop, no GPU** (6 clips, after weights were cached):
Whisper-turbo int8 scoring ≈ 49 s (≈ 8 s/clip including model load), UTMOS ≈ 3 s, ECAPA ≈ 5 s, prosody ≈ 6 s,
signal/efficiency/robustness < 1 s. The *TTS itself* dominates: Chatterbox on this CPU took 15-30 s per short clip
(RTF 6-16). One-time downloads (Chatterbox ≈ 3 GB, Whisper-turbo ≈ 1.6 GB) are not included.

## Sweeps

```yaml
adapter: examples/adapters/chatterbox_adapter.py:ChatterboxAdapter
sweep: {exaggeration: [0.3, 0.5, 0.8], cfg_weight: [0.2, 0.5]}   # 6 variants
gates: [{field: wer, op: "<=", value: 0.05}, {field: mos, op: ">=", value: 3.5}]
```

Only parameters the adapter *declares* can be swept; bad names or out-of-range values fail before any audio is generated.

## Layout

```
src/voxeval/   core: types, registry, config, suite, sweep, runner, aggregate, profiles, cli
  adapters/    base, folder (bring-your-own-audio), http (generic endpoint)
  metrics/     one file per metric
  suites/      built-in prompt suites
  ui/app.py    Streamlit viewer
examples/      reference Chatterbox adapter + configs        configs/   core-only example configs
docs/          architecture, metrics, scoring, extending      tests/     stubbed, no downloads
```

## Status and honest limitations

v0.1: a solid foundation, not a finished benchmark.

- **Verified**: 30 unit tests (stub adapter and metrics, no downloads) pass; an end-to-end CPU run with a real TTS
  (Chatterbox) and every laptop-profile metric; re-scoring the same audio through the `folder` adapter reproduces
  identical scores; the viewer runs on real output.
- **Not exercised yet**: the `judge` metric on a real GPU (its parsing and prompt are unit-tested, the model path is
  not), `profile: gpu` models, and the `http` adapter against a real remote service (tested against a local server only).
  `emotion` is implemented but was not run (opt-in; licence).
- Predicted MOS and open audio-LLM judges are **proxies for human judgement**: good for ranking variants,
  weak for absolute claims. Speaker-similarity values are specific to the embedding model.
- Robustness flags are heuristic (e.g. a single repeated word such as "Saturday? Saturday?" raises WER but is not flagged).
- Language-specific normalisation covers en/es/fr/de. Languages without spaces (zh, ja) need CER and are not shipped.
- Not included: FAD/TTSDS (corpus-level), DNSMOS/NISQA/Audiobox Aesthetics, pairwise win-rate, streaming latency.
  See [docs/architecture.md](docs/architecture.md) for where they would plug in.

## Contributing

New metrics, adapters, languages and bug reports are welcome: [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence and attribution

Code: Apache-2.0. Models keep their own licences (see [docs/metrics.md](docs/metrics.md)); non-commercial
weights are never defaults. Reference voices fetched by `voxeval fetch-voices` are CC-BY-4.0
(LibriSpeech, Multilingual LibriSpeech) and carry an `ATTRIBUTION.md`.
Prior work this builds on: EmergentTTS-Eval, Seed-TTS-eval, TTSDS, VERSA, UTMOS, SpeechBrain, Whisper.
