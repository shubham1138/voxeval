# Architecture

```
suite (prompts) ─┐
voices (opt.)  ──┼─► Runner ─► 1. GENERATE  adapter × variants ─► audio/ + generation.jsonl   (cached, resumable)
config.yaml   ───┘           2. SCORE     one metric at a time (load, score, free) ─► scores/<metric>.jsonl (cached)
                             3. AGGREGATE bootstrap CI, gates, composite ─► samples.csv, summary.csv/json
                                                                               └─► Streamlit viewer
```

## Principles

1. **The core knows no TTS model.** Everything goes through `TTSAdapter` + declared `Capabilities`.
   A feature a system lacks (cloning, a parameter, a language) is skipped, never faked.
2. **Metrics declare what they need.** `requires` (ref audio, transcript) and `depends_on` (other metrics)
   are checked by the runner. An item that cannot be scored is *skipped with a reason*, not failed.
3. **Generation and scoring are separate, cached stages.** Interrupted runs resume. Adding a metric later
   never re-synthesizes. Re-scoring after a metric upgrade means deleting `scores/<metric>.jsonl`.
4. **One model in memory at a time.** Each metric loads, scores everything, and frees memory, so a
   16 GB GPU can host Whisper-large *and* the judge in turn.
5. **Failures are data.** A bad sample, a missing dependency or an unavailable model is recorded
   (`run_meta.json`, `generation.jsonl`) and shown in the UI. Nothing is silently dropped.
6. **Everything about a run is in its folder:** resolved config, environment (`env.json`: profile, device,
   versions, GPU), raw per-sample scores, aggregates.

## Run folder

```
runs/<name>-<timestamp>/
  config.resolved.yaml   env.json   run_meta.json   (metric status, licences, timings)
  audio/<variant>/<prompt>.wav
  generation.jsonl       scores/<metric>.jsonl
  samples.csv            (one row per clip, every metric field, gates, composite)
  summary.csv  summary.json   (mean + 95% bootstrap CI per variant × {overall, category, language})
```

## Extension points (and what is deliberately *not* built yet)

| Extension | How today | Natural next step |
|---|---|---|
| New TTS | `TTSAdapter` subclass, loaded by import path or registered by name / entry point | adapter packages on PyPI |
| New metric | `Metric` subclass, same mechanisms | corpus-level metrics (FAD, TTSDS): add `level: corpus` and a second hook |
| New judge | `JudgeBackend` in `metrics/judge.py` | OpenAI-compatible backend (vLLM-served judges); pairwise win-rate mode |
| New language | a `<lang>_core.yaml` suite | per-language ASR/normaliser table incl. CER languages |
| New suites | YAML / JSONL / CSV | loaders for public benchmark sets |
