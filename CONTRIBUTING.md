# Contributing

voxeval is deliberately small. The most valuable contributions are **new metrics**, **new
adapters** (as examples or plugins), **new prompt suites / languages**, and **bug reports with a
reproducible config**.

## Setup

```bash
git clone <repo> && cd voxeval
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,asr,prosody]"
pytest -q            # no network, no model downloads
ruff check . && ruff format --check .
```

Tests never download models: use the stubs in `tests/conftest.py` (a tone-generating adapter and
trivial metrics). If your metric wraps a model, unit-test its pure logic (parsing, scoring maths,
thresholds) and keep model loading inside `setup()`.

## Add a metric

See [docs/extending.md](docs/extending.md). In short: subclass `Metric`, declare `fields`
(+ `higher_is_better`), `requires`, `license`, `extra`; import heavy libraries lazily in `setup()`;
register with `@register_metric("name")`. Put the licence of the *weights*, not just the code, in `license`.

## Add an adapter

Prefer an example under `examples/adapters/` (own `requirements.txt`) over a core adapter. The core
stays free of model-specific dependencies. See [examples/adapters/README.md](examples/adapters/README.md).

## Add a language to the built-in suite

Add `src/voxeval/suites/<lang>_core.yaml` using the same categories as the others (`short, long,
question, numbers, contact, tongue_twister, foreign_words, expressive`). Prompts must be original text
(no copyrighted material). If Whisper cannot transcribe a form literally (URLs, symbols), set
`asr_reference` to the spoken form. Languages written without spaces (zh, ja) need CER and a different
normaliser: open an issue first.

## Ground rules

- No API keys or accounts required for the default path. A metric that needs one must be opt-in and say so.
- Licences matter: non-commercial weights must be opt-in with a warning, never a default.
- Keep results reproducible: no hidden randomness, record versions in `env.json`.
