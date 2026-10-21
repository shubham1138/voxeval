# Extending voxeval

## A metric in ~25 lines

```python
from voxeval.metrics.base import Metric, lazy_import
from voxeval.registry import register_metric
from voxeval.types import Item


@register_metric("my_metric")
class MyMetric(Metric):
    description = "What it measures."
    license = "MIT (code), CC-BY-4.0 (weights)"  # licence of the WEIGHTS too
    extra = "mymetric"  # pip extra that provides the deps
    fields = {"score": True}  # name -> higher_is_better (True/False/None)
    info = ("label",)  # non-numeric, kept per sample only
    requires = frozenset({"ref_audio"})  # skip items without a reference clip
    depends_on = ()  # e.g. ("intelligibility",) to read its transcript

    def setup(self, device):  # heavy imports + model load go here
        self.model = lazy_import("some_pkg", "mymetric").load(device=device)

    def score(self, item: Item) -> dict:  # item.wav_path, item.ref_audio, item.text, item.results
        return {"score": self.model(item.wav_path), "label": "ok"}

    def teardown(self):  # free memory for the next metric
        self.model = None
```

Use it in a config by import path, with no changes to voxeval:

```yaml
metrics:
  signal: {}
  "my_pkg/my_metric.py:MyMetric": {some_arg: 1}
```

Field names must be unique across enabled metrics (checked at start-up). Booleans aggregate as rates.
Anything in `fields` can be used in `gates` and (with an anchor) in the `composite`.

## Gates and composite

```yaml
gates:                                  # pass/fail per sample; pass rate per variant
  - {field: wer, op: "<=", value: 0.05}
composite:
  weights: {wer: 0.3, mos: 0.3, sim: 0.2}
  anchors: {my_score: [higher, 0.0, 10.0]}      # direction, worst, best -> clipped linear map to 0..1
```

After editing gates or composite in a run's `config.resolved.yaml`, run `voxeval aggregate <run dir>`
to recompute the tables without regenerating or re-scoring anything.

## An adapter

See [../examples/adapters/README.md](../examples/adapters/README.md).
