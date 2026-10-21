# Scoring and aggregation

## Per sample → per group

Every clip gets raw metric fields (`samples.csv`). They are then grouped per **variant** and, within each
variant, **overall / by category / by language** (`summary.csv`, `summary.json`). Each group reports
`mean`, a **95% percentile-bootstrap confidence interval** (1000 resamples over samples, fixed seed) and `n`.

> Small n means wide intervals. If two variants' intervals overlap, you have not shown a difference.
> Quick runs (a handful of prompts) are smoke tests, not benchmarks.

## Gates (pass/fail)

```yaml
gates:
  - {field: wer, op: "<=", value: 0.05}
  - {field: mos, op: ">=", value: 3.5}
  - {field: is_empty, op: "==", value: 0}
```

Each gate is evaluated per sample; the report shows the **pass rate per variant**. Gates answer "is it good
enough?", whereas the means answer "which is better?". Reasonable starting points (published rules of thumb,
not guarantees): WER <= 3-5%, UTMOS >= 3.5 (>= 4.0 is good), speaker similarity >= 0.6 with WavLM-SV
(ECAPA values differ: calibrate on your own reference clips), clipping_ratio <= 0.001, no empty clips.

## Composite (optional, 0-100)

A single number is convenient and easy to misread, so it is explicit and optional:

```
score_f = clip((x_f - worst_f) / (best_f - worst_f), 0, 1)        # per metric, anchors below
composite = 100 * Σ w_f · score_f / Σ w_f                          # over metrics PRESENT for that sample
```

Default anchors (override with `composite.anchors`):

| field | direction | worst → best |
|---|---|---|
| `wer`, `cer` | lower | 0.20 → 0.00 |
| `mos` | higher | 1 → 5 |
| `sim` | higher | 0.2 → 0.8 |
| `judge_overall` | higher | 1 → 5 |

Default weights: `wer 0.3, mos 0.3, sim 0.2, judge_overall 0.2`, renormalised over what exists (so a run
without cloning or without the judge is still scored). Composites are only comparable between runs that
used the **same fields, weights, anchors and profile**. Disable with `composite: {enabled: false}`.

## Profiles

`profile: laptop` (default) and `profile: gpu` only change default models per metric (see
`src/voxeval/profiles.py`); `env.json` records which was used and the viewer warns when runs with
different profiles are compared.
