"""Hardware profiles: default model per metric. Any metric can override its own args."""

from __future__ import annotations

SKIP = "_skip"
OPTIN = "_optin"  # enabled only when the user lists the metric explicitly

_EMO_NOTE = "emotion2vec weights use a non-standard licence; add `emotion` to `metrics` to enable"

PROFILES: dict[str, dict[str, dict]] = {
    "laptop": {
        "intelligibility": {"model": "large-v3-turbo", "compute_type": "int8", "beam_size": 1},
        "speaker_similarity": {},
        "naturalness": {},
        "signal": {},
        "efficiency": {},
        "robustness": {},
        "prosody": {},
        "emotion": {
            OPTIN: _EMO_NOTE,
            "model": "iic/emotion2vec_plus_base",
        },
        "judge": {SKIP: "needs ~16 GB GPU; use `profile: gpu`"},
    },
    "gpu": {
        "intelligibility": {"model": "large-v3", "compute_type": "float16", "beam_size": 5},
        "speaker_similarity": {},
        "naturalness": {},
        "signal": {},
        "efficiency": {},
        "robustness": {},
        "prosody": {},
        "emotion": {
            OPTIN: _EMO_NOTE,
            "model": "iic/emotion2vec_plus_large",
        },
        "judge": {"model": "Qwen/Qwen2.5-Omni-7B"},
    },
}
