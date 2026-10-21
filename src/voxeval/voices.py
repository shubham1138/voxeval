"""Fetch a few openly licensed (CC-BY-4.0) reference voices for cloning evaluations.

Streams only the first row groups of a public parquet file, so nothing large is downloaded.
No account or token is needed.
"""

from __future__ import annotations

import io
from pathlib import Path

import soundfile as sf

from .audio import save_audio
from .types import VoxevalError

# lang -> (HF dataset, parquet path inside it, revision, attribution)
SOURCES = {
    "en": (
        "openslr/librispeech_asr",
        "all/test.clean/0000.parquet",
        "main",
        "LibriSpeech test-clean (Panayotov et al., 2015), CC-BY-4.0",
    ),
    "es": (
        "facebook/multilingual_librispeech",
        "spanish/test/0000.parquet",
        "refs%2Fconvert%2Fparquet",
        "Multilingual LibriSpeech (Pratap et al., 2020), CC-BY-4.0",
    ),
    "fr": (
        "facebook/multilingual_librispeech",
        "french/test/0000.parquet",
        "refs%2Fconvert%2Fparquet",
        "Multilingual LibriSpeech (Pratap et al., 2020), CC-BY-4.0",
    ),
    "de": (
        "facebook/multilingual_librispeech",
        "german/test/0000.parquet",
        "refs%2Fconvert%2Fparquet",
        "Multilingual LibriSpeech (Pratap et al., 2020), CC-BY-4.0",
    ),
}


def fetch(out: Path, lang: str, n: int = 4, min_s: float = 5.0, max_s: float = 12.0) -> list[Path]:
    if lang not in SOURCES:
        raise VoxevalError(f"no built-in voice source for '{lang}'. Available: {sorted(SOURCES)}")
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem
    except ImportError as e:
        raise VoxevalError("install with: pip install 'voxeval[voices]'") from e
    repo, path, rev, credit = SOURCES[lang]
    dest = Path(out) / lang
    dest.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    saved: dict[str, Path] = {}
    with fs.open(f"datasets/{repo}@{rev}/{path}", "rb") as f:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["audio", "speaker_id"]).to_pylist()
            for row in tbl:
                spk = str(row["speaker_id"])
                if spk in saved:
                    continue
                audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                if min_s <= len(audio) / sr <= max_s:
                    p = dest / f"{lang}_spk{spk}.wav"
                    save_audio(p, audio, sr)
                    saved[spk] = p
                if len(saved) >= n:
                    break
            if len(saved) >= n:
                break
    if not saved:
        raise VoxevalError("no suitable clips found in the source file")
    (dest / "ATTRIBUTION.md").write_text(
        f"Reference voices from {credit}.\nSource: https://huggingface.co/datasets/{repo}\n"
        "Licence: CC-BY-4.0 (attribution required if you redistribute these clips).\n"
    )
    return list(saved.values())
