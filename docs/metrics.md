# Metrics

Every metric is automated, uses open weights, and needs no API key. Licences below are for the
**weights** as far as they could be verified; check them before commercial use.

| Metric | Measures | Default model | Output fields | Licence | Needs |
|---|---|---|---|---|---|
| `intelligibility` | Did it say the right words? | faster-whisper (`large-v3-turbo` int8 on laptop, `large-v3` fp16 on gpu) | `wer`, `cer`, `wer_sub/del/ins`, `transcript` | MIT | `[asr]` |
| `speaker_similarity` | Does it sound like the reference voice? | SpeechBrain ECAPA-TDNN | `sim` (cosine) | Apache-2.0 | `[sim]`, **reference audio** |
| `naturalness` | Does it sound human? (predicted MOS) | UTMOS22-strong | `mos` (1-5) | code MIT; weights: check upstream | `[mos]` |
| `signal` | Is the file sane? | none | `clipping_ratio`, `silence_ratio`, `lead/trail_silence_s`, `lufs`, `is_empty`, `duration_s` | n/a | core |
| `efficiency` | What did generation cost? | none | `latency_s`, `rtf`, `peak_vram_mb` | n/a | core |
| `robustness` | Hard-case failure modes | none (reads `intelligibility`) | `truncated`, `hallucinated`, `repetition_score`, `duration_ratio`, `robust_fail` | n/a | core |
| `prosody` | Pitch, rate, pauses | librosa pYIN | `f0_mean_hz`, `f0_std_st`, `f0_range_st`, `voiced_ratio`, `speaking_rate_wps`, `pause_count`, `pause_mean_s`, `f0_mean_delta_st` | ISC | `[prosody]` |
| `emotion` | Emotion of the delivery; style match | emotion2vec+ (base / large) | `emotion_label`, `emotion_confidence`, `style_match` | **non-standard: opt-in** | `[emotion]` |
| `judge` | Audio-LLM rubric rating | Qwen2.5-Omni-7B | `judge_naturalness/prosody/pronunciation/style_adherence/overall` (1-5), `judge_reason` | Apache-2.0 | `[judge]`, ~16 GB GPU (`gpu` profile only) |

## How to read them

- **Intelligibility.** Text is normalised with Whisper's normalisers before comparing (English-specific
  for `en`; language-agnostic otherwise). Per-prompt `asr_reference` sets the spoken form for URLs,
  symbols and acronyms. A smaller ASR inflates absolute WER: compare runs that used the same profile.
  Languages without spaces need CER; the shipped languages (en/es/fr/de) use WER. Digits in non-English
  prompts are noisy because Whisper may write numbers either way.
- **Speaker similarity.** Cosine between ECAPA embeddings of reference and output. Values depend on the
  embedding model: do **not** compare with published WavLM-large-SV numbers (e.g. Seed-TTS-eval), only
  with your own runs. Skipped automatically when the adapter has no cloning or the prompt has no reference.
- **Naturalness.** A *predicted* MOS from a model trained on human ratings. Good for ranking variants;
  weak at absolute claims, and biased toward clean read speech (expressive or emotional speech can score low).
- **Robustness.** Flags what an average WER hides. `truncated`: deleted-word rate >= 30% or audio far shorter
  than the text implies. `hallucinated`: inserted-word rate >= 30%, audio far longer than expected, or
  n-gram repetition the text does not contain. Thresholds are metric args (`max_deletion_rate`, ...).
  Per-category WER (numbers, URLs, tongue twisters...) comes from the aggregate tables.
- **Signal.** Cheap, model-free sanity gates: clipping, silence, loudness (LUFS, ITU-R BS.1770), empty/NaN output.
- **Efficiency.** `rtf = latency / audio duration` (< 1 is faster than real time). Pre-generated audio
  (`folder` adapter) has no latency and reports none. `peak_vram_mb` is only recorded when the adapter runs in
  the same process on a CUDA GPU. Always compare efficiency on the same hardware.
- **Prosody.** Descriptors, not a quality score: use them to see *how* two variants differ (flatter pitch,
  faster rate, more pauses). `f0_mean_delta_st` is the pitch offset (semitones) from the reference voice.
- **Emotion.** Only meaningful for prompts with a `style`. emotion2vec's weights carry a non-standard
  licence, so the metric is opt-in. Research also reports that emotion embeddings correlate weakly with
  human similarity judgements: treat as a coarse check.
- **Judge.** An open audio-LLM rating on a fixed rubric (temperature 0, strict JSON, one retry with
  sampling, parse failures counted as errors). Published results show open judges agree with humans
  less than the best closed ones. Use it to compare variants under the *same* judge and model, read the
  `judge_reason`s, and never mix numbers across judge models.

## Not included (yet)

Distributional metrics (FAD, TTSDS), DNSMOS/NISQA/Audiobox Aesthetics, pairwise win-rate judging,
streaming latency (time to first audio), CER-language support. See [architecture.md](architecture.md)
for where each would plug in. NISQA and SQUIM-subjective weights are non-commercial, which is why they
are not defaults.
