# TTS Model Selection Report

Auto-generated from the Kaggle Arabic TTS bake-off.
Combine this report with **listening** to each WAV before choosing a production voice.

## Run summary

- Prompt chars: **1693**
- Models attempted: **4**
- OK: **0** | Failed: **4** | Skipped: **0**

## Recommended picks (speed/resources)

## Robot realtime leaderboard

| Rank | Model | Robot | RTF | xRT | Chars/s | Gen(s) | VRAM pk | Realtime |
|---:|---|---:|---:|---:|---:|---:|---:|:---:|

## Per-model detail

### `NAMAA-Egyptian-TTS` — error

- Error: ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject

### `Chatterbox-Multilingual-V3` — error

- Error: ModuleNotFoundError: Could not import module 'LlamaModel'. Are this object's requirements defined correctly?

### `VoiceTut-TTS` — error

- Error: ModuleNotFoundError: Could not import module 'HiggsAudioV2TokenizerModel'. Are this object's requirements defined correctly?

### `SILMA-TTS` — error

- Error: ImportError: libcudart.so.13: cannot open shared object file: No such file or directory

## Listening checklist (manual quality)

For each OK WAV, score 1–5:

1. Egyptian dialect naturalness
2. Arabic/English code-switching
3. Numbers / dates / times
4. Prosody / pauses / artifacts
5. Overall robot-voice suitability

## How to use these files

1. Start with `tts_recommendations.json`.
2. Sort `tts_leaderboard.csv` by robot/speed/VRAM.
3. Listen to the shortlisted WAVs.
4. Confirm resources in `tts_analytics.csv`.

## Notes

- No successful TTS runs; cannot recommend a model.
