# TTS Model Selection Report

Auto-generated from the Kaggle Arabic TTS bake-off.
Combine this report with **listening** to each WAV before choosing a production voice.

## Run summary

- Prompt chars: **1693**
- Models attempted: **4**
- OK: **4** | Failed: **0** | Skipped: **0**

## Recommended picks (speed/resources)

### `best_for_robot_realtime`

- **Model:** `VoiceTut-TTS`
- **Why:** Best composite of RTF + throughput + low VRAM + fast load.
- **Metrics:** score_robot_realtime=90.02, rtf=0.157, chars_per_second=82.7, peak_vram_mb=3150.2
- **Listen:** `VoiceTut-TTS.wav`

### `best_speed`

- **Model:** `VoiceTut-TTS`
- **Why:** Lowest RTF (generation faster relative to audio length).
- **Metrics:** rtf=0.157, x_realtime=6.39, generation_seconds=20.48

### `best_throughput`

- **Model:** `VoiceTut-TTS`
- **Why:** Highest characters synthesized per second.
- **Metrics:** chars_per_second=82.7

### `lowest_vram`

- **Model:** `VoiceTut-TTS`
- **Why:** Lowest peak VRAM — better for 6–16 GB GPUs.
- **Metrics:** peak_vram_mb=3150.2

### `best_balanced`

- **Model:** `VoiceTut-TTS`
- **Why:** Balanced speed/throughput/VRAM/generation time.
- **Metrics:** score_balanced=100.0

### `best_realtime_capable`

- **Model:** `VoiceTut-TTS`
- **Why:** RTF < 1.0 (faster than real time).
- **Metrics:** rtf=0.157

## Robot realtime leaderboard

| Rank | Model | Robot | RTF | xRT | Chars/s | Gen(s) | VRAM pk | Realtime |
|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| 1 | `VoiceTut-TTS` | 90.02 | 0.157 | 6.39 | 82.7 | 20.48 | 3150.2 | yes |
| 2 | `SILMA-TTS` | 77.42 | 0.238 | 4.21 | 46.8 | 36.16 | 3340.2 | yes |
| 3 | `Chatterbox-Multilingual-V3` | 28.63 | 1.101 | 0.91 | 13.4 | 126.29 | 4790.2 | no |
| 4 | `NAMAA-Egyptian-TTS` | 0.61 | 1.232 | 0.81 | 15.1 | 112.46 | 6066.2 | no |

## Per-model detail

### `SILMA-TTS` — ok

- Timing: load=70.27s, gen=36.16s, wall=304.74s, audio=152.19s
- Speed: RTF=0.238, xRT=4.21, chars/s=46.8, sec/1k chars=21.36
- Resources: CPU pk=180.3%, RAM pk=3293.5MB, GPU pk=99.0%, VRAM pk=3340.2MB, model VRAM=2788.0MB
- Output: `SILMA-TTS.wav` (7305308 bytes, sr=24000)
- Note: silma-ai/silma-tts; official Arabic ref=ar.ref.24k.wav; force_tashkeel=False; chunks=7

### `NAMAA-Egyptian-TTS` — ok

- Timing: load=152.01s, gen=112.46s, wall=334.1s, audio=91.28s
- Speed: RTF=1.232, xRT=0.81, chars/s=15.1, sec/1k chars=66.42
- Resources: CPU pk=193.6%, RAM pk=5626.4MB, GPU pk=97.0%, VRAM pk=6066.2MB, model VRAM=5514.0MB
- Output: `NAMAA-Egyptian-TTS.wav` (4381518 bytes, sr=24000)
- Note: chunks=9

### `Chatterbox-Multilingual-V3` — ok

- Timing: load=25.45s, gen=126.29s, wall=193.18s, audio=114.68s
- Speed: RTF=1.101, xRT=0.91, chars/s=13.4, sec/1k chars=74.6
- Resources: CPU pk=123.7%, RAM pk=5189.2MB, GPU pk=98.0%, VRAM pk=4790.2MB, model VRAM=4238.0MB
- Output: `Chatterbox-Multilingual-V3.wav` (5504718 bytes, sr=24000)
- Note: default multilingual; chunks=9

### `VoiceTut-TTS` — ok

- Timing: load=109.67s, gen=20.48s, wall=179.07s, audio=130.77s
- Speed: RTF=0.157, xRT=6.39, chars/s=82.7, sec/1k chars=12.1
- Resources: CPU pk=181.4%, RAM pk=3840.8MB, GPU pk=100.0%, VRAM pk=3150.2MB, model VRAM=2598.0MB
- Output: `VoiceTut-TTS.wav` (6277004 bytes, sr=24000)
- Note: speaker=Mohamed; chunks=11

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

- IMPORTANT: Naturalness / Egyptian dialect / code-switch quality must be judged by listening to WAVs.
- Automated scores only cover latency and resource fit for the robot pipeline.
- RTF < 1.0 is preferred for near-real-time synthesis; streaming TTFA is not measured here.
- For production, also measure time-to-first-audio with streaming APIs.
