# ASR Model Selection Report

Auto-generated from the Kaggle Arabic ASR bake-off. Use this with the CSVs/JSON
to pick the production ASR model for the ESP32 Arabic voice robot.

## Run summary

- Total runs: **12**
- OK: **12**
- Failed: **0**
- Models with ≥1 OK run: **12**

## Recommended picks

### `best_for_robot_realtime`

- **Model:** `Whisper-Large-v3-Turbo-CT2`
- **Why:** Highest composite robot score (accuracy + speed + VRAM + load).
- **Metrics:** score_robot_realtime=89.62, avg_accuracy_percent=68.4, avg_rtf=0.046, peak_vram_mb=2492.2

### `best_accuracy`

- **Model:** `Whisper-Large-v3-CT2`
- **Why:** Highest average word accuracy / lowest WER.
- **Metrics:** avg_accuracy_percent=72.92, avg_wer=0.2708, avg_cer=0.1132

### `best_speed`

- **Model:** `MMS-1B-all`
- **Why:** Lowest average RTF (fastest relative to audio length).
- **Metrics:** avg_rtf=0.044, avg_x_realtime=22.88, avg_transcribe_seconds=5.64

### `lowest_vram`

- **Model:** `Whisper-Small-CT2`
- **Why:** Lowest peak VRAM — useful for 6–16 GB GPUs or multi-model co-residency.
- **Metrics:** peak_vram_mb=1340.2

### `best_balanced`

- **Model:** `Whisper-Large-v3-Turbo-CT2`
- **Why:** Balanced accuracy/speed/VRAM tradeoff.
- **Metrics:** score_balanced=90.17

### `best_realtime_with_accuracy`

- **Model:** `Whisper-Large-v3-CT2`
- **Why:** RTF < 1 and best available accuracy among realtime-capable models.
- **Metrics:** avg_rtf=0.133, avg_accuracy_percent=72.92

## Robot realtime leaderboard

| Rank | Model | Robot score | Acc% | WER | RTF | xRT | VRAM pk MB | Success% |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `Whisper-Large-v3-Turbo-CT2` | 89.62 | 68.4 | 0.316 | 0.046 | 21.75 | 2492.2 | 100.0|
| 2 | `Whisper-Large-v3-CT2` | 87.92 | 72.92 | 0.2708 | 0.133 | 7.54 | 4316.2 | 100.0|
| 3 | `Whisper-Small-CT2` | 86.19 | 63.19 | 0.3681 | 0.084 | 11.86 | 1340.2 | 100.0|
| 4 | `Arabic-Whisper-Turbo-FT-CT2` | 83.38 | 60.76 | 0.3924 | 0.046 | 21.91 | 1692.2 | 100.0|
| 5 | `Arabic-Whisper-Large-v3-FT-CT2` | 79.11 | 63.19 | 0.3681 | 0.135 | 7.4 | 2812.2 | 100.0|
| 6 | `Voxtral-Mini-3B` | 60.19 | 71.53 | 0.2847 | 0.281 | 3.56 | 11154.2 | 100.0|
| 7 | `Audar-ASR-V1-Flash` | 58.9 | 49.31 | 0.5069 | 0.191 | 5.23 | 3480.2 | 100.0|
| 8 | `SeamlessM4T-v2-Large` | 57.34 | 51.74 | 0.4826 | 0.135 | 7.42 | 4072.2 | 100.0|
| 9 | `MMS-1B-all` | 55.21 | 42.36 | 0.5764 | 0.044 | 22.88 | 5206.2 | 100.0|
| 10 | `Qwen3-ASR-0.6B` | 48.84 | 39.93 | 0.6007 | 0.191 | 5.23 | 3740.2 | 100.0|
| 11 | `Qwen3-ASR-1.7B` | 33.28 | 34.72 | 0.6528 | 0.215 | 4.65 | 6044.2 | 100.0|
| 12 | `QwenCleo-ASR` | 18.08 | 33.68 | 0.6632 | 0.582 | 1.72 | 5870.2 | 100.0|

## Per-model aggregate detail

### `Whisper-Small-CT2`

- Runs: 1/1 OK (100.0%)
- Accuracy: avg=63.19% (min=63.19, max=63.19, std=0.0)
- WER/CER: avg_wer=0.3681, avg_cer=0.1686
- Speed: avg_rtf=0.084, min_rtf=0.084, max_rtf=0.084, realtime_capable=100.0%
- Timing: load_avg=6.16s, transcribe_avg=10.88s
- Resources: CPU pk=118.8%, RAM pk=1390.6MB, GPU pk=76.0%, VRAM pk=1340.2MB, model VRAM=788.0MB

### `Whisper-Large-v3-Turbo-CT2`

- Runs: 1/1 OK (100.0%)
- Accuracy: avg=68.4% (min=68.4, max=68.4, std=0.0)
- WER/CER: avg_wer=0.316, avg_cer=0.1662
- Speed: avg_rtf=0.046, min_rtf=0.046, max_rtf=0.046, realtime_capable=100.0%
- Timing: load_avg=48.59s, transcribe_avg=5.93s
- Resources: CPU pk=172.2%, RAM pk=2404.4MB, GPU pk=100.0%, VRAM pk=2492.2MB, model VRAM=1940.0MB

### `Arabic-Whisper-Turbo-FT-CT2`

- Runs: 1/1 OK (100.0%)
- Accuracy: avg=60.76% (min=60.76, max=60.76, std=0.0)
- WER/CER: avg_wer=0.3924, avg_cer=0.1782
- Speed: avg_rtf=0.046, min_rtf=0.046, max_rtf=0.046, realtime_capable=100.0%
- Timing: load_avg=29.69s, transcribe_avg=5.89s
- Resources: CPU pk=121.4%, RAM pk=1749.3MB, GPU pk=99.0%, VRAM pk=1692.2MB, model VRAM=1140.0MB

### `Arabic-Whisper-Large-v3-FT-CT2`

- Runs: 1/1 OK (100.0%)
- Accuracy: avg=63.19% (min=63.19, max=63.19, std=0.0)
- WER/CER: avg_wer=0.3681, avg_cer=0.1794
- Speed: avg_rtf=0.135, min_rtf=0.135, max_rtf=0.135, realtime_capable=100.0%
- Timing: load_avg=35.17s, transcribe_avg=17.43s
- Resources: CPU pk=119.0%, RAM pk=2821.4MB, GPU pk=100.0%, VRAM pk=2812.2MB, model VRAM=2260.0MB

### `QwenCleo-ASR`

- Runs: 1/1 OK (100.0%)
- Accuracy: avg=33.68% (min=33.68, max=33.68, std=0.0)
- WER/CER: avg_wer=0.6632, avg_cer=0.5888
- Speed: avg_rtf=0.582, min_rtf=0.582, max_rtf=0.582, realtime_capable=100.0%
- Timing: load_avg=0.0s, transcribe_avg=75.09s
- Resources: CPU pk=188.1%, RAM pk=4888.8MB, GPU pk=100.0%, VRAM pk=5870.2MB, model VRAM=5318.0MB

### `Whisper-Large-v3-CT2`

- Runs: 1/1 OK (100.0%)
- Accuracy: avg=72.92% (min=72.92, max=72.92, std=0.0)
- WER/CER: avg_wer=0.2708, avg_cer=0.1132
- Speed: avg_rtf=0.133, min_rtf=0.133, max_rtf=0.133, realtime_capable=100.0%
- Timing: load_avg=37.64s, transcribe_avg=17.1s
- Resources: CPU pk=194.1%, RAM pk=3787.3MB, GPU pk=100.0%, VRAM pk=4316.2MB, model VRAM=3764.0MB

### `Qwen3-ASR-0.6B`

- Runs: 1/1 OK (100.0%)
- Accuracy: avg=39.93% (min=39.93, max=39.93, std=0.0)
- WER/CER: avg_wer=0.6007, avg_cer=0.401
- Speed: avg_rtf=0.191, min_rtf=0.191, max_rtf=0.191, realtime_capable=100.0%
- Timing: load_avg=21.39s, transcribe_avg=24.65s
- Resources: CPU pk=191.7%, RAM pk=3064.2MB, GPU pk=100.0%, VRAM pk=3740.2MB, model VRAM=3188.0MB

### `Qwen3-ASR-1.7B`

- Runs: 1/1 OK (100.0%)
- Accuracy: avg=34.72% (min=34.72, max=34.72, std=0.0)
- WER/CER: avg_wer=0.6528, avg_cer=0.5328
- Speed: avg_rtf=0.215, min_rtf=0.215, max_rtf=0.215, realtime_capable=100.0%
- Timing: load_avg=90.55s, transcribe_avg=27.74s
- Resources: CPU pk=586.4%, RAM pk=5483.1MB, GPU pk=100.0%, VRAM pk=6044.2MB, model VRAM=5492.0MB

### `Audar-ASR-V1-Flash`

- Runs: 1/1 OK (100.0%)
- Accuracy: avg=49.31% (min=49.31, max=49.31, std=0.0)
- WER/CER: avg_wer=0.5069, avg_cer=0.413
- Speed: avg_rtf=0.191, min_rtf=0.191, max_rtf=0.191, realtime_capable=100.0%
- Timing: load_avg=37.32s, transcribe_avg=24.66s
- Resources: CPU pk=119.4%, RAM pk=3242.3MB, GPU pk=100.0%, VRAM pk=3480.2MB, model VRAM=2928.0MB

### `Voxtral-Mini-3B`

- Runs: 1/1 OK (100.0%)
- Accuracy: avg=71.53% (min=71.53, max=71.53, std=0.0)
- WER/CER: avg_wer=0.2847, avg_cer=0.103
- Speed: avg_rtf=0.281, min_rtf=0.281, max_rtf=0.281, realtime_capable=100.0%
- Timing: load_avg=146.24s, transcribe_avg=36.22s
- Resources: CPU pk=191.4%, RAM pk=6382.0MB, GPU pk=100.0%, VRAM pk=11154.2MB, model VRAM=10602.0MB

### `SeamlessM4T-v2-Large`

- Runs: 1/1 OK (100.0%)
- Accuracy: avg=51.74% (min=51.74, max=51.74, std=0.0)
- WER/CER: avg_wer=0.4826, avg_cer=0.2896
- Speed: avg_rtf=0.135, min_rtf=0.135, max_rtf=0.135, realtime_capable=100.0%
- Timing: load_avg=133.44s, transcribe_avg=17.39s
- Resources: CPU pk=193.6%, RAM pk=7432.8MB, GPU pk=100.0%, VRAM pk=4072.2MB, model VRAM=3520.0MB

### `MMS-1B-all`

- Runs: 1/1 OK (100.0%)
- Accuracy: avg=42.36% (min=42.36, max=42.36, std=0.0)
- WER/CER: avg_wer=0.5764, avg_cer=0.2697
- Speed: avg_rtf=0.044, min_rtf=0.044, max_rtf=0.044, realtime_capable=100.0%
- Timing: load_avg=56.16s, transcribe_avg=5.64s
- Resources: CPU pk=185.9%, RAM pk=5012.7MB, GPU pk=100.0%, VRAM pk=5206.2MB, model VRAM=4654.0MB

## Per-run accuracy ranking (top 20)

1. `Whisper-Large-v3-CT2` / `VoiceTut-TTS.wav` — Acc=72.92% WER=0.2708 CER=0.1132 RTF=0.133
2. `Voxtral-Mini-3B` / `VoiceTut-TTS.wav` — Acc=71.53% WER=0.2847 CER=0.103 RTF=0.281
3. `Whisper-Large-v3-Turbo-CT2` / `VoiceTut-TTS.wav` — Acc=68.4% WER=0.316 CER=0.1662 RTF=0.046
4. `Whisper-Small-CT2` / `VoiceTut-TTS.wav` — Acc=63.19% WER=0.3681 CER=0.1686 RTF=0.084
5. `Arabic-Whisper-Large-v3-FT-CT2` / `VoiceTut-TTS.wav` — Acc=63.19% WER=0.3681 CER=0.1794 RTF=0.135
6. `Arabic-Whisper-Turbo-FT-CT2` / `VoiceTut-TTS.wav` — Acc=60.76% WER=0.3924 CER=0.1782 RTF=0.046
7. `SeamlessM4T-v2-Large` / `VoiceTut-TTS.wav` — Acc=51.74% WER=0.4826 CER=0.2896 RTF=0.135
8. `Audar-ASR-V1-Flash` / `VoiceTut-TTS.wav` — Acc=49.31% WER=0.5069 CER=0.413 RTF=0.191
9. `MMS-1B-all` / `VoiceTut-TTS.wav` — Acc=42.36% WER=0.5764 CER=0.2697 RTF=0.044
10. `Qwen3-ASR-0.6B` / `VoiceTut-TTS.wav` — Acc=39.93% WER=0.6007 CER=0.401 RTF=0.191
11. `Qwen3-ASR-1.7B` / `VoiceTut-TTS.wav` — Acc=34.72% WER=0.6528 CER=0.5328 RTF=0.215
12. `QwenCleo-ASR` / `VoiceTut-TTS.wav` — Acc=33.68% WER=0.6632 CER=0.5888 RTF=0.582

## How to use these files

1. Open `asr_recommendations.json` for the primary pick.
2. Confirm with `asr_leaderboard.csv` (sortable in Excel/Sheets).
3. Drill into `asr_analytics.csv` for every audio×model run.
4. Use `asr_analytics_by_model.csv` for min/max/std stability.
5. If accuracy refs exist, also check `asr_accuracy_ranking.csv`.

## Notes

- score_robot_realtime weights accuracy (45%) + speed/RTF (30%) + low VRAM (15%) + fast load (10%).
- RTF < 1.0 means transcription finishes faster than audio duration (good for near-real-time).
- If WER/accuracy is missing, listen-quality still needs human review of transcripts.
- For ESP32 robot VPS: prefer low RTF + high accuracy + VRAM that fits your GPU with headroom.
