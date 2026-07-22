# Kaggle Arabic ASR & TTS Comparison — Summary

Two Kaggle-ready scripts for comparing open Arabic speech models on **T4 / T4x2**. Both isolate each model in its own subprocess (with `CUDA_VISIBLE_DEVICES`) so one CUDA crash cannot poison the notebook session, and both collect timing plus CPU/RAM/GPU analytics.

---

## Shared design

| Concern | Approach |
|---|---|
| Isolation | Each model runs in a worker subprocess; parent collects JSON metadata |
| Multi-GPU | Round-robin across GPUs on T4x2 |
| Disk | Results → `/kaggle/working/...` (persisted); checkpoints/repos → scratch (`/kaggle/tmp` or `/tmp/kaggle_scratch`) |
| Install | Auto-install on Kaggle; `--no-install` for repeat runs |
| Analytics | `ResourceMonitor` samples CPU%, RAM, GPU util, VRAM while the worker runs |
| Metrics | Load time, generation/transcribe time, audio duration, **RTF** (real-time factor) |

---

## 1. ASR — `kaggle_arabic_asr_compare.py`

**Goal:** Transcribe the same Arabic (and code-switched) audio with many ASR backends and compare accuracy + speed + resource use.

### Models enabled

| Model | Backend / source | Role |
|---|---|---|
| Whisper-Small-CT2 | faster-whisper `small` | Fast general baseline |
| Whisper-Large-v3-Turbo-CT2 | Systran faster-whisper turbo | Fast large baseline |
| Whisper-Large-v3-CT2 | Systran faster-whisper large-v3 | Stronger general baseline |
| Arabic-Whisper-Turbo-FT-CT2 | `dev-ahmedhany/...-arabic-ft-ct2-int8` | Dialectal Arabic, production-friendly |
| Arabic-Whisper-Large-v3-FT-CT2 | `dev-ahmedhany/...-arabic-ft-v3-ct2-int8` | Stronger dialectal Arabic CT2 |
| QwenCleo-ASR | QwenCleo / qwen-asr | Targeted at Egyptian Arabic + AR/EN code-switch |
| Qwen3-ASR-0.6B / 1.7B | `Qwen/Qwen3-ASR-*` | Newer Qwen ASR family |
| Audar-ASR-H-Turbo | `audarai/audar-asr-h-turbo-merged` | Heavy Arabic ASR (Qwen3-style) |
| Voxtral-Mini-3B | `mistralai/Voxtral-Mini-3B-2507` | Multilingual transformers ASR |
| SeamlessM4T-v2-Large | `facebook/seamless-m4t-v2-large` | Meta MSA-oriented baseline |
| MMS-1B-all | `facebook/mms-1b-all` | Meta CTC, 1100+ languages incl. Arabic |

### Inputs

- Audio from `/kaggle/input/...`, `asr_inputs/`, or `tts_outputs/` (`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.opus`, …)
- Optional ground truth: same-stem `.txt` beside each audio (or `--reference-json`) for **WER / CER**
- Default language hint: `ar`
- CLI filters: `--only`, `--skip`, `--audio`, `--no-install`

### Outputs

```
asr_outputs/
  transcripts/<model>/<audio>.txt   (or <model>__<audio>.txt)
  summary.json
  summary.csv
  asr_analytics.csv                 # every audio×model run
  asr_analytics_by_model.csv        # mean/min/max/std + success rate
  asr_leaderboard.csv               # ranked composite scores
  asr_accuracy_ranking.csv          # WER/Acc ranking when refs exist
  asr_recommendations.json          # best_for_robot / accuracy / speed / VRAM picks
  asr_selection_report.md           # human-readable selection guide
```

Analytics include per-run and per-model aggregates: load/transcribe/wall time, RTF, x-realtime, WER/CER, peak/avg CPU/RAM/GPU/VRAM, stability (min/max/std), robot composite score, and explicit selection picks.

### Research notes baked into the script

- **QwenCleo-ASR** — strongest open pick found for Egyptian Arabic + Arabic/English code-switching
- **Arabic Whisper FT CT2** — production-friendly int8 dialect baselines
- Heavier models (Qwen3, Audar, etc.) are optional after a first stable pass

---

## 2. TTS — `kaggle_arabic_tts_compare.py`

**Goal:** Synthesize one long Egyptian-Arabic + English code-switched prompt with several TTS models and produce listen-comparable WAVs plus performance stats.

### Default prompt

A long mixed text (Egyptian dialect + English tech terms) covering meetings, dashboards, APIs, pricing, KPIs, etc. — designed to stress naturalness on long, information-dense, code-switched speech.

### Models (currently enabled)

| Model | Source / notes |
|---|---|
| NAMAA-Egyptian-TTS | `NAMAA-Space/NAMAA-Egyptian-TTS` on Chatterbox multilingual |
| Chatterbox-Multilingual-V3 | Base Chatterbox multilingual (`t3_model=v3` when available) |
| VoiceTut-TTS | `mohammedaly22/VoiceTut-TTS`, speaker `Mohamed`; text chunked (~180 chars) |
| SILMA-TTS | `silma-ai/silma-tts` with packaged Arabic reference WAV |

### Models present but disabled (commented in `ENABLE`)

Kokoro-82M, Qwen3-TTS-0.6B, CosyVoice-0.5B, Fish-Speech — runners exist; enable when needed.

### Kaggle / dependency hardening

- Installs `espeak-ng`, `sox`, `ffmpeg`, `git-lfs`, and model-specific pip packages
- Forces `transformers>=5.3` for VoiceTut (HiggsAudio tokenizer)
- Replaces broken `onnxruntime-gpu` (CUDA 13) with CPU `onnxruntime` on T4 images
- Patches Qwen-TTS decorator quirks when that path is enabled
- Chunks long text for models that struggle with very long inputs

### Outputs

```
tts_outputs/
  <ModelName>.wav
  summary.json
  tts_analytics.csv
  tts_analytics_by_model.csv
  tts_leaderboard.csv
  tts_recommendations.json
  tts_selection_report.md
```

Analytics: load/generation seconds, audio duration, RTF, chars/s, realtime capability, peak CPU/RAM/GPU util/VRAM, robot composite ranking. **Listening quality is still manual** — the report includes a 1–5 listening checklist.

---

## 3. LLM — `kaggle_arabic_llm_compare.py`

**Goal:** Compare Arabic conversational LLMs on Egyptian / MSA / code-switch / robot prompts for latency + resource fit.

### Outputs

```
llm_outputs/
  responses/<model>/<prompt_id>.txt
  summary.json
  summary.csv
  llm_analytics.csv                 # every prompt×model run
  llm_analytics_by_model.csv        # mean/min/max/std + success rate
  llm_analytics_by_category.csv     # egyptian / msa / code_switch / …
  llm_leaderboard.csv               # ranked composite scores
  llm_recommendations.json          # robot / TTFT / tok/s / VRAM picks
  llm_selection_report.md           # human-readable selection guide
```

Analytics: TTFT, tok/s, load mode, CPU/RAM/sysRAM/GPU/VRAM, category specialists, robot composite score. Response quality still needs manual review under `responses/`.

---

## How to run (Kaggle)

**TTS**

```python
%run /kaggle/working/kaggle_arabic_tts_compare.py
# later:
%run /kaggle/working/kaggle_arabic_tts_compare.py --no-install
```

**ASR**

```python
%run /kaggle/working/kaggle_arabic_asr_compare.py
%run /kaggle/working/kaggle_arabic_asr_compare.py --no-install
%run /kaggle/working/kaggle_arabic_asr_compare.py --only QwenCleo-ASR,Arabic-Whisper-Large-v3-FT-CT2
```

Tip: restart the Kaggle session before a fresh TTS pass if a previous run poisoned CUDA.

---

## What this work delivers

1. **Side-by-side listening** for Egyptian / code-switched TTS candidates  
2. **Measurable ASR bake-off** (WER/CER + RTF + VRAM + leaderboard + recommendations)  
3. **Measurable LLM bake-off** (TTFT + tok/s + category breakdown + recommendations)  
4. **Kaggle-safe orchestration** (subprocess isolation, scratch vs working disk, auto-install)  
5. **Selection artifacts** — for each modality: per-run CSV, per-model aggregates, leaderboard, `*_recommendations.json`, and `*_selection_report.md` so you can pick production models from data, not guesswork
