# Arabic Robot Research — Master Summary

**Prepared:** July 2026  
**Hardware baseline:** NVIDIA Tesla T4 (~14.9 GB VRAM)  
**Scope:** Combined summary of all deliverables in this folder.

| Source report | Topic |
|---------------|--------|
| `ASR_Model_Selection_Report.pdf` | Egyptian Arabic speech recognition bake-off |
| `LLM_Model_Selection_Report.pdf` | Conversational LLM bake-off |
| `TTS_Model_Selection_Report.pdf` | Arabic TTS bake-off |
| `Combined_Families_Summary.pdf` | Cross-family stack + VRAM budget |
| `VPS_Provider_Research_Report.pdf` | GPU cloud / VPS selection + capacity |

---

## 1. Bottom line

### Production default stack

| Family | Winner | Why |
|--------|--------|-----|
| **ASR** | **Whisper-Large-v3-Turbo-CT2** | Best accuracy/speed/VRAM blend — robot score **89.62**, Acc **68.4%**, RTF **0.046** (~22×), VRAM **~2.5 GB** |
| **LLM** | **Nile-Chat-4B** | Best turn-taking feel — robot score **95.41**, TTFT **0.96 s**, **11.56 tok/s**, VRAM **~8.4 GB** |
| **TTS** | **VoiceTut-TTS** | Best speed/resources — robot score **90.02**, RTF **0.157** (~6.4×), **82.7 chars/s**, VRAM **~3.15 GB** |

**Combined naive peak VRAM ≈ 14.0 GB** — fits a T4 tightly; deploy on **≥24 GB** (RTX 4090 class) for production headroom.

### Hosting default

**RunPod Secure Cloud — RTX 4090 24 GB** (~$0.69/hr). Use Community / Vast.ai for cheap R&D; move live traffic to Secure before client demos.

---

## 2. ASR summary

**12 models** evaluated on a shared Egyptian Arabic clip (100% success).  
Robot score weights: **45% accuracy + 30% speed/RTF + 15% VRAM + 10% load**.

### Recommended picks

| Role | Model | Key metrics |
|------|-------|-------------|
| Production default | Whisper-Large-v3-Turbo-CT2 | Robot **89.62**, Acc **68.4%**, RTF **0.046**, VRAM **~2.5 GB** |
| Best accuracy | Whisper-Large-v3-CT2 | Acc **72.92%**, WER **0.271**, RTF **0.133**, VRAM **~4.3 GB** |
| Lowest VRAM | Whisper-Small-CT2 | VRAM **~1.3 GB**, Acc **63.2%** |
| Fastest RTF | MMS-1B-all | RTF **0.044** (weak accuracy — not production) |

### Top of leaderboard

| Rank | Model | Robot | Acc% | WER | RTF | VRAM MB |
|-----:|-------|------:|-----:|----:|----:|--------:|
| 1 | Whisper-Large-v3-Turbo-CT2 | 89.62 | 68.40 | 0.316 | 0.046 | 2492 |
| 2 | Whisper-Large-v3-CT2 | 87.92 | 72.92 | 0.271 | 0.133 | 4316 |
| 3 | Whisper-Small-CT2 | 86.19 | 63.19 | 0.368 | 0.084 | 1340 |
| 4 | Arabic-Whisper-Turbo-FT-CT2 | 83.38 | 60.76 | 0.392 | 0.046 | 1692 |
| 5 | Arabic-Whisper-Large-v3-FT-CT2 | 79.11 | 63.19 | 0.368 | 0.135 | 2812 |
| 6 | Voxtral-Mini-3B | 60.19 | 71.53 | 0.285 | 0.281 | 11154 |

### Takeaways

- Whisper CT2 (faster-whisper) models dominate the top of the robot leaderboard.
- Arabic fine-tunes were competitive on speed but did **not** beat base Whisper Large on accuracy in this bake-off.
- Voxtral-Mini is accurate but **~11 GB VRAM** — poor co-resident with LLM+TTS.
- Qwen ASR / QwenCleo underperformed on WER for this Egyptian clip — avoid for production.

---

## 3. LLM summary

**4 models** completed the prompt suite successfully (Nile-Chat-4B, Qwen3-4B-Instruct-2507, Qwen3-8B int4, ALLaM-7B). Automated scores cover **latency / throughput / VRAM**; Arabic answer quality was reviewed separately.

### Recommended picks

| Role | Model | Key metrics |
|------|-------|-------------|
| Production default (speed UX) | Nile-Chat-4B | Robot **95.41**, TTFT **0.96 s**, **11.56 tok/s**, VRAM **~8.4 GB** |
| Fastest TTFT | Qwen3-4B-Instruct-2507 | TTFT **0.95 s**, Quality **~4.41/5**, auto-pass **~80%** |
| Quality alternative | Qwen3-8B (int4) | Quality **~4.65/5**, auto-pass **~85%**, TTFT **1.94 s**, VRAM **~6.9 GB** |
| Avoid on ≤16 GB | ALLaM-7B | VRAM **~15.3 GB**, slow TTFT **2.47 s** |

### Leaderboard

| Rank | Model | Robot | TTFT s | tok/s | VRAM MB | Quality /5 | Auto-pass |
|-----:|-------|------:|-------:|------:|--------:|-----------:|----------:|
| 1 | Nile-Chat-4B | 95.41 | 0.96 | 11.56 | 8394 | ~2.81 | ~20% |
| 2 | Qwen3-4B-Instruct-2507 | 91.61 | 0.95 | 10.48 | 8758 | ~4.41 | ~80% |
| 3 | Qwen3-8B (int4) | 30.66 | 1.94 | 5.16 | 6900 | ~4.65 | ~85% |
| 4 | ALLaM-7B | 4.15 | 2.47 | 5.62 | 15280 | ~4.25 | ~85% |

### Decision rule

- Prefer **Nile-Chat-4B** when turn-taking feel / low latency matters.
- Switch to **Qwen3-8B** (or Qwen3-4B) when answer correctness / TTS-fit matter more.
- Nile can be verbose — spot-check before freezing if TTS latency is tight.

---

## 4. TTS summary

**4 models** completed synthesis. Automated scores cover **speed / resources only** — Egyptian dialect and code-switch quality must be confirmed by **listening to WAVs**.

### Recommended picks

| Role | Model | Key metrics |
|------|-------|-------------|
| Production default | VoiceTut-TTS | Robot **90.02**, RTF **0.157**, **82.7 chars/s**, VRAM **~3.15 GB** |
| Realtime alternative | SILMA-TTS | Robot **77.42**, RTF **0.238**, VRAM **~3.34 GB** |
| Not realtime | Chatterbox / NAMAA | RTF **> 1** — too slow for live robot |

### Leaderboard

| Rank | Model | Robot | RTF | chars/s | VRAM MB | Realtime? |
|-----:|-------|------:|----:|--------:|--------:|-----------|
| 1 | VoiceTut-TTS | 90.02 | 0.157 | 82.7 | 3150 | yes |
| 2 | SILMA-TTS | 77.42 | 0.238 | 46.8 | 3340 | yes |
| 3 | Chatterbox-Multilingual-V3 | 28.63 | 1.101 | 13.4 | 4790 | no |
| 4 | NAMAA-Egyptian-TTS | 0.61 | 1.232 | 15.1 | 6066 | no |

### Listening checklist (manual)

Score each WAV 1–5 for: Egyptian naturalness, Arabic/English code-switch, numbers/dates, prosody/artifacts, overall robot-voice fit.

---

## 5. Combined families (stack budget)

| Stack | ASR | LLM | TTS | Naive peak | Fits |
|-------|-----|-----|-----|------------|------|
| **Speed (default)** | Turbo ~2.5 GB | Nile ~8.4 GB | VoiceTut ~3.2 GB | **~14.0 GB** | T4 tight / **24 GB OK** |
| **Quality** | Large-v3 ~4.3 GB | Qwen3-8B ~6.9 GB | VoiceTut ~3.2 GB | **~14.4 GB** | T4 tight / **24 GB OK** |
| Avoid | + ALLaM ~15 GB | — | — | overflows mid GPUs | need ≥24–48 GB |

**Rule:** ship on **≥24 GB** even if T4 “fits,” so KV-cache, CUDA context, and queued turns do not OOM.

---

## 6. VPS / GPU hosting summary

### Primary pick: RunPod Secure — RTX 4090 24 GB

| Criterion | Why |
|-----------|-----|
| Fit | 24 GB hosts ASR+LLM+TTS with headroom |
| Interaction | SSH/root, Docker templates, REST API, serverless |
| Billing | Per-second; stop pod → stop compute |
| Reliability | Datacenter Secure tier (~99.9%+ class) vs Community marketplace |

**Practical rule:** develop on **Community** (~$0.34/hr) → go live on **Secure** (~$0.69/hr).

### Provider scorecard (July 2026 listings — re-check before buy)

| Provider | RTX 4090-class $/hr | Best use | Score /10 |
|----------|--------------------:|----------|----------:|
| **RunPod Secure** | ~0.69 | **Production default** | **9.0** |
| RunPod Community | ~0.34 | Dev / staging | 8.2 |
| Vast.ai | ~0.27–0.37 | Cheap experiments | 6.5 |
| Hetzner | ~0.33 (Ada 20 GB) | Always-on EU | 7.8 |
| DigitalOcean | RTX 4000 ~0.76 | Simple DX | 7.0 |
| AWS / GCP / Azure | 2–3× rates | Compliance only | 4.5 |

### Cost snapshots (1× RTX 4090)

| Usage | RunPod Community | RunPod Secure | Vast.ai (~$0.30) |
|-------|-----------------:|--------------:|-----------------:|
| 3 h/day metered | ~$31 | ~$62 | ~$27 |
| Business hours (10×22) | ~$75 | ~$152 | ~$66 |
| Soft launch 24/7 | ~$245 | ~$497 | ~$216 |
| Growth (2 GPUs 24/7) | ~$490 | ~$994 | ~$432 |

### Capacity (production speech lengths)

Assumptions: ~60 s user utterance, ~35 s robot reply, multi-turn sessions.

| Stage | DAU | Turns/day | GPUs (24 GB) guidance |
|-------|----:|----------:|----------------------:|
| Pilot | 100 | ~600 | **1** |
| Soft launch | 1,000 | ~8,000 | **1–2** |
| Growth | 5,000 | ~50,000 | **2–6** |
| Scale | 25,000 | ~300,000 | **8–12+ / serverless** |

One RTX 4090 ≈ **20–40** connected VAD sessions, **1–3** GPU-busy turns at once. Size from **peak concurrency**, not DAU alone.

### Suggested architecture

```
Clients → Edge CPU (TLS/auth) → RunPod Secure RTX 4090
                                  (Whisper Turbo + Nile/Qwen + VoiceTut)
                               → Object storage (logs / analytics)
```

---

## 7. Final selection matrix

| Decision | Choice |
|----------|--------|
| Default ASR | **Whisper-Large-v3-Turbo-CT2** |
| Accuracy ASR alt | Whisper-Large-v3-CT2 |
| Default LLM | **Nile-Chat-4B** (speed UX) |
| Quality LLM alt | **Qwen3-8B int4** |
| Default TTS | **VoiceTut-TTS** (confirm by listening) |
| TTS alt | SILMA-TTS |
| GPU class | **24 GB** (RTX 4090 or better) |
| Provider | **RunPod Secure Cloud** |
| Dev / cost lane | RunPod Community or Vast.ai |

---

## 8. Open sign-off items

1. Re-validate ASR on the real mic / noise / dialect mix (single-clip bake-off).
2. Listen to VoiceTut (and SILMA) WAVs for Egyptian + code-switch quality.
3. Spot-check Nile verbosity vs Qwen quality for the live UX priority.
4. Re-check live GPU prices before committing budget.

---

*Summary of the Arabic Robot Research deliverables. Full detail remains in the individual PDF reports in this folder.*

**Research by Eng. Mohamed Soltan.**
