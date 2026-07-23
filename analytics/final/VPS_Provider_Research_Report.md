# Arabic Voice Robot — Model Stack + VPS Provider Research

**Prepared:** July 2026  
**Hardware baseline used in tests:** NVIDIA Tesla T4 (~14.9 GB VRAM)  
**Purpose:** Lock the best production model stack, then choose a VPS / GPU cloud provider with capacity, bandwidth, and cost estimates.

---

## 1. Executive recommendation

### Best model stack (production default)

| Family | Winner | Role | Key evidence |
|--------|--------|------|--------------|
| **ASR** | **Whisper-Large-v3-Turbo-CT2** | Default realtime listener | Robot score **89.62**, Acc **68.4%**, RTF **0.046** (~22× realtime), VRAM **~2.5 GB** |
| **LLM** | **Nile-Chat-4B** | Default conversational brain | Robot score **95.41**, TTFT **0.96 s**, **11.56 tok/s**, VRAM **~8.4 GB** |
| **TTS** | **VoiceTut-TTS** | Default speaker | Robot score **90.02**, RTF **0.157** (~6.4×), **82.7 chars/s**, VRAM **~3.15 GB** |

**Why this trio:** Best overall accuracy/speed/VRAM blend for a live Arabic (Egyptian) voice robot on one mid-tier GPU. Combined naive peak VRAM ≈ **2.5 + 8.4 + 3.15 ≈ 14.0 GB**, which fits a T4 tightly and fits comfortably on **24 GB** (RTX 4090 / A40-class) with concurrency headroom.

### Quality / alternate picks (when UX priority flips)

| Need | Pick | Tradeoff |
|------|------|----------|
| Higher ASR word accuracy | Whisper-Large-v3-CT2 | Acc **72.92%**, but VRAM **~4.3 GB**, slower RTF **0.133** |
| Higher LLM answer quality | **Qwen3-8B (int4)** | Quality **4.65/5**, auto-pass **85%** vs Nile **2.81/5** / **20%** — but slower TTFT **1.94 s** |
| Balanced LLM mid-GPU | Qwen3-4B-Instruct-2507 | Quality **4.41/5**, TTFT **0.95 s**, VRAM **~8.8 GB** |
| TTS voice naturalness wins | **VoiceTut-TTS** (listen first) | Keep VoiceTut after human listening QA for Egyptian dialect / code-switch quality |

> **TTS caveat (from bake-off):** robot scores cover speed/resources only. Confirm Egyptian dialect / code-switch naturalness by listening to WAVs before freezing VoiceTut.

### Best VPS / GPU cloud to interact with

**Primary pick: RunPod (Secure Cloud) — RTX 4090 24 GB**

| Criterion | Why RunPod wins for this project |
|-----------|----------------------------------|
| Interaction | Full SSH/root pods, Docker templates, REST API, serverless endpoints |
| Fit for stack | 24 GB VRAM = ASR+LLM+TTS co-resident with headroom |
| Billing | Per-second (good for iteration + scale-to-zero patterns) |
| Ops | Template deploy in ~30s; Community for cheap R&D, Secure for production |
| Cost vs peers | RTX 4090 ~**$0.34/hr** Community / ~**$0.69/hr** Secure (July 2026 listings) |

**Secondary:** Vast.ai (cheapest RTX 4090, weaker SLA) · **Hetzner** (predictable EU dedicated / monthly) · **DigitalOcean** (simplest DX, higher $/hr).

#### RunPod Community vs Secure — detailed difference

RunPod sells the **same GPU SKUs** (e.g. RTX 4090) in two clouds. The GPU chip can be identical; **the hosting environment, reliability, and price are not**.

| Dimension | **Community Cloud** | **Secure Cloud** |
|-----------|---------------------|------------------|
| What it is | Peer-to-peer style marketplace of **third-party / partner hosts** invited and vetted by RunPod | Hardware in **Tier 3 / Tier 4 datacenters** run with RunPod’s trusted partners (enterprise-style facilities) |
| Price (RTX 4090, July 2026) | ~**$0.34/hr** (cheaper) | ~**$0.69/hr** (~2× Community) |
| Reliability | Good hosts exist, but **no strong formal uptime SLA**; machines can go offline / be less redundant on power & network | Built for production: redundancy, faster incident response; marketed with high uptime (RunPod cites **99.9%+ / up to 99.99%** class SLAs on Secure / platform docs) |
| Security / compliance | Multi-tenant container isolation; weaker story for regulated data | Better isolation story; SOC 2 Type II / partner certs (SOC 2, ISO 27001, etc. depending on site) — preferred for **client voice audio** and production PII |
| Availability | Supply/price can vary by region and time of day | More predictable capacity for always-on APIs |
| Best for | Dev, staging, bake-offs, non-critical batch jobs | **Live Arabic voice robot serving real users** |
| Risk if it fails mid-call | User hears timeout / dropped session; hard to explain to a client | Much lower chance of random host disappearance |

**Practical rule for this project**

1. Build and test on **Community** (save money while iterating).  
2. Move the public API to **Secure** before client demos / soft launch.  
3. Same Docker image and model weights transfer between the two — you are mostly paying for **reliability + compliance**, not a different GPU model.

---

## 2. Bake-off evidence summary

### 2.1 ASR leaderboard (top)

| Rank | Model | Robot | Acc% | WER | RTF | VRAM MB |
|-----:|-------|------:|-----:|----:|----:|--------:|
| 1 | Whisper-Large-v3-Turbo-CT2 | 89.62 | 68.40 | 0.316 | 0.046 | 2492 |
| 2 | Whisper-Large-v3-CT2 | 87.92 | 72.92 | 0.271 | 0.133 | 4316 |
| 3 | Whisper-Small-CT2 | 86.19 | 63.19 | 0.368 | 0.084 | 1340 |
| 4 | Arabic-Whisper-Turbo-FT-CT2 | 83.38 | 60.76 | 0.392 | 0.046 | 1692 |

Arabic fine-tunes were competitive on speed but did **not** beat base Whisper Large on accuracy in this clip bake-off. Voxtral-Mini is accurate but VRAM-heavy (~11 GB) — poor co-resident with LLM+TTS.

### 2.2 LLM leaderboard

| Rank | Model | Robot | TTFT s | tok/s | VRAM MB | Quality /5 | Auto-pass |
|-----:|-------|------:|-------:|------:|--------:|-----------:|----------:|
| 1 | Nile-Chat-4B | 95.41 | 0.96 | 11.56 | 8394 | 2.81 | 20% |
| 2 | Qwen3-4B-Instruct-2507 | 91.61 | 0.95 | 10.48 | 8758 | 4.41 | 80% |
| 3 | Qwen3-8B (int4) | 30.66 | 1.94 | 5.16 | 6900 | **4.65** | **85%** |
| 4 | ALLaM-7B | 4.15 | 2.47 | 5.62 | 15280 | 4.25 | 85% |

**Decision rule from client PDFs:** prefer Nile when turn-taking feel matters; switch to Qwen3-8B when answer correctness / TTS-fit matter more. ALLaM-7B exceeds a T4 alone — avoid on ≤16 GB cards.

### 2.3 TTS leaderboard

| Rank | Model | Robot | RTF | chars/s | VRAM MB | Realtime? |
|-----:|-------|------:|----:|--------:|--------:|-----------|
| 1 | VoiceTut-TTS | 90.02 | 0.157 | 82.7 | 3150 | yes |
| 2 | SILMA-TTS | 77.42 | 0.238 | 46.8 | 3340 | yes |
| 3 | Chatterbox-Multilingual-V3 | 28.63 | 1.101 | 13.4 | 4790 | no |
| 4 | NAMAA-Egyptian-TTS | 0.61 | 1.232 | 15.1 | 6066 | no |

### 2.4 Combined VRAM budget

| Stack | ASR | LLM | TTS | Naive peak sum | Fits |
|-------|-----|-----|-----|----------------|------|
| **Speed (default)** | Turbo ~2.5 GB | Nile ~8.4 GB | VoiceTut ~3.2 GB | **~14.0 GB** | T4 tight / **24 GB OK** |
| **Quality** | Large-v3 ~4.3 GB | Qwen3-8B ~6.9 GB | VoiceTut ~3.2 GB | **~14.4 GB** | T4 tight / **24 GB OK** |
| Avoid | + ALLaM 15 GB | — | — | overflows mid GPUs | need ≥24–48 GB |

Production rule: deploy on **≥24 GB** even if T4 “fits,” so KV-cache, CUDA context, and 2–3 queued turns do not OOM.

---

## 3. Workload model (users, usage, bandwidth)

This section sizes **how many people** the robot can serve, **how much they talk**, and **how much network/GPU** that consumes — using **production-style** assumptions (not lab bake-off clip lengths).

### Glossary (key metrics)

| Term | Meaning |
|------|---------|
| **DAU** | **Daily Active Users** — unique people who complete **at least one real voice session** that calendar day |
| **MAU** | **Monthly Active Users** — unique people active in a 30-day window. For sticky consumer apps, DAU is often ~**15–25%** of MAU (industry stickiness band); AI products often land near ~**20%** DAU/MAU |
| **Session** | One continuous conversation (user opens the robot, talks, then leaves) |
| **Turn** | One user utterance + one robot reply |
| **Peak concurrent sessions** | How many users are **in a live call at the same second** during the busiest minute — this drives GPU count, not DAU alone |
| **GPU-busy concurrency** | How many turns are **actively using the GPU** at once (usually lower than “connected” sessions if you use VAD / push-to-talk) |

### Production assumptions (realistic)

Voice products in the wild are **not** 5–6 second lab clips. Conversational / assistant sessions often run **many minutes**; individual user speaking stretches commonly land in the **30 seconds – 2 minutes** range when users explain a problem, dictate, or give multi-step instructions. Voice-heavy platforms also report longer average sessions than text chat (often tens of minutes when users stay in a voice room; for a **task robot**, expect shorter sessions but still multi-turn).

| Parameter | Production assumption | Why this number |
|-----------|----------------------|-----------------|
| Avg **user utterance** (speech in) | **~60 s** (range **30 s – 2 min**) | Mid-point of real explanatory speech; short commands exist, but capacity must plan for long turns |
| Avg **robot reply** (speech out) | **~35 s** | Assistants usually speak less than long user monologues, but Arabic answers + explanations are not 5 s |
| Sessions / active user / day | Pilot **1** → Scale **2–3** | Most users open the robot 1–2 times/day early; power users more |
| Turns / session | **5–8** | Multi-step tasks (ask → clarify → confirm → follow-up) |

### 3.1 Audio & GPU usage by stage

**What each stage means**

| Stage | Meaning in product terms | Who is using it |
|-------|--------------------------|-----------------|
| **Pilot** | Closed alpha / friends & family / internal QA | Team + invited testers; traffic is low and spiky |
| **Soft launch** | Limited public beta (one city, one partner, waitlist) | Early adopters; still manually supported |
| **Growth** | Open product with marketing; usage climbing week over week | Real customers; need SLA and monitoring |
| **Scale** | Established product; multiple campaigns / B2B seats | Must autoscaling; one GPU is no longer enough |
| **Mature** (extra) | Large consumer or multi-tenant B2B footprint | Multi-region / multi-GPU fleet |

**Workload table (production speech lengths)**

Formulas used:

- Audio-in hours/day ≈ `DAU × turns/user/day × 60 s / 3600`
- Audio-out hours/day ≈ `DAU × turns/user/day × 35 s / 3600`
- GPU-seconds/day ≈ turns/day × **~18–25 s** wall time per turn on one sequential ASR→LLM→TTS pipeline (Whisper Turbo RTF ~0.05 on ~60 s ≈ 3 s; LLM ~3–8 s; VoiceTut RTF ~0.16 on ~35 s ≈ 6 s; plus queue/IO)

| Stage | DAU | Approx. MAU* | Turns/user/day | Audio-in h/day | Audio-out h/day | Turns/day | Est. GPU-seconds / day |
|-------|----:|-------------:|---------------:|---------------:|----------------:|----------:|-----------------------:|
| Pilot | **100** | ~500 | 6 | **10** | **5.8** | 600 | ~12,000–15,000 |
| Soft launch | **1,000** | ~5,000 | 8 | **133** | **78** | 8,000 | ~160,000–200,000 |
| Growth | **5,000** | ~25,000 | 10 | **833** | **486** | 50,000 | ~1.0M–1.3M |
| Scale | **25,000** | ~100,000 | 12 | **5,000** | **2,920** | 300,000 | ~6M–7.5M |
| Mature | **100,000** | ~400,000 | 12 | **20,000** | **11,700** | 1,200,000 | fleet / multi-region |

\*MAU ≈ DAU / 0.20 using a ~**20% DAU/MAU** stickiness assumption (typical AI / consumer band).

These audio-hour numbers are in the same order of magnitude as high-traffic self-hosted speech systems once you leave “demo clip” lengths — **hundreds to thousands of audio hours/day** at Growth/Scale — which is why GPU count and queueing matter more than raw disk bandwidth.

### 3.2 Concurrent capacity (one GPU) — explained

With ASR + LLM + TTS on **one RTX 4090**, the GPU usually handles **one full turn at a time** (or a tiny queue). Longer turns (~60 s in + ~35 s out) take roughly **20 s** of GPU work each, so one card sustains about **3–5 turns per minute**.

Users who are silent (VAD / push-to-talk) do not burn the GPU, so one pod can keep roughly **20–40 live sessions** connected. Size the fleet from **peak concurrent sessions**, not from DAU alone — e.g. 5,000 DAU at a 2% peak ⇒ ~100 concurrent ⇒ you need several GPUs, not one.

### Scaling rule of thumb — why these numbers

We size GPUs from **peak concurrent connected sessions**, assuming:

- ~**1 GPU handles ~20–40** comfortable connected sessions (VAD), with **1–3** GPU-busy turns  
- Above that, add GPUs **linearly** and put a queue / load balancer in front  
- Beyond ~100 concurrent, prefer **RunPod Serverless workers** or many pods so cold capacity can appear only at peak

| Peak concurrent voice sessions | GPUs needed (24 GB) | Why |
|-------------------------------:|--------------------:|-----|
| ≤ 25 | **1** | Fits one 4090 connected budget with margin |
| 26–50 | **2** | Split traffic; failover if one pod dies |
| 51–120 | **4–6** | Growth-stage peaks; keep p95 latency stable |
| 121–250 | **8–12** | Scale-stage; still pod-based is fine |
| 250+ | **Serverless / large pod fleet** | Spiky load; paying 24/7 for max peak wastes money |

### 3.3 Bandwidth estimates (realistic)

**Per turn** (Opus 32 kbps ≈ 4 KB/s each way):

- User speech ~60 s → **~240 KB** in  
- Robot speech ~35 s → **~140 KB** out  
- +25% overhead → **~475 KB / turn** (~0.46 MB)

| Stage | Turns/day | Data/day | Data/month | Notes |
|-------|----------:|---------:|-----------:|-------|
| Pilot (100 DAU) | 600 | ~0.28 GB | **~8 GB** | Still small |
| Soft (1,000) | 8,000 | ~3.7 GB | **~110 GB** | Usually inside generous VPS allowances |
| Growth (5,000) | 50,000 | ~23 GB | **~700 GB** | Watch egress fees on hyperscalers |
| Scale (25,000) | 300,000 | ~140 GB | **~4.2 TB** | Need cheap egress (Hetzner-like) or CDN; don’t archive WAV on the GPU box |
| Mature (100,000) | 1,200,000 | ~560 GB | **~17 TB** | Object storage + egress contracts required |
| + raw WAV archive (16-bit/16 kHz mono) | — | — | **×10–20** vs Opus | Keep compressed Opus for transport; archive selectively |

**Always-open mic worst case** (no VAD): ~8 KB/s bidirectional ≈ **~0.7 GB/hour/session**. Twenty always-on sessions ≈ **~10 TB/month** — avoid this design; use **VAD or push-to-talk**.

**Egress watchlist:** AWS / some DigitalOcean paths bill egress heavily; Hetzner / Contabo often include multi-TB; RunPod / Vast.ai compute is separate from **public egress and volume storage** — confirm in the console before Scale.

### 3.4 Storage

| Item | Size guidance |
|------|----------------|
| Model weights (Turbo + Nile + VoiceTut) | ~**15–25 GB** on disk |
| Docker / CUDA / deps | ~**20–40 GB** |
| Working / logs (7 days) | ~**20–100 GB** at Growth |
| Recommended disk | **150–300 GB** NVMe on the GPU pod + object storage for archives |

---

## 4. VPS / GPU provider comparison (July 2026)

Prices move weekly. Figures below are **public on-demand listings** from provider pages and 2026 comparison aggregators. **Re-check before purchase.**

### 4.1 Scorecard (fit for *this* Arabic voice stack)

| Provider | Interact (SSH/API) | RTX 4090-class $/hr | Reliability | Always-on monthly* | Best use | Score /10 |
|----------|--------------------|--------------------:|-------------|-------------------:|----------|----------:|
| **RunPod Secure** | Excellent | ~0.69 | High (DC) | ~$497 | **Production default** | **9.0** |
| **RunPod Community** | Excellent | ~0.34 | Medium | ~$245 | Dev / staging | 8.2 |
| **Vast.ai** | Good | ~0.27–0.37 | Variable (P2P) | ~$195–270 | Cheap experiments | 6.5 |
| **Hetzner** | Excellent | ~0.33 (RTX 4000 Ada 20GB) | High EU | ~$240 | Always-on EU, predictable | 7.8 |
| **DigitalOcean** | Excellent | RTX 4000 ~0.76; H100 3.39 | High | RTX4k ~$547 | Simple DX / team | 7.0 |
| **Vultr** | Excellent | A100 from ~1.29; H100 ~2.30 | High | A100 ~$930 | Enterprise-ish | 6.8 |
| **Lambda Labs** | Good | No 4090; A100/H100 focus | High + SLA | A100 ~$940+ | Training / big models | 6.0 |
| **Contabo** | Good | 4090 ~0.89 listed; monthly AI boxes | Medium | 4090 ~$640+ / Ada4000 ~€549 | Budget monthly EU | 6.2 |
| **AWS / GCP / Azure** | Excellent | 2–3× specialist rates | Highest | Very high | Compliance only | 4.5 |

\*24/7 ≈ ×720 hours. Spot / reserved can cut 20–50%.

### 4.2 Why RunPod over Vast.ai if the GPU is “the same”?

Yes — both can rent an **RTX 4090 24 GB**. You still pick **RunPod Secure** for production because you are buying **operations quality**, not only FLOPS:

| Factor | RunPod Secure | Vast.ai marketplace |
|--------|---------------|---------------------|
| Who owns/runs the box | Datacenter partners under RunPod’s Secure program | Random hosts bidding prices |
| Uptime / eviction | Production-oriented; formal SLA story | Host can be slow, full, or interruptible; reliability scores vary |
| API / DX | Strong pod + **Serverless** product, templates, docs | Good CLI/API, but more “DIY marketplace” |
| Client voice data | Better compliance / isolation narrative | Harder to defend in a client security review |
| Debugging a live outage | One vendor support path | “Which host failed?” becomes your problem |
| Price | Higher | Often **cheapest** 4090 |

**Use Vast.ai** when: training experiments, load tests, non-customer traffic, or you accept interrupt risk to save ~30–50%.  
**Use RunPod Secure** when: real users speak Arabic into your robot and downtime = lost trust.

Community RunPod sits **between** them: cheaper than Secure, more curated than raw Vast.ai, still weaker than Secure for go-live.

### 4.3 Serverless / pay-for-what-you-use — 3 hours/day vs full day

**Important:** “Serverless” and “Pods” bill differently.

| Product style | Providers | If you only need **3 GPU hours / day** | If you leave it on **24/7** |
|---------------|-----------|----------------------------------------|-----------------------------|
| **Serverless / scale-to-zero** | **RunPod Serverless**, **Vast.ai Serverless** | You pay roughly **3 hours × $/hr** of worker time (plus short idle timeout after last request, often seconds–minutes; plus storage) — **not** a full day | Only if traffic is continuous or you keep “active workers” warm |
| **On-demand Pod / instance you start & stop** | RunPod Pods, Vast.ai instances | **3 hours × $/hr** if you **stop/destroy** after use | Full **24 × $/hr** if you forget to stop |
| **VM that still bills when powered off** | **DigitalOcean GPU Droplets** (official caveat) | Power-off may **still bill** reserved GPU — you must **destroy** to stop compute | Full month either way if the droplet exists |
| **Monthly dedicated** | Contabo / some Hetzner dedicated | You usually pay the **full month** even if you use 3 hours/day | Same monthly price |

**Example — RTX 4090 ~$0.69/hr Secure Pod**

| Usage pattern | Approx. monthly compute |
|---------------|-------------------------|
| 3 hours/day × 30 days = **90 h** | 90 × 0.69 ≈ **$62** (if stopped when idle) |
| Business day 10 h × 22 days = **220 h** | ≈ **$152** |
| Always on 720 h | ≈ **$497** |

Same math on Community (~$0.34/hr) or Vast.ai (~$0.30/hr): **3 hours/day ≈ 90 × rate**, **not** the 24/7 price — **only if** the product is truly metered and you (or Serverless) scale to zero.

**Serverless caveat for voice:** cold start (container + load ~14 GB of models) can add **seconds to tens of seconds** unless you keep a warm worker. For a live robot, many teams keep **1 warm Secure pod** during business hours and use Serverless only for overflow.

### 4.4 Detailed provider notes

#### RunPod (recommended)

- **Products:** Pods (persistent), Serverless (autoscaling endpoints), Instant Clusters.
- **Tiers:** Community (cheaper hosts) vs Secure (vetted datacenters) — see §1 detailed table.
- **Why it fits:** Deploy Docker with CUDA, expose HTTP for ASR/LLM/TTS APIs, scale pods when concurrency rises, stop billing when terminated (per-second).
- **Suggested SKUs:**
  - Dev: **1× RTX 4090 Community** (~$0.34/hr)
  - Prod: **1× RTX 4090 Secure** (~$0.69/hr) → add pods as concurrency grows
  - Optional later: Serverless workers for burst traffic
- **Watch:** Network volume storage is billed even when compute is stopped; keep models on a volume, destroy unused pods.

#### Vast.ai

- Marketplace / bidding; often cheapest 4090 (~$0.27–0.37/hr median).
- Also has **Serverless** (per-second, autoscaling) similar in *billing shape* to RunPod Serverless.
- **Pros:** Cost. **Cons:** Host variability, interrupt risk, weaker production SLA / client compliance story.
- Use for bake-offs and load tests, not sole production path for a client-facing robot.

#### Hetzner

- Strong EU footprint, generous traffic pools on many plans, good SSH/root UX.
- RTX 4000 Ada ~20 GB (~$0.33/hr listed) — **borderline** for quality stack; prefer RTX 6000 Ada / larger if available.
- Best when you want **stable monthly** EU hosting and lower surprise egress bills (often closer to “full month” economics).

#### DigitalOcean

- Excellent docs, VPC, managed DB, simple GPU Droplets.
- RTX 4000 Ada **$0.76/hr**; L40S **$1.57/hr**; H100 **$3.39/hr** (official docs, 2026).
- **Caveat:** powered-off GPU Droplets **still bill** — destroy to stop charges (not true scale-to-zero serverless for this stack).
- Good if the team already lives in DO; otherwise RunPod is cheaper for 4090-class inference.

#### Vultr / Lambda / Contabo

- **Vultr:** solid Cloud GPU (A100/H100); overkill $ for this ≤24 GB stack.
- **Lambda:** great for training clusters / SLA; weak consumer-GPU catalog.
- **Contabo:** monthly AI dedicated (e.g. Ada 4000 ~€549/mo listed) can beat 24/7 RunPod Secure if utilization is always-on — but **3 hours/day still costs the full month**.

### 4.5 Cost scenarios for *our* stack

Assumptions: **1× RTX 4090**, public API.

| Scenario | Hours GPU / mo | RunPod Community | RunPod Secure | Vast.ai (~$0.30) | Notes |
|----------|---------------:|-----------------:|--------------:|-----------------:|-------|
| 3 h/day metered (Serverless or start/stop) | 90 | ~$31 | ~$62 | ~$27 | **True pay-per-use** |
| Nightly tests only | 60 | ~$20 | ~$41 | ~$18 | CI / bake-offs |
| Business hours API (10×22) | 220 | ~$75 | ~$152 | ~$66 | Stop pod nights |
| Soft launch 24/7 | 720 | ~$245 | ~$497 | ~$216 | 1 GPU warm always |
| Growth (2 GPUs 24/7) | 1,440 | ~$490 | ~$994 | ~$432 | ~5k DAU class peaks |
| Scale (6 GPUs 24/7) | 4,320 | ~$1,470 | ~$2,980 | ~$1,300 | ~25k DAU class |

Add ~**$10–40/mo** object storage + ~**$5–20/mo** small CPU edge (TLS / gateway) if split.

**Break-even insight:** If real usage is only a few hours/day, **RunPod/Vast metered or Serverless** wins. If the robot must answer with **low latency 24/7**, compare Secure 24/7 vs Contabo/Hetzner monthly quotes.

---

## 5. Recommended deployment architecture

```
Clients (mobile / web)
        │  Opus / WebSocket
        ▼
Edge VPS (CPU, optional)  ← TLS, auth, rate limits, session router
        │
        ▼
GPU Pod(s) — RunPod Secure RTX 4090
   ┌─────────────────────────────────────┐
   │  Whisper-Large-v3-Turbo-CT2 (ASR)   │
   │  Nile-Chat-4B (LLM)  [or Qwen3-8B]  │
   │  VoiceTut-TTS (TTS)                 │
   └─────────────────────────────────────┘
        │
        ▼
Object storage (WAV logs, analytics) — optional region near EG/EU
```

**Interaction checklist (why RunPod works for the team)**

1. Create Secure pod with CUDA + PyTorch template  
2. `scp` / volume-mount weights; expose FastAPI/gRPC ports  
3. Automate via RunPod API (create/stop pods from CI)  
4. Add second pod when p95 turn latency > target (e.g. 3 s)  
5. Move burst traffic to Serverless workers if traffic is spiky  

---

## 6. Final selection matrix

| Decision | Choice |
|----------|--------|
| Default ASR | **Whisper-Large-v3-Turbo-CT2** |
| Default LLM | **Nile-Chat-4B** (speed UX) |
| Quality LLM alt | **Qwen3-8B int4** |
| Default TTS | **VoiceTut-TTS** (listen to confirm) |
| GPU class | **24 GB** (RTX 4090 or better) |
| Provider | **RunPod Secure Cloud** |
| Dev / cost lane | RunPod Community or Vast.ai |
| Always-on EU alt | Hetzner / Contabo monthly quote |
| Pilot capacity | **1 GPU → ~100–500 DAU** |
| Soft launch | **1–2 GPUs → ~1,000–2,500 DAU** |
| Growth | **2–6 GPUs → ~5,000 DAU class** |
| Scale | **Autoscaling pods / Serverless; plan multi-TB egress if WAV archived** |

---

## 7. Sources

**External (pricing / comparison, July 2026)**

- https://klymentiev.com/blog/vps-with-gpu  
- https://klymentiev.com/blog/runpod-vs-lambda-vs-vast  
- https://www.runpod.io/product/cloud-gpus  
- https://docs.runpod.io/serverless/pricing  
- https://docs.runpod.io/serverless/overview  
- https://www.gpucloudlist.com/en/blog/lambda-labs-vs-runpod-vs-vast-ai  
- https://docs.digitalocean.com/products/droplets/details/pricing/  
- https://docs.vast.ai/examples/migrations/runpod-to-vast  
- https://getdeploying.com/gpus/nvidia-rtx-4000-ada  
- https://gputracker.dev/provider/contabo  
- https://aicostcalculators.com/gpu-cloud-pricing-comparison/  
- https://mixpanel.com/blog/mau/  
- https://diyai.io/ai-tools/hosting/best-gpu-hosting-for-ai/  

---

*This file is a planning deliverable. Spot-check live provider dashboards before committing budget — GPU spot rates change frequently.*

**All of this research made by Eng. Mohamed Soltan.**
