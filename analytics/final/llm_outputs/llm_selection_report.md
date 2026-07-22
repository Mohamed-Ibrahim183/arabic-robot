# LLM Model Selection Report

Auto-generated from the Kaggle Arabic LLM bake-off.
Combine latency/resource rankings with manual review of Arabic response quality.

## Run summary

- Total prompt runs: **83**
- OK: **80** | Failed: **3**
- Models with ≥1 OK run: **4**

## Recommended picks

### `best_for_robot_realtime`

- **Model:** `Nile-Chat-4B`
- **Why:** Best composite of TTFT + tok/s + VRAM + load for conversational turns.
- **Metrics:** score_robot_realtime=95.41, avg_first_token_seconds=0.96, avg_tokens_per_second=11.56, peak_vram_mb=8394.2

### `lowest_ttft`

- **Model:** `Qwen3-4B-Instruct-2507`
- **Why:** Fastest average time-to-first-token (best turn-taking feel).
- **Metrics:** avg_first_token_seconds=0.947

### `highest_throughput`

- **Model:** `Nile-Chat-4B`
- **Why:** Highest average tokens/second.
- **Metrics:** avg_tokens_per_second=11.56

### `lowest_vram`

- **Model:** `Qwen3-8B`
- **Why:** Lowest peak VRAM — better for small GPUs / co-residency.
- **Metrics:** peak_vram_mb=6900.2

### `best_balanced`

- **Model:** `Nile-Chat-4B`
- **Why:** Balanced latency/throughput/VRAM tradeoff.
- **Metrics:** score_balanced=93.75

## Category specialists (lowest TTFT per category)

- **asr_noise:** `Nile-Chat-4B` (TTFT=0.551, tok/s=11.8)
- **code_switching:** `Nile-Chat-4B` (TTFT=0.529, tok/s=13.54)
- **egyptian_arabic:** `Nile-Chat-4B` (TTFT=0.504, tok/s=12.19)
- **instruction_following:** `Nile-Chat-4B` (TTFT=0.61, tok/s=13.21)
- **multi_turn_memory:** `Qwen3-4B-Instruct-2507` (TTFT=0.719, tok/s=5.19)
- **structured_output:** `Nile-Chat-4B` (TTFT=0.747, tok/s=13.12)
- **tool_calling:** `Qwen3-4B-Instruct-2507` (TTFT=2.22, tok/s=7.77)

## Robot realtime leaderboard

| Rank | Model | Robot | TTFT avg | tok/s | VRAM pk | Success% | load |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `Nile-Chat-4B` | 95.41 | 0.96 | 11.56 | 8394.2 | 100.0 | bf16 |
| 2 | `Qwen3-4B-Instruct-2507` | 91.61 | 0.947 | 10.48 | 8758.2 | 100.0 | bf16 |
| 3 | `Qwen3-8B` | 30.66 | 1.942 | 5.16 | 6900.2 | 100.0 | int4 |
| 4 | `ALLaM-7B` | 4.15 | 2.473 | 5.62 | 15280.2 | 100.0 | bf16 |

## Per-model aggregate detail

### `Qwen3-4B-Instruct-2507`

- Runs: 20/20 OK (100.0%)
- Latency: TTFT avg=0.947 (min=0.551, max=2.276, std=0.659)
- Throughput: tok/s avg=10.48 (min=4.71, max=16.96, std=3.77)
- Timing: load_avg=138.72s, generate_avg=3.774s
- Resources: CPU pk=139.8%, RAM pk=6116.8MB, sysRAM pk=4027.7MB, GPU pk=100.0%, VRAM pk=8758.2MB, model VRAM=8206.0MB
- Load mode: bf16 / dtype=bfloat16

### `Qwen3-8B`

- Runs: 20/20 OK (100.0%)
- Latency: TTFT avg=1.942 (min=1.185, max=4.452, std=1.229)
- Throughput: tok/s avg=5.16 (min=1.55, max=8.23, std=1.93)
- Timing: load_avg=313.72s, generate_avg=4.634s
- Resources: CPU pk=142.2%, RAM pk=6077.5MB, sysRAM pk=4352.3MB, GPU pk=100.0%, VRAM pk=6900.2MB, model VRAM=6348.0MB
- Load mode: int4 / dtype=bfloat16

### `Gemma3-4B-IT`

- Runs: 0/1 OK (0.0%)
- Latency: TTFT avg= (min=, max=, std=)
- Throughput: tok/s avg= (min=, max=, std=)
- Timing: load_avg=s, generate_avg=s
- Resources: CPU pk=%, RAM pk=MB, sysRAM pk=MB, GPU pk=%, VRAM pk=MB, model VRAM=MB
- Load mode:  / dtype=

### `Gemma3-12B-IT`

- Runs: 0/1 OK (0.0%)
- Latency: TTFT avg= (min=, max=, std=)
- Throughput: tok/s avg= (min=, max=, std=)
- Timing: load_avg=s, generate_avg=s
- Resources: CPU pk=%, RAM pk=MB, sysRAM pk=MB, GPU pk=%, VRAM pk=MB, model VRAM=MB
- Load mode:  / dtype=

### `ALLaM-7B`

- Runs: 20/20 OK (100.0%)
- Latency: TTFT avg=2.473 (min=1.291, max=7.216, std=2.318)
- Throughput: tok/s avg=5.62 (min=1.27, max=10.53, std=2.91)
- Timing: load_avg=278.82s, generate_avg=4.294s
- Resources: CPU pk=139.6%, RAM pk=6073.8MB, sysRAM pk=4146.1MB, GPU pk=100.0%, VRAM pk=15280.2MB, model VRAM=14728.0MB
- Load mode: bf16 / dtype=bfloat16

### `Jais-2-8B`

- Runs: 0/1 OK (0.0%)
- Latency: TTFT avg= (min=, max=, std=)
- Throughput: tok/s avg= (min=, max=, std=)
- Timing: load_avg=s, generate_avg=s
- Resources: CPU pk=%, RAM pk=MB, sysRAM pk=MB, GPU pk=%, VRAM pk=MB, model VRAM=MB
- Load mode:  / dtype=

### `Nile-Chat-4B`

- Runs: 20/20 OK (100.0%)
- Latency: TTFT avg=0.96 (min=0.499, max=2.435, std=0.754)
- Throughput: tok/s avg=11.56 (min=3.6, max=14.11, std=2.99)
- Timing: load_avg=165.63s, generate_avg=13.081s
- Resources: CPU pk=138.2%, RAM pk=6163.4MB, sysRAM pk=4518.4MB, GPU pk=100.0%, VRAM pk=8394.2MB, model VRAM=7842.0MB
- Load mode: bf16 / dtype=bfloat16

## Category breakdown

| Model | Category | OK | TTFT avg | tok/s | out tok avg |
|---|---|---:|---:|---:|---:|
| `Qwen3-4B-Instruct-2507` | egyptian_arabic | 3/3 | 0.559 | 13.38 | 59.3 |
| `Qwen3-4B-Instruct-2507` | code_switching | 2/2 | 0.595 | 14.03 | 120.5 |
| `Qwen3-4B-Instruct-2507` | asr_noise | 4/4 | 0.58 | 12.3 | 23.5 |
| `Qwen3-4B-Instruct-2507` | instruction_following | 3/3 | 0.642 | 12.72 | 77.7 |
| `Qwen3-4B-Instruct-2507` | multi_turn_memory | 3/3 | 0.719 | 5.19 | 5.7 |
| `Qwen3-4B-Instruct-2507` | tool_calling | 4/4 | 2.22 | 7.77 | 32.2 |
| `Qwen3-4B-Instruct-2507` | structured_output | 1/1 | 0.789 | 7.32 | 27.0 |
| `Qwen3-8B` | egyptian_arabic | 3/3 | 1.198 | 6.19 | 31.3 |
| `Qwen3-8B` | code_switching | 2/2 | 1.224 | 7.33 | 40.0 |
| `Qwen3-8B` | asr_noise | 4/4 | 1.24 | 6.07 | 23.8 |
| `Qwen3-8B` | instruction_following | 3/3 | 1.366 | 5.84 | 23.0 |
| `Qwen3-8B` | multi_turn_memory | 3/3 | 1.616 | 2.18 | 5.0 |
| `Qwen3-8B` | tool_calling | 4/4 | 4.311 | 4.03 | 31.8 |
| `Qwen3-8B` | structured_output | 1/1 | 1.653 | 5.49 | 24.0 |
| `Gemma3-4B-IT` | general | 0/1 |  |  |  |
| `Gemma3-12B-IT` | general | 0/1 |  |  |  |
| `ALLaM-7B` | egyptian_arabic | 3/3 | 1.295 | 8.15 | 28.7 |
| `ALLaM-7B` | code_switching | 2/2 | 1.32 | 9.5 | 35.0 |
| `ALLaM-7B` | asr_noise | 4/4 | 1.326 | 4.48 | 9.5 |
| `ALLaM-7B` | instruction_following | 3/3 | 1.35 | 6.98 | 19.7 |
| `ALLaM-7B` | multi_turn_memory | 3/3 | 1.387 | 2.13 | 3.7 |
| `ALLaM-7B` | tool_calling | 4/4 | 6.989 | 3.89 | 45.2 |
| `ALLaM-7B` | structured_output | 1/1 | 1.456 | 8.17 | 31.0 |
| `Jais-2-8B` | general | 0/1 |  |  |  |
| `Nile-Chat-4B` | egyptian_arabic | 3/3 | 0.504 | 12.19 | 103.7 |
| `Nile-Chat-4B` | code_switching | 2/2 | 0.529 | 13.54 | 246.5 |
| `Nile-Chat-4B` | asr_noise | 4/4 | 0.551 | 11.8 | 176.2 |
| `Nile-Chat-4B` | instruction_following | 3/3 | 0.61 | 13.21 | 179.0 |
| `Nile-Chat-4B` | multi_turn_memory | 3/3 | 0.721 | 13.22 | 243.0 |
| `Nile-Chat-4B` | tool_calling | 4/4 | 2.421 | 6.99 | 76.8 |
| `Nile-Chat-4B` | structured_output | 1/1 | 0.747 | 13.12 | 243.0 |

## Failures

- `Gemma3-4B-IT` | ``: Gated model google/gemma-3-4b-it requires HF auth. Accept the license on Hugging Face, then set HF_TOKEN (Kaggle Secrets / Colab Secrets / env) and re-run.
- `Gemma3-12B-IT` | ``: Gated model google/gemma-3-12b-it requires HF auth. Accept the license on Hugging Face, then set HF_TOKEN (Kaggle Secrets / Colab Secrets / env) and re-run.
- `Jais-2-8B` | ``: Gated model inceptionai/Jais-2-8B-Chat requires HF auth. Accept the license on Hugging Face, then set HF_TOKEN (Kaggle Secrets / Colab Secrets / env) and re-run.

## How to use these files

1. Open `llm_recommendations.json` for the primary pick.
2. Confirm with `llm_leaderboard.csv`.
3. Check category fit in `llm_analytics_by_category.csv`.
4. Drill into `llm_analytics.csv` + response texts under `responses/`.

## Notes

- Automated scores cover latency/throughput/resources only.
- Manually review Egyptian / MSA / code-switch response quality in llm_outputs/responses/.
- For robot UX, prioritize low TTFT even if peak tok/s is slightly lower.
- Disable thinking modes where available to reduce TTFT further.
