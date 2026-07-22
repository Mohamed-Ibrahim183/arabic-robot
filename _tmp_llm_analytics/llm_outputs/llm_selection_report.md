# LLM Model Selection Report

Auto-generated from the Kaggle Arabic LLM bake-off.
Combine latency/resource rankings with manual review of Arabic response quality.

## Run summary

- Total prompt runs: **7**
- OK: **4** | Failed: **3**
- Models with ≥1 OK run: **4**

## Recommended picks

### `best_for_robot_realtime`

- **Model:** `Qwen3-4B-Instruct-2507`
- **Why:** Best composite of TTFT + tok/s + VRAM + load for conversational turns.
- **Metrics:** score_robot_realtime=92.53, avg_first_token_seconds=0.618, avg_tokens_per_second=11.98, peak_vram_mb=8430.2

### `lowest_ttft`

- **Model:** `Nile-Chat-4B`
- **Why:** Fastest average time-to-first-token (best turn-taking feel).
- **Metrics:** avg_first_token_seconds=0.538

### `highest_throughput`

- **Model:** `Qwen3-4B-Instruct-2507`
- **Why:** Highest average tokens/second.
- **Metrics:** avg_tokens_per_second=11.98

### `lowest_vram`

- **Model:** `Qwen3-8B`
- **Why:** Lowest peak VRAM — better for small GPUs / co-residency.
- **Metrics:** peak_vram_mb=6722.2

### `best_balanced`

- **Model:** `Qwen3-4B-Instruct-2507`
- **Why:** Balanced latency/throughput/VRAM tradeoff.
- **Metrics:** score_balanced=91.59

## Category specialists (lowest TTFT per category)

- **egyptian_arabic:** `Nile-Chat-4B` (TTFT=0.538, tok/s=8.9)

## Robot realtime leaderboard

| Rank | Model | Robot | TTFT avg | tok/s | VRAM pk | Success% | load |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `Qwen3-4B-Instruct-2507` | 92.53 | 0.618 | 11.98 | 8430.2 | 100.0 | bf16 |
| 2 | `Nile-Chat-4B` | 85.42 | 0.538 | 8.9 | 8148.2 | 100.0 | bf16 |
| 3 | `Qwen3-8B` | 21.36 | 1.303 | 3.76 | 6722.2 | 100.0 | int4 |
| 4 | `ALLaM-7B` | 4.88 | 1.429 | 4.29 | 14194.2 | 100.0 | bf16 |

## Per-model aggregate detail

### `Qwen3-4B-Instruct-2507`

- Runs: 1/1 OK (100.0%)
- Latency: TTFT avg=0.618 (min=0.618, max=0.618, std=0.0)
- Throughput: tok/s avg=11.98 (min=11.98, max=11.98, std=0.0)
- Timing: load_avg=174.68s, generate_avg=2.588s
- Resources: CPU pk=141.5%, RAM pk=5571.9MB, sysRAM pk=3991.3MB, GPU pk=100.0%, VRAM pk=8430.2MB, model VRAM=7878.0MB
- Load mode: bf16 / dtype=bfloat16

### `Qwen3-8B`

- Runs: 1/1 OK (100.0%)
- Latency: TTFT avg=1.303 (min=1.303, max=1.303, std=0.0)
- Throughput: tok/s avg=3.76 (min=3.76, max=3.76, std=0.0)
- Timing: load_avg=387.37s, generate_avg=2.126s
- Resources: CPU pk=143.0%, RAM pk=5255.4MB, sysRAM pk=4035.8MB, GPU pk=100.0%, VRAM pk=6722.2MB, model VRAM=6170.0MB
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

- Runs: 1/1 OK (100.0%)
- Latency: TTFT avg=1.429 (min=1.429, max=1.429, std=0.0)
- Throughput: tok/s avg=4.29 (min=4.29, max=4.29, std=0.0)
- Timing: load_avg=324.8s, generate_avg=2.097s
- Resources: CPU pk=146.8%, RAM pk=5123.5MB, sysRAM pk=3965.1MB, GPU pk=100.0%, VRAM pk=14194.2MB, model VRAM=13642.0MB
- Load mode: bf16 / dtype=bfloat16

### `Jais-2-8B`

- Runs: 0/1 OK (0.0%)
- Latency: TTFT avg= (min=, max=, std=)
- Throughput: tok/s avg= (min=, max=, std=)
- Timing: load_avg=s, generate_avg=s
- Resources: CPU pk=%, RAM pk=MB, sysRAM pk=MB, GPU pk=%, VRAM pk=MB, model VRAM=MB
- Load mode:  / dtype=

### `Nile-Chat-4B`

- Runs: 1/1 OK (100.0%)
- Latency: TTFT avg=0.538 (min=0.538, max=0.538, std=0.0)
- Throughput: tok/s avg=8.9 (min=8.9, max=8.9, std=0.0)
- Timing: load_avg=184.85s, generate_avg=2.696s
- Resources: CPU pk=136.3%, RAM pk=5002.7MB, sysRAM pk=4342.5MB, GPU pk=98.0%, VRAM pk=8148.2MB, model VRAM=7596.0MB
- Load mode: bf16 / dtype=bfloat16

## Category breakdown

| Model | Category | OK | TTFT avg | tok/s | out tok avg |
|---|---|---:|---:|---:|---:|
| `Qwen3-4B-Instruct-2507` | egyptian_arabic | 1/1 | 0.618 | 11.98 | 31.0 |
| `Qwen3-8B` | egyptian_arabic | 1/1 | 1.303 | 3.76 | 8.0 |
| `Gemma3-4B-IT` | general | 0/1 |  |  |  |
| `Gemma3-12B-IT` | general | 0/1 |  |  |  |
| `ALLaM-7B` | egyptian_arabic | 1/1 | 1.429 | 4.29 | 9.0 |
| `Jais-2-8B` | general | 0/1 |  |  |  |
| `Nile-Chat-4B` | egyptian_arabic | 1/1 | 0.538 | 8.9 | 24.0 |

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
