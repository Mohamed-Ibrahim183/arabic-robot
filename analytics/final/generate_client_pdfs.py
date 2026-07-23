"""Generate balanced text+chart client PDF reports from analytics/final.

Outputs (in analytics/final/client_pdfs/):
  1. LLM_Model_Selection_Report.pdf
  2. ASR_Model_Selection_Report.pdf
  3. TTS_Model_Selection_Report.pdf
  4. Combined_Families_Summary.pdf
"""

from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "client_pdfs"

C_BLUE = "#2b6cb0"
C_GREEN = "#2f855a"
C_AMBER = "#c05621"
C_RED = "#c53030"
C_PURPLE = "#6b46c1"
C_TEAL = "#319795"
C_GRAY = "#718096"
C_NAVY = "#1a365d"
C_SOFT = "#edf2f7"
C_LINE = "#cbd5e0"
PALETTE = [C_BLUE, C_GREEN, C_AMBER, C_RED, C_PURPLE, C_TEAL, "#dd6b20", "#38a169", "#3182ce", "#805ad5", "#d69e2e", "#e53e3e"]

for candidate in ("Segoe UI", "Arial", "Tahoma", "DejaVu Sans"):
    try:
        path = font_manager.findfont(candidate, fallback_to_default=False)
        if path:
            plt.rcParams["font.family"] = candidate
            break
    except Exception:
        continue

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#fafbfc",
        "axes.edgecolor": C_LINE,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.color": "#e2e8f0",
        "axes.titlesize": 12,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 15,
    }
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(row: dict, key: str, default: float | None = None) -> float | None:
    raw = (row.get(key) or "").strip()
    if raw == "" or raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def short(model: str) -> str:
    mapping = {
        "Qwen3-4B-Instruct-2507": "Qwen3-4B",
        "Nile-Chat-4B": "Nile-Chat-4B",
        "Qwen3-8B": "Qwen3-8B",
        "ALLaM-7B": "ALLaM-7B",
        "Whisper-Large-v3-Turbo-CT2": "Whisper-L-Turbo",
        "Whisper-Large-v3-CT2": "Whisper-L-v3",
        "Whisper-Small-CT2": "Whisper-Small",
        "Arabic-Whisper-Turbo-FT-CT2": "Ar-Whisper-Turbo",
        "Arabic-Whisper-Large-v3-FT-CT2": "Ar-Whisper-L",
        "Voxtral-Mini-3B": "Voxtral-Mini",
        "Audar-ASR-V1-Flash": "Audar-Flash",
        "SeamlessM4T-v2-Large": "SeamlessM4T",
        "MMS-1B-all": "MMS-1B",
        "Qwen3-ASR-0.6B": "QwenASR-0.6B",
        "Qwen3-ASR-1.7B": "QwenASR-1.7B",
        "QwenCleo-ASR": "QwenCleo",
        "NAMAA-Egyptian-TTS": "NAMAA",
        "Chatterbox-Multilingual-V3": "Chatterbox",
        "VoiceTut-TTS": "VoiceTut",
        "SILMA-TTS": "SILMA",
    }
    return mapping.get(model, model)


def footer(fig: plt.Figure, page: int, total: int, family: str) -> None:
    fig.text(
        0.5,
        0.01,
        f"Arabic Voice Robot  ·  {family}  ·  {page}/{total}",
        ha="center",
        va="bottom",
        fontsize=8,
        color=C_GRAY,
    )


def bar_labels(ax, bars, fmt: str = "{:.1f}", offset: float = 0.5) -> None:
    for bar in bars:
        h = bar.get_height()
        if np.isnan(h):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#2d3748",
        )


def hbar_labels(ax, bars, fmt: str = "{:.1f}", offset: float = 0.8) -> None:
    for bar in bars:
        w = bar.get_width()
        if np.isnan(w):
            continue
        ax.text(
            w + offset,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(w),
            ha="left",
            va="center",
            fontsize=7,
            color="#2d3748",
        )


def draw_text_block(fig: plt.Figure, x: float, y: float, w: float, h: float, title: str, lines: list[str], title_color=C_NAVY) -> None:
    """Draw a bordered text panel on the figure (figure coordinates)."""
    ax = fig.add_axes([x, y, w, h])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        plt.Rectangle((0, 0), 1, 1, facecolor=C_SOFT, edgecolor=C_LINE, linewidth=1, transform=ax.transAxes, clip_on=False)
    )
    ax.text(0.04, 0.90, title, fontsize=10, fontweight="bold", color=title_color, transform=ax.transAxes, va="top")
    body = "\n".join(lines)
    ax.text(0.04, 0.72, body, fontsize=8.2, color="#2d3748", transform=ax.transAxes, va="top", linespacing=1.35)


def insight_caption(fig: plt.Figure, text: str, y: float = 0.035) -> None:
    wrapped = "\n".join(textwrap.wrap(text, width=145))
    fig.text(0.06, y, wrapped, ha="left", va="bottom", fontsize=8.2, color="#2d3748", style="italic")


def cover(
    pdf: PdfPages,
    title: str,
    subtitle: str,
    callouts: list[tuple[str, str]],
    overview: list[str],
    how_to_read: list[str],
    family: str,
    total: int,
) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(title, fontsize=18, fontweight="bold", color=C_NAVY, y=0.955)
    fig.text(0.5, 0.905, subtitle, ha="center", fontsize=10.5, color=C_GRAY)

    for i, (label, value) in enumerate(callouts[:4]):
        x = 0.07 + (i % 4) * 0.23
        ax = fig.add_axes([x, 0.76, 0.21, 0.10])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=C_SOFT, edgecolor=C_LINE, linewidth=1, transform=ax.transAxes))
        ax.text(0.06, 0.68, label, fontsize=7.5, color=C_GRAY, transform=ax.transAxes)
        ax.text(0.06, 0.22, value, fontsize=9.5, fontweight="bold", color=C_NAVY, transform=ax.transAxes)

    draw_text_block(fig, 0.07, 0.38, 0.86, 0.34, "Overview", overview)
    draw_text_block(fig, 0.07, 0.08, 0.86, 0.26, "How to read this report", how_to_read)

    footer(fig, 1, total, family)
    pdf.savefig(fig)
    plt.close(fig)


def recommendations_page(
    pdf: PdfPages,
    title: str,
    picks: list[tuple[str, str, str, str]],
    notes: list[str],
    family: str,
    page: int,
    total: int,
) -> None:
    """picks: (label, model, why, metrics_line)"""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(title, fontsize=15, fontweight="bold", color=C_NAVY, y=0.96)

    y = 0.86
    for label, model, why, metrics in picks:
        ax = fig.add_axes([0.07, y - 0.11, 0.86, 0.11])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=C_LINE, linewidth=1, transform=ax.transAxes))
        ax.add_patch(plt.Rectangle((0, 0), 0.012, 1, facecolor=C_BLUE, transform=ax.transAxes, clip_on=False))
        ax.text(0.04, 0.72, label, fontsize=8, color=C_GRAY, transform=ax.transAxes)
        ax.text(0.04, 0.38, model, fontsize=12, fontweight="bold", color=C_NAVY, transform=ax.transAxes)
        ax.text(0.38, 0.55, why, fontsize=8.2, color="#2d3748", transform=ax.transAxes, wrap=True)
        ax.text(0.38, 0.18, metrics, fontsize=7.5, color=C_TEAL, transform=ax.transAxes)
        y -= 0.125

    draw_text_block(fig, 0.07, 0.06, 0.86, 0.18, "Selection notes", notes, title_color=C_AMBER)
    footer(fig, page, total, family)
    pdf.savefig(fig)
    plt.close(fig)


# Metric glossary: (term, direction, meaning)
SHARED_METRICS = [
    (
        "Peak VRAM (MB)",
        "LOWER better",
        "GPU memory used at peak. Lower leaves room to run ASR + LLM + TTS together on one GPU.",
    ),
    (
        "Peak RAM (MB)",
        "LOWER better",
        "System (CPU) memory used at peak. Lower is safer on smaller servers / VPS hosts.",
    ),
    (
        "Peak CPU %",
        "LOWER better",
        "CPU utilization spike during the run. Lower means less contention with other services.",
    ),
    (
        "Cold load time (s)",
        "LOWER better",
        "Seconds to load the model into memory the first time. Matters for startup / cold starts.",
    ),
    (
        "RTF (Real-Time Factor)",
        "LOWER better",
        "Processing time / audio duration. RTF < 1 = faster than realtime (good). RTF > 1 = slower than the audio.",
    ),
    (
        "x Realtime (xRT)",
        "HIGHER better",
        "How many times faster than realtime (about 1/RTF). Example: 6x = 6s of audio in ~1s of compute.",
    ),
    (
        "Robot realtime score (0-100)",
        "HIGHER better",
        "Composite score for robot use (speed + resources + accuracy where relevant). Higher = better fit.",
    ),
    (
        "Balanced score (0-100)",
        "HIGHER better",
        "Normalized blend of the main tradeoffs. Higher = better all-rounder.",
    ),
    (
        "Success rate %",
        "HIGHER better",
        "Share of runs that completed without error. Prefer 100% for production reliability.",
    ),
]

LLM_METRICS = [
    (
        "TTFT - Time To First Token (s)",
        "LOWER better",
        "Delay until the model starts generating. Lower = snappier turn-taking (robot starts sooner).",
    ),
    (
        "Throughput (tokens/s)",
        "HIGHER better",
        "How fast tokens are generated after the first one. Higher = long answers finish sooner.",
    ),
    (
        "Generate time (s)",
        "LOWER better",
        "Total time to finish the reply. Lower is better (also depends on answer length).",
    ),
    (
        "Quality / overall score (/5)",
        "HIGHER better",
        "Rubric score for answer quality. Higher = better answers for the robot.",
    ),
    (
        "Auto-pass rate %",
        "HIGHER better",
        "Share of prompts that passed automated checks. Higher = more reliable across the suite.",
    ),
    (
        "Correctness / Instruction / Conciseness / TTS fit (/5)",
        "HIGHER better",
        "Quality sub-scores. Higher on each is better. Conciseness + TTS fit matter for spoken replies.",
    ),
    (
        "Latency / Throughput / VRAM / Load scores (0-100)",
        "HIGHER better",
        "Normalized component scores inside the robot composite. Higher = better on that axis.",
    ),
]

ASR_METRICS = [
    (
        "Accuracy % (word accuracy)",
        "HIGHER better",
        "Share of words recognized correctly (about 1 - WER). Higher = fewer recognition mistakes.",
    ),
    (
        "WER - Word Error Rate",
        "LOWER better",
        "Fraction of words wrong. 0 = perfect. Lower = better recognition.",
    ),
    (
        "CER - Character Error Rate",
        "LOWER better",
        "Same idea as WER at character level. Useful for Arabic. Lower = better.",
    ),
    (
        "Transcribe time (s)",
        "LOWER better",
        "Wall time to finish transcription. Lower = faster handoff to the LLM.",
    ),
    (
        "Realtime capable",
        "Prefer YES",
        "Yes when RTF < 1. Needed for near-live listening on the robot.",
    ),
]

TTS_METRICS = [
    (
        "Chars/s (characters per second)",
        "HIGHER better",
        "Input characters synthesized per second of compute. Higher = faster speech generation.",
    ),
    (
        "Generation time (s)",
        "LOWER better",
        "Time to synthesize the full utterance. Lower = robot speaks sooner after the LLM.",
    ),
    (
        "Audio length (s)",
        "Context only",
        "Duration of the produced WAV. Not good/bad alone — used with generate time to get RTF.",
    ),
    (
        "Sec per 1k chars",
        "LOWER better",
        "Seconds of compute per 1000 characters. Lower = more efficient synthesis.",
    ),
    (
        "Realtime capable",
        "Prefer YES",
        "Yes when RTF < 1 (generate time shorter than the spoken audio).",
    ),
]


def glossary_page(
    pdf: PdfPages,
    family: str,
    family_metrics: list[tuple[str, str, str]],
    page: int,
    total: int,
    extra_notes: list[str] | None = None,
) -> None:
    """Full-page glossary: what each metric means + whether lower/higher is better."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(f"{family} - Metrics glossary (what to prefer)", fontsize=15, fontweight="bold", color=C_NAVY, y=0.965)
    fig.text(
        0.5,
        0.925,
        "Use this page while reading the charts. Direction labels show what you want for a production robot.",
        ha="center",
        fontsize=9,
        color=C_GRAY,
    )

    entries = family_metrics + SHARED_METRICS
    mid = (len(entries) + 1) // 2
    cols = [entries[:mid], entries[mid:]]
    for col_i, col in enumerate(cols):
        x0 = 0.05 + col_i * 0.48
        y = 0.88
        for term, direction, meaning in col:
            ax = fig.add_axes([x0, y - 0.095, 0.45, 0.09])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            ax.add_patch(
                plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=C_LINE, linewidth=0.8, transform=ax.transAxes)
            )
            ax.text(0.03, 0.72, term, fontsize=7.8, fontweight="bold", color=C_NAVY, transform=ax.transAxes)
            if "Context" in direction:
                dir_color = C_GRAY
            else:
                dir_color = C_GREEN
            ax.text(0.03, 0.42, direction, fontsize=7.5, fontweight="bold", color=dir_color, transform=ax.transAxes)
            wrapped = "\n".join(textwrap.wrap(meaning, width=54))
            ax.text(0.03, 0.08, wrapped, fontsize=6.6, color="#2d3748", transform=ax.transAxes, va="bottom")
            y -= 0.10

    notes = extra_notes or [
        "Rule of thumb: prefer LOWER latency, LOWER memory (RAM/VRAM), LOWER error rates;",
        "prefer HIGHER accuracy, throughput, quality, and robot scores.",
        "A model can win on speed but lose on quality (or vice versa) — check both sides.",
    ]
    draw_text_block(fig, 0.05, 0.04, 0.90, 0.12, "Remember", notes, title_color=C_AMBER)
    footer(fig, page, total, family)
    pdf.savefig(fig)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM
# ═══════════════════════════════════════════════════════════════════════════════


def build_llm_pdf() -> Path:
    src = ROOT / "llm_outputs"
    recs = read_json(src / "llm_recommendations.json")
    lb = read_csv(src / "llm_leaderboard.csv")
    by_model = [r for r in read_csv(src / "llm_analytics_by_model.csv") if fnum(r, "ok", 0)]
    by_cat = [r for r in read_csv(src / "llm_analytics_by_category.csv") if fnum(r, "ok", 0)]
    scores = read_csv(src / "llm_quality_scores.csv")
    all_models = read_csv(src / "llm_analytics_by_model.csv")

    names = [short(r["model"]) for r in lb]
    models = [r["model"] for r in lb]
    qmap = {r["model"]: r for r in by_model}
    picks = recs.get("picks") or {}
    best_q = max(by_model, key=lambda r: fnum(r, "avg_overall_score", 0) or 0) if by_model else {}
    gated = [r["model"] for r in all_models if not fnum(r, "ok", 0)]

    out = OUT_DIR / "LLM_Model_Selection_Report.pdf"
    total = 10
    with PdfPages(out) as pdf:
        cover(
            pdf,
            "LLM Model Selection Report",
            "Egyptian Arabic conversational models — bake-off results",
            [
                ("Best robot realtime", picks.get("best_for_robot_realtime", {}).get("model", "—")),
                ("Best quality", f"{best_q.get('model', '—')}"),
                ("Lowest TTFT", picks.get("lowest_ttft", {}).get("model", "—")),
                ("OK / attempted", f"{len(by_model)} / {len(all_models)}"),
            ],
            [
                f"This report compares {len(all_models)} LLM candidates for an Egyptian Arabic voice robot.",
                f"{len(by_model)} models completed the suite successfully; {len(gated)} failed (typically gated Hugging Face auth).",
                "Two rankings matter: (1) robot realtime = latency + throughput + VRAM + load, and",
                "(2) response quality = rubric scores for correctness, instruction following, and TTS suitability.",
                f"Fastest composite pick: {picks.get('best_for_robot_realtime', {}).get('model', '—')}.",
                f"Highest quality pick: {best_q.get('model', '—')} ({fnum(best_q, 'avg_overall_score', 0):.2f}/5).",
                "These two leaders may differ — choose based on whether turn-taking speed or answer quality is the priority.",
            ],
            [
                "Page 2: metrics glossary — what every term means and whether LOWER or HIGHER is better.",
                "Page 3: recommended picks with rationale.",
                "Pages 4-6: latency, throughput, VRAM, and normalized score charts.",
                "Pages 7-9: quality dimensions, category heatmaps, and score distributions.",
                "Page 10: decision guidance for production selection.",
            ],
            "LLM",
            total,
        )

        glossary_page(
            pdf,
            "LLM",
            LLM_METRICS,
            2,
            total,
            [
                "For LLM robots: want LOWER TTFT, LOWER VRAM/RAM, LOWER load time;",
                "want HIGHER tokens/s, HIGHER quality scores, HIGHER auto-pass %, HIGHER robot score.",
                "Prefer quality when answers are spoken aloud; prefer TTFT when turn-taking feel matters most.",
            ],
        )

        pick_rows = []
        for key, label in [
            ("best_for_robot_realtime", "Best for robot realtime"),
            ("lowest_ttft", "Lowest time-to-first-token (LOWER better)"),
            ("highest_throughput", "Highest throughput (HIGHER better)"),
            ("lowest_vram", "Lowest peak VRAM (LOWER better)"),
            ("best_balanced", "Best balanced"),
        ]:
            p = picks.get(key) or {}
            metrics = p.get("metrics") or {}
            metric_txt = "  |  ".join(f"{k}={v}" for k, v in metrics.items())
            pick_rows.append((label, p.get("model", "—"), p.get("why", ""), metric_txt))

        recommendations_page(
            pdf,
            "Recommended picks (automated)",
            pick_rows,
            [
                "Automated scores cover latency, throughput, and resources only — not dialect naturalness.",
                "Manually spot-check Egyptian / MSA / code-switch responses under llm_outputs/responses/.",
                "For robot UX, prioritize LOW TTFT (lower better) even if peak tok/s is slightly lower.",
                "Disable thinking modes where available to reduce TTFT further.",
            ],
            "LLM",
            3,
            total,
        )

        # P3 — Robot vs Quality + text panel
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Robot realtime score  vs  Response quality", fontweight="bold", color=C_NAVY, y=0.97)
        ax0 = fig.add_axes([0.08, 0.42, 0.40, 0.48])
        ax1 = fig.add_axes([0.56, 0.42, 0.40, 0.48])
        robot = [fnum(r, "score_robot_realtime", 0) or 0 for r in lb]
        bars = ax0.barh(names[::-1], robot[::-1], color=[PALETTE[i % len(PALETTE)] for i in range(len(names))][::-1])
        ax0.set_xlabel("Robot score (0-100)  |  HIGHER better")
        ax0.set_title("Latency / VRAM composite (HIGHER better)")
        ax0.set_xlim(0, 115)
        hbar_labels(ax0, bars)

        q_names = [short(r["model"]) for r in by_model]
        q_vals = [fnum(r, "avg_overall_score", 0) or 0 for r in by_model]
        order = np.argsort(q_vals)
        bars = ax1.barh([q_names[i] for i in order], [q_vals[i] for i in order], color=C_GREEN)
        ax1.set_xlabel("Overall quality / 5  |  HIGHER better")
        ax1.set_title("Rubric quality (HIGHER better)")
        ax1.set_xlim(0, 5.6)
        hbar_labels(ax1, bars, "{:.2f}", 0.08)

        speed_model = picks.get("best_for_robot_realtime", {}).get("model", "—")
        speed_q = fnum(qmap.get(speed_model, {}), "avg_overall_score", 0) or 0
        draw_text_block(
            fig,
            0.07,
            0.08,
            0.86,
            0.28,
            "What this means",
            [
                "Robot score (HIGHER better): blends TTFT, tokens/s, VRAM efficiency, and load time.",
                "Quality /5 (HIGHER better): rubric for correctness, instructions, conciseness, TTS fit.",
                f"Left chart: {speed_model} leads robot realtime.",
                f"Right chart: {best_q.get('model', '—')} leads at {fnum(best_q, 'avg_overall_score', 0):.2f}/5.",
                f"{speed_model} quality is {speed_q:.2f}/5 — strong on speed, weaker on rubric quality.",
                "Prefer left leader for snappy turns; prefer right leader for better spoken answers.",
            ],
        )
        footer(fig, 4, total, "LLM")
        pdf.savefig(fig)
        plt.close(fig)

        # P5 — TTFT & throughput
        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
        fig.subplots_adjust(bottom=0.22, top=0.88)
        fig.suptitle("Latency & throughput", fontweight="bold", color=C_NAVY)
        ttft = [fnum(r, "avg_first_token_seconds", 0) or 0 for r in lb]
        tps = [fnum(r, "avg_tokens_per_second", 0) or 0 for r in lb]
        bars = axes[0].bar(names, ttft, color=C_AMBER)
        axes[0].set_ylabel("Seconds  |  LOWER better")
        axes[0].set_title("TTFT = Time To First Token (LOWER better)")
        bar_labels(axes[0], bars, "{:.3f}", max(ttft) * 0.02 if ttft else 0.02)
        axes[0].tick_params(axis="x", rotation=20)
        bars = axes[1].bar(names, tps, color=C_GREEN)
        axes[1].set_ylabel("Tokens / second  |  HIGHER better")
        axes[1].set_title("Throughput (HIGHER better)")
        bar_labels(axes[1], bars, "{:.2f}", max(tps) * 0.02 if tps else 0.1)
        axes[1].tick_params(axis="x", rotation=20)
        insight_caption(
            fig,
            "TTFT (LOWER better): delay before the first token — drives how fast the robot starts speaking. "
            "Throughput / tokens-per-second (HIGHER better): how quickly the rest of the answer is generated. "
            "Nile-Chat-4B and Qwen3-4B lead both; Qwen3-8B and ALLaM-7B are slower.",
        )
        footer(fig, 5, total, "LLM")
        pdf.savefig(fig)
        plt.close(fig)

        # P6 — scatter + VRAM
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Quality-speed tradeoff  &  VRAM", fontweight="bold", color=C_NAVY, y=0.97)
        ax0 = fig.add_axes([0.08, 0.40, 0.40, 0.50])
        ax1 = fig.add_axes([0.56, 0.40, 0.40, 0.50])
        for i, r in enumerate(lb):
            q = qmap.get(r["model"], {})
            x = fnum(r, "avg_first_token_seconds", 0) or 0
            y = fnum(q, "avg_overall_score", 0) or 0
            s = (fnum(r, "peak_vram_mb", 0) or 1000) / 40
            ax0.scatter(x, y, s=s, color=PALETTE[i % len(PALETTE)], alpha=0.85, edgecolors="white", linewidths=1)
            ax0.annotate(short(r["model"]), (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax0.set_xlabel("TTFT (s)  |  LOWER better  -->")
        ax0.set_ylabel("Quality / 5  |  HIGHER better")
        ax0.set_title("Ideal zone: top-left (LOW TTFT + HIGH quality)")
        ax0.set_ylim(0, 5.5)

        vram = [fnum(r, "peak_vram_mb", 0) or 0 for r in lb]
        bars = ax1.bar(names, vram, color=C_RED)
        ax1.axhline(14913, color=C_GRAY, linestyle="--", linewidth=1, label="T4 ~ 14.9 GB")
        ax1.set_ylabel("MB  |  LOWER better")
        ax1.set_title("Peak VRAM (LOWER better)")
        ax1.legend(fontsize=8)
        bar_labels(ax1, bars, "{:.0f}", max(vram) * 0.02 if vram else 50)
        ax1.tick_params(axis="x", rotation=20)

        draw_text_block(
            fig,
            0.07,
            0.07,
            0.86,
            0.26,
            "Tradeoff reading",
            [
                "Bubble size = peak VRAM (LOWER better for co-residency with ASR/TTS).",
                "Top-left of scatter is ideal: LOW TTFT + HIGH quality.",
                "Qwen3-8B: higher quality, higher TTFT (slower), lowest VRAM (int4) — good if GPU is tight.",
                "ALLaM-7B: near T4 ceiling (~15 GB) — risky when ASR+TTS share the GPU (VRAM LOWER better).",
                "Nile-Chat-4B: fast TTFT but lower quality — often verbose (hurts TTS suitability).",
                "Leave GPU headroom if ASR + LLM + TTS must share one card.",
            ],
        )
        footer(fig, 6, total, "LLM")
        pdf.savefig(fig)
        plt.close(fig)

        # P7 score breakdown
        fig, ax = plt.subplots(figsize=(11, 8.5))
        fig.subplots_adjust(bottom=0.22, top=0.88)
        fig.suptitle("Normalized score breakdown (all scores: HIGHER better)", fontweight="bold", color=C_NAVY)
        metrics = [
            ("Latency", "score_latency"),
            ("Throughput", "score_throughput"),
            ("VRAM eff.", "score_vram_efficiency"),
            ("Load", "score_load"),
            ("Balanced", "score_balanced"),
            ("Robot", "score_robot_realtime"),
        ]
        x = np.arange(len(names))
        width = 0.13
        for i, (label, key) in enumerate(metrics):
            vals = [fnum(r, key, 0) or 0 for r in lb]
            ax.bar(x + (i - 2.5) * width, vals, width, label=label, color=PALETTE[i])
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel("Score 0-100  |  HIGHER better")
        ax.set_ylim(0, 115)
        ax.legend(ncol=3, loc="upper right")
        insight_caption(
            fig,
            "These are min-max normalized scores (HIGHER better on every bar). "
            "Latency score rewards LOW TTFT; Throughput rewards HIGH tok/s; VRAM efficiency rewards LOW peak VRAM; "
            "Load rewards LOW cold-load time. Robot score blends them for conversational robots.",
        )
        footer(fig, 7, total, "LLM")
        pdf.savefig(fig)
        plt.close(fig)

        # P8 quality dimensions
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Quality dimensions & auto-pass rate (HIGHER better)", fontweight="bold", color=C_NAVY, y=0.97)
        ax0 = fig.add_axes([0.08, 0.40, 0.40, 0.50])
        ax1 = fig.add_axes([0.56, 0.40, 0.40, 0.50])
        dims = [
            ("Overall", "avg_overall_score"),
            ("Correct", "avg_correctness_score"),
            ("Instr.", "avg_instruction_following_score"),
            ("Concise", "avg_conciseness_score"),
            ("TTS fit", "avg_tts_suitability_score"),
        ]
        x = np.arange(len(dims))
        width = 0.18
        for i, r in enumerate(by_model):
            vals = [fnum(r, k, 0) or 0 for _, k in dims]
            ax0.bar(x + (i - 1.5) * width, vals, width, label=short(r["model"]), color=PALETTE[i])
        ax0.set_xticks(x)
        ax0.set_xticklabels([d[0] for d in dims])
        ax0.set_ylim(0, 5.5)
        ax0.set_ylabel("Score / 5  |  HIGHER better")
        ax0.set_title("Quality rubric breakdown (HIGHER better)")
        ax0.legend(fontsize=7)

        pass_rates = [fnum(r, "auto_pass_rate_percent", 0) or 0 for r in by_model]
        pnames = [short(r["model"]) for r in by_model]
        bars = ax1.bar(pnames, pass_rates, color=C_TEAL)
        ax1.set_ylabel("%  |  HIGHER better")
        ax1.set_ylim(0, 110)
        ax1.set_title("Auto-pass rate (HIGHER better)")
        bar_labels(ax1, bars, "{:.0f}%", 2)
        ax1.tick_params(axis="x", rotation=20)

        draw_text_block(
            fig,
            0.07,
            0.07,
            0.86,
            0.26,
            "Quality interpretation",
            [
                "All quality metrics are HIGHER better (0-5 scale unless noted).",
                "Correctness + instruction following matter most for robot reliability.",
                "Conciseness and TTS suitability matter because spoken replies must stay short and clean.",
                f"Qwen3-8B leads overall quality ({fnum(best_q, 'avg_overall_score', 0):.2f}/5) and auto-pass rate.",
                "Nile-Chat-4B often over-generates (lower conciseness / TTS fit) despite fast tokens.",
                "Language score was uniformly high for OK models — dialect capability is not the differentiator here.",
            ],
        )
        footer(fig, 8, total, "LLM")
        pdf.savefig(fig)
        plt.close(fig)

        # P9 heatmaps
        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
        fig.subplots_adjust(bottom=0.20, top=0.88)
        fig.suptitle("Category performance heatmaps", fontweight="bold", color=C_NAVY)
        cats = sorted({r["category"] for r in by_cat if r.get("category") and r["category"] != "general"})
        ok_models = [r["model"] for r in by_model]
        ttft_m = np.full((len(ok_models), len(cats)), np.nan)
        tps_m = np.full((len(ok_models), len(cats)), np.nan)
        lookup = {(r["model"], r["category"]): r for r in by_cat}
        for i, m in enumerate(ok_models):
            for j, c in enumerate(cats):
                row = lookup.get((m, c))
                if row:
                    ttft_m[i, j] = fnum(row, "avg_first_token_seconds")
                    tps_m[i, j] = fnum(row, "avg_tokens_per_second")
        cat_labels = [c.replace("_", "\n") for c in cats]
        im0 = axes[0].imshow(ttft_m, aspect="auto", cmap="YlOrRd")
        axes[0].set_xticks(range(len(cats)))
        axes[0].set_xticklabels(cat_labels, fontsize=7)
        axes[0].set_yticks(range(len(ok_models)))
        axes[0].set_yticklabels([short(m) for m in ok_models])
        axes[0].set_title("TTFT by category (s) — LOWER better")
        for i in range(len(ok_models)):
            for j in range(len(cats)):
                v = ttft_m[i, j]
                if not np.isnan(v):
                    axes[0].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5)
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(tps_m, aspect="auto", cmap="YlGn")
        axes[1].set_xticks(range(len(cats)))
        axes[1].set_xticklabels(cat_labels, fontsize=7)
        axes[1].set_yticks(range(len(ok_models)))
        axes[1].set_yticklabels([short(m) for m in ok_models])
        axes[1].set_title("Throughput by category (tok/s) — HIGHER better")
        for i in range(len(ok_models)):
            for j in range(len(cats)):
                v = tps_m[i, j]
                if not np.isnan(v):
                    axes[1].text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        insight_caption(
            fig,
            "Left heatmap: TTFT seconds (LOWER better — darker usually means slower). "
            "Right heatmap: tokens/s (HIGHER better — darker green usually means faster generation). "
            "Tool-calling is the slowest category for all models.",
        )
        footer(fig, 9, total, "LLM")
        pdf.savefig(fig)
        plt.close(fig)

        # P10 decision page
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Score distribution & decision guidance", fontweight="bold", color=C_NAVY, y=0.97)
        ax0 = fig.add_axes([0.08, 0.42, 0.42, 0.48])
        ax1 = fig.add_axes([0.58, 0.42, 0.34, 0.48])
        data_by, labels_box = [], []
        for m in models:
            vals = [fnum(r, "overall_score") for r in scores if r.get("model") == m and fnum(r, "overall_score") is not None]
            if vals:
                data_by.append(vals)
                labels_box.append(short(m))
        if data_by:
            bp = ax0.boxplot(data_by, tick_labels=labels_box, patch_artist=True)
            for patch, color in zip(bp["boxes"], PALETTE):
                patch.set_facecolor(color)
                patch.set_alpha(0.65)
            ax0.set_ylabel("Overall score / 5  |  HIGHER better")
            ax0.set_title("Per-prompt quality spread (HIGHER better)")
            ax0.tick_params(axis="x", rotation=20)
            ax0.set_ylim(0, 5.5)

        load = [fnum(r, "avg_load_seconds", 0) or 0 for r in lb]
        bars = ax1.bar(names, load, color=C_GRAY)
        ax1.set_ylabel("Seconds  |  LOWER better")
        ax1.set_title("Cold load time (LOWER better)")
        bar_labels(ax1, bars, "{:.0f}", max(load) * 0.02 if load else 5)
        ax1.tick_params(axis="x", rotation=20)

        draw_text_block(
            fig,
            0.07,
            0.06,
            0.86,
            0.30,
            "How to choose",
            [
                f"1) Need fastest conversational feel (LOW TTFT) -> {speed_model}.",
                f"2) Need best answer quality / auto-pass (HIGHER better) -> {best_q.get('model', '—')}.",
                "3) Need a balance on a mid GPU -> Qwen3-4B (strong TTFT + solid quality).",
                "4) Need lowest VRAM (LOWER better) for co-residency -> Qwen3-8B (int4).",
                "5) Avoid ALLaM-7B on T4-class GPUs unless dialect needs outweigh HIGH VRAM cost.",
                "6) Gated models (Gemma / Jais) were not scored — re-run after HF auth if required.",
            ],
        )
        footer(fig, 10, total, "LLM")
        pdf.savefig(fig)
        plt.close(fig)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# ASR
# ═══════════════════════════════════════════════════════════════════════════════


def build_asr_pdf() -> Path:
    src = ROOT / "asr_outputs"
    recs = read_json(src / "asr_recommendations.json")
    lb = read_csv(src / "asr_leaderboard.csv")
    ranking = read_csv(src / "asr_accuracy_ranking.csv")
    picks = recs.get("picks") or {}
    names = [short(r["model"]) for r in lb]

    out = OUT_DIR / "ASR_Model_Selection_Report.pdf"
    total = 9
    with PdfPages(out) as pdf:
        cover(
            pdf,
            "ASR Model Selection Report",
            "Egyptian Arabic speech recognition — bake-off results",
            [
                ("Best robot realtime", picks.get("best_for_robot_realtime", {}).get("model", "—")),
                ("Best accuracy", picks.get("best_accuracy", {}).get("model", "—")),
                ("Best speed", picks.get("best_speed", {}).get("model", "—")),
                ("Models OK", str(recs.get("model_count_ok", len(lb)))),
            ],
            [
                f"{len(lb)} ASR models were evaluated successfully on the Arabic test clip.",
                "Robot realtime score weights: accuracy 45% + speed/RTF 30% + low VRAM 15% + fast load 10%.",
                "RTF (Real-Time Factor) = transcribe_time / audio_duration. LOWER better. RTF < 1 = realtime.",
                "WER (Word Error Rate): LOWER better. Accuracy %: HIGHER better (approx. 1 - WER).",
                f"Top composite pick: {picks.get('best_for_robot_realtime', {}).get('model', '—')}.",
                f"Top accuracy pick: {picks.get('best_accuracy', {}).get('model', '—')}.",
            ],
            [
                "Page 2: metrics glossary — RTF, WER, CER, VRAM, and what to prefer.",
                "Page 3: recommended picks with rationale.",
                "Pages 4-6: accuracy, speed, WER, and accuracy-RTF tradeoff charts.",
                "Pages 7-8: VRAM/load and normalized score breakdown.",
                "Page 9: full accuracy ranking and selection guidance.",
            ],
            "ASR",
            total,
        )

        glossary_page(
            pdf,
            "ASR",
            ASR_METRICS,
            2,
            total,
            [
                "For ASR: want HIGHER accuracy %, HIGHER robot score, HIGHER xRealtime;",
                "want LOWER WER, LOWER CER, LOWER RTF, LOWER VRAM/RAM, LOWER load/transcribe time.",
                "Prefer realtime-capable (RTF < 1) models for live robot listening.",
            ],
        )

        pick_rows = []
        for key, label in [
            ("best_for_robot_realtime", "Best for robot realtime (HIGHER score better)"),
            ("best_accuracy", "Best accuracy % (HIGHER better)"),
            ("best_speed", "Best speed / lowest RTF (LOWER better)"),
            ("lowest_vram", "Lowest peak VRAM (LOWER better)"),
            ("best_balanced", "Best balanced"),
            ("best_realtime_with_accuracy", "Best realtime + accuracy"),
        ]:
            p = picks.get(key) or {}
            metrics = p.get("metrics") or {}
            metric_txt = "  |  ".join(f"{k}={v}" for k, v in metrics.items())
            pick_rows.append((label, p.get("model", "—"), p.get("why", ""), metric_txt))
        recommendations_page(
            pdf,
            "Recommended picks (automated)",
            pick_rows,
            [
                "Prefer LOW RTF + HIGH accuracy + LOW VRAM so LLM/TTS still fit on the same GPU.",
                "Whisper CT2 (faster-whisper) models dominate the top of the robot leaderboard.",
                "Arabic fine-tunes were competitive on speed but did not beat base Whisper Large on accuracy here.",
                "Voxtral-Mini has strong accuracy but HIGH VRAM — costly for co-residency.",
            ],
            "ASR",
            3,
            total,
        )

        # P3 robot + accuracy
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Robot score  &  Accuracy", fontweight="bold", color=C_NAVY, y=0.97)
        ax0 = fig.add_axes([0.08, 0.40, 0.40, 0.50])
        ax1 = fig.add_axes([0.56, 0.40, 0.40, 0.50])
        robot = [fnum(r, "score_robot_realtime", 0) or 0 for r in lb]
        bars = ax0.barh(names[::-1], robot[::-1], color=[PALETTE[i % len(PALETTE)] for i in range(len(names))][::-1])
        ax0.set_xlabel("Robot score (0-100)  |  HIGHER better")
        ax0.set_title("Composite robot ranking (HIGHER better)")
        ax0.set_xlim(0, 115)
        hbar_labels(ax0, bars)
        acc = [fnum(r, "avg_accuracy_percent", 0) or 0 for r in lb]
        order = np.argsort(acc)
        bars = ax1.barh([names[i] for i in order], [acc[i] for i in order], color=C_GREEN)
        ax1.set_xlabel("Word accuracy %  |  HIGHER better")
        ax1.set_title("Accuracy ranking (HIGHER better)")
        ax1.set_xlim(0, 100)
        hbar_labels(ax1, bars, "{:.1f}")
        draw_text_block(
            fig,
            0.07,
            0.07,
            0.86,
            0.27,
            "What this means",
            [
                "Left: overall robot fitness (HIGHER better). Right: pure recognition accuracy (HIGHER better).",
                "Whisper-Large-v3-Turbo leads robot score by combining solid accuracy with excellent RTF and moderate VRAM.",
                f"Whisper-Large-v3 leads accuracy ({(picks.get('best_accuracy', {}).get('metrics') or {}).get('avg_accuracy_percent', '—')}%).",
                "Smaller Whisper variants trade some accuracy for LOWER VRAM — useful on tight GPUs.",
                "Bottom of the list (Qwen ASR family / QwenCleo) is too weak on accuracy for production Egyptian Arabic.",
            ],
        )
        footer(fig, 4, total, "ASR")
        pdf.savefig(fig)
        plt.close(fig)

        # P4 scatter + WER
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Accuracy–speed tradeoff  &  WER", fontweight="bold", color=C_NAVY, y=0.97)
        ax0 = fig.add_axes([0.08, 0.40, 0.40, 0.50])
        ax1 = fig.add_axes([0.56, 0.40, 0.40, 0.50])
        for i, r in enumerate(lb):
            x = fnum(r, "avg_rtf", 0) or 0
            y = fnum(r, "avg_accuracy_percent", 0) or 0
            s = (fnum(r, "peak_vram_mb", 0) or 1000) / 30
            ax0.scatter(x, y, s=s, color=PALETTE[i % len(PALETTE)], alpha=0.85, edgecolors="white")
            ax0.annotate(short(r["model"]), (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7)
        ax0.axvline(1.0, color=C_RED, linestyle="--", linewidth=1, label="RTF = 1")
        ax0.set_xlabel("RTF (LOWER better)")
        ax0.set_ylabel("Accuracy % (HIGHER better)")
        ax0.set_title("Ideal zone: top-left (LOW RTF + HIGH accuracy)")
        ax0.legend(fontsize=8)
        wer = [fnum(r, "avg_wer", 0) or 0 for r in lb]
        bars = ax1.bar(names, wer, color=C_AMBER)
        ax1.set_ylabel("WER  |  LOWER better")
        ax1.set_title("Word Error Rate (LOWER better)")
        bar_labels(ax1, bars, "{:.3f}", max(wer) * 0.02 if wer else 0.01)
        ax1.tick_params(axis="x", rotation=55)
        draw_text_block(
            fig,
            0.07,
            0.07,
            0.86,
            0.26,
            "Tradeoff reading",
            [
                "RTF = transcribe_time / audio_length (LOWER better). RTF < 1 = realtime.",
                "WER = Word Error Rate (LOWER better). Accuracy % ~ (1 - WER) x 100 (HIGHER better).",
                "Bubble size = peak VRAM (LOWER better for co-residency with LLM/TTS).",
                "Top-left models are both accurate and fast — preferred for voice robots.",
                "All evaluated models were realtime-capable (RTF < 1) on this clip.",
                "Voxtral is accurate but HIGH VRAM — expensive to host beside an LLM.",
            ],
        )
        footer(fig, 5, total, "ASR")
        pdf.savefig(fig)
        plt.close(fig)

        # P5 speed
        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
        fig.subplots_adjust(bottom=0.22, top=0.88)
        fig.suptitle("Speed metrics", fontweight="bold", color=C_NAVY)
        rtf = [fnum(r, "avg_rtf", 0) or 0 for r in lb]
        bars = axes[0].bar(names, rtf, color=C_BLUE)
        axes[0].axhline(1.0, color=C_RED, linestyle="--", linewidth=1, label="Realtime threshold")
        axes[0].set_ylabel("RTF  |  LOWER better")
        axes[0].set_title("Real-Time Factor (LOWER better)")
        axes[0].legend(fontsize=8)
        bar_labels(axes[0], bars, "{:.3f}", max(rtf) * 0.02 if rtf else 0.01)
        axes[0].tick_params(axis="x", rotation=55)
        xrt = [fnum(r, "avg_x_realtime", 0) or 0 for r in lb]
        bars = axes[1].bar(names, xrt, color=C_TEAL)
        axes[1].set_ylabel("x realtime  |  HIGHER better")
        axes[1].set_title("Speedup vs audio duration (HIGHER better)")
        bar_labels(axes[1], bars, "{:.1f}", max(xrt) * 0.02 if xrt else 0.3)
        axes[1].tick_params(axis="x", rotation=55)
        insight_caption(
            fig,
            "MMS-1B and Whisper Turbo variants are the speed leaders. Fast alone is not enough — pair speed charts with accuracy "
            "before selecting. A very fast but low-accuracy model will force clarification loops and hurt robot UX.",
        )
        footer(fig, 6, total, "ASR")
        pdf.savefig(fig)
        plt.close(fig)

        # P6 resources
        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
        fig.subplots_adjust(bottom=0.22, top=0.88)
        fig.suptitle("Resources (LOWER better on both charts)", fontweight="bold", color=C_NAVY)
        vram = [fnum(r, "peak_vram_mb", 0) or 0 for r in lb]
        bars = axes[0].bar(names, vram, color=C_RED)
        axes[0].axhline(14913, color=C_GRAY, linestyle="--", linewidth=1, label="T4 ≈ 14.9 GB")
        axes[0].set_ylabel("MB  |  LOWER better")
        axes[0].set_title("Peak VRAM (LOWER better)")
        axes[0].legend(fontsize=8)
        bar_labels(axes[0], bars, "{:.0f}", max(vram) * 0.02 if vram else 50)
        axes[0].tick_params(axis="x", rotation=55)
        load = [fnum(r, "avg_load_seconds", 0) or 0 for r in lb]
        bars = axes[1].bar(names, load, color=C_GRAY)
        axes[1].set_ylabel("Seconds  |  LOWER better")
        axes[1].set_title("Cold load time (LOWER better)")
        bar_labels(axes[1], bars, "{:.0f}", max(load) * 0.02 if load else 2)
        axes[1].tick_params(axis="x", rotation=55)
        insight_caption(
            fig,
            "Whisper-Small is the VRAM/load champion (~1.3 GB). Whisper-Large-v3-Turbo stays under ~2.5 GB while ranking #1 on robot score — "
            "usually the best default when LLM must share the GPU. Avoid Voxtral (~11 GB) unless ASR runs alone.",
        )
        footer(fig, 7, total, "ASR")
        pdf.savefig(fig)
        plt.close(fig)

        # P7 score breakdown
        fig, ax = plt.subplots(figsize=(11, 8.5))
        fig.subplots_adjust(bottom=0.22, top=0.88)
        fig.suptitle("Normalized score breakdown (top 6) - all HIGHER better", fontweight="bold", color=C_NAVY)
        top = lb[:6]
        tnames = [short(r["model"]) for r in top]
        metrics = [
            ("Accuracy", "score_accuracy"),
            ("Speed", "score_speed"),
            ("VRAM eff.", "score_vram_efficiency"),
            ("Load", "score_load"),
            ("Balanced", "score_balanced"),
            ("Robot", "score_robot_realtime"),
        ]
        x = np.arange(len(tnames))
        width = 0.13
        for i, (label, key) in enumerate(metrics):
            vals = [fnum(r, key, 0) or 0 for r in top]
            ax.bar(x + (i - 2.5) * width, vals, width, label=label, color=PALETTE[i])
        ax.set_xticks(x)
        ax.set_xticklabels(tnames, rotation=20)
        ax.set_ylabel("Score 0-100  |  HIGHER better")
        ax.set_ylim(0, 115)
        ax.legend(ncol=3, loc="upper right")
        insight_caption(
            fig,
            "Use this breakdown when two models have similar robot scores. Example: Turbo wins on speed+VRAM blend; "
            "full Large-v3 wins accuracy. Pick based on whether recognition errors or latency hurt your robot more.",
        )
        footer(fig, 8, total, "ASR")
        pdf.savefig(fig)
        plt.close(fig)

        # P8 ranking + guidance
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Accuracy ranking & selection guidance", fontweight="bold", color=C_NAVY, y=0.97)
        ax = fig.add_axes([0.22, 0.40, 0.70, 0.50])
        if ranking:
            rnames = [short(r.get("model", "")) for r in ranking]
            racc = [fnum(r, "accuracy_percent", 0) or 0 for r in ranking]
            bars = ax.barh(rnames[::-1], racc[::-1], color=[PALETTE[i % len(PALETTE)] for i in range(len(rnames))][::-1])
            ax.set_xlabel("Accuracy %  |  HIGHER better")
            ax.set_xlim(0, 100)
            hbar_labels(ax, bars, "{:.1f}")
            ax.set_title("Per-run word accuracy (HIGHER better)")
        draw_text_block(
            fig,
            0.07,
            0.06,
            0.86,
            0.28,
            "How to choose",
            [
                "1) Default production pick → Whisper-Large-v3-Turbo-CT2 (best robot composite).",
                "2) If accuracy is the hard constraint → Whisper-Large-v3-CT2.",
                "3) If GPU is tight / multi-model → Whisper-Small-CT2 (lowest VRAM, still decent).",
                "4) Skip Qwen ASR / QwenCleo for this Egyptian Arabic use case based on current WER.",
                "5) Re-validate on your own microphone noise and dialect samples before freeze.",
            ],
        )
        footer(fig, 9, total, "ASR")
        pdf.savefig(fig)
        plt.close(fig)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# TTS
# ═══════════════════════════════════════════════════════════════════════════════


def build_tts_pdf() -> Path:
    src = ROOT / "tts_outputs"
    recs = read_json(src / "tts_recommendations.json")
    lb = read_csv(src / "tts_leaderboard.csv")
    by_model = read_csv(src / "tts_analytics_by_model.csv") or read_csv(src / "tts_analytics.csv")
    picks = recs.get("picks") or {}

    # Prefer leaderboard order when available
    if lb:
        rows = lb
    else:
        rows = by_model

    names = [short(r.get("model", "")) for r in rows]
    ok_n = sum(1 for r in by_model if (r.get("status") or "").lower() == "ok" or str(r.get("success", "")).lower() in {"1", "true"})
    fail_n = len(by_model) - ok_n
    best = picks.get("best_for_robot_realtime", {})

    out = OUT_DIR / "TTS_Model_Selection_Report.pdf"
    total = 9
    with PdfPages(out) as pdf:
        cover(
            pdf,
            "TTS Model Selection Report",
            "Egyptian Arabic text-to-speech — bake-off results",
            [
                ("Best robot realtime", best.get("model", "—")),
                ("Best speed", (picks.get("best_speed") or {}).get("model", "—")),
                ("Lowest VRAM", (picks.get("lowest_vram") or {}).get("model", "—")),
                ("OK / attempted", f"{ok_n} / {len(by_model)}"),
            ],
            [
                f"{ok_n} of {len(by_model)} TTS models generated audio successfully.",
                "Automated scores cover RTF, throughput (chars/s), VRAM, and load — not voice naturalness.",
                "IMPORTANT: listen to each WAV before choosing a production voice.",
                "RTF = generate_time / audio_length. LOWER better. RTF < 1 = realtime-capable.",
                "Chars/s (HIGHER better). Peak VRAM / RAM / load time (LOWER better).",
                f"Top robot realtime pick: {best.get('model', '—')} "
                f"(score {(best.get('metrics') or {}).get('score_robot_realtime', '—')}).",
            ],
            [
                "Page 2: metrics glossary — RTF, chars/s, VRAM, and what to prefer.",
                "Page 3: recommended picks with rationale.",
                "Pages 4-6: robot score, RTF/throughput, and timing charts.",
                "Pages 7-8: VRAM/load and normalized score breakdown.",
                "Page 9: listening checklist and final selection guidance.",
            ],
            "TTS",
            total,
        )

        glossary_page(
            pdf,
            "TTS",
            TTS_METRICS,
            2,
            total,
            [
                "For TTS: want LOWER RTF, LOWER generate time, LOWER VRAM/RAM, LOWER load time;",
                "want HIGHER chars/s, HIGHER xRealtime, HIGHER robot score.",
                "Prefer realtime-capable (RTF < 1). Always listen to WAVs — scores ignore naturalness.",
            ],
        )

        pick_rows = []
        for key, label in [
            ("best_for_robot_realtime", "Best for robot realtime (HIGHER score better)"),
            ("best_speed", "Best speed / lowest RTF (LOWER better)"),
            ("best_throughput", "Best throughput chars/s (HIGHER better)"),
            ("lowest_vram", "Lowest peak VRAM (LOWER better)"),
            ("best_balanced", "Best balanced"),
            ("best_realtime_capable", "Best realtime-capable (prefer YES)"),
        ]:
            p = picks.get(key) or {}
            metrics = p.get("metrics") or {}
            metric_txt = "  |  ".join(f"{k}={v}" for k, v in metrics.items())
            listen = p.get("listen_file") or ""
            why = p.get("why", "")
            if listen:
                why = f"{why}  Listen: {listen}"
            pick_rows.append((label, p.get("model", "—"), why, metric_txt))
        recommendations_page(
            pdf,
            "Recommended picks (automated - listen to confirm)",
            pick_rows,
            [
                "Naturalness / Egyptian dialect / code-switch quality must be judged by listening to WAVs.",
                "Automated scores only cover latency and resource fit for the robot pipeline.",
                "RTF < 1.0 (LOWER better) is preferred for near-real-time synthesis.",
                "For production, also measure time-to-first-audio with streaming APIs.",
            ],
            "TTS",
            3,
            total,
        )

        # P3 robot + realtime
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Robot realtime score  &  Realtime capability", fontweight="bold", color=C_NAVY, y=0.97)
        ax0 = fig.add_axes([0.08, 0.40, 0.40, 0.50])
        ax1 = fig.add_axes([0.56, 0.40, 0.40, 0.50])
        robot = [fnum(r, "score_robot_realtime", 0) or 0 for r in rows]
        bars = ax0.barh(names[::-1], robot[::-1], color=[PALETTE[i % len(PALETTE)] for i in range(len(names))][::-1])
        ax0.set_xlabel("Robot score (0-100)  |  HIGHER better")
        ax0.set_title("Composite: RTF + throughput + VRAM + load (HIGHER better)")
        ax0.set_xlim(0, 115)
        hbar_labels(ax0, bars)

        rt_yes = sum(1 for r in rows if str(r.get("realtime_capable", "")).lower() in {"yes", "1", "true"})
        rt_no = len(rows) - rt_yes
        ax1.pie(
            [rt_yes, rt_no] if rt_no else [rt_yes],
            labels=[f"Realtime RTF<1 ({rt_yes})", f"Slower than realtime ({rt_no})"] if rt_no else [f"Realtime ({rt_yes})"],
            colors=[C_GREEN, C_AMBER] if rt_no else [C_GREEN],
            autopct="%1.0f%%",
            startangle=90,
            textprops={"fontsize": 10},
            wedgeprops={"width": 0.45, "edgecolor": "white"},
        )
        ax1.set_title("Realtime capable?")

        draw_text_block(
            fig,
            0.07,
            0.07,
            0.86,
            0.27,
            "What this means",
            [
                f"{best.get('model', '—')} leads robot realtime with the best blend of speed and low VRAM.",
                "VoiceTut and SILMA are realtime-capable (RTF < 1); Chatterbox and NAMAA are not.",
                "Non-realtime models can still be used offline or for offline rendering, but hurt live robot replies.",
                "Listen before finalizing — a fast voice that sounds unnatural is still a bad production pick.",
            ],
        )
        footer(fig, 4, total, "TTS")
        pdf.savefig(fig)
        plt.close(fig)

        # P4 RTF + throughput
        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
        fig.subplots_adjust(bottom=0.22, top=0.88)
        fig.suptitle("Speed & throughput", fontweight="bold", color=C_NAVY)
        rtf = [fnum(r, "rtf", 0) or 0 for r in rows]
        bars = axes[0].bar(names, rtf, color=C_BLUE)
        axes[0].axhline(1.0, color=C_RED, linestyle="--", linewidth=1, label="Realtime threshold")
        axes[0].set_ylabel("RTF  |  LOWER better")
        axes[0].set_title("Real-Time Factor (LOWER better)")
        axes[0].legend(fontsize=8)
        bar_labels(axes[0], bars, "{:.3f}", max(rtf) * 0.02 if rtf else 0.01)
        axes[0].tick_params(axis="x", rotation=15)

        cps = [fnum(r, "chars_per_second", 0) or 0 for r in rows]
        bars = axes[1].bar(names, cps, color=C_GREEN)
        axes[1].set_ylabel("Chars / second  |  HIGHER better")
        axes[1].set_title("Synthesis throughput (HIGHER better)")
        bar_labels(axes[1], bars, "{:.1f}", max(cps) * 0.02 if cps else 1)
        axes[1].tick_params(axis="x", rotation=15)
        insight_caption(
            fig,
            "VoiceTut is clearly fastest (RTF 0.157, ~83 chars/s). SILMA is a solid second. "
            "Chatterbox and NAMAA take longer than the audio they produce — weak for live robot turns.",
        )
        footer(fig, 5, total, "TTS")
        pdf.savefig(fig)
        plt.close(fig)

        # P5 gen time + xRealtime
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Generation time  &  Speedup", fontweight="bold", color=C_NAVY, y=0.97)
        ax0 = fig.add_axes([0.08, 0.40, 0.40, 0.50])
        ax1 = fig.add_axes([0.56, 0.40, 0.40, 0.50])
        gen = [fnum(r, "generation_seconds", 0) or 0 for r in rows]
        audio = [fnum(r, "audio_seconds", 0) or 0 for r in rows]
        x = np.arange(len(names))
        w = 0.35
        ax0.bar(x - w / 2, gen, w, label="Generate (s)", color=C_AMBER)
        ax0.bar(x + w / 2, audio, w, label="Audio length (s)", color=C_TEAL)
        ax0.set_xticks(x)
        ax0.set_xticklabels(names, rotation=15)
        ax0.set_ylabel("Seconds")
        ax0.set_title("Generate (LOWER better) vs audio length (context)")
        ax0.legend(fontsize=8)

        xrt = [fnum(r, "x_realtime", 0) or 0 for r in rows]
        bars = ax1.bar(names, xrt, color=C_PURPLE)
        ax1.axhline(1.0, color=C_RED, linestyle="--", linewidth=1, label="1x realtime")
        ax1.set_ylabel("x realtime  |  HIGHER better")
        ax1.set_title("Speedup vs audio duration (HIGHER better)")
        ax1.legend(fontsize=8)
        bar_labels(ax1, bars, "{:.2f}", max(xrt) * 0.02 if xrt else 0.05)
        ax1.tick_params(axis="x", rotation=15)

        draw_text_block(
            fig,
            0.07,
            0.07,
            0.86,
            0.26,
            "Reading the timing charts",
            [
                "When generate time < audio length, the model is faster than realtime (good).",
                "VoiceTut generates ~131s of audio in ~20s (~6.4x). SILMA ~4.2x.",
                "Chatterbox and NAMAA generate slower than the audio they output.",
                "Cold load time is separate — it hits startup, not every reply, if the model stays warm.",
            ],
        )
        footer(fig, 6, total, "TTS")
        pdf.savefig(fig)
        plt.close(fig)

        # P6 VRAM + load
        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
        fig.subplots_adjust(bottom=0.22, top=0.88)
        fig.suptitle("Resources (LOWER better on both charts)", fontweight="bold", color=C_NAVY)
        vram = [fnum(r, "peak_vram_mb", 0) or 0 for r in rows]
        bars = axes[0].bar(names, vram, color=C_RED)
        axes[0].axhline(14913, color=C_GRAY, linestyle="--", linewidth=1, label="T4 ≈ 14.9 GB")
        axes[0].set_ylabel("MB  |  LOWER better")
        axes[0].set_title("Peak VRAM (LOWER better)")
        axes[0].legend(fontsize=8)
        bar_labels(axes[0], bars, "{:.0f}", max(vram) * 0.02 if vram else 50)
        axes[0].tick_params(axis="x", rotation=15)

        load = [fnum(r, "load_seconds", 0) or 0 for r in rows]
        bars = axes[1].bar(names, load, color=C_GRAY)
        axes[1].set_ylabel("Seconds  |  LOWER better")
        axes[1].set_title("Cold load time (LOWER better)")
        bar_labels(axes[1], bars, "{:.0f}", max(load) * 0.02 if load else 2)
        axes[1].tick_params(axis="x", rotation=15)
        insight_caption(
            fig,
            "VoiceTut uses the least peak VRAM (~3.15 GB) among OK models — best for co-residency with ASR+LLM. "
            "NAMAA is heaviest (~6.1 GB). Chatterbox loads fastest but is not realtime and uses more VRAM.",
        )
        footer(fig, 7, total, "TTS")
        pdf.savefig(fig)
        plt.close(fig)

        # P7 score breakdown
        fig, ax = plt.subplots(figsize=(11, 8.5))
        fig.subplots_adjust(bottom=0.22, top=0.88)
        fig.suptitle("Normalized score breakdown (all HIGHER better)", fontweight="bold", color=C_NAVY)
        metrics = [
            ("Speed", "score_speed"),
            ("Throughput", "score_throughput"),
            ("VRAM eff.", "score_vram_efficiency"),
            ("Load", "score_load"),
            ("Balanced", "score_balanced"),
            ("Robot", "score_robot_realtime"),
        ]
        x = np.arange(len(names))
        width = 0.13
        for i, (label, key) in enumerate(metrics):
            vals = [fnum(r, key, 0) or 0 for r in rows]
            ax.bar(x + (i - 2.5) * width, vals, width, label=label, color=PALETTE[i])
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15)
        ax.set_ylabel("Score 0-100  |  HIGHER better")
        ax.set_ylim(0, 115)
        ax.legend(ncol=3, loc="upper right")
        insight_caption(
            fig,
            "VoiceTut sweeps speed/throughput/VRAM/balanced. SILMA is the clear #2. "
            "Chatterbox only leads on cold-load score — not enough for live robot use. NAMAA ranks last on robot score.",
        )
        footer(fig, 8, total, "TTS")
        pdf.savefig(fig)
        plt.close(fig)

        # P8 listening + decision
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Listening checklist & how to choose", fontweight="bold", color=C_NAVY, y=0.95)
        draw_text_block(
            fig,
            0.07,
            0.55,
            0.86,
            0.32,
            "Listen to these WAVs (score 1–5 each)",
            [
                "1) Egyptian dialect naturalness",
                "2) Arabic / English code-switching",
                "3) Numbers, dates, and times",
                "4) Prosody, pauses, and artifacts",
                "5) Overall robot-voice suitability",
                "Files: VoiceTut-TTS.wav · SILMA-TTS.wav · Chatterbox-Multilingual-V3.wav · NAMAA-Egyptian-TTS.wav",
            ],
            title_color=C_GREEN,
        )
        draw_text_block(
            fig,
            0.07,
            0.08,
            0.86,
            0.42,
            "How to choose",
            [
                "1) Default speed/resources pick → VoiceTut-TTS (also listen to confirm voice quality).",
                "2) If VoiceTut fails the listen test → try SILMA-TTS next (still realtime, similar VRAM).",
                "3) Prefer Chatterbox/NAMAA only if their voice quality is clearly superior AND latency is acceptable.",
                "4) Confirm combined VRAM with ASR + LLM on the target GPU.",
                "5) For production, also measure streaming time-to-first-audio (not covered in this bake-off).",
            ],
        )
        footer(fig, 9, total, "TTS")
        pdf.savefig(fig)
        plt.close(fig)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Combined
# ═══════════════════════════════════════════════════════════════════════════════


def build_combined_pdf() -> Path:
    llm_recs = read_json(ROOT / "llm_outputs" / "llm_recommendations.json")
    asr_recs = read_json(ROOT / "asr_outputs" / "asr_recommendations.json")
    tts_recs = read_json(ROOT / "tts_outputs" / "tts_recommendations.json")
    llm_lb = read_csv(ROOT / "llm_outputs" / "llm_leaderboard.csv")
    llm_by = [r for r in read_csv(ROOT / "llm_outputs" / "llm_analytics_by_model.csv") if fnum(r, "ok", 0)]
    asr_lb = read_csv(ROOT / "asr_outputs" / "asr_leaderboard.csv")
    tts_lb = read_csv(ROOT / "tts_outputs" / "tts_leaderboard.csv")
    tts_rows = read_csv(ROOT / "tts_outputs" / "tts_analytics_by_model.csv") or read_csv(
        ROOT / "tts_outputs" / "tts_analytics.csv"
    )

    llm_best = (llm_recs.get("picks") or {}).get("best_for_robot_realtime", {}).get("model", "—")
    asr_best = (asr_recs.get("picks") or {}).get("best_for_robot_realtime", {}).get("model", "—")
    asr_acc = (asr_recs.get("picks") or {}).get("best_accuracy", {}).get("model", "—")
    tts_best = (tts_recs.get("picks") or {}).get("best_for_robot_realtime", {}).get("model", "—")
    best_q = max(llm_by, key=lambda r: fnum(r, "avg_overall_score", 0) or 0) if llm_by else {}
    tts_ok = sum(
        1
        for r in tts_rows
        if str(r.get("success", "")).lower() in {"1", "true"} or (r.get("status") or "").lower() == "ok"
    )

    out = OUT_DIR / "Combined_Families_Summary.pdf"
    total = 8
    with PdfPages(out) as pdf:
        cover(
            pdf,
            "Combined Families Summary",
            "LLM · ASR · TTS bake-off — cross-family overview",
            [
                ("LLM (speed)", llm_best),
                ("LLM (quality)", best_q.get("model", "—")),
                ("ASR pick", asr_best),
                ("TTS pick", tts_best),
            ],
            [
                "This summary helps select one model from each family for the Arabic voice robot stack.",
                "All three families now have successful bake-off results and shortlists.",
                f"Suggested ASR default: {asr_best}. Accuracy alternative: {asr_acc}.",
                f"Suggested LLM speed default: {llm_best}. Quality alternative: {best_q.get('model', '—')}.",
                f"Suggested TTS default (speed/resources): {tts_best} - confirm by listening to the WAV.",
                "Prefer LOWER latency/memory/error rates; prefer HIGHER accuracy/throughput/quality/robot scores.",
            ],
            [
                "Page 2: shared metrics glossary (RTF, TTFT, WER, VRAM, ...).",
                "Page 3: shortlist table and decision path.",
                "Page 4: family readiness and top scores.",
                "Pages 5-7: LLM, ASR, and TTS snapshot charts.",
                "Page 8: package contents and bottom line.",
            ],
            "Combined",
            total,
        )

        glossary_page(
            pdf,
            "Combined",
            LLM_METRICS[:3] + ASR_METRICS[:3] + TTS_METRICS[:3],
            2,
            total,
            [
                "Shared rule: LOWER = TTFT, RTF, WER, CER, VRAM, RAM, CPU%, load/generate/transcribe time.",
                "Shared rule: HIGHER = accuracy%, tokens/s, chars/s, quality scores, robot/balanced scores, xRealtime.",
                "See each family PDF glossary for the full term list.",
            ],
        )

        # P2 shortlist
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Recommended shortlist", fontweight="bold", color=C_NAVY, y=0.95)
        shortlist = [
            ("ASR — robot default", asr_best, "Best accuracy/speed/VRAM blend for realtime input"),
            ("ASR — accuracy alt", asr_acc, "Highest word accuracy when recognition quality is critical"),
            ("LLM — speed default", llm_best, "Best TTFT/throughput/VRAM composite for turn-taking"),
            (
                "LLM — quality alt",
                best_q.get("model", "—"),
                f"Best rubric quality ({fnum(best_q, 'avg_overall_score', 0):.2f}/5) and auto-pass rate",
            ),
            (
                "TTS — speed/resources",
                tts_best,
                "Best RTF/throughput/VRAM composite — listen to WAV before freeze",
            ),
        ]
        y = 0.82
        for label, model, why in shortlist:
            ax = fig.add_axes([0.07, y - 0.10, 0.86, 0.10])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=C_LINE, linewidth=1, transform=ax.transAxes))
            ax.add_patch(plt.Rectangle((0, 0), 0.012, 1, facecolor=C_TEAL, transform=ax.transAxes))
            ax.text(0.04, 0.65, label, fontsize=8, color=C_GRAY, transform=ax.transAxes)
            ax.text(0.04, 0.22, model, fontsize=12, fontweight="bold", color=C_NAVY, transform=ax.transAxes)
            ax.text(0.42, 0.35, why, fontsize=8.5, color="#2d3748", transform=ax.transAxes)
            y -= 0.115
        draw_text_block(
            fig,
            0.07,
            0.06,
            0.86,
            0.22,
            "Decision path",
            [
                "1) Lock ASR first — input quality gates the whole pipeline.",
                "2) Lock LLM next — choose speed (Nile-Chat-4B) or quality (Qwen3-8B / Qwen3-4B).",
                f"3) Lock TTS after listening — default {tts_best}; fallback SILMA-TTS if voice quality wins.",
                "4) Validate the combined VRAM budget on the deployment GPU.",
            ],
        )
        footer(fig, 3, total, "Combined")
        pdf.savefig(fig)
        plt.close(fig)

        # P3 readiness
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Family readiness & top robot scores", fontweight="bold", color=C_NAVY, y=0.97)
        ax0 = fig.add_axes([0.08, 0.40, 0.40, 0.50])
        ax1 = fig.add_axes([0.56, 0.40, 0.40, 0.50])
        families = ["LLM", "ASR", "TTS"]
        ok_counts = [len(llm_by), len(asr_lb), tts_ok]
        fail_counts = [
            len(read_csv(ROOT / "llm_outputs" / "llm_analytics_by_model.csv")) - len(llm_by),
            0,
            len(tts_rows) - tts_ok,
        ]
        x = np.arange(len(families))
        ax0.bar(x, ok_counts, 0.55, label="OK models", color=C_GREEN)
        ax0.bar(x, fail_counts, 0.55, bottom=ok_counts, label="Failed / gated", color=C_RED)
        ax0.set_xticks(x)
        ax0.set_xticklabels(families)
        ax0.set_ylabel("Models")
        ax0.set_title("Bake-off coverage")
        ax0.legend()

        top_scores = [
            fnum(llm_lb[0], "score_robot_realtime", 0) or 0 if llm_lb else 0,
            fnum(asr_lb[0], "score_robot_realtime", 0) or 0 if asr_lb else 0,
            fnum(tts_lb[0], "score_robot_realtime", 0) or 0 if tts_lb else 0,
        ]
        top_labels = [
            short(llm_lb[0]["model"]) if llm_lb else "—",
            short(asr_lb[0]["model"]) if asr_lb else "—",
            short(tts_lb[0]["model"]) if tts_lb else "—",
        ]
        bars = ax1.bar(families, top_scores, color=[C_BLUE, C_TEAL, C_PURPLE])
        ax1.set_ylim(0, 115)
        ax1.set_ylabel("Robot score")
        ax1.set_title("Top robot score per family")
        for i, (bar, lab) in enumerate(zip(bars, top_labels)):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{top_scores[i]:.1f}\n{lab}",
                ha="center",
                fontsize=8,
            )
        draw_text_block(
            fig,
            0.07,
            0.07,
            0.86,
            0.26,
            "Readiness reading",
            [
                "ASR is fully covered (12/12 OK). LLM has 4 OK models + 3 gated failures.",
                f"TTS is now ready for shortlisting ({tts_ok}/{len(tts_rows)} OK) — VoiceTut leads robot score.",
                "All three families can be shortlisted together; only TTS still needs a human listening pass.",
                "Plan GPU headroom for Whisper + LLM + VoiceTut/SILMA together.",
            ],
        )
        footer(fig, 4, total, "Combined")
        pdf.savefig(fig)
        plt.close(fig)

        # P4 LLM snapshot
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("LLM snapshot", fontweight="bold", color=C_NAVY, y=0.97)
        ax0 = fig.add_axes([0.08, 0.40, 0.40, 0.50])
        ax1 = fig.add_axes([0.56, 0.40, 0.40, 0.50])
        names = [short(r["model"]) for r in llm_lb]
        robot = [fnum(r, "score_robot_realtime", 0) or 0 for r in llm_lb]
        qmap = {r["model"]: fnum(r, "avg_overall_score", 0) or 0 for r in llm_by}
        quality = [qmap.get(r["model"], 0) for r in llm_lb]
        x = np.arange(len(names))
        w = 0.35
        ax0.bar(x - w / 2, robot, w, label="Robot /100", color=C_BLUE)
        ax0.bar(x + w / 2, [q * 20 for q in quality], w, label="Quality x20 (/5->100)", color=C_GREEN)
        ax0.set_xticks(x)
        ax0.set_xticklabels(names, rotation=20)
        ax0.set_ylim(0, 115)
        ax0.legend(fontsize=8)
        ax0.set_title("Speed composite vs quality")

        ttft = [fnum(r, "avg_first_token_seconds", 0) or 0 for r in llm_lb]
        vram = [(fnum(r, "peak_vram_mb", 0) or 0) / 1000 for r in llm_lb]
        ax1.scatter(
            ttft,
            vram,
            s=[40 + (qmap.get(r["model"], 0) * 40) for r in llm_lb],
            c=[PALETTE[i] for i in range(len(llm_lb))],
            alpha=0.85,
        )
        for i, r in enumerate(llm_lb):
            ax1.annotate(short(r["model"]), (ttft[i], vram[i]), textcoords="offset points", xytext=(5, 4), fontsize=8)
        ax1.set_xlabel("TTFT (s)")
        ax1.set_ylabel("Peak VRAM (GB)")
        ax1.set_title("Bubble size ~ quality")
        draw_text_block(
            fig,
            0.07,
            0.07,
            0.86,
            0.26,
            "LLM takeaway",
            [
                f"{llm_best} wins robot realtime; {best_q.get('model', '—')} wins quality.",
                "If spoken answers must stay short and correct, lean quality (Qwen).",
                "If the robot must start speaking immediately, lean Nile-Chat-4B — then spot-check verbosity.",
                "Qwen3-4B is the practical middle ground for many deployments.",
            ],
        )
        footer(fig, 5, total, "Combined")
        pdf.savefig(fig)
        plt.close(fig)

        # P5 ASR snapshot
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("ASR snapshot (top 8)", fontweight="bold", color=C_NAVY, y=0.97)
        ax0 = fig.add_axes([0.10, 0.40, 0.38, 0.50])
        ax1 = fig.add_axes([0.56, 0.40, 0.40, 0.50])
        top = asr_lb[:8]
        anames = [short(r["model"]) for r in top]
        bars = ax0.barh(anames[::-1], [fnum(r, "score_robot_realtime", 0) or 0 for r in top][::-1], color=C_TEAL)
        ax0.set_xlabel("Robot score")
        ax0.set_xlim(0, 115)
        hbar_labels(ax0, bars)
        ax0.set_title("Robot leaderboard")
        for i, r in enumerate(top):
            ax1.scatter(
                fnum(r, "avg_rtf", 0) or 0,
                fnum(r, "avg_accuracy_percent", 0) or 0,
                s=(fnum(r, "peak_vram_mb", 0) or 1000) / 25,
                color=PALETTE[i % len(PALETTE)],
                alpha=0.85,
            )
            ax1.annotate(
                short(r["model"]),
                (fnum(r, "avg_rtf", 0) or 0, fnum(r, "avg_accuracy_percent", 0) or 0),
                fontsize=7,
                textcoords="offset points",
                xytext=(4, 3),
            )
        ax1.set_xlabel("RTF")
        ax1.set_ylabel("Accuracy %")
        ax1.set_title("Accuracy vs speed (bubble = VRAM)")
        draw_text_block(
            fig,
            0.07,
            0.07,
            0.86,
            0.26,
            "ASR takeaway",
            [
                f"Default: {asr_best} for robot balance.",
                f"Accuracy alternative: {asr_acc}.",
                "Whisper CT2 family occupies the useful top band; Qwen ASR variants lag badly on WER.",
                "Keep ASR VRAM low so the LLM + TTS still fit on the same GPU.",
            ],
        )
        footer(fig, 6, total, "Combined")
        pdf.savefig(fig)
        plt.close(fig)

        # P6 TTS snapshot
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("TTS snapshot", fontweight="bold", color=C_NAVY, y=0.97)
        ax0 = fig.add_axes([0.08, 0.40, 0.40, 0.50])
        ax1 = fig.add_axes([0.56, 0.40, 0.40, 0.50])
        tnames = [short(r["model"]) for r in tts_lb] if tts_lb else [short(r.get("model", "")) for r in tts_rows]
        trobot = [fnum(r, "score_robot_realtime", 0) or 0 for r in (tts_lb or tts_rows)]
        bars = ax0.barh(tnames[::-1], trobot[::-1], color=[PALETTE[i % len(PALETTE)] for i in range(len(tnames))][::-1])
        ax0.set_xlabel("Robot score")
        ax0.set_xlim(0, 115)
        hbar_labels(ax0, bars)
        ax0.set_title("TTS robot leaderboard")

        for i, r in enumerate(tts_lb or tts_rows):
            ax1.scatter(
                fnum(r, "rtf", 0) or 0,
                fnum(r, "chars_per_second", 0) or 0,
                s=(fnum(r, "peak_vram_mb", 0) or 1000) / 20,
                color=PALETTE[i % len(PALETTE)],
                alpha=0.85,
            )
            ax1.annotate(
                short(r.get("model", "")),
                (fnum(r, "rtf", 0) or 0, fnum(r, "chars_per_second", 0) or 0),
                fontsize=8,
                textcoords="offset points",
                xytext=(5, 4),
            )
        ax1.axvline(1.0, color=C_RED, linestyle="--", linewidth=1, label="RTF=1")
        ax1.set_xlabel("RTF (lower better)")
        ax1.set_ylabel("Chars / second")
        ax1.set_title("Speed vs throughput (bubble = VRAM)")
        ax1.legend(fontsize=8)
        draw_text_block(
            fig,
            0.07,
            0.07,
            0.86,
            0.26,
            "TTS takeaway",
            [
                f"Default speed/resources pick: {tts_best}.",
                "SILMA-TTS is the best realtime alternative if VoiceTut fails the listen test.",
                "Chatterbox and NAMAA are not realtime on this bake-off — use only if voice quality clearly wins.",
                "Always listen to WAVs before freezing the production voice.",
            ],
        )
        footer(fig, 7, total, "Combined")
        pdf.savefig(fig)
        plt.close(fig)

        # P7 package + bottom line
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Package contents & bottom line", fontweight="bold", color=C_NAVY, y=0.95)
        draw_text_block(
            fig,
            0.07,
            0.48,
            0.86,
            0.38,
            "PDF package",
            [
                "LLM_Model_Selection_Report.pdf — charts + picks + quality guidance",
                "ASR_Model_Selection_Report.pdf — charts + accuracy/speed guidance",
                "TTS_Model_Selection_Report.pdf — charts + picks + listening checklist",
                "Combined_Families_Summary.pdf — this cross-family overview",
                "",
                "WAV files for listening: analytics/final/tts_outputs/*.wav",
            ],
        )
        draw_text_block(
            fig,
            0.07,
            0.08,
            0.86,
            0.34,
            "Bottom line",
            [
                f"ASR = {asr_best} (or {asr_acc} for max accuracy).",
                f"LLM = {llm_best} (speed) or {best_q.get('model', '—')} (quality).",
                f"TTS = {tts_best} pending listening confirmation (SILMA as backup).",
                "See family PDFs for full evidence before final sign-off.",
            ],
            title_color=C_GREEN,
        )
        footer(fig, 8, total, "Combined")
        pdf.savefig(fig)
        plt.close(fig)

    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        build_llm_pdf(),
        build_asr_pdf(),
        build_tts_pdf(),
        build_combined_pdf(),
    ]
    print(f"Wrote {len(paths)} PDFs to {OUT_DIR}")
    for p in paths:
        print(f"  {p.name}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
