"""Export analytics/llm_outputs bake-off charts into a PDF report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent / "llm_outputs"
OUT_PDF = ROOT / "llm_analytics_report.pdf"

# Prefer Windows fonts that cover Arabic + Latin.
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
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fnum(row: dict[str, str], key: str, default: float = 0.0) -> float:
    raw = (row.get(key) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def short_name(model: str) -> str:
    mapping = {
        "Qwen3-4B-Instruct-2507": "Qwen3-4B",
        "Nile-Chat-4B": "Nile-Chat-4B",
        "Qwen3-8B": "Qwen3-8B",
        "ALLaM-7B": "ALLaM-7B",
    }
    return mapping.get(model, model)


def add_footer(fig: plt.Figure, page: int, total: int) -> None:
    fig.text(
        0.5,
        0.015,
        f"Source: analytics/llm_outputs  ·  Tesla T4  ·  suite=first / A001  ·  page {page}/{total}",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#666666",
    )


def page_cover(pdf: PdfPages, recs: dict, ok: int, failed: int, total_pages: int) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Arabic LLM Bake-off Analytics", fontsize=20, fontweight="bold", y=0.92)
    fig.text(
        0.5,
        0.86,
        "Report generated from analytics/llm_outputs",
        ha="center",
        fontsize=11,
        color="#444444",
    )

    picks = recs.get("picks", {})
    lines = [
        f"Models OK: {ok} / {ok + failed}",
        f"Best robot realtime: {picks.get('best_for_robot_realtime', {}).get('model', '—')}",
        f"Lowest TTFT: {picks.get('lowest_ttft', {}).get('model', '—')}",
        f"Highest throughput: {picks.get('highest_throughput', {}).get('model', '—')}",
        f"Lowest VRAM: {picks.get('lowest_vram', {}).get('model', '—')}",
        "",
        "Note: single-prompt pilot (A001 only).",
        "Use ranks for latency/VRAM screening, not final quality selection.",
    ]
    fig.text(0.12, 0.72, "\n".join(lines), va="top", fontsize=12, family="monospace")

    notes = recs.get("notes") or []
    if notes:
        fig.text(0.12, 0.38, "Notes", fontsize=13, fontweight="bold")
        fig.text(0.12, 0.34, "\n".join(f"• {n}" for n in notes), va="top", fontsize=10)

    add_footer(fig, 1, total_pages)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_robot_and_outcomes(
    pdf: PdfPages,
    leaderboard: list[dict[str, str]],
    ok: int,
    failed: int,
    page: int,
    total_pages: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
    fig.suptitle("Robot score & run outcomes", fontsize=16, fontweight="bold")

    names = [short_name(r["model"]) for r in leaderboard]
    robot = [fnum(r, "score_robot_realtime") for r in leaderboard]
    colors = ["#2f6fed", "#3d9a6a", "#c98512", "#b42318"]
    axes[0].barh(names[::-1], robot[::-1], color=colors[: len(names)][::-1])
    axes[0].set_xlabel("Robot realtime score (0–100)")
    axes[0].set_title("Robot realtime score by model")
    for y, v in enumerate(robot[::-1]):
        axes[0].text(v + 1, y, f"{v:.1f}", va="center", fontsize=9)
    axes[0].set_xlim(0, 110)

    axes[1].pie(
        [ok, failed],
        labels=[f"OK ({ok})", f"HF auth fail ({failed})"],
        autopct="%1.0f%%",
        colors=["#3d9a6a", "#b42318"],
        startangle=90,
        wedgeprops={"width": 0.45},
    )
    axes[1].set_title("Run outcomes (7 models × 1 prompt)")

    add_footer(fig, page, total_pages)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_latency_throughput(
    pdf: PdfPages,
    leaderboard: list[dict[str, str]],
    page: int,
    total_pages: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
    fig.suptitle("Latency & throughput", fontsize=16, fontweight="bold")

    names = [short_name(r["model"]) for r in leaderboard]
    ttft = [fnum(r, "avg_first_token_seconds") for r in leaderboard]
    tps = [fnum(r, "avg_tokens_per_second") for r in leaderboard]

    axes[0].bar(names, ttft, color="#c98512")
    axes[0].set_ylabel("Seconds")
    axes[0].set_title("Time to first token (lower is better)")
    for x, v in enumerate(ttft):
        axes[0].text(x, v + 0.02, f"{v:.3f}s", ha="center", fontsize=8)

    axes[1].bar(names, tps, color="#3d9a6a")
    axes[1].set_ylabel("Tokens / second")
    axes[1].set_title("Generation throughput (higher is better)")
    for x, v in enumerate(tps):
        axes[1].text(x, v + 0.15, f"{v:.2f}", ha="center", fontsize=8)

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)

    add_footer(fig, page, total_pages)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_vram_load(
    pdf: PdfPages,
    leaderboard: list[dict[str, str]],
    page: int,
    total_pages: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
    fig.suptitle("VRAM & model load time", fontsize=16, fontweight="bold")

    names = [short_name(r["model"]) for r in leaderboard]
    vram = [fnum(r, "peak_vram_mb") for r in leaderboard]
    load = [fnum(r, "avg_load_seconds") for r in leaderboard]

    axes[0].bar(names, vram, color="#b42318")
    axes[0].axhline(14913, color="#666666", linestyle="--", linewidth=1, label="T4 total ≈ 14.9 GB")
    axes[0].set_ylabel("MB")
    axes[0].set_title("Peak VRAM on Tesla T4")
    axes[0].legend(fontsize=8)
    for x, v in enumerate(vram):
        axes[0].text(x, v + 120, f"{v:.0f}", ha="center", fontsize=8)

    axes[1].bar(names, load, color="#555555")
    axes[1].set_ylabel("Seconds")
    axes[1].set_title("Cold load time")
    for x, v in enumerate(load):
        axes[1].text(x, v + 5, f"{v:.0f}s", ha="center", fontsize=8)

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)

    add_footer(fig, page, total_pages)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_score_breakdown(
    pdf: PdfPages,
    leaderboard: list[dict[str, str]],
    page: int,
    total_pages: int,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.suptitle("Normalized score breakdown (OK models)", fontsize=16, fontweight="bold")

    names = [short_name(r["model"]) for r in leaderboard]
    metrics = [
        ("Latency", "score_latency"),
        ("Throughput", "score_throughput"),
        ("VRAM efficiency", "score_vram_efficiency"),
        ("Load", "score_load"),
    ]
    x = range(len(names))
    width = 0.18
    for i, (label, key) in enumerate(metrics):
        vals = [fnum(r, key) for r in leaderboard]
        offset = (i - 1.5) * width
        ax.bar([xi + offset for xi in x], vals, width=width, label=label)

    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel("Score (0–100)")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.set_title("Source: llm_leaderboard.csv · score_* columns")

    add_footer(fig, page, total_pages)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_tables(
    pdf: PdfPages,
    leaderboard: list[dict[str, str]],
    analytics: list[dict[str, str]],
    page: int,
    total_pages: int,
) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Leaderboard & timing detail", fontsize=16, fontweight="bold")

    ax1 = fig.add_axes([0.05, 0.52, 0.90, 0.38])
    ax1.axis("off")
    ax1.set_title("Robot realtime leaderboard", loc="left", fontsize=12, pad=12)
    lb_rows = [
        [
            str(i + 1),
            short_name(r["model"]),
            f"{fnum(r, 'score_robot_realtime'):.2f}",
            f"{fnum(r, 'avg_first_token_seconds'):.3f}",
            f"{fnum(r, 'avg_tokens_per_second'):.2f}",
            f"{fnum(r, 'peak_vram_mb'):.0f}",
            r.get("load_mode", ""),
        ]
        for i, r in enumerate(leaderboard)
    ]
    t1 = ax1.table(
        cellText=lb_rows,
        colLabels=["Rank", "Model", "Robot", "TTFT (s)", "tok/s", "VRAM pk", "Mode"],
        loc="upper center",
        cellLoc="center",
    )
    t1.auto_set_font_size(False)
    t1.set_fontsize(9)
    t1.scale(1, 1.4)

    ok_rows = [r for r in analytics if r.get("status") == "ok"]
    ax2 = fig.add_axes([0.05, 0.08, 0.90, 0.38])
    ax2.axis("off")
    ax2.set_title("Timing & resources (A001)", loc="left", fontsize=12, pad=12)
    detail = [
        [
            short_name(r["model"]),
            f"{fnum(r, 'generate_seconds'):.2f}",
            r.get("completion_tokens", ""),
            f"{fnum(r, 'chars_per_second'):.1f}",
            f"{fnum(r, 'peak_ram_mb'):.0f}",
            f"{fnum(r, 'model_vram_mb'):.0f}",
        ]
        for r in ok_rows
    ]
    t2 = ax2.table(
        cellText=detail,
        colLabels=["Model", "Generate (s)", "Tok", "Chars/s", "Peak RAM", "Model VRAM"],
        loc="upper center",
        cellLoc="center",
    )
    t2.auto_set_font_size(False)
    t2.set_fontsize(9)
    t2.scale(1, 1.4)

    add_footer(fig, page, total_pages)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_responses(
    pdf: PdfPages,
    analytics: list[dict[str, str]],
    page: int,
    total_pages: int,
) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Sample responses — A001: عامل إيه النهارده؟", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.90,
        "Auto quality scores 5/5 for OK models — manual Arabic review still required.",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    ok_rows = [r for r in analytics if r.get("status") == "ok"]
    # Prefer response_text from summary.csv if present; else leave blank.
    y = 0.82
    for r in ok_rows:
        model = r.get("model", "")
        resp = (r.get("response_text") or "").strip()
        if not resp:
            # fallback files
            resp_path = ROOT / "responses" / model / "A001.txt"
            if resp_path.exists():
                resp = resp_path.read_text(encoding="utf-8").strip()
        block = (
            f"{model}  |  TTFT {fnum(r, 'first_token_seconds'):.3f}s  |  "
            f"{fnum(r, 'tokens_per_second'):.2f} tok/s  |  pass 5/5\n"
            f"{resp}"
        )
        fig.text(0.08, y, block, va="top", fontsize=10, wrap=True)
        y -= 0.18

    failed = [r for r in analytics if r.get("status") == "error"]
    if failed:
        fig.text(0.08, y, "Failed models (no response):", fontsize=11, fontweight="bold")
        y -= 0.04
        for r in failed:
            fig.text(
                0.08,
                y,
                f"• {r.get('model')}: gated HF auth required",
                fontsize=9,
                color="#b42318",
            )
            y -= 0.03

    add_footer(fig, page, total_pages)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_picks(
    pdf: PdfPages,
    recs: dict,
    page: int,
    total_pages: int,
) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Recommended picks", fontsize=16, fontweight="bold")

    picks = recs.get("picks", {})
    y = 0.82
    for key, title in [
        ("best_for_robot_realtime", "Best for robot realtime"),
        ("lowest_ttft", "Lowest TTFT"),
        ("highest_throughput", "Highest throughput"),
        ("lowest_vram", "Lowest VRAM"),
        ("best_balanced", "Best balanced"),
    ]:
        item = picks.get(key) or {}
        metrics = item.get("metrics") or {}
        metric_txt = ", ".join(f"{k}={v}" for k, v in metrics.items())
        text = (
            f"{title}\n"
            f"  Model: {item.get('model', '—')}\n"
            f"  Why: {item.get('why', '')}\n"
            f"  Metrics: {metric_txt}"
        )
        fig.text(0.1, y, text, va="top", fontsize=11)
        y -= 0.14

    cats = recs.get("category_specialists") or {}
    if cats:
        fig.text(0.1, y, "Category specialists", fontsize=12, fontweight="bold")
        y -= 0.04
        for cat, info in cats.items():
            fig.text(
                0.1,
                y,
                f"• {cat}: {info.get('model')} "
                f"(TTFT={info.get('avg_first_token_seconds')}, "
                f"tok/s={info.get('avg_tokens_per_second')})",
                fontsize=10,
            )
            y -= 0.03

    add_footer(fig, page, total_pages)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> Path:
    leaderboard = read_csv(ROOT / "llm_leaderboard.csv")
    analytics = read_csv(ROOT / "summary.csv")
    if not analytics:
        analytics = read_csv(ROOT / "llm_analytics.csv")
    recs = json.loads((ROOT / "llm_recommendations.json").read_text(encoding="utf-8"))

    ok = sum(1 for r in analytics if r.get("status") == "ok")
    failed = sum(1 for r in analytics if r.get("status") == "error")

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    total_pages = 8
    with PdfPages(OUT_PDF) as pdf:
        page_cover(pdf, recs, ok, failed, total_pages)
        page_picks(pdf, recs, 2, total_pages)
        page_robot_and_outcomes(pdf, leaderboard, ok, failed, 3, total_pages)
        page_latency_throughput(pdf, leaderboard, 4, total_pages)
        page_vram_load(pdf, leaderboard, 5, total_pages)
        page_score_breakdown(pdf, leaderboard, 6, total_pages)
        page_tables(pdf, leaderboard, analytics, 7, total_pages)
        page_responses(pdf, analytics, 8, total_pages)

    return OUT_PDF


if __name__ == "__main__":
    path = main()
    print(path)
    print(f"size_bytes={path.stat().st_size}")
