#!/usr/bin/env python3
"""
Kaggle Arabic LLM comparison (T4 / T4x2 safe)
=============================================
Each model runs in its own subprocess so one CUDA/model failure does not poison
the rest of the benchmark.

Usage on Kaggle:
  1) Upload this file AND arabic_voice_robot_llm_dataset.py to /kaggle/working/
     (optional: also upload arabic_voice_robot_llm_test_dataset.md for reference).
  2) Prompts are built from the Arabic voice-robot dataset (first-round 20 critical
     tests by default). Override with --suite full|critical|first or a custom
     --prompts-file JSON.
  3) Add HF_TOKEN as a Kaggle/Colab secret (and accept each model license on Hugging Face)
     for gated models: Gemma / Mistral / Jais / SILMA. Nile-Chat is usually public.
  4) Run:
       %run /kaggle/working/kaggle_arabic_llm_compare.py
     Full dataset suite:
       %run /kaggle/working/kaggle_arabic_llm_compare.py --suite full
     Repeat after packages are installed:
       %run /kaggle/working/kaggle_arabic_llm_compare.py --no-install
     Run specific models:
       %run /kaggle/working/kaggle_arabic_llm_compare.py --only Qwen3-8B,ALLaM-7B,Jais-2-8B
     Skip other heavy ones:
       %run /kaggle/working/kaggle_arabic_llm_compare.py --skip Gemma3-12B-IT
     Opt in to 14B (needs enough RAM; can kill single-T4 Colab sessions):
       %run /kaggle/working/kaggle_arabic_llm_compare.py --only Qwen3-14B
     Force dependency install on Colab / custom WORK_DIR:
       %run .../kaggle_arabic_llm_compare.py --install

After the run finishes, a zip of all llm_outputs is written next to the output folder.
Notes for T4 / T4x2:
  - Checkpoints live on /kaggle/tmp and are purged between models (working disk is small).
  - Qwen3-14B, Qwen3-30B-A3B and Mistral-24B are OFF by default (14B RAM-spikes the
    runtime; 24B/30B ~45–65 GB downloads fill temp disk → worker exit -7 / SIGBUS).

Outputs:
  /kaggle/working/llm_outputs/responses/<model>/<prompt_id>.txt
  /kaggle/working/llm_outputs/summary.json
  /kaggle/working/llm_outputs/summary.csv
  /kaggle/working/llm_outputs/llm_analytics.csv
  /kaggle/working/llm_outputs/llm_analytics_by_model.csv
  /kaggle/working/llm_outputs/llm_analytics_by_category.csv
  /kaggle/working/llm_outputs/llm_leaderboard.csv
  /kaggle/working/llm_outputs/llm_recommendations.json
  /kaggle/working/llm_outputs/llm_selection_report.md
      (TTFT, tok/s, resources, category breakdowns, rankings, robot picks)

Model selection notes (web / model-card research, mid-2026):
  - Qwen3-4B/8B/14B/30B-A3B: strong multilingual baselines; good Arabic + English.
  - Gemma 3 4B/12B IT: compact multilingual alternatives.
  - Mistral Small 3.1 24B: conversational / tool-use candidate (heavier).
  - ALLaM-7B: Saudi Arabic/English specialized open model.
  - Jais-2-8B: Arabic-centric, dialects + Arabic/English code-switching.
  - Nile-Chat-4B: Egyptian dialect specialist (Arabic script + Arabizi).
  - SILMA-9B / Fanar-1-9B: additional Arabic open-weight candidates (optional).
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

WORK_DIR = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
if not WORK_DIR.exists():
    WORK_DIR = Path.cwd() / "kaggle_working"
WORK_DIR.mkdir(parents=True, exist_ok=True)


def _scratch_dir() -> Path:
    """Big ephemeral disk for model caches.

    /kaggle/working is limited to ~19 GB and persisted as notebook output, so
    only results should live there. Downloaded checkpoints go to the temp disk
    instead (not persisted, but much larger).
    """
    override = os.environ.get("KAGGLE_SCRATCH_DIR")
    candidates = [Path(override)] if override else []
    if Path("/kaggle/working").exists():  # on Kaggle
        candidates += [Path("/kaggle/tmp"), Path("/tmp/kaggle_scratch")]
    for cand in candidates:
        try:
            cand.mkdir(parents=True, exist_ok=True)
            probe = cand / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return cand
        except Exception:
            continue
    return WORK_DIR


SCRATCH_DIR = _scratch_dir()

OUTPUT_DIR = WORK_DIR / "llm_outputs"
RESPONSE_DIR = OUTPUT_DIR / "responses"
META_DIR = WORK_DIR / "llm_meta"
INPUT_DIR = WORK_DIR / "llm_inputs"
CACHE_DIR = SCRATCH_DIR / "llm_cache"

for d in (OUTPUT_DIR, RESPONSE_DIR, META_DIR, CACHE_DIR, INPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

DEFAULT_PROMPTS_PATH = INPUT_DIR / "prompts.json"
MAX_NEW_TOKENS = int(os.environ.get("LLM_MAX_NEW_TOKENS", "256"))
# Deterministic defaults from arabic_voice_robot_llm_test_dataset.md §1
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
TOP_P = float(os.environ.get("LLM_TOP_P", "1.0"))
DEFAULT_SUITE = os.environ.get("LLM_SUITE", "first")


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

# load_mode:
#   auto  — bf16/fp16 on GPU if it likely fits; else 4-bit
#   bf16  — force half precision
#   int4  — force bitsandbytes 4-bit
#
# Kaggle T4 notes:
#   - HF still downloads full-precision weights even for bitsandbytes int4.
#   - /kaggle/tmp fills fast if large checkpoints accumulate; we purge between models.
#   - 24B/30B downloads (~45–65 GB) routinely exhaust temp disk → worker SIGBUS (-7)
#     and then OSError Errno 5 when spawning the next python. Keep them opt-in.
ENABLE = {
    # User-requested core set
    "Qwen3-4B-Instruct-2507": True,
    "Qwen3-8B": True,
    "Qwen3-14B": False,  # ~30GB download + RAM spike; kills single-T4 Colab/Kaggle sessions
    "Qwen3-30B-A3B-Instruct-2507": False,  # ~60GB download; enable with --only if you have disk headroom
    "Gemma3-4B-IT": True,
    "Gemma3-12B-IT": True,
    "Mistral-Small-3.1-24B": False,        # ~48GB download; opt-in on T4x2 with clean scratch
    "ALLaM-7B": True,
    # Research extras for Arabic / Egyptian / code-switching
    "Jais-2-8B": True,
    "Nile-Chat-4B": True,
    "SILMA-9B": False,          # optional Arabic open-weight baseline
    "Fanar-1-9B": False,        # optional QCRI Arabic/English baseline
}

MODEL_ORDER = [
    "Qwen3-4B-Instruct-2507",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-30B-A3B-Instruct-2507",
    "Gemma3-4B-IT",
    "Gemma3-12B-IT",
    "Mistral-Small-3.1-24B",
    "ALLaM-7B",
    "Jais-2-8B",
    "Nile-Chat-4B",
    "SILMA-9B",
    "Fanar-1-9B",
]

MODELS: dict[str, dict[str, Any]] = {
    "Qwen3-4B-Instruct-2507": {
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "role": "Small multilingual Apache-2.0 baseline",
        "load_mode": "bf16",
        "enable_thinking": False,
        "approx_params_b": 4.0,
        "approx_disk_gb": 9.0,
    },
    "Qwen3-8B": {
        "model_id": "Qwen/Qwen3-8B",
        "role": "Main production quality/speed candidate",
        # bf16 barely fits a 16GB T4 and often CPU-offloads (tok/s collapses); int4 stays on-GPU.
        "load_mode": "int4",
        "enable_thinking": False,
        "approx_params_b": 8.0,
        "approx_disk_gb": 17.0,
    },
    "Qwen3-14B": {
        "model_id": "Qwen/Qwen3-14B",
        "role": "Higher-quality reasoning / conversation candidate",
        "load_mode": "int4",
        "enable_thinking": False,
        "approx_params_b": 14.0,
        "approx_disk_gb": 30.0,
    },
    "Qwen3-30B-A3B-Instruct-2507": {
        "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "role": "30B MoE (~3B active) efficiency experiment",
        "load_mode": "int4",
        "enable_thinking": False,
        "approx_params_b": 30.5,
        "approx_disk_gb": 62.0,
    },
    "Gemma3-4B-IT": {
        "model_id": "google/gemma-3-4b-it",
        "role": "Compact multilingual alternative",
        "load_mode": "bf16",
        "approx_params_b": 4.0,
        "approx_disk_gb": 9.0,
        "gated": True,
    },
    "Gemma3-12B-IT": {
        "model_id": "google/gemma-3-12b-it",
        "role": "Medium multilingual competitor to Qwen-14B",
        "load_mode": "int4",
        "approx_params_b": 12.0,
        "approx_disk_gb": 26.0,
        "gated": True,
    },
    "Mistral-Small-3.1-24B": {
        "model_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "role": "Conversational / function-calling premium local candidate",
        "load_mode": "int4",
        "approx_params_b": 24.0,
        "approx_disk_gb": 48.0,
        "gated": True,
    },
    "ALLaM-7B": {
        "model_id": "ALLaM-AI/ALLaM-7B-Instruct-preview",
        "role": "Arabic-specialized bilingual (SDAIA)",
        "load_mode": "bf16",
        "approx_params_b": 7.0,
        "approx_disk_gb": 15.0,
    },
    "Jais-2-8B": {
        "model_id": "inceptionai/Jais-2-8B-Chat",
        "role": "Arabic-centric + dialects + Arabic/English code-switching",
        "load_mode": "int4",
        "approx_params_b": 8.0,
        "approx_disk_gb": 17.0,
        "gated": True,
    },
    "Nile-Chat-4B": {
        "model_id": "MBZUAI-Paris/Nile-Chat-4B",
        "role": "Egyptian Arabic specialist (script + Arabizi)",
        "load_mode": "bf16",
        "approx_params_b": 4.0,
        "approx_disk_gb": 9.0,
        # Public download in practice (Gemma-family weights, no HF gate observed).
        "gated": False,
    },
    "SILMA-9B": {
        "model_id": "silma-ai/SILMA-9B-Instruct-v1.0",
        "role": "Arabic open-weight Gemma-based baseline",
        "load_mode": "int4",
        "approx_params_b": 9.0,
        "approx_disk_gb": 19.0,
        "gated": True,
    },
    "Fanar-1-9B": {
        "model_id": "QCRI/Fanar-1-9B-Instruct",
        "role": "QCRI Arabic/English + dialect open model",
        "load_mode": "int4",
        "approx_params_b": 8.7,
        "approx_disk_gb": 18.0,
    },
}

DEFAULT_SYSTEM = (
    "You are the conversational assistant for an Arabic voice robot.\n\n"
    "Respond in concise, natural Egyptian Arabic unless the user requests another language or style.\n"
    "Use English only for common technical terms, product names, or when the user uses English.\n"
    "Put the direct answer first.\n"
    "Avoid Markdown tables, long headings, URLs, emojis, and unnecessary introductions.\n"
    "Keep ordinary answers under two short sentences unless more detail is required.\n"
    "Resolve clear self-corrections using the user's latest value.\n"
    "Do not invent missing information.\n"
    "Ask one short clarification only when a required value is missing.\n"
    "For tool calls, return valid arguments that exactly match the provided schema.\n"
    "Never claim that an action succeeded before receiving a successful tool result.\n"
    "Protect private information and refuse unsafe or unauthorized requests."
)

# Legacy single-turn prompts kept only as emergency fallback if the dataset module
# cannot be imported. Prefer arabic_voice_robot_llm_dataset.VOICE_ROBOT_CASES.
BUILTIN_PROMPTS: list[dict[str, Any]] = [
    {
        "id": "A001",
        "category": "egyptian_arabic",
        "priority": "critical",
        "user": "عامل إيه النهارده؟",
        "turns": [{"role": "user", "content": "عامل إيه النهارده؟"}],
        "expected_behavior": ["Respond naturally in Egyptian Arabic."],
        "forbidden_behavior": ["Respond only in English."],
        "checks": {"max_sentences": 3, "expect_lang": "ar"},
    },
]


# ---------------------------------------------------------------------------
# Install / environment helpers
# ---------------------------------------------------------------------------

INSTALL_BASE = r'''
import subprocess, sys

def _pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

_pip("-U", "pip", "setuptools", "wheel")
# Keep protobuf in the 5.x band so Kaggle's preinstalled google-* stack stays happier.
_pip("-U", "huggingface_hub", "accelerate", "sentencepiece", "einops")
_pip("protobuf>=5.26.1,<6")
_pip("-U", "transformers>=4.53.0")
# transformers 4-bit path requires bitsandbytes>=0.46.1
_pip("-U", "bitsandbytes>=0.46.1")
_pip("psutil", "nvidia-ml-py")
# Helpful for some gated / multimodal instruct checkpoints.
_pip("-U", "mistral-common")
'''

BNB_MIN_VERSION = (0, 46, 1)


def install_packages(_model_names: list[str]) -> None:
    print("Installing Kaggle LLM benchmark dependencies...")
    exec(INSTALL_BASE, {"__name__": "__main__"})  # noqa: S102


def _pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])


def ensure_module(module_name: str, *packages: str, upgrade: bool = False, no_deps: bool = False) -> None:
    if importlib.util.find_spec(module_name) is not None:
        return
    cmd: list[str] = []
    if upgrade:
        cmd.append("-U")
    if no_deps:
        cmd.append("--no-deps")
    cmd.extend(packages)
    print(f"Installing missing dependency for {module_name}: {' '.join(packages)}")
    _pip(*cmd)
    if importlib.util.find_spec(module_name) is None:
        raise ModuleNotFoundError(f"No module named {module_name!r} after installing {' '.join(packages)}")


def _parse_version_tuple(raw: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in re.split(r"[^\d]+", str(raw).strip()):
        if chunk.isdigit():
            parts.append(int(chunk))
        if len(parts) >= 3:
            break
    return tuple(parts) if parts else (0,)


def ensure_bitsandbytes(min_version: tuple[int, ...] = BNB_MIN_VERSION) -> None:
    """Install/upgrade bitsandbytes so transformers int4 loading works."""
    need_install = importlib.util.find_spec("bitsandbytes") is None
    if not need_install:
        try:
            import bitsandbytes as bnb  # type: ignore

            ver = _parse_version_tuple(getattr(bnb, "__version__", "0"))
            if ver < min_version:
                need_install = True
                print(
                    f"bitsandbytes {getattr(bnb, '__version__', '?')} is too old; "
                    f"upgrading to >={'.'.join(map(str, min_version))}"
                )
        except Exception as exc:
            need_install = True
            print(f"bitsandbytes import failed ({exc!r}); reinstalling…")
    if need_install:
        print("Installing bitsandbytes>=0.46.1 (required for int4 / 4-bit models)…")
        _pip("-U", "bitsandbytes>=0.46.1")
    try:
        import bitsandbytes as bnb  # type: ignore

        ver = _parse_version_tuple(getattr(bnb, "__version__", "0"))
        if ver < min_version:
            raise ImportError(
                f"bitsandbytes {getattr(bnb, '__version__', '?')} < "
                f"{'.'.join(map(str, min_version))} after install"
            )
    except Exception as exc:
        raise ImportError(
            "Using 4-bit quantization requires bitsandbytes>=0.46.1. "
            "Run with --install, or: pip install -U 'bitsandbytes>=0.46.1'"
        ) from exc


def resolve_hf_token() -> Optional[str]:
    """HF token from env, then Kaggle secrets, then Colab userdata."""
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        val = (os.environ.get(key) or "").strip()
        if val:
            return val
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "huggingface"):
            try:
                val = (UserSecretsClient().get_secret(key) or "").strip()
                if val:
                    os.environ.setdefault("HF_TOKEN", val)
                    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", val)
                    return val
            except Exception:
                continue
    except Exception:
        pass
    try:
        from google.colab import userdata  # type: ignore

        for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            try:
                val = (userdata.get(key) or "").strip()
                if val:
                    os.environ.setdefault("HF_TOKEN", val)
                    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", val)
                    return val
            except Exception:
                continue
    except Exception:
        pass
    return None


def ensure_hf_login() -> Optional[str]:
    """Make the Hub token visible to huggingface_hub / transformers."""
    token = resolve_hf_token()
    if not token:
        return None
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    except Exception as exc:
        # Still pass token= into from_pretrained even if login() is picky.
        print(f"HF login() warning: {exc!r} (will still pass token= to downloads)")
    return token


def gated_auth_skip_reason(model_name: str, token: Optional[str]) -> Optional[str]:
    cfg = MODELS.get(model_name) or {}
    if not cfg.get("gated"):
        return None
    if token:
        return None
    model_id = cfg.get("model_id", model_name)
    return (
        f"Gated model {model_id} requires HF auth. "
        "Accept the license on Hugging Face, then set HF_TOKEN "
        "(Kaggle Secrets / Colab Secrets / env) and re-run."
    )


def gpu_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        return 0


def list_gpus() -> list[dict[str, Any]]:
    info: list[dict[str, Any]] = []
    try:
        import torch

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            info.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_mb": round(total / 1024**2, 1),
                    "free_mb": round(free / 1024**2, 1),
                }
            )
    except Exception as exc:
        info.append({"error": str(exc)})
    return info


def cleanup_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _bytes_to_gb(n: float) -> float:
    return round(n / (1024**3), 2)


def disk_usage(path: Path) -> dict[str, Any]:
    """Return total/used/free GB for the filesystem containing path."""
    try:
        usage = shutil.disk_usage(path)
        return {
            "total_gb": _bytes_to_gb(usage.total),
            "used_gb": _bytes_to_gb(usage.used),
            "free_gb": _bytes_to_gb(usage.free),
        }
    except Exception as exc:
        return {"error": str(exc), "total_gb": 0.0, "used_gb": 0.0, "free_gb": 0.0}


def dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


def purge_model_caches(*, keep_hub_token_cache: bool = True) -> dict[str, Any]:
    """Delete downloaded HF checkpoints from scratch so the next model has room.

    Kaggle temp disk is shared across models; leaving 4B+8B+14B weights around
    before fetching a 30B MoE is what triggers mid-download SIGBUS / Errno 5.
    """
    targets = [
        CACHE_DIR / "hf" / "hub",
        CACHE_DIR / "hf" / "transformers",
        CACHE_DIR / "hf" / "xet",
        Path(os.environ.get("HF_HOME", CACHE_DIR / "hf")) / "hub",
        Path(os.environ.get("HUGGINGFACE_HUB_CACHE", CACHE_DIR / "hf")),
        Path(os.environ.get("TRANSFORMERS_CACHE", CACHE_DIR / "hf")),
    ]
    # Deduplicate resolved paths.
    seen: set[Path] = set()
    removed_gb = 0.0
    removed_paths: list[str] = []
    for raw in targets:
        try:
            path = raw.resolve()
        except Exception:
            path = raw
        if path in seen or not path.exists():
            continue
        seen.add(path)
        before = dir_size_bytes(path)
        # Prefer deleting model snapshot trees; keep tiny token/auth files if asked.
        if keep_hub_token_cache and path.name == "hub":
            for child in list(path.iterdir()):
                name = child.name
                if name.startswith("models--") or name.startswith("datasets--") or name == ".locks":
                    try:
                        shutil.rmtree(child, ignore_errors=True)
                    except Exception:
                        pass
                elif child.is_file() and name.endswith(".incomplete"):
                    try:
                        child.unlink()
                    except Exception:
                        pass
        else:
            shutil.rmtree(path, ignore_errors=True)
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        after = dir_size_bytes(path) if path.exists() else 0
        freed = max(0, before - after)
        if freed:
            removed_gb += freed / (1024**3)
            removed_paths.append(str(path))
    # Incomplete downloads / tmp shards anywhere under CACHE_DIR.
    if CACHE_DIR.exists():
        for p in CACHE_DIR.rglob("*.incomplete"):
            try:
                p.unlink()
            except Exception:
                pass
    gc.collect()
    return {
        "freed_gb": round(removed_gb, 2),
        "paths": removed_paths,
        "scratch": disk_usage(SCRATCH_DIR),
    }


def model_disk_budget_gb(model_name: str) -> float:
    cfg = MODELS.get(model_name) or {}
    if cfg.get("approx_disk_gb") is not None:
        return float(cfg["approx_disk_gb"])
    params = float(cfg.get("approx_params_b") or 8.0)
    # Full-precision safetensors download ≈ 2 bytes/param (+ ~10% overhead).
    return round(params * 2.2, 1)


def ensure_disk_for_model(model_name: str, *, reserve_gb: float = 3.0) -> Optional[str]:
    """Return a skip reason if scratch disk is too small for this checkpoint."""
    need = model_disk_budget_gb(model_name) + reserve_gb
    usage = disk_usage(SCRATCH_DIR)
    free = float(usage.get("free_gb") or 0.0)
    if "error" in usage:
        return None  # can't measure; try anyway
    if free < need:
        return (
            f"Insufficient scratch disk for {model_name}: need ~{need:.1f} GB free "
            f"(model≈{model_disk_budget_gb(model_name):.1f} + reserve), have {free:.1f} GB at {SCRATCH_DIR}. "
            f"Caches were purged; skip or free space / use --only with smaller models."
        )
    return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "item"


# ---------------------------------------------------------------------------
# Resource monitoring (CPU / RAM / GPU sampled in a background thread)
# ---------------------------------------------------------------------------

class ResourceMonitor:
    """Samples process-tree CPU%, RSS RAM, and GPU util/VRAM while a model runs."""

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.samples: list[dict[str, float]] = []
        self._psutil = None
        self._proc = None
        self._nvml = None
        self._gpu_handle = None
        self.gpu_name = ""
        self.baseline_vram_mb = 0.0

        try:
            import psutil

            self._psutil = psutil
            self._proc = psutil.Process(os.getpid())
            self._proc.cpu_percent(None)
        except Exception:
            pass

        try:
            import pynvml

            pynvml.nvmlInit()
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            idx = 0
            if visible:
                try:
                    idx = int(visible.split(",")[0])
                except ValueError:
                    idx = 0
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            self._nvml = pynvml
            name = pynvml.nvmlDeviceGetName(self._gpu_handle)
            self.gpu_name = name.decode() if isinstance(name, bytes) else str(name)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
            self.baseline_vram_mb = mem.used / 1024**2
        except Exception:
            self._nvml = None

    def _tree_cpu_ram(self) -> tuple[float, float]:
        if self._psutil is None or self._proc is None:
            return 0.0, 0.0
        cpu = 0.0
        rss = 0.0
        procs = [self._proc]
        try:
            procs += self._proc.children(recursive=True)
        except Exception:
            pass
        for p in procs:
            try:
                cpu += p.cpu_percent(None)
                rss += p.memory_info().rss
            except Exception:
                continue
        return cpu, rss / 1024**2

    def _sample_once(self) -> None:
        entry: dict[str, float] = {"t": time.time()}
        cpu, ram_mb = self._tree_cpu_ram()
        entry["cpu_percent"] = cpu
        entry["ram_mb"] = ram_mb
        if self._psutil is not None:
            try:
                entry["sys_ram_used_mb"] = self._psutil.virtual_memory().used / 1024**2
                entry["sys_ram_percent"] = self._psutil.virtual_memory().percent
            except Exception:
                pass
        if self._nvml is not None and self._gpu_handle is not None:
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                mem = self._nvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                entry["gpu_util_percent"] = float(util.gpu)
                entry["vram_mb"] = mem.used / 1024**2
            except Exception:
                pass
        with self._lock:
            self.samples.append(entry)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self._sample_once()
            except Exception:
                pass

    def start(self) -> "ResourceMonitor":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._sample_once()
        with self._lock:
            samples = list(self.samples)

        def agg(key: str) -> tuple[float, float]:
            vals = [s[key] for s in samples if key in s]
            if not vals:
                return 0.0, 0.0
            return max(vals), sum(vals) / len(vals)

        peak_cpu, avg_cpu = agg("cpu_percent")
        peak_ram, avg_ram = agg("ram_mb")
        peak_gpu, avg_gpu = agg("gpu_util_percent")
        peak_vram, avg_vram = agg("vram_mb")
        peak_sys_ram, avg_sys_ram = agg("sys_ram_used_mb")
        peak_sys_pct, avg_sys_pct = agg("sys_ram_percent")
        stats = {
            "samples": len(samples),
            "peak_cpu_percent": round(peak_cpu, 1),
            "avg_cpu_percent": round(avg_cpu, 1),
            "peak_ram_mb": round(peak_ram, 1),
            "avg_ram_mb": round(avg_ram, 1),
            "peak_sys_ram_mb": round(peak_sys_ram, 1),
            "avg_sys_ram_mb": round(avg_sys_ram, 1),
            "peak_sys_ram_percent": round(peak_sys_pct, 1),
            "avg_sys_ram_percent": round(avg_sys_pct, 1),
            "peak_gpu_util_percent": round(peak_gpu, 1),
            "avg_gpu_util_percent": round(avg_gpu, 1),
            "peak_vram_mb": round(peak_vram, 1),
            "avg_vram_mb": round(avg_vram, 1),
            "baseline_vram_mb": round(self.baseline_vram_mb, 1),
            "model_vram_mb": round(max(0.0, peak_vram - self.baseline_vram_mb), 1),
            "gpu_name": self.gpu_name,
        }
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        return stats


# ---------------------------------------------------------------------------
# Voice-robot dataset loading
# ---------------------------------------------------------------------------

def _import_voice_robot_dataset() -> Any:
    """Load arabic_voice_robot_llm_dataset from script dir / work dir / sys.path."""
    candidates = []
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent)
    candidates.extend(
        [
            WORK_DIR,
            Path.cwd(),
            Path("/kaggle/working"),
            Path("/content/kaggle_working"),
        ]
    )
    for parent in candidates:
        mod_path = parent / "arabic_voice_robot_llm_dataset.py"
        if not mod_path.exists():
            continue
        spec = importlib.util.spec_from_file_location("arabic_voice_robot_llm_dataset", mod_path)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    try:
        import arabic_voice_robot_llm_dataset as mod  # type: ignore

        return mod
    except Exception:
        return None


def build_dataset_payload(suite: str = DEFAULT_SUITE) -> dict[str, Any]:
    mod = _import_voice_robot_dataset()
    if mod is not None and hasattr(mod, "build_prompts_payload"):
        return mod.build_prompts_payload(suite)
    print("WARNING: arabic_voice_robot_llm_dataset.py not found — using tiny fallback prompts.")
    return {
        "system": DEFAULT_SYSTEM,
        "tool_definitions": [],
        "tool_instruction": "",
        "suite": suite,
        "source": "fallback",
        "prompts": list(BUILTIN_PROMPTS),
    }


def ensure_default_prompts_file(suite: str = DEFAULT_SUITE) -> Path:
    """Always refresh prompts.json from the voice-robot dataset (suite-filtered)."""
    payload = build_dataset_payload(suite)
    write_json(DEFAULT_PROMPTS_PATH, payload)
    return DEFAULT_PROMPTS_PATH


def load_prompts(path: Optional[Path] = None) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    prompt_path = path or ensure_default_prompts_file()
    meta: dict[str, Any] = {"tool_instruction": "", "suite": DEFAULT_SUITE, "source": "file"}
    if prompt_path.exists():
        data = json.loads(prompt_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return DEFAULT_SYSTEM, data, meta
        system = str(data.get("system") or DEFAULT_SYSTEM)
        prompts = list(data.get("prompts") or BUILTIN_PROMPTS)
        meta["tool_instruction"] = str(data.get("tool_instruction") or "")
        meta["suite"] = data.get("suite") or DEFAULT_SUITE
        meta["source"] = data.get("source") or "file"
        meta["tool_definitions"] = data.get("tool_definitions") or []
        return system, prompts, meta
    return DEFAULT_SYSTEM, list(BUILTIN_PROMPTS), meta


# ---------------------------------------------------------------------------
# Auto evaluation (dataset §4 / §8 heuristics)
# ---------------------------------------------------------------------------

_AR_CHARS = re.compile(r"[\u0600-\u06FF]")
_LATIN_WORDS = re.compile(r"[A-Za-z]{2,}")


def _sentence_count(text: str) -> int:
    parts = [p for p in re.split(r"[.!?؟\n]+", text.strip()) if p.strip()]
    return max(1, len(parts)) if text.strip() else 0


def _extract_json_object(text: str) -> Optional[Any]:
    raw = text.strip()
    if not raw:
        return None
    # Strip accidental markdown fences
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except Exception:
            return None
    return None


def _parse_tool_call(text: str) -> Optional[dict[str, Any]]:
    obj = _extract_json_object(text)
    if isinstance(obj, dict) and ("tool" in obj or "name" in obj) and (
        "arguments" in obj or "parameters" in obj or "args" in obj
    ):
        return {
            "tool": obj.get("tool") or obj.get("name"),
            "arguments": obj.get("arguments") or obj.get("parameters") or obj.get("args") or {},
        }
    # Loose pattern: TOOL_CALL name / ARGUMENTS {...}
    m = re.search(
        r'(?:TOOL_CALL|tool)\s*[:=]?\s*["\']?([A-Za-z0-9_]+)["\']?.*?'
        r'(?:ARGUMENTS|arguments)\s*[:=]?\s*(\{[\s\S]*\})',
        text,
        flags=re.IGNORECASE,
    )
    if m:
        try:
            return {"tool": m.group(1), "arguments": json.loads(m.group(2))}
        except Exception:
            return {"tool": m.group(1), "arguments": {}}
    return None


def evaluate_response(case: dict[str, Any], response_text: str) -> dict[str, Any]:
    """Heuristic pass/fail + coarse 1–5 scores from dataset checks."""
    text = (response_text or "").strip()
    checks = dict(case.get("checks") or {})
    failures: list[str] = []
    notes: list[str] = []

    ar_chars = len(_AR_CHARS.findall(text))
    latin_words = len(_LATIN_WORDS.findall(text))
    expect_lang = checks.get("expect_lang")
    if expect_lang == "ar" and latin_words > ar_chars and ar_chars < 8:
        failures.append("expected Arabic-dominant reply")
    if expect_lang == "en" and ar_chars > 20 and latin_words < 5:
        failures.append("expected English reply")

    for needle in checks.get("must_contain") or []:
        if needle and needle not in text:
            failures.append(f"missing required text: {needle!r}")
    any_needles = checks.get("must_contain_any") or []
    if any_needles and not any(n and n in text for n in any_needles):
        failures.append(f"missing any of: {any_needles!r}")
    for needle in checks.get("must_not_contain") or []:
        if needle and needle in text:
            failures.append(f"forbidden text present: {needle!r}")

    max_sent = checks.get("max_sentences")
    sent_n = _sentence_count(text)
    if isinstance(max_sent, int) and max_sent > 0 and sent_n > max_sent * 2:
        # Dataset §8: fail when more than twice the requested length
        failures.append(f"too many sentences ({sent_n} > 2×{max_sent})")
    elif isinstance(max_sent, int) and max_sent > 0 and sent_n > max_sent:
        notes.append(f"slightly over sentence budget ({sent_n}>{max_sent})")

    if checks.get("require_question"):
        if "?" not in text and "؟" not in text and not re.search(r"\b(إيه|فين|امتى|مين|هل|what|which|when|who)\b", text, re.I):
            failures.append("expected a clarification question")

    tool = _parse_tool_call(text)
    expect_tool = checks.get("expect_tool")
    if expect_tool:
        if not tool or str(tool.get("tool")) != str(expect_tool):
            failures.append(f"expected tool {expect_tool!r}")
        else:
            for k, v in (checks.get("expect_tool_args") or {}).items():
                got = (tool.get("arguments") or {}).get(k)
                if got != v and str(got) != str(v):
                    # soft numeric compare
                    try:
                        if float(got) != float(v):
                            failures.append(f"tool arg {k}={got!r} != {v!r}")
                    except Exception:
                        failures.append(f"tool arg {k}={got!r} != {v!r}")
    if checks.get("forbid_tool") and tool:
        failures.append(f"unexpected tool call: {tool.get('tool')}")

    valid_json = None
    parsed = None
    if checks.get("require_json") or checks.get("expected_json") is not None:
        parsed = _extract_json_object(text)
        valid_json = isinstance(parsed, dict)
        if not valid_json:
            failures.append("invalid / missing JSON")
        else:
            expected = checks.get("expected_json") or {}
            for k, v in expected.items():
                if k not in parsed:
                    failures.append(f"JSON missing key {k!r}")
                elif parsed.get(k) != v and str(parsed.get(k)) != str(v):
                    failures.append(f"JSON {k}={parsed.get(k)!r} != {v!r}")
            if "```" in text:
                notes.append("JSON wrapped in markdown fences")

    # Coarse scores (1–5); tool_accuracy 0/1/null
    conciseness = 5
    if isinstance(max_sent, int) and max_sent > 0:
        if sent_n <= max_sent:
            conciseness = 5
        elif sent_n <= max_sent * 2:
            conciseness = 3
        else:
            conciseness = 1
    elif len(text) > 600:
        conciseness = 2
    elif len(text) > 350:
        conciseness = 3

    tts = 5
    if re.search(r"https?://|www\.|^\s*[-*]\s|^\s*\d+\.\s", text, re.M):
        tts = 2
    if "|" in text and "---" in text:
        tts = 1
    if len(text) > 500:
        tts = min(tts, 2)

    language = 5 if expect_lang != "en" else (5 if ar_chars < 10 else 2)
    if expect_lang == "ar" and ar_chars < 5:
        language = 2

    instruction = 5 if not failures else (2 if len(failures) == 1 else 1)
    correctness = 5 if not failures else (3 if notes and len(failures) <= 1 else 1)
    if notes and not failures:
        instruction = 4
        correctness = 4

    tool_accuracy: Any = None
    if expect_tool or checks.get("forbid_tool"):
        tool_accuracy = 1.0 if not any("tool" in f.lower() for f in failures) else 0.0
        # refine: if only tool failures
        if expect_tool:
            tool_accuracy = 1.0 if tool and str(tool.get("tool")) == str(expect_tool) and not any(
                f.startswith("tool arg") or f.startswith("expected tool") for f in failures
            ) else 0.0
        elif checks.get("forbid_tool"):
            tool_accuracy = 0.0 if tool else 1.0

    if tool_accuracy is None:
        overall = (
            correctness * 0.35
            + language * 0.18
            + instruction * 0.18
            + tts * 0.18
            + conciseness * 0.11
        )
    else:
        overall = (
            correctness * 0.30
            + language * 0.15
            + instruction * 0.15
            + tts * 0.15
            + float(tool_accuracy) * 5 * 0.15
            + conciseness * 0.10
        )

    passed = len(failures) == 0
    return {
        "auto_pass": passed,
        "auto_failures": failures,
        "auto_notes": notes,
        "valid_json": valid_json,
        "parsed_tool": tool,
        "language_score": language,
        "correctness_score": correctness,
        "instruction_following_score": instruction,
        "conciseness_score": conciseness,
        "tts_suitability_score": tts,
        "tool_accuracy": tool_accuracy,
        "overall_score": round(overall, 3),
        "sentence_count": sent_n,
    }


def _build_messages_from_case(
    system: str,
    case: dict[str, Any],
    *,
    tool_instruction: str = "",
) -> list[dict[str, str]]:
    """Build chat messages including multi-turn history, context, and tools."""
    messages: list[dict[str, str]] = []
    sys_parts = [(case.get("system") or system or "").strip()]
    ctx = str(case.get("context") or "").strip()
    if ctx:
        sys_parts.append("Context for this conversation:\n" + ctx)
    if case.get("current_datetime") or case.get("timezone"):
        sys_parts.append(
            "Current datetime: "
            + str(case.get("current_datetime") or "")
            + "\nTimezone: "
            + str(case.get("timezone") or "")
        )
    cat = str(case.get("category") or "")
    if cat == "tool_calling" and tool_instruction:
        sys_parts.append(tool_instruction.strip())
    elif cat == "structured_output":
        sys_parts.append("When JSON is requested, reply with JSON only — no markdown fences, no prose.")
    sys_text = "\n\n".join(p for p in sys_parts if p)
    if sys_text:
        messages.append({"role": "system", "content": sys_text})

    turns = list(case.get("turns") or [])
    if not turns:
        user = str(case.get("user") or "").strip()
        if user:
            messages.append({"role": "user", "content": user})
        return messages

    for turn in turns:
        role = str(turn.get("role") or "user").lower()
        content = str(turn.get("content") or "")
        if role == "tool":
            # Many chat templates lack a tool role; fold into user.
            messages.append({"role": "user", "content": f"[tool_result]\n{content}"})
        elif role in {"system", "user", "assistant"}:
            messages.append({"role": role, "content": content})
        else:
            messages.append({"role": "user", "content": content})
    return messages


# ---------------------------------------------------------------------------
# Model loading + generation
# ---------------------------------------------------------------------------

def _resolve_load_mode(cfg: dict[str, Any], n_gpu: int) -> str:
    mode = str(cfg.get("load_mode", "auto")).lower()
    if mode in {"bf16", "fp16", "int4", "4bit"}:
        return "int4" if mode in {"int4", "4bit"} else "bf16"
    # auto heuristic for a single 16 GB T4
    params = float(cfg.get("approx_params_b") or 8.0)
    if n_gpu >= 2 and params <= 14:
        return "bf16"
    if params > 10:
        return "int4"
    return "bf16"


def _load_model_and_tokenizer(model_name: str) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cfg = MODELS[model_name]
    model_id = cfg["model_id"]
    n_gpu = gpu_count()
    load_mode = _resolve_load_mode(cfg, n_gpu)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    hf_token = ensure_hf_login()

    load_info: dict[str, Any] = {
        "model_id": model_id,
        "load_mode": load_mode,
        "dtype": str(dtype).replace("torch.", ""),
        "device_map": "auto" if n_gpu > 0 else None,
        "n_gpu": n_gpu,
        "hf_auth": bool(hf_token),
    }

    if load_mode == "int4" and n_gpu > 0:
        ensure_bitsandbytes()

    t0 = time.perf_counter()
    tok_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if hf_token:
        tok_kwargs["token"] = hf_token
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto" if n_gpu > 0 else None,
        "low_cpu_mem_usage": True,
    }
    if hf_token:
        kwargs["token"] = hf_token

    if load_mode == "int4" and n_gpu > 0:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        # Prefer dtype=; fall back handled below if an older transformers rejects it.
        kwargs["dtype"] = dtype

    # Keep weights on the visible GPU(s). Silent CPU offload makes tok/s look like ~1.
    if n_gpu > 0:
        max_memory: dict[Any, str] = {}
        for i in range(n_gpu):
            # Leave a little headroom for activations / KV on a 15–16 GB T4.
            max_memory[i] = os.environ.get("LLM_MAX_GPU_MEM", "14GiB")
        max_memory["cpu"] = os.environ.get("LLM_MAX_CPU_MEM", "8GiB")
        kwargs["max_memory"] = max_memory

    def _from_pretrained(cls: Any) -> Any:
        try:
            return cls.from_pretrained(model_id, **kwargs)
        except TypeError as exc:
            if "dtype" in kwargs and "torch_dtype" not in kwargs:
                kwargs.pop("dtype", None)
                kwargs["torch_dtype"] = dtype
                return cls.from_pretrained(model_id, **kwargs)
            raise exc

    try:
        model = _from_pretrained(AutoModelForCausalLM)
    except Exception as first_exc:
        # Gemma 3 multimodal instruct checkpoints sometimes need this class.
        if "gemma-3" in model_id.lower() or "nile-chat" in model_id.lower():
            try:
                from transformers import Gemma3ForConditionalGeneration

                model = _from_pretrained(Gemma3ForConditionalGeneration)
            except Exception:
                raise first_exc from None
        else:
            raise

    if n_gpu <= 0:
        model = model.to("cpu")

    model.eval()
    load_info["load_seconds"] = round(time.perf_counter() - t0, 3)
    load_info["max_memory"] = kwargs.get("max_memory")
    return model, tokenizer, load_info


def _build_messages(system: str, user: str, prompt_system: Optional[str] = None) -> list[dict[str, str]]:
    sys_text = (prompt_system or system or "").strip()
    messages: list[dict[str, str]] = []
    if sys_text:
        messages.append({"role": "system", "content": sys_text})
    messages.append({"role": "user", "content": user})
    return messages


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], enable_thinking: Optional[bool]) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    # Qwen3 supports enable_thinking; other tokenizers ignore unknown kwargs via try/except.
    if enable_thinking is not None:
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
        except TypeError:
            pass
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        # Fallback for rare tokenizers without chat templates.
        parts = []
        for m in messages:
            parts.append(f"{m['role'].upper()}: {m['content']}")
        parts.append("ASSISTANT:")
        return "\n".join(parts)


def generate_one(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    enable_thinking: Optional[bool],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    import torch
    from transformers import TextIteratorStreamer

    prompt_text = _apply_chat_template(tokenizer, messages, enable_thinking)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    if hasattr(model, "device"):
        device = model.device
    else:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_tokens = int(inputs["input_ids"].shape[-1])

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    do_sample = temperature is not None and float(temperature) > 0
    gen_kwargs: dict[str, Any] = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = max(float(temperature), 1e-5)
        gen_kwargs["top_p"] = top_p

    output_chunks: list[str] = []
    first_token_seconds: Optional[float] = None
    t_gen0 = time.perf_counter()

    def _run_generate() -> None:
        with torch.inference_mode():
            model.generate(**gen_kwargs)

    thread = threading.Thread(target=_run_generate, daemon=True)
    thread.start()
    for chunk in streamer:
        if first_token_seconds is None:
            first_token_seconds = time.perf_counter() - t_gen0
        output_chunks.append(chunk)
    thread.join()
    generate_seconds = time.perf_counter() - t_gen0

    text = "".join(output_chunks).strip()
    # Strip residual Qwen thinking blocks if any leaked through.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    completion_ids = tokenizer.encode(text, add_special_tokens=False) if text else []
    completion_tokens = len(completion_ids)
    tok_s = (completion_tokens / generate_seconds) if generate_seconds > 0 else 0.0

    return {
        "prompt_text": prompt_text,
        "response_text": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "response_chars": len(text),
        "first_token_seconds": round(first_token_seconds or generate_seconds, 4),
        "generate_seconds": round(generate_seconds, 4),
        "tokens_per_second": round(tok_s, 3),
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }


def run_model_prompts(
    model_name: str,
    system: str,
    prompts: list[dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    *,
    tool_instruction: str = "",
) -> dict[str, Any]:
    cfg = MODELS[model_name]
    model, tokenizer, load_info = _load_model_and_tokenizer(model_name)
    enable_thinking = cfg.get("enable_thinking")
    runs: list[dict[str, Any]] = []

    # Warmup tiny generate so first measured prompt is less cold-cache biased.
    try:
        warm = _build_messages(system, "قل مرحباً فقط.")
        _ = generate_one(
            model,
            tokenizer,
            warm,
            enable_thinking=enable_thinking if enable_thinking is not None else False,
            max_new_tokens=8,
            temperature=0.0,
            top_p=1.0,
        )
    except Exception as exc:
        print(f"[worker] warmup skipped: {exc!r}")

    for prompt in prompts:
        pid = str(prompt.get("id") or safe_name(str(prompt.get("user", "prompt"))[:40]))
        category = str(prompt.get("category") or "general")
        priority = str(prompt.get("priority") or "")
        user = str(prompt.get("user") or "").strip()
        if not user:
            turns = prompt.get("turns") or []
            for t in reversed(turns):
                if t.get("role") == "user":
                    user = str(t.get("content") or "")
                    break
        messages = _build_messages_from_case(system, prompt, tool_instruction=tool_instruction)
        print(f"[worker] generate model={model_name} prompt={pid} cat={category}")
        try:
            gen = generate_one(
                model,
                tokenizer,
                messages,
                enable_thinking=enable_thinking,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            eval_result = evaluate_response(prompt, gen.get("response_text") or "")
            out_path = RESPONSE_DIR / safe_name(model_name) / f"{safe_name(pid)}.txt"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(gen["response_text"] + "\n", encoding="utf-8")
            # Side-car JSON with eval for human review
            eval_path = RESPONSE_DIR / safe_name(model_name) / f"{safe_name(pid)}.eval.json"
            write_json(
                eval_path,
                {
                    "prompt_id": pid,
                    "category": category,
                    "priority": priority,
                    "expected_behavior": prompt.get("expected_behavior") or [],
                    "forbidden_behavior": prompt.get("forbidden_behavior") or [],
                    "response_text": gen.get("response_text"),
                    **eval_result,
                },
            )
            runs.append(
                {
                    "status": "ok",
                    "model": model_name,
                    "model_id": cfg["model_id"],
                    "role": cfg.get("role", ""),
                    "prompt_id": pid,
                    "category": category,
                    "priority": priority,
                    "user": user,
                    "response_file": str(out_path),
                    "eval_file": str(eval_path),
                    "load_seconds": load_info["load_seconds"],
                    "load_mode": load_info["load_mode"],
                    "dtype": load_info["dtype"],
                    "device": "cuda" if gpu_count() > 0 else "cpu",
                    **gen,
                    **eval_result,
                }
            )
            mark = "PASS" if eval_result.get("auto_pass") else "FAIL"
            print(
                f"[worker] {mark} {pid} overall={eval_result.get('overall_score')} "
                f"ttft={gen.get('first_token_seconds')}s tok/s={gen.get('tokens_per_second')}"
            )
        except Exception as exc:
            runs.append(
                {
                    "status": "error",
                    "model": model_name,
                    "model_id": cfg["model_id"],
                    "prompt_id": pid,
                    "category": category,
                    "priority": priority,
                    "user": user,
                    "error": repr(exc),
                    "traceback": traceback.format_exc()[-4000:],
                    "load_seconds": load_info.get("load_seconds"),
                    "load_mode": load_info.get("load_mode"),
                    "auto_pass": False,
                    "overall_score": 0,
                }
            )
            print(f"[worker] FAIL prompt={pid}: {exc!r}")

    del model
    cleanup_gpu()
    ok_runs = [r for r in runs if r.get("status") == "ok"]
    pass_n = sum(1 for r in ok_runs if r.get("auto_pass"))
    return {
        "status": "ok" if ok_runs else "error",
        "model": model_name,
        "model_id": cfg["model_id"],
        "role": cfg.get("role", ""),
        "load": load_info,
        "runs": runs,
        "quality": {
            "prompts_ok": len(ok_runs),
            "auto_pass": pass_n,
            "auto_pass_rate": round(pass_n / len(ok_runs), 3) if ok_runs else 0.0,
            "avg_overall_score": round(
                sum(float(r.get("overall_score") or 0) for r in ok_runs) / len(ok_runs), 3
            )
            if ok_runs
            else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# Worker + orchestration
# ---------------------------------------------------------------------------

def worker_main(
    model_name: str,
    prompts_file: Path,
    meta_file: Path,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> int:
    print(f"[worker] model={model_name} cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    for mod, pkg in (("psutil", "psutil"), ("pynvml", "nvidia-ml-py")):
        try:
            ensure_module(mod, pkg)
        except Exception:
            pass

    # Resolve HF token inside the worker too (Colab/Kaggle secrets are process-local).
    token = ensure_hf_login()
    skip_auth = gated_auth_skip_reason(model_name, token)
    if skip_auth:
        write_json(
            meta_file,
            {
                "status": "error",
                "model": model_name,
                "model_id": MODELS.get(model_name, {}).get("model_id", ""),
                "error": skip_auth,
                "runs": [],
                "skipped": True,
            },
        )
        print(f"[worker] SKIP {model_name}: {skip_auth}")
        return 1

    cfg = MODELS.get(model_name) or {}
    if str(cfg.get("load_mode", "")).lower() in {"int4", "4bit"}:
        try:
            ensure_bitsandbytes()
        except Exception as exc:
            write_json(
                meta_file,
                {
                    "status": "error",
                    "model": model_name,
                    "model_id": cfg.get("model_id", ""),
                    "error": repr(exc),
                    "runs": [],
                },
            )
            print(f"[worker] FAIL {model_name}: {exc!r}")
            return 1

    monitor = ResourceMonitor().start()
    try:
        system, prompts, prompt_meta = load_prompts(prompts_file)
        payload = run_model_prompts(
            model_name,
            system,
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            tool_instruction=str(prompt_meta.get("tool_instruction") or ""),
        )
        payload["resources"] = monitor.stop()
        write_json(meta_file, payload)
        print(f"[worker] OK {model_name} prompts={len(payload.get('runs', []))}")
        return 0 if payload.get("status") == "ok" else 1
    except Exception as exc:
        payload = {
            "status": "error",
            "model": model_name,
            "model_id": MODELS.get(model_name, {}).get("model_id", ""),
            "error": repr(exc),
            "traceback": traceback.format_exc()[-6000:],
            "resources": monitor.stop(),
            "runs": [],
        }
        write_json(meta_file, payload)
        print(f"[worker] FAIL {model_name}: {payload['error']}")
        traceback.print_exc()
        return 1
    finally:
        cleanup_gpu()


def _notebook_cell_source() -> Optional[str]:
    try:
        from IPython import get_ipython

        ip = get_ipython()
        hist = getattr(ip, "history_manager", None)
        if not hist:
            return None
        cells = [entry[2] for entry in hist.get_range()]
        for cell in reversed(list(cells)):
            if isinstance(cell, str) and "def run_model_isolated" in cell and "def worker_main" in cell:
                return cell
    except Exception:
        return None
    return None


def _this_script() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve()
    src = _notebook_cell_source()
    if src:
        dest = WORK_DIR / "_kaggle_arabic_llm_compare_worker.py"
        dest.write_text(src, encoding="utf-8")
        return dest.resolve()
    raise RuntimeError(
        "Cannot resolve script path for worker subprocess. Upload "
        "kaggle_arabic_llm_compare.py to /kaggle/working/ and run it with %run."
    )


def run_model_isolated(
    model_name: str,
    prompts_file: Path,
    gpu_index: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    model_key = safe_name(model_name)
    meta_path = META_DIR / f"{model_key}.json"
    if meta_path.exists():
        meta_path.unlink()

    # Free previous checkpoints before download; skip if still too tight.
    purge_info = purge_model_caches()
    print(
        f"Scratch free: {purge_info.get('scratch', {}).get('free_gb', '?')} GB "
        f"(purged ~{purge_info.get('freed_gb', 0)} GB of old HF caches)"
    )
    skip_reason = ensure_disk_for_model(model_name)
    if skip_reason:
        print(f"SKIP {model_name}: {skip_reason}")
        return {
            "status": "error",
            "model": model_name,
            "model_id": MODELS.get(model_name, {}).get("model_id", ""),
            "error": skip_reason,
            "runs": [],
            "skipped": True,
            "disk": purge_info.get("scratch"),
        }

    hf_token = resolve_hf_token()
    auth_skip = gated_auth_skip_reason(model_name, hf_token)
    if auth_skip:
        print(f"SKIP {model_name}: {auth_skip}")
        return {
            "status": "error",
            "model": model_name,
            "model_id": MODELS.get(model_name, {}).get("model_id", ""),
            "error": auth_skip,
            "runs": [],
            "skipped": True,
            "disk": purge_info.get("scratch"),
        }

    env = os.environ.copy()
    if gpu_index >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env.setdefault("HF_HOME", str(CACHE_DIR / "hf"))
    env.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR / "hf"))
    env.setdefault("HUGGINGFACE_HUB_CACHE", str(CACHE_DIR / "hf" / "hub"))
    env.setdefault("HF_HUB_DISABLE_XET", "1")  # fewer temp-disk surprises on Kaggle
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token

    cmd = [
        sys.executable,
        str(_this_script()),
        "--worker",
        "--model",
        model_name,
        "--prompts-file",
        str(prompts_file),
        "--meta",
        str(meta_path),
        "--max-new-tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
    ]

    print(f"\n{'=' * 72}\nLLM {model_name} | GPU {gpu_index if gpu_index >= 0 else 'CPU'}\n{'=' * 72}")
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, env=env)
        returncode = proc.returncode
        spawn_error = None
    except OSError as exc:
        # After a disk-full SIGBUS, Kaggle sometimes cannot exec python (Errno 5).
        returncode = -1
        spawn_error = repr(exc)
        print(f"FAIL spawning worker for {model_name}: {spawn_error}")
        # Best-effort recovery so later models can still run.
        try:
            purge_model_caches()
        except Exception:
            pass
        cleanup_gpu()
    wall_s = time.perf_counter() - started

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        signal_hint = ""
        if isinstance(returncode, int) and returncode < 0:
            # subprocess negative codes are -signal on POSIX (e.g. -7 = SIGBUS).
            sig = -returncode
            signal_hint = f" (signal {sig}; often disk-full mmap / SIGBUS on Kaggle temp)"
        err = spawn_error or f"Worker exited {returncode} without writing meta{signal_hint}"
        meta = {"status": "error", "error": err, "runs": []}
    meta["wall_seconds"] = round(wall_s, 3)
    meta["gpu_index"] = gpu_index
    meta["disk_after"] = disk_usage(SCRATCH_DIR)

    # Always reclaim checkpoint disk before the next model.
    try:
        post = purge_model_caches()
        meta["cache_purged_gb"] = post.get("freed_gb")
    except Exception as exc:
        meta["cache_purge_error"] = repr(exc)
    cleanup_gpu()
    return meta


def zip_llm_outputs(output_dir: Path = OUTPUT_DIR) -> Path:
    """Zip the entire llm_outputs tree for easy Kaggle/Colab download."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    zip_base = WORK_DIR / f"llm_outputs_{stamp}"
    # Also write a stable name that overwrites each run.
    stable = WORK_DIR / "llm_outputs"
    archive = Path(
        shutil.make_archive(str(zip_base), "zip", root_dir=str(output_dir.parent), base_dir=output_dir.name)
    )
    stable_zip = Path(str(stable) + ".zip")
    try:
        if stable_zip.exists():
            stable_zip.unlink()
        shutil.copy2(archive, stable_zip)
    except Exception:
        stable_zip = archive
    print(f"Outputs zip: {archive}")
    if stable_zip != archive:
        print(f"Outputs zip (stable): {stable_zip}")
    return archive


def select_models(only: Optional[str], skip: Optional[str]) -> list[str]:
    enabled = dict(ENABLE)
    if only:
        requested = [x.strip() for x in only.split(",") if x.strip()]
        unknown = [x for x in requested if x not in MODEL_ORDER]
        if unknown:
            raise ValueError(f"Unknown model(s): {unknown}. Available: {MODEL_ORDER}")
        return requested
    if skip:
        for name in [x.strip() for x in skip.split(",") if x.strip()]:
            enabled[name] = False
    return [name for name in MODEL_ORDER if enabled.get(name, False)]


# ---------------------------------------------------------------------------
# Analytics reporting (printed tables + CSV/JSON/MD for model selection)
# ---------------------------------------------------------------------------

ANALYTICS_COLUMNS = [
    "model",
    "model_id",
    "prompt_id",
    "category",
    "priority",
    "status",
    "auto_pass",
    "overall_score",
    "language_score",
    "correctness_score",
    "instruction_following_score",
    "conciseness_score",
    "tts_suitability_score",
    "tool_accuracy",
    "valid_json",
    "gpu_index",
    "gpu_name",
    "load_mode",
    "dtype",
    "device",
    "load_seconds",
    "first_token_seconds",
    "generate_seconds",
    "wall_seconds",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "tokens_per_second",
    "response_chars",
    "chars_per_second",
    "peak_cpu_percent",
    "avg_cpu_percent",
    "peak_ram_mb",
    "avg_ram_mb",
    "peak_sys_ram_mb",
    "peak_sys_ram_percent",
    "peak_gpu_util_percent",
    "avg_gpu_util_percent",
    "peak_vram_mb",
    "avg_vram_mb",
    "model_vram_mb",
    "error",
]


def _num(vals: list[Any]) -> list[float]:
    out: list[float] = []
    for v in vals:
        if v is None or v == "":
            continue
        try:
            out.append(float(v))
        except Exception:
            continue
    return out


def _stat_mean(vals: list[float], ndigits: int = 3) -> Any:
    return round(sum(vals) / len(vals), ndigits) if vals else ""


def _stat_std(vals: list[float], ndigits: int = 3) -> Any:
    if len(vals) < 2:
        return 0.0 if vals else ""
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
    return round(var**0.5, ndigits)


def _stat_min(vals: list[float], ndigits: int = 3) -> Any:
    return round(min(vals), ndigits) if vals else ""


def _stat_max(vals: list[float], ndigits: int = 3) -> Any:
    return round(max(vals), ndigits) if vals else ""


def _norm_map(values: dict[str, float], *, lower_is_better: bool = False) -> dict[str, float]:
    if not values:
        return {}
    nums = list(values.values())
    lo, hi = min(nums), max(nums)
    if hi <= lo:
        return {k: 100.0 for k in values}
    out: dict[str, float] = {}
    for k, v in values.items():
        score = (hi - v) / (hi - lo) if lower_is_better else (v - lo) / (hi - lo)
        out[k] = round(100.0 * score, 2)
    return out


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: Optional[list[str]] = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        cols = fieldnames or []
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
        return path
    cols = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def flatten_runs(model_metas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for meta in model_metas:
        res = meta.get("resources", {}) or {}
        load = meta.get("load", {}) or {}
        prompt_runs = meta.get("runs") or []
        if not prompt_runs and meta.get("status") == "error":
            rows.append(
                {
                    "model": meta.get("model", ""),
                    "model_id": meta.get("model_id", ""),
                    "prompt_id": "",
                    "category": "",
                    "priority": "",
                    "status": "error",
                    "auto_pass": False,
                    "overall_score": "",
                    "language_score": "",
                    "correctness_score": "",
                    "instruction_following_score": "",
                    "conciseness_score": "",
                    "tts_suitability_score": "",
                    "tool_accuracy": "",
                    "valid_json": "",
                    "gpu_index": meta.get("gpu_index", ""),
                    "gpu_name": res.get("gpu_name", ""),
                    "load_mode": load.get("load_mode", ""),
                    "dtype": load.get("dtype", ""),
                    "device": "cuda" if (meta.get("gpu_index", -1) or -1) >= 0 else "cpu",
                    "load_seconds": load.get("load_seconds", ""),
                    "first_token_seconds": "",
                    "generate_seconds": "",
                    "wall_seconds": meta.get("wall_seconds", ""),
                    "prompt_tokens": "",
                    "completion_tokens": "",
                    "total_tokens": "",
                    "tokens_per_second": "",
                    "response_chars": "",
                    "chars_per_second": "",
                    "peak_cpu_percent": res.get("peak_cpu_percent", ""),
                    "avg_cpu_percent": res.get("avg_cpu_percent", ""),
                    "peak_ram_mb": res.get("peak_ram_mb", ""),
                    "avg_ram_mb": res.get("avg_ram_mb", ""),
                    "peak_sys_ram_mb": res.get("peak_sys_ram_mb", ""),
                    "peak_sys_ram_percent": res.get("peak_sys_ram_percent", ""),
                    "peak_gpu_util_percent": res.get("peak_gpu_util_percent", ""),
                    "avg_gpu_util_percent": res.get("avg_gpu_util_percent", ""),
                    "peak_vram_mb": res.get("peak_vram_mb", ""),
                    "avg_vram_mb": res.get("avg_vram_mb", ""),
                    "model_vram_mb": res.get("model_vram_mb", ""),
                    "error": meta.get("error", ""),
                }
            )
            continue

        for run in prompt_runs:
            gen_s = run.get("generate_seconds", "")
            resp_chars = run.get("response_chars", "")
            cps = ""
            try:
                if gen_s not in ("", None) and resp_chars not in ("", None) and float(gen_s) > 0:
                    cps = round(float(resp_chars) / float(gen_s), 1)
            except Exception:
                cps = ""
            rows.append(
                {
                    "model": run.get("model") or meta.get("model", ""),
                    "model_id": run.get("model_id") or meta.get("model_id", ""),
                    "prompt_id": run.get("prompt_id", ""),
                    "category": run.get("category", ""),
                    "priority": run.get("priority", ""),
                    "status": run.get("status", ""),
                    "auto_pass": run.get("auto_pass", ""),
                    "overall_score": run.get("overall_score", ""),
                    "language_score": run.get("language_score", ""),
                    "correctness_score": run.get("correctness_score", ""),
                    "instruction_following_score": run.get("instruction_following_score", ""),
                    "conciseness_score": run.get("conciseness_score", ""),
                    "tts_suitability_score": run.get("tts_suitability_score", ""),
                    "tool_accuracy": run.get("tool_accuracy", ""),
                    "valid_json": run.get("valid_json", ""),
                    "gpu_index": meta.get("gpu_index", ""),
                    "gpu_name": res.get("gpu_name", ""),
                    "load_mode": run.get("load_mode") or load.get("load_mode", ""),
                    "dtype": run.get("dtype") or load.get("dtype", ""),
                    "device": run.get("device", ""),
                    "load_seconds": run.get("load_seconds") or load.get("load_seconds", ""),
                    "first_token_seconds": run.get("first_token_seconds", ""),
                    "generate_seconds": gen_s,
                    "wall_seconds": meta.get("wall_seconds", ""),
                    "prompt_tokens": run.get("prompt_tokens", ""),
                    "completion_tokens": run.get("completion_tokens", ""),
                    "total_tokens": run.get("total_tokens", ""),
                    "tokens_per_second": run.get("tokens_per_second", ""),
                    "response_chars": resp_chars,
                    "chars_per_second": cps,
                    "peak_cpu_percent": res.get("peak_cpu_percent", ""),
                    "avg_cpu_percent": res.get("avg_cpu_percent", ""),
                    "peak_ram_mb": res.get("peak_ram_mb", ""),
                    "avg_ram_mb": res.get("avg_ram_mb", ""),
                    "peak_sys_ram_mb": res.get("peak_sys_ram_mb", ""),
                    "peak_sys_ram_percent": res.get("peak_sys_ram_percent", ""),
                    "peak_gpu_util_percent": res.get("peak_gpu_util_percent", ""),
                    "avg_gpu_util_percent": res.get("avg_gpu_util_percent", ""),
                    "peak_vram_mb": res.get("peak_vram_mb", ""),
                    "avg_vram_mb": res.get("avg_vram_mb", ""),
                    "model_vram_mb": res.get("model_vram_mb", ""),
                    "error": run.get("error", ""),
                    "response_text": run.get("response_text", ""),
                    "auto_failures": run.get("auto_failures", []),
                }
            )
    return rows


def aggregate_by_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for r in rows:
        name = str(r.get("model") or "")
        if name not in by_model:
            by_model[name] = []
            order.append(name)
        by_model[name].append(r)

    agg: list[dict[str, Any]] = []
    for model in order:
        items = by_model[model]
        ok = [x for x in items if x.get("status") == "ok"]

        def col(key: str, source: Optional[list[dict[str, Any]]] = None) -> list[float]:
            src = source if source is not None else ok
            return _num([x.get(key) for x in src])

        ttft = col("first_token_seconds")
        tps = col("tokens_per_second")
        gen = col("generate_seconds")
        load = col("load_seconds")
        out_tok = col("completion_tokens")
        in_tok = col("prompt_tokens")
        cps = col("chars_per_second")
        success_rate = round(100.0 * len(ok) / len(items), 1) if items else 0.0
        passed = [x for x in ok if x.get("auto_pass") in (True, "True", "true", 1, "1")]
        auto_pass_rate = round(100.0 * len(passed) / len(ok), 1) if ok else 0.0
        overall = col("overall_score")

        agg.append(
            {
                "model": model,
                "model_id": items[0].get("model_id", ""),
                "load_mode": next((x.get("load_mode") for x in ok if x.get("load_mode")), items[0].get("load_mode", "")),
                "dtype": next((x.get("dtype") for x in ok if x.get("dtype")), items[0].get("dtype", "")),
                "runs": len(items),
                "ok": len(ok),
                "failed": len(items) - len(ok),
                "success_rate_percent": success_rate,
                "auto_pass": len(passed),
                "auto_pass_rate_percent": auto_pass_rate,
                "avg_overall_score": _stat_mean(overall, 3),
                "avg_language_score": _stat_mean(col("language_score"), 2),
                "avg_correctness_score": _stat_mean(col("correctness_score"), 2),
                "avg_instruction_following_score": _stat_mean(col("instruction_following_score"), 2),
                "avg_conciseness_score": _stat_mean(col("conciseness_score"), 2),
                "avg_tts_suitability_score": _stat_mean(col("tts_suitability_score"), 2),
                "avg_tool_accuracy": _stat_mean(col("tool_accuracy"), 3),
                "avg_load_seconds": _stat_mean(load, 2),
                "avg_first_token_seconds": _stat_mean(ttft, 3),
                "min_first_token_seconds": _stat_min(ttft, 3),
                "max_first_token_seconds": _stat_max(ttft, 3),
                "std_first_token_seconds": _stat_std(ttft, 3),
                "avg_generate_seconds": _stat_mean(gen, 3),
                "min_generate_seconds": _stat_min(gen, 3),
                "max_generate_seconds": _stat_max(gen, 3),
                "avg_tokens_per_second": _stat_mean(tps, 2),
                "min_tokens_per_second": _stat_min(tps, 2),
                "max_tokens_per_second": _stat_max(tps, 2),
                "std_tokens_per_second": _stat_std(tps, 2),
                "avg_completion_tokens": _stat_mean(out_tok, 1),
                "avg_prompt_tokens": _stat_mean(in_tok, 1),
                "avg_chars_per_second": _stat_mean(cps, 1),
                "peak_cpu_percent": _stat_max(col("peak_cpu_percent", items), 1),
                "peak_ram_mb": _stat_max(col("peak_ram_mb", items), 1),
                "peak_sys_ram_mb": _stat_max(col("peak_sys_ram_mb", items), 1),
                "peak_sys_ram_percent": _stat_max(col("peak_sys_ram_percent", items), 1),
                "peak_gpu_util_percent": _stat_max(col("peak_gpu_util_percent", items), 1),
                "peak_vram_mb": _stat_max(col("peak_vram_mb", items), 1),
                "model_vram_mb": _stat_max(col("model_vram_mb", items), 1),
            }
        )
    return agg


def aggregate_by_category(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    order: list[tuple[str, str]] = []
    for r in rows:
        key = (str(r.get("model") or ""), str(r.get("category") or "general"))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)

    out: list[dict[str, Any]] = []
    for model, category in order:
        items = groups[(model, category)]
        ok = [x for x in items if x.get("status") == "ok"]
        ttft = _num([x.get("first_token_seconds") for x in ok])
        tps = _num([x.get("tokens_per_second") for x in ok])
        gen = _num([x.get("generate_seconds") for x in ok])
        out.append(
            {
                "model": model,
                "category": category,
                "runs": len(items),
                "ok": len(ok),
                "success_rate_percent": round(100.0 * len(ok) / len(items), 1) if items else 0.0,
                "avg_first_token_seconds": _stat_mean(ttft, 3),
                "avg_generate_seconds": _stat_mean(gen, 3),
                "avg_tokens_per_second": _stat_mean(tps, 2),
                "avg_completion_tokens": _stat_mean(_num([x.get("completion_tokens") for x in ok]), 1),
                "avg_response_chars": _stat_mean(_num([x.get("response_chars") for x in ok]), 1),
            }
        )
    return out


def build_llm_leaderboard(agg_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok = [r for r in agg_rows if int(r.get("ok") or 0) > 0]
    if not ok:
        return []

    ttft_map = {
        str(r["model"]): float(r["avg_first_token_seconds"])
        for r in ok
        if r.get("avg_first_token_seconds") not in ("", None)
    }
    tps_map = {
        str(r["model"]): float(r["avg_tokens_per_second"])
        for r in ok
        if r.get("avg_tokens_per_second") not in ("", None)
    }
    vram_map = {
        str(r["model"]): float(r["peak_vram_mb"])
        for r in ok
        if r.get("peak_vram_mb") not in ("", None)
    }
    load_map = {
        str(r["model"]): float(r["avg_load_seconds"])
        for r in ok
        if r.get("avg_load_seconds") not in ("", None)
    }

    latency_s = _norm_map(ttft_map, lower_is_better=True)
    thr_s = _norm_map(tps_map, lower_is_better=False)
    vram_s = _norm_map(vram_map, lower_is_better=True)
    load_s = _norm_map(load_map, lower_is_better=True)

    board: list[dict[str, Any]] = []
    for r in ok:
        name = str(r["model"])
        lat = latency_s.get(name, 50.0)
        th = thr_s.get(name, 50.0)
        vr = vram_s.get(name, 50.0)
        ld = load_s.get(name, 50.0)
        # Robot: low TTFT matters most for turn-taking; then tok/s and VRAM.
        robot = round(0.45 * lat + 0.30 * th + 0.15 * vr + 0.10 * ld, 2)
        balanced = round(0.30 * lat + 0.35 * th + 0.25 * vr + 0.10 * ld, 2)
        board.append(
            {
                "model": name,
                "model_id": r.get("model_id", ""),
                "load_mode": r.get("load_mode", ""),
                "ok_runs": r.get("ok", ""),
                "success_rate_percent": r.get("success_rate_percent", ""),
                "avg_first_token_seconds": r.get("avg_first_token_seconds", ""),
                "avg_tokens_per_second": r.get("avg_tokens_per_second", ""),
                "avg_generate_seconds": r.get("avg_generate_seconds", ""),
                "avg_load_seconds": r.get("avg_load_seconds", ""),
                "peak_vram_mb": r.get("peak_vram_mb", ""),
                "peak_ram_mb": r.get("peak_ram_mb", ""),
                "model_vram_mb": r.get("model_vram_mb", ""),
                "score_latency": lat,
                "score_throughput": th,
                "score_vram_efficiency": vr,
                "score_load": ld,
                "score_balanced": balanced,
                "score_robot_realtime": robot,
            }
        )
    board.sort(key=lambda x: (-float(x["score_robot_realtime"]), str(x["model"])))
    for i, row in enumerate(board, 1):
        row["rank_robot"] = i
    by_ttft = sorted(
        board,
        key=lambda x: float(x["avg_first_token_seconds"])
        if x.get("avg_first_token_seconds") not in ("", None)
        else 999.0,
    )
    for i, row in enumerate(by_ttft, 1):
        row["rank_ttft"] = i
    by_tps = sorted(
        board,
        key=lambda x: -float(x["avg_tokens_per_second"])
        if x.get("avg_tokens_per_second") not in ("", None)
        else 0.0,
    )
    for i, row in enumerate(by_tps, 1):
        row["rank_throughput"] = i
    return board


def build_llm_recommendations(
    leaderboard: list[dict[str, Any]],
    category_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not leaderboard:
        return {
            "status": "no_successful_models",
            "picks": {},
            "notes": ["No successful LLM runs; cannot recommend a model."],
        }

    def pick(key: str, reverse: bool = False) -> Optional[dict[str, Any]]:
        eligible = [r for r in leaderboard if r.get(key) not in ("", None)]
        if not eligible:
            return None
        return sorted(eligible, key=lambda r: float(r[key]), reverse=reverse)[0]

    best_robot = leaderboard[0]
    best_ttft = pick("avg_first_token_seconds", reverse=False)
    best_tps = pick("avg_tokens_per_second", reverse=True)
    best_vram = pick("peak_vram_mb", reverse=False)
    best_balanced = pick("score_balanced", reverse=True)

    # Category specialists by lowest TTFT among models with OK runs in that category.
    category_picks: dict[str, Any] = {}
    cats = sorted({str(r.get("category") or "") for r in category_rows if r.get("category")})
    for cat in cats:
        cat_ok = [
            r
            for r in category_rows
            if r.get("category") == cat
            and int(r.get("ok") or 0) > 0
            and r.get("avg_first_token_seconds") not in ("", None)
        ]
        if not cat_ok:
            continue
        # Prefer low TTFT, then high tok/s within category.
        cat_ok.sort(
            key=lambda r: (
                float(r["avg_first_token_seconds"]),
                -float(r["avg_tokens_per_second"])
                if r.get("avg_tokens_per_second") not in ("", None)
                else 0.0,
            )
        )
        top = cat_ok[0]
        category_picks[cat] = {
            "model": top.get("model"),
            "avg_first_token_seconds": top.get("avg_first_token_seconds"),
            "avg_tokens_per_second": top.get("avg_tokens_per_second"),
        }

    return {
        "status": "ok",
        "picks": {
            "best_for_robot_realtime": {
                "model": best_robot.get("model"),
                "why": "Best composite of TTFT + tok/s + VRAM + load for conversational turns.",
                "metrics": {
                    "score_robot_realtime": best_robot.get("score_robot_realtime"),
                    "avg_first_token_seconds": best_robot.get("avg_first_token_seconds"),
                    "avg_tokens_per_second": best_robot.get("avg_tokens_per_second"),
                    "peak_vram_mb": best_robot.get("peak_vram_mb"),
                },
            },
            "lowest_ttft": {
                "model": (best_ttft or {}).get("model"),
                "why": "Fastest average time-to-first-token (best turn-taking feel).",
                "metrics": {"avg_first_token_seconds": (best_ttft or {}).get("avg_first_token_seconds")},
            },
            "highest_throughput": {
                "model": (best_tps or {}).get("model"),
                "why": "Highest average tokens/second.",
                "metrics": {"avg_tokens_per_second": (best_tps or {}).get("avg_tokens_per_second")},
            },
            "lowest_vram": {
                "model": (best_vram or {}).get("model"),
                "why": "Lowest peak VRAM — better for small GPUs / co-residency.",
                "metrics": {"peak_vram_mb": (best_vram or {}).get("peak_vram_mb")},
            },
            "best_balanced": {
                "model": (best_balanced or {}).get("model"),
                "why": "Balanced latency/throughput/VRAM tradeoff.",
                "metrics": {"score_balanced": (best_balanced or {}).get("score_balanced")},
            },
        },
        "category_specialists": category_picks,
        "leaderboard_top3": leaderboard[:3],
        "model_count_ok": len(leaderboard),
        "notes": [
            "Automated scores cover latency/throughput/resources only.",
            "Manually review Egyptian / MSA / code-switch response quality in llm_outputs/responses/.",
            "For robot UX, prioritize low TTFT even if peak tok/s is slightly lower.",
            "Disable thinking modes where available to reduce TTFT further.",
        ],
    }


def write_llm_selection_report(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    agg_rows: list[dict[str, Any]],
    category_rows: list[dict[str, Any]],
    leaderboard: list[dict[str, Any]],
    recommendations: dict[str, Any],
) -> Path:
    ok = [r for r in rows if r.get("status") == "ok"]
    bad = [r for r in rows if r.get("status") == "error"]
    lines = [
        "# LLM Model Selection Report",
        "",
        "Auto-generated from the Kaggle Arabic LLM bake-off.",
        "Combine latency/resource rankings with manual review of Arabic response quality.",
        "",
        "## Run summary",
        "",
        f"- Total prompt runs: **{len(rows)}**",
        f"- OK: **{len(ok)}** | Failed: **{len(bad)}**",
        f"- Models with ≥1 OK run: **{len(leaderboard)}**",
        "",
        "## Recommended picks",
        "",
    ]
    for key, payload in (recommendations.get("picks") or {}).items():
        if not isinstance(payload, dict):
            continue
        model = payload.get("model") or "(none)"
        lines.append(f"### `{key}`")
        lines.append("")
        lines.append(f"- **Model:** `{model}`")
        lines.append(f"- **Why:** {payload.get('why', '')}")
        metrics = payload.get("metrics") or {}
        metric_txt = ", ".join(f"{k}={v}" for k, v in metrics.items() if v not in ("", None))
        if metric_txt:
            lines.append(f"- **Metrics:** {metric_txt}")
        lines.append("")

    cat_specs = recommendations.get("category_specialists") or {}
    if cat_specs:
        lines.extend(["## Category specialists (lowest TTFT per category)", ""])
        for cat, payload in cat_specs.items():
            lines.append(
                f"- **{cat}:** `{payload.get('model')}` "
                f"(TTFT={payload.get('avg_first_token_seconds')}, "
                f"tok/s={payload.get('avg_tokens_per_second')})"
            )
        lines.append("")

    lines.extend(
        [
            "## Robot realtime leaderboard",
            "",
            "| Rank | Model | Robot | TTFT avg | tok/s | VRAM pk | Success% | load |",
            "|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in leaderboard:
        lines.append(
            "| {rank} | `{model}` | {robot} | {ttft} | {tps} | {vram} | {ok} | {mode} |".format(
                rank=row.get("rank_robot", ""),
                model=row.get("model", ""),
                robot=row.get("score_robot_realtime", ""),
                ttft=row.get("avg_first_token_seconds", "-"),
                tps=row.get("avg_tokens_per_second", "-"),
                vram=row.get("peak_vram_mb", "-"),
                ok=row.get("success_rate_percent", "-"),
                mode=row.get("load_mode", "-"),
            )
        )
    lines.append("")

    lines.extend(["## Per-model aggregate detail", ""])
    for r in agg_rows:
        lines.append(f"### `{r.get('model')}`")
        lines.append("")
        lines.append(
            f"- Runs: {r.get('ok')}/{r.get('runs')} OK ({r.get('success_rate_percent')}%)"
        )
        lines.append(
            f"- Latency: TTFT avg={r.get('avg_first_token_seconds')} "
            f"(min={r.get('min_first_token_seconds')}, max={r.get('max_first_token_seconds')}, "
            f"std={r.get('std_first_token_seconds')})"
        )
        lines.append(
            f"- Throughput: tok/s avg={r.get('avg_tokens_per_second')} "
            f"(min={r.get('min_tokens_per_second')}, max={r.get('max_tokens_per_second')}, "
            f"std={r.get('std_tokens_per_second')})"
        )
        lines.append(
            f"- Timing: load_avg={r.get('avg_load_seconds')}s, "
            f"generate_avg={r.get('avg_generate_seconds')}s"
        )
        lines.append(
            f"- Resources: CPU pk={r.get('peak_cpu_percent')}%, "
            f"RAM pk={r.get('peak_ram_mb')}MB, sysRAM pk={r.get('peak_sys_ram_mb')}MB, "
            f"GPU pk={r.get('peak_gpu_util_percent')}%, "
            f"VRAM pk={r.get('peak_vram_mb')}MB, model VRAM={r.get('model_vram_mb')}MB"
        )
        lines.append(f"- Load mode: {r.get('load_mode')} / dtype={r.get('dtype')}")
        lines.append("")

    if category_rows:
        lines.extend(["## Category breakdown", ""])
        lines.append("| Model | Category | OK | TTFT avg | tok/s | out tok avg |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for r in category_rows:
            lines.append(
                f"| `{r.get('model')}` | {r.get('category')} | {r.get('ok')}/{r.get('runs')} | "
                f"{r.get('avg_first_token_seconds', '-')} | {r.get('avg_tokens_per_second', '-')} | "
                f"{r.get('avg_completion_tokens', '-')} |"
            )
        lines.append("")

    if bad:
        lines.extend(["## Failures", ""])
        for r in bad:
            lines.append(f"- `{r.get('model')}` | `{r.get('prompt_id')}`: {r.get('error')}")
        lines.append("")

    lines.extend(
        [
            "## How to use these files",
            "",
            "1. Open `llm_recommendations.json` for the primary pick.",
            "2. Confirm with `llm_leaderboard.csv`.",
            "3. Check category fit in `llm_analytics_by_category.csv`.",
            "4. Drill into `llm_analytics.csv` + response texts under `responses/`.",
            "",
            "## Notes",
            "",
        ]
    )
    for note in recommendations.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _print_table(title: str, columns: list[str], rows: list[dict[str, Any]]) -> None:
    print(f"\n{title}")
    if not rows:
        print("(no rows)")
        return
    labels = {
        "model": "model",
        "prompt_id": "prompt",
        "category": "cat",
        "status": "status",
        "load_seconds": "load_s",
        "first_token_seconds": "ttft_s",
        "generate_seconds": "gen_s",
        "wall_seconds": "wall_s",
        "tokens_per_second": "tok/s",
        "completion_tokens": "out_tok",
        "prompt_tokens": "in_tok",
        "peak_cpu_percent": "CPU% pk",
        "avg_cpu_percent": "CPU% avg",
        "peak_ram_mb": "RAM pk MB",
        "avg_ram_mb": "RAM avg MB",
        "peak_sys_ram_mb": "sysRAM pk",
        "peak_sys_ram_percent": "sysRAM%",
        "peak_gpu_util_percent": "GPU% pk",
        "avg_gpu_util_percent": "GPU% avg",
        "peak_vram_mb": "VRAM pk MB",
        "model_vram_mb": "VRAM mdl MB",
        "avg_load_seconds": "avg load_s",
        "avg_first_token_seconds": "avg ttft_s",
        "avg_generate_seconds": "avg gen_s",
        "avg_tokens_per_second": "avg tok/s",
        "avg_completion_tokens": "avg out_tok",
        "runs": "runs",
        "ok": "ok",
        "success_rate_percent": "OK%",
        "load_mode": "load",
        "gpu_index": "gpu",
        "rank_robot": "rank",
        "score_robot_realtime": "robot",
        "score_balanced": "balanced",
    }
    header = [labels.get(c, c) for c in columns]
    data = []
    for r in rows:
        data.append([str(r.get(c, ""))[:28] for c in columns])
    widths = [max(len(h), *(len(row[i]) for row in data)) for i, h in enumerate(header)]
    line = "-+-".join("-" * w for w in widths)
    print(line)
    print(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
    print(line)
    for row in data:
        print(" | ".join(val.ljust(w) for val, w in zip(row, widths)))
    print(line)


def print_analytics(
    rows: list[dict[str, Any]],
    agg_rows: list[dict[str, Any]],
    leaderboard: Optional[list[dict[str, Any]]] = None,
    category_rows: Optional[list[dict[str, Any]]] = None,
) -> None:
    print("\n" + "=" * 72)
    print("LLM MODEL ANALYTICS")
    print("=" * 72)
    _print_table(
        "Per run: timing (TTFT = time to first token; tok/s = completion tokens / generate seconds)",
        [
            "model",
            "prompt_id",
            "category",
            "status",
            "load_mode",
            "load_seconds",
            "first_token_seconds",
            "generate_seconds",
            "tokens_per_second",
            "completion_tokens",
            "prompt_tokens",
        ],
        rows,
    )
    _print_table(
        "Per run: resources (worker process tree + physical GPU, sampled every 0.5s)",
        [
            "model",
            "prompt_id",
            "peak_cpu_percent",
            "avg_cpu_percent",
            "peak_ram_mb",
            "avg_ram_mb",
            "peak_sys_ram_mb",
            "peak_sys_ram_percent",
            "peak_gpu_util_percent",
            "avg_gpu_util_percent",
            "peak_vram_mb",
            "model_vram_mb",
        ],
        rows,
    )
    _print_table(
        "Per model: averages / stability across prompts",
        [
            "model",
            "runs",
            "ok",
            "success_rate_percent",
            "avg_load_seconds",
            "avg_first_token_seconds",
            "avg_generate_seconds",
            "avg_tokens_per_second",
            "peak_cpu_percent",
            "peak_ram_mb",
            "peak_gpu_util_percent",
            "peak_vram_mb",
            "model_vram_mb",
        ],
        agg_rows,
    )
    if category_rows:
        _print_table(
            "Per model × category",
            [
                "model",
                "category",
                "ok",
                "avg_first_token_seconds",
                "avg_tokens_per_second",
                "avg_generate_seconds",
            ],
            category_rows,
        )
    if leaderboard:
        _print_table(
            "Leaderboard: robot realtime composite (higher is better)",
            [
                "rank_robot",
                "model",
                "score_robot_realtime",
                "score_balanced",
                "avg_first_token_seconds",
                "avg_tokens_per_second",
                "peak_vram_mb",
                "success_rate_percent",
            ],
            leaderboard,
        )
    failed = [r for r in rows if r.get("status") == "error"]
    if failed:
        print("\nFailures:")
        for r in failed:
            print(f"  - {r.get('model')} | {r.get('prompt_id')}: {r.get('error')}")


def write_analytics_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    return write_csv_rows(path, rows, ANALYTICS_COLUMNS)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ANALYTICS_COLUMNS + ["response_text"]
    write_csv_rows(path, rows, fields)


def export_llm_analytics(
    output_dir: Path,
    flat_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    analytics_rows = [{k: r.get(k, "") for k in ANALYTICS_COLUMNS} for r in flat_rows]
    agg_rows = aggregate_by_model(analytics_rows)
    category_rows = aggregate_by_category(analytics_rows)
    leaderboard = build_llm_leaderboard(agg_rows)
    recommendations = build_llm_recommendations(leaderboard, category_rows)

    paths = {
        "analytics": write_analytics_csv(analytics_rows, output_dir / "llm_analytics.csv"),
        "by_model": write_csv_rows(output_dir / "llm_analytics_by_model.csv", agg_rows),
        "by_category": write_csv_rows(output_dir / "llm_analytics_by_category.csv", category_rows),
        "leaderboard": write_csv_rows(output_dir / "llm_leaderboard.csv", leaderboard),
        "recommendations": output_dir / "llm_recommendations.json",
        "report": output_dir / "llm_selection_report.md",
    }
    write_json(paths["recommendations"], recommendations)
    write_llm_selection_report(
        paths["report"],
        rows=analytics_rows,
        agg_rows=agg_rows,
        category_rows=category_rows,
        leaderboard=leaderboard,
        recommendations=recommendations,
    )
    print_analytics(analytics_rows, agg_rows, leaderboard, category_rows)

    print("\n" + "=" * 72)
    print("LLM SELECTION PICKS")
    print("=" * 72)
    for key, payload in (recommendations.get("picks") or {}).items():
        if isinstance(payload, dict) and payload.get("model"):
            print(f"  {key}: {payload['model']}")
    cat_specs = recommendations.get("category_specialists") or {}
    if cat_specs:
        print("  category specialists:")
        for cat, payload in cat_specs.items():
            print(f"    - {cat}: {payload.get('model')}")

    return {
        "analytics_rows": analytics_rows,
        "agg_rows": agg_rows,
        "category_rows": category_rows,
        "leaderboard": leaderboard,
        "recommendations": recommendations,
        "paths": paths,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    only: Optional[str] = None,
    skip: Optional[str] = None,
    prompts_file: Optional[Path] = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    run_install: bool = False,
    suite: str = DEFAULT_SUITE,
) -> dict[str, Any]:
    model_names = select_models(only, skip)
    if run_install:
        install_packages(model_names)

    hf_token = ensure_hf_login()

    # int4 models need bitsandbytes even when the user passed --no-install.
    needs_bnb = any(
        str((MODELS.get(n) or {}).get("load_mode", "")).lower() in {"int4", "4bit"}
        for n in model_names
    )
    if needs_bnb:
        try:
            ensure_bitsandbytes()
        except Exception as exc:
            print(f"WARNING: bitsandbytes not ready ({exc}). int4 models will fail until you --install.")

    if SCRATCH_DIR != WORK_DIR:
        leftover = WORK_DIR / "llm_cache"
        if leftover.exists() and leftover.is_dir():
            print(f"Removing old {leftover} from the limited working disk...")
            shutil.rmtree(leftover, ignore_errors=True)

    # Start from a clean HF cache so the first large model is less likely to SIGBUS.
    boot_purge = purge_model_caches()
    if prompts_file:
        prompts_path = Path(prompts_file)
    else:
        prompts_path = ensure_default_prompts_file(suite)
    system, prompts, prompt_meta = load_prompts(prompts_path)

    n_gpu = gpu_count()
    gpus = list_gpus()
    scratch = disk_usage(SCRATCH_DIR)
    print(f"GPUs detected: {n_gpu}")
    print(json.dumps(gpus, indent=2))
    print(f"Output (persisted): {OUTPUT_DIR}")
    print(f"Scratch for model caches (temp disk): {SCRATCH_DIR} | {scratch}")
    if boot_purge.get("freed_gb"):
        print(f"Boot cache purge freed ~{boot_purge['freed_gb']} GB")
    print(
        f"Prompts file: {prompts_path} | suite={prompt_meta.get('suite') or suite} "
        f"| source={prompt_meta.get('source')} | cases={len(prompts)}"
    )
    print(f"Models: {model_names}")
    disabled = [n for n, on in ENABLE.items() if not on and n in MODEL_ORDER]
    if disabled and not only:
        print(f"Disabled by default (opt-in via --only): {disabled}")
    print(f"max_new_tokens={max_new_tokens} temperature={temperature} top_p={top_p}")
    gated_pending = [n for n in model_names if (MODELS.get(n) or {}).get("gated")]
    if hf_token:
        print(f"HF_TOKEN detected — gated models can authenticate ({len(gated_pending)} gated in this run).")
    elif gated_pending:
        print(
            "Note: no HF_TOKEN set — gated models will be skipped: "
            f"{gated_pending}. Add HF_TOKEN via Kaggle/Colab Secrets (and accept each HF license)."
        )

    results: dict[str, Any] = {
        "output_dir": str(OUTPUT_DIR),
        "response_dir": str(RESPONSE_DIR),
        "gpu_count": n_gpu,
        "gpus": gpus,
        "prompts_file": str(prompts_path),
        "suite": prompt_meta.get("suite") or suite,
        "dataset_source": prompt_meta.get("source"),
        "system": system,
        "prompt_count": len(prompts),
        "models": model_names,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "runs": [],
        "research_notes": {
            "dataset": "arabic_voice_robot_llm_test_dataset.md — Egyptian/MSA/code-switch/ASR-noise/tools/JSON/safety",
            "Qwen3 family": "Strong multilingual open models; disable thinking for lower robot latency.",
            "Gemma3 IT": "Compact/medium multilingual alternatives; may require HF gate acceptance.",
            "Mistral-Small-3.1-24B": "Strong conversational + native function calling; heavy on T4 (4-bit).",
            "ALLaM-7B": "Arabic/English specialized open model from SDAIA.",
            "Jais-2-8B": "Arabic-centric; MSA, dialects, and Arabic/English code-switching.",
            "Nile-Chat-4B": "Egyptian dialect specialist (Arabic script + Arabizi).",
            "SILMA-9B / Fanar-1-9B": "Additional Arabic open-weight candidates (disabled by default).",
        },
    }

    model_metas: list[dict[str, Any]] = []
    for i, model_name in enumerate(model_names):
        gpu = -1 if n_gpu <= 0 else (i % n_gpu)
        meta = run_model_isolated(
            model_name,
            prompts_path,
            gpu,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        results["runs"].append(meta)
        model_metas.append(meta)

        res = meta.get("resources", {}) or {}
        ok_runs = [r for r in (meta.get("runs") or []) if r.get("status") == "ok"]
        if meta.get("status") == "ok" and ok_runs:
            avg_tps = sum(float(r.get("tokens_per_second") or 0) for r in ok_runs) / len(ok_runs)
            avg_ttft = sum(float(r.get("first_token_seconds") or 0) for r in ok_runs) / len(ok_runs)
            pass_n = sum(1 for r in ok_runs if r.get("auto_pass"))
            avg_score = sum(float(r.get("overall_score") or 0) for r in ok_runs) / len(ok_runs)
            print(
                f"OK {model_name} | prompts_ok={len(ok_runs)}/{len(meta.get('runs') or [])} "
                f"auto_pass={pass_n}/{len(ok_runs)} avg_score={avg_score:.2f} "
                f"avg_ttft={avg_ttft:.3f}s avg_tok/s={avg_tps:.2f}"
            )
            print(
                f"   load={meta.get('load', {}).get('load_seconds', '-')}s  "
                f"cpu_peak={res.get('peak_cpu_percent', '-')}%  "
                f"ram_peak={res.get('peak_ram_mb', '-')}MB  "
                f"gpu_peak={res.get('peak_gpu_util_percent', '-')}%  "
                f"vram_peak={res.get('peak_vram_mb', '-')}MB"
            )
        else:
            print(f"FAIL {model_name}: {meta.get('error')}")

    flat_rows = flatten_runs(model_metas)
    summary_path = OUTPUT_DIR / "summary.json"
    csv_path = OUTPUT_DIR / "summary.csv"
    write_summary_csv(csv_path, flat_rows)

    exported = export_llm_analytics(OUTPUT_DIR, flat_rows)
    paths = exported["paths"]
    results["analytics"] = {
        "leaderboard": exported["leaderboard"],
        "recommendations": exported["recommendations"],
        "category_rows": exported["category_rows"],
        "files": {k: str(v) for k, v in paths.items()},
    }
    write_json(summary_path, results)

    # Quality scoreboard CSV
    quality_rows = [
        {
            "model": r.get("model"),
            "prompt_id": r.get("prompt_id"),
            "category": r.get("category"),
            "priority": r.get("priority"),
            "auto_pass": r.get("auto_pass"),
            "overall_score": r.get("overall_score"),
            "language_score": r.get("language_score"),
            "correctness_score": r.get("correctness_score"),
            "instruction_following_score": r.get("instruction_following_score"),
            "conciseness_score": r.get("conciseness_score"),
            "tts_suitability_score": r.get("tts_suitability_score"),
            "tool_accuracy": r.get("tool_accuracy"),
            "valid_json": r.get("valid_json"),
            "first_token_seconds": r.get("first_token_seconds"),
            "tokens_per_second": r.get("tokens_per_second"),
            "auto_failures": "; ".join(r.get("auto_failures") or [])
            if isinstance(r.get("auto_failures"), list)
            else r.get("auto_failures", ""),
        }
        for r in flat_rows
        if r.get("status") == "ok" or r.get("prompt_id")
    ]
    quality_path = write_csv_rows(OUTPUT_DIR / "llm_quality_scores.csv", quality_rows)
    paths["quality"] = quality_path

    zip_path = zip_llm_outputs(OUTPUT_DIR)
    results["outputs_zip"] = str(zip_path)
    write_json(summary_path, results)

    analytics_rows = exported["analytics_rows"]
    print(f"\nAnalytics CSV: {paths['analytics']}")
    print(f"Quality scores CSV: {quality_path}")
    print(f"Per-model analytics CSV: {paths['by_model']}")
    print(f"Per-category analytics CSV: {paths['by_category']}")
    print(f"Leaderboard CSV: {paths['leaderboard']}")
    print(f"Recommendations JSON: {paths['recommendations']}")
    print(f"Selection report MD: {paths['report']}")
    print(f"Summary JSON: {summary_path}")
    print(f"Summary CSV: {csv_path}")
    print(f"Outputs ZIP: {zip_path}")

    ok = [r for r in analytics_rows if r.get("status") == "ok"]
    bad = [r for r in analytics_rows if r.get("status") == "error"]
    passed = [r for r in ok if r.get("auto_pass") in (True, "True", "true", 1, "1")]
    print(f"OK runs: {len(ok)} / {len(analytics_rows)}")
    print(f"Auto-pass: {len(passed)} / {len(ok) if ok else 0}")
    if bad:
        print("Failed runs:")
        for row in bad:
            print(f"  - {row.get('model')} | {row.get('prompt_id')}: {row.get('error')}")
    return results


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument("--meta", type=Path)
    parser.add_argument("--only", help="Comma-separated model names to run")
    parser.add_argument("--skip", help="Comma-separated model names to skip")
    parser.add_argument("--suite", default=DEFAULT_SUITE, help="Dataset suite: first|critical|full")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--no-install", action="store_true")
    args, _unknown = parser.parse_known_args(argv)
    return args


if __name__ == "__main__":
    args = _parse_args()
    if args.worker:
        if not args.model or not args.meta or not args.prompts_file:
            raise SystemExit("--worker requires --model --meta --prompts-file")
        raise SystemExit(
            worker_main(
                args.model,
                args.prompts_file,
                args.meta,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        )

    # Auto-install on Kaggle, Colab (/content), or when KAGGLE_WORKING_DIR is set.
    on_notebook = (
        Path("/kaggle/working").exists()
        or Path("/content").exists()
        or bool(os.environ.get("KAGGLE_WORKING_DIR"))
        or bool(os.environ.get("COLAB_RELEASE_TAG"))
    )
    auto_install = on_notebook and os.environ.get("AUTO_INSTALL", "1") != "0"
    main(
        only=args.only,
        skip=args.skip,
        prompts_file=args.prompts_file,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        suite=args.suite,
        run_install=(args.install or os.environ.get("RUN_INSTALL", "0") == "1" or auto_install)
        and not args.no_install,
    )
