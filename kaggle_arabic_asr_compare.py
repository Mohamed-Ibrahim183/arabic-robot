#!/usr/bin/env python3
"""
Kaggle Arabic ASR comparison (T4 / T4x2 safe)
=============================================
Each model runs in its own subprocess so one CUDA/model failure does not poison
the rest of the benchmark.

Usage on Kaggle:
  1) Upload this file to /kaggle/working/.
  2) Put test audio in one of:
       /kaggle/input/<dataset>/
       /kaggle/working/asr_inputs/
       /kaggle/working/tts_outputs/
     Supported: .wav, .mp3, .flac, .m4a, .ogg, .opus.
  3) The spoken reference text is embedded in this script. Each model transcript
     is scored with WER/CER and accuracy % = (1 - WER) * 100. You can optionally
     pass --reference-text or place a .txt beside an audio file to override it.
  4) Run:
       %run /kaggle/working/kaggle_arabic_asr_compare.py
     Repeat after packages are installed:
       %run /kaggle/working/kaggle_arabic_asr_compare.py --no-install
     Run specific models:
       %run /kaggle/working/kaggle_arabic_asr_compare.py --only QwenCleo-ASR,Arabic-Whisper-Large-v3-FT-CT2

Outputs:
  /kaggle/working/asr_outputs/<model>__<audio>.txt
  /kaggle/working/asr_outputs/summary.json
  /kaggle/working/asr_outputs/summary.csv
  /kaggle/working/asr_outputs/asr_analytics.csv
  /kaggle/working/asr_outputs/asr_analytics_by_model.csv
  /kaggle/working/asr_outputs/asr_leaderboard.csv
  /kaggle/working/asr_outputs/asr_accuracy_ranking.csv
  /kaggle/working/asr_outputs/asr_recommendations.json
  /kaggle/working/asr_outputs/asr_selection_report.md
  (detailed timing, RTF, WER/CER, accuracy, resources, rankings, robot picks)

Model selection notes, based on current web/model-card research:
  - QwenCleo-ASR: strongest open model found for Egyptian Arabic + Arabic/English code-switching.
  - Arabic Whisper large-v3 FT CT2: strong production-friendly dialectal Arabic baseline.
  - Arabic Whisper turbo FT CT2: faster production-friendly dialectal baseline.
  - Whisper large-v3 / turbo: useful general baselines.
  - Qwen3-ASR and Audar runners are optional because they are heavier and move faster.
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
import wave
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
    return WORK_DIR  # local / fallback: keep old behavior


SCRATCH_DIR = _scratch_dir()

OUTPUT_DIR = WORK_DIR / "asr_outputs"       # results → persisted notebook output
TRANSCRIPT_DIR = OUTPUT_DIR
META_DIR = WORK_DIR / "asr_meta"            # small JSON metadata → persisted
INPUT_DIR = WORK_DIR / "asr_inputs"         # user-provided audio → persisted
CACHE_DIR = SCRATCH_DIR / "asr_cache"       # model checkpoints → big temp disk

# Change this to your audio file or folder on Kaggle.
# Examples:
#   Path("/kaggle/input/my-audio/sample.wav")
#   Path("/kaggle/input/my-audio/")
#   Path("/kaggle/working/asr_inputs/")
# AUDIO_LOCATION = Path(os.environ.get("ASR_AUDIO_LOCATION", str(INPUT_DIR)))
AUDIO_LOCATION = Path("/content/VoiceTut-TTS.wav")

for d in (OUTPUT_DIR, TRANSCRIPT_DIR, META_DIR, CACHE_DIR, INPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".webm"}

# Ground-truth spoken words embedded here so testing needs only this script.
EMBEDDED_REFERENCE_TEXT = """
السلام عليكم يا روبوت، صباح الخير. أنا محتاج أعمل اختبار لجودة تحويل النص إلى كلام باللهجة المصرية، عشان أقدر أقارن بين أكتر من موديل. فكرني إن عندي اجتماع بكرة الساعة عشرة ونص مع فريق التطوير، وبعدها Presentation الساعة اتنين، وبعدها Call مع العميل الساعة خمسة. كمان محتاجك تراجع معاي الـ dashboard، وتشوف الـ API response time، لازم يكون أقل من ميتين ملي ثانية. درجة الحرارة حوالي سبعة وثلاثين، وسعر الدولار حوالي خمسين جنيه. لو حصل delay في التقرير، ابعتلي notification قبل الساعة تلاتة ونص، وقولي لو في أي risk على الـ deadline. Please open the dashboard and send the report to the customer, and remind me again before the meeting by half an hour. وبالمناسبة، لو العميل سأل عن الـ pricing، قول له إن العرض سارٍ لمدة أسبوعين، وإن في خصم للعقد السنوي. شكراً ليك، وياريت الصوت يبقى واضح وطبيعي. كمان ممكن تضيف ملاحظة عن جدول الأعمال الإسبوعي، بحيث أعرف أولوية المهام الأساسيّة، وعايز قائمة تفصيلية بالـ KPIs المرتبطة بالمشروع الحالي. لو في أي changes في المواصفات، ابعتلي نسخة بالتحديثات فوراً على البريد الإلكتروني. عايز أعرف آخر status للـ deployment وتأكدلي إن الباك أب اتعمل الليلة الماضية. لو فيه مشاكل في التكامل مع الواجهات البرمجية، حاول تراجع اللوجات وترسللي ملخص بالأخطاء الشائعة. بالنسبة للعقود الجديدة، راجع مع قسم المالية لو فيه أي ملاحظات على العروض، وأكد مع العميل ميعاد توقيع العقد. لو احتاجنا نتواصل مع الدعم الفني، جهزلي كل التفاصيل، وسجل أي تذاكر technical تم فتحها مؤخراً. راجع أيضاً نسبة رضا العميل من خلال survey الأخير، ولو وصلت أي شكاوى لازم تبلغني على طول. أنا مهتم يكون الصوت طبيعي وثابت حتى مع جمل طويلة وزحام معلومات، ولو ممكن تضيف بعض intonation لتحسين السماع. وأخيراً، بعد انتهاء كل المهام المذكورة، ابعتلي تقرير نهائي يلخص العمل الأسبوعي وتوصيات للتحسين القادم.
""".strip()


def resolve_default_reference_text(explicit: Optional[Path] = None) -> Optional[Path]:
    """Locate an optional file that overrides the embedded reference text."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    env = os.environ.get("ASR_REFERENCE_TEXT")
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            WORK_DIR / "test-text.txt",
            Path.cwd() / "test-text.txt",
        ]
    )
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent / "test-text.txt")
    for cand in candidates:
        try:
            if cand is not None and cand.is_file() and cand.stat().st_size > 0:
                return cand.resolve()
        except OSError:
            continue
    return None


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

ENABLE = {
    "Whisper-Small-CT2": True,
    "Whisper-Large-v3-Turbo-CT2": True,
    "Arabic-Whisper-Turbo-FT-CT2": True,
    "Arabic-Whisper-Large-v3-FT-CT2": True,
    "QwenCleo-ASR": True,
    # Heavier or less predictable on Kaggle. Enable after the default pass works.
    "Whisper-Large-v3-CT2": True,
    "Qwen3-ASR-0.6B": True,
    "Qwen3-ASR-1.7B": True,
    "Audar-ASR-V1-Flash": True,
    # Additional suggested models (transformers backends, all fit on a T4):
    "Voxtral-Mini-3B": True,        # Mistral, strong multilingual ASR incl. Arabic
    "SeamlessM4T-v2-Large": True,   # Meta, good MSA ASR baseline
    "MMS-1B-all": True,             # Meta CTC, 1100+ languages incl. Arabic
}

MODEL_ORDER = [
    "Whisper-Small-CT2",
    "Whisper-Large-v3-Turbo-CT2",
    "Arabic-Whisper-Turbo-FT-CT2",
    "Arabic-Whisper-Large-v3-FT-CT2",
    "QwenCleo-ASR",
    "Whisper-Large-v3-CT2",
    "Qwen3-ASR-0.6B",
    "Qwen3-ASR-1.7B",
    "Audar-ASR-V1-Flash",
    "Voxtral-Mini-3B",
    "SeamlessM4T-v2-Large",
    "MMS-1B-all",
]

FAST_WHISPER_MODELS = {
    "Whisper-Small-CT2": "small",
    # Systran's turbo repo was removed from HF; deepdml hosts a public CT2 conversion.
    "Whisper-Large-v3-Turbo-CT2": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "Whisper-Large-v3-CT2": "Systran/faster-whisper-large-v3",
    "Arabic-Whisper-Turbo-FT-CT2": "dev-ahmedhany/whisper-large-v3-turbo-arabic-ft-ct2-int8",
    "Arabic-Whisper-Large-v3-FT-CT2": "dev-ahmedhany/whisper-large-v3-arabic-ft-v3-ct2-int8",
}

QWEN_ASR_MODELS = {
    "Qwen3-ASR-0.6B": "Qwen/Qwen3-ASR-0.6B",
    "Qwen3-ASR-1.7B": "Qwen/Qwen3-ASR-1.7B",
    # Audar's old "audar-asr-h-turbo-merged" repo was removed from HF.
    # Audar-ASR-V1-Flash is their transformers-compatible (qwen3_asr) release;
    # Audar-ASR-V1-Turbo only ships GGUF/vLLM weights, which this script can't load.
    "Audar-ASR-V1-Flash": "audarai/Audar-ASR-V1-Flash",
}

HF_ASR_MODELS = {
    "Voxtral-Mini-3B": "mistralai/Voxtral-Mini-3B-2507",
    "SeamlessM4T-v2-Large": "facebook/seamless-m4t-v2-large",
    "MMS-1B-all": "facebook/mms-1b-all",
}


# ---------------------------------------------------------------------------
# Install / environment helpers
# ---------------------------------------------------------------------------

INSTALL_BASE = r'''
import subprocess, sys

def _pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

subprocess.check_call(["apt-get", "update", "-qq"])
subprocess.check_call(["apt-get", "install", "-y", "-qq", "ffmpeg", "libsndfile1", "git", "git-lfs"])

_pip("-U", "faster-whisper", "jiwer", "soundfile", "librosa", "pandas", "huggingface_hub")

# Analytics dependencies (CPU/RAM/GPU sampling)
_pip("psutil", "nvidia-ml-py")
'''

INSTALL_QWEN = r'''
import subprocess, sys

def _pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

# Do not let qwencleo-asr replace Kaggle's working torch install.
_pip("-U", "qwen-asr>=0.0.6")
_pip("--no-deps", "qwencleo-asr")
'''

INSTALL_HF_ASR = r'''
import subprocess, sys

def _pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

_pip("-U", "accelerate", "sentencepiece")
# Voxtral audio handling in transformers requires mistral-common with audio extras.
_pip("-U", "mistral-common[audio]>=1.8.1")
'''


def install_packages(model_names: list[str]) -> None:
    print("Installing Kaggle ASR benchmark dependencies...")
    exec(INSTALL_BASE, {"__name__": "__main__"})  # noqa: S102
    if any(name == "QwenCleo-ASR" or name in QWEN_ASR_MODELS for name in model_names):
        exec(INSTALL_QWEN, {"__name__": "__main__"})  # noqa: S102
    if any(name in HF_ASR_MODELS for name in model_names):
        exec(INSTALL_HF_ASR, {"__name__": "__main__"})  # noqa: S102


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Resource monitoring (CPU / RAM / GPU sampled in a background thread)
# ---------------------------------------------------------------------------

class ResourceMonitor:
    """Samples process-tree CPU%, RSS RAM, and GPU util/VRAM while a model runs.

    Runs inside the worker process. GPU stats come from NVML for the physical
    GPU selected via CUDA_VISIBLE_DEVICES, so all GPU work is captured.
    """

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
            self._proc.cpu_percent(None)  # prime the counter
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
        stats = {
            "samples": len(samples),
            "peak_cpu_percent": round(peak_cpu, 1),
            "avg_cpu_percent": round(avg_cpu, 1),
            "peak_ram_mb": round(peak_ram, 1),
            "avg_ram_mb": round(avg_ram, 1),
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
# Audio / scoring helpers
# ---------------------------------------------------------------------------

def discover_audio(paths: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    search_roots: list[Path] = []
    if paths:
        search_roots = paths
    else:
        search_roots.extend([AUDIO_LOCATION, WORK_DIR / "tts_outputs"])
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            search_roots.append(kaggle_input)
        local_audio = Path.cwd() / "audio"
        if local_audio.exists():
            search_roots.append(local_audio)

    for root in search_roots:
        root = Path(root)  # tolerate plain strings (e.g. hand-edited AUDIO_LOCATION)
        if root.is_file() and root.suffix.lower() in AUDIO_EXTENSIONS:
            candidates.append(root)
        elif root.is_dir():
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
                    candidates.append(p)

    unique: dict[str, Path] = {}
    for p in candidates:
        unique[str(p.resolve())] = p
    return sorted(unique.values(), key=lambda p: str(p).lower())


def load_reference_map(path: Optional[Path]) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("reference JSON must be an object mapping filename/stem/path to text")
    return {str(k): str(v) for k, v in data.items()}


def reference_for_audio(
    audio_path: Path,
    reference_map: dict[str, str],
    default_reference: Optional[str] = None,
) -> Optional[str]:
    sidecar = audio_path.with_suffix(".txt")
    if sidecar.exists():
        return sidecar.read_text(encoding="utf-8").strip()
    keys = [str(audio_path), str(audio_path.resolve()), audio_path.name, audio_path.stem]
    for key in keys:
        if key in reference_map:
            return reference_map[key].strip()
    if default_reference:
        return default_reference.strip()
    return None


def audio_duration_seconds(path: Path) -> Optional[float]:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.frames and info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as handle:
                return float(handle.getnframes()) / float(handle.getframerate())
        except Exception:
            pass
    try:
        import librosa

        return float(librosa.get_duration(path=str(path)))
    except Exception:
        return None


_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_PUNCT = re.compile(r"[^\w\s\u0600-\u06ff]+", re.UNICODE)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = _ARABIC_DIACRITICS.sub("", text)
    text = text.replace("\u0640", "")
    # Light Arabic spelling normalization. Keep English code-switch words intact.
    text = re.sub("[إأآا]", "ا", text)
    text = text.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    text = _PUNCT.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (0 if ca == cb else 1),
                )
            )
        prev = cur
    return prev[-1]


def score_transcript(reference: Optional[str], hypothesis: str) -> dict[str, Any]:
    if not reference:
        return {}
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)
    scores: dict[str, Any] = {
        "reference": reference,
        "reference_normalized": ref_norm,
        "hypothesis_normalized": hyp_norm,
    }
    try:
        from jiwer import cer, wer

        scores["wer"] = float(wer(ref_norm, hyp_norm))
        scores["cer"] = float(cer(ref_norm, hyp_norm))
    except Exception:
        ref_words = ref_norm.split()
        hyp_words = hyp_norm.split()
        scores["wer"] = _edit_distance(ref_words, hyp_words) / max(1, len(ref_words))
        scores["cer"] = _edit_distance(list(ref_norm), list(hyp_norm)) / max(1, len(ref_norm))
    # Accuracy %: how close the transcript is to the reference (100% = perfect).
    scores["accuracy_percent"] = round(max(0.0, (1.0 - float(scores["wer"])) * 100.0), 2)
    scores["char_accuracy_percent"] = round(max(0.0, (1.0 - float(scores["cer"])) * 100.0), 2)
    return scores


def safe_name(path_or_name: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", path_or_name)
    return text.strip("._") or "item"


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------

def transcribe_faster_whisper(model_name: str, audio_path: Path, language: Optional[str]) -> dict[str, Any]:
    ensure_module("faster_whisper", "faster-whisper")
    from faster_whisper import WhisperModel

    model_id = FAST_WHISPER_MODELS[model_name]
    device = "cuda" if gpu_count() > 0 else "cpu"
    if device == "cuda" and "ct2-int8" in model_id:
        compute_type = os.environ.get("ASR_COMPUTE_TYPE", "int8_float16")
    elif device == "cuda":
        compute_type = os.environ.get("ASR_COMPUTE_TYPE", "float16")
    else:
        compute_type = os.environ.get("ASR_COMPUTE_TYPE", "int8")

    t0 = time.perf_counter()
    model = WhisperModel(
        model_id,
        device=device,
        compute_type=compute_type,
        cpu_threads=int(os.environ.get("ASR_CPU_THREADS", "8")),
        download_root=str(CACHE_DIR / "faster_whisper"),
    )
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        task="transcribe",
        beam_size=int(os.environ.get("ASR_BEAM_SIZE", "5")),
        vad_filter=os.environ.get("ASR_VAD", "1") != "0",
        condition_on_previous_text=False,
    )
    segments = list(segments)
    transcribe_s = time.perf_counter() - t1
    text = " ".join(s.text.strip() for s in segments if s.text.strip()).strip()
    segment_rows = [
        {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
        for s in segments
    ]
    return {
        "backend": "faster-whisper",
        "model_id": model_id,
        "device": device,
        "compute_type": compute_type,
        "load_seconds": load_s,
        "transcribe_seconds": transcribe_s,
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "text": text,
        "segments": segment_rows,
    }


def _import_qwen_asr_class():
    """Import qwen_asr's model class, pinning transformers if the import breaks.

    qwen-asr pins transformers==4.57.x; Kaggle images may carry 5.x which can
    break qwen_asr at import time. Nothing else in this benchmark needs 5.x.
    """
    try:
        from qwen_asr import Qwen3ASRModel

        return Qwen3ASRModel
    except Exception:
        print("qwen_asr import failed - pinning transformers==4.57.3 and retrying...")
        _pip("transformers==4.57.3")
        from qwen_asr import Qwen3ASRModel

        return Qwen3ASRModel


def transcribe_qwencleo(audio_path: Path, language: Optional[str]) -> dict[str, Any]:
    ensure_module("qwen_asr", "qwen-asr>=0.0.6", upgrade=True)
    ensure_module("qwencleo_asr", "qwencleo-asr", no_deps=True)
    _import_qwen_asr_class()  # verify/repair transformers compatibility first
    from qwencleo_asr import QwenCleoASR

    forced_language = "Arabic" if language == "ar" else language
    t0 = time.perf_counter()
    asr = QwenCleoASR()
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    result = asr.transcribe(str(audio_path), language=forced_language or "Arabic")
    transcribe_s = time.perf_counter() - t1
    if isinstance(result, list):
        result = result[0] if result else None
    text = getattr(result, "text", str(result) if result is not None else "").strip()
    return {
        "backend": "qwencleo-asr",
        "model_id": "mohammedaly22/QwenCleo-ASR",
        "device": "cuda" if gpu_count() > 0 else "cpu",
        "load_seconds": load_s,
        "transcribe_seconds": transcribe_s,
        "language": forced_language or "Arabic",
        "text": text,
    }


def transcribe_qwen_asr(model_name: str, audio_path: Path, language: Optional[str]) -> dict[str, Any]:
    ensure_module("qwen_asr", "qwen-asr>=0.0.6", upgrade=True)
    import torch

    Qwen3ASRModel = _import_qwen_asr_class()

    model_id = QWEN_ASR_MODELS[model_name]
    forced_language = "Arabic" if language == "ar" else language
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"

    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": device_map,
        "max_inference_batch_size": int(os.environ.get("QWEN_ASR_BATCH", "1")),
        "max_new_tokens": int(os.environ.get("QWEN_ASR_MAX_NEW_TOKENS", "384")),
    }
    if model_name == "Audar-ASR-V1-Flash":
        kwargs["trust_remote_code"] = True
        kwargs["attn_implementation"] = os.environ.get("AUDAR_ATTN", "sdpa")

    t0 = time.perf_counter()
    model = Qwen3ASRModel.from_pretrained(model_id, **kwargs)
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    results = model.transcribe(
        audio=str(audio_path),
        language=forced_language or "Arabic",
        return_time_stamps=False,
    )
    transcribe_s = time.perf_counter() - t1
    result = results[0] if isinstance(results, list) and results else results
    text = getattr(result, "text", str(result) if result is not None else "").strip()
    detected = getattr(result, "language", forced_language or "Arabic")
    return {
        "backend": "qwen-asr",
        "model_id": model_id,
        "device": device_map,
        "dtype": str(dtype).replace("torch.", ""),
        "load_seconds": load_s,
        "transcribe_seconds": transcribe_s,
        "language": detected,
        "text": text,
    }


def _load_audio_16k(audio_path: Path):
    import librosa
    import numpy as np

    audio, _sr = librosa.load(str(audio_path), sr=16000, mono=True)
    return np.asarray(audio, dtype="float32"), 16000


def _chunk_audio(audio, sr: int, chunk_s: float = 25.0):
    step = int(chunk_s * sr)
    return [audio[i : i + step] for i in range(0, len(audio), step)] or [audio]


def transcribe_voxtral(audio_path: Path, language: Optional[str]) -> dict[str, Any]:
    """mistralai/Voxtral-Mini-3B-2507 via transformers (dedicated transcription mode)."""
    ensure_module("mistral_common", "mistral-common[audio]>=1.8.1", upgrade=True)
    import torch
    from transformers import AutoProcessor, VoxtralForConditionalGeneration

    model_id = HF_ASR_MODELS["Voxtral-Mini-3B"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32  # T4 has no fast bf16

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=str(CACHE_DIR / "hf"))
    model = VoxtralForConditionalGeneration.from_pretrained(
        model_id, dtype=dtype, device_map=device, cache_dir=str(CACHE_DIR / "hf")
    )
    load_s = time.perf_counter() - t0

    # Method was renamed; old transformers versions only have the typo'd name.
    apply_req = getattr(processor, "apply_transcription_request", None) or getattr(
        processor, "apply_transcrition_request"
    )

    t1 = time.perf_counter()
    inputs = apply_req(language=language or "ar", audio=str(audio_path), model_id=model_id)
    inputs = inputs.to(device, dtype=dtype)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=2048)
    decoded = processor.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    transcribe_s = time.perf_counter() - t1
    return {
        "backend": "transformers-voxtral",
        "model_id": model_id,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "load_seconds": load_s,
        "transcribe_seconds": transcribe_s,
        "language": language or "ar",
        "text": " ".join(d.strip() for d in decoded).strip(),
    }


def transcribe_seamless(audio_path: Path, language: Optional[str]) -> dict[str, Any]:
    """facebook/seamless-m4t-v2-large speech-to-text; chunked for long audio."""
    ensure_module("sentencepiece", "sentencepiece")
    import torch
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

    model_id = HF_ASR_MODELS["SeamlessM4T-v2-Large"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tgt_lang = "arb" if (language or "ar").startswith("ar") else language

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=str(CACHE_DIR / "hf"))
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        model_id, dtype=dtype, cache_dir=str(CACHE_DIR / "hf")
    ).to(device)
    load_s = time.perf_counter() - t0

    audio, sr = _load_audio_16k(audio_path)
    t1 = time.perf_counter()
    pieces: list[str] = []
    with torch.inference_mode():
        for chunk in _chunk_audio(audio, sr):
            inputs = processor(audios=chunk, sampling_rate=sr, return_tensors="pt").to(device)
            if dtype == torch.float16 and "input_features" in inputs:
                inputs["input_features"] = inputs["input_features"].to(dtype)
            tokens = model.generate(**inputs, tgt_lang=tgt_lang)
            pieces.append(processor.decode(tokens[0].tolist(), skip_special_tokens=True).strip())
    transcribe_s = time.perf_counter() - t1
    return {
        "backend": "transformers-seamless",
        "model_id": model_id,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "load_seconds": load_s,
        "transcribe_seconds": transcribe_s,
        "language": tgt_lang,
        "text": " ".join(p for p in pieces if p).strip(),
    }


def transcribe_mms(audio_path: Path, language: Optional[str]) -> dict[str, Any]:
    """facebook/mms-1b-all CTC with the Arabic adapter; chunked for long audio."""
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    model_id = HF_ASR_MODELS["MMS-1B-all"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()
    processor = None
    model = None
    lang_used = None
    last_err: Optional[Exception] = None
    # MMS uses ISO 639-3 codes; Standard Arabic is usually "arb".
    for code in (os.environ.get("MMS_LANG", "arb"), "ara"):
        try:
            processor = AutoProcessor.from_pretrained(
                model_id, target_lang=code, cache_dir=str(CACHE_DIR / "hf")
            )
            model = Wav2Vec2ForCTC.from_pretrained(
                model_id,
                target_lang=code,
                ignore_mismatched_sizes=True,
                cache_dir=str(CACHE_DIR / "hf"),
            ).to(device)
            lang_used = code
            break
        except Exception as exc:
            last_err = exc
    if model is None or processor is None:
        raise RuntimeError(f"MMS Arabic adapter load failed: {last_err}")
    load_s = time.perf_counter() - t0

    audio, sr = _load_audio_16k(audio_path)
    t1 = time.perf_counter()
    pieces: list[str] = []
    with torch.inference_mode():
        for chunk in _chunk_audio(audio, sr):
            inputs = processor(chunk, sampling_rate=sr, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            ids = torch.argmax(logits, dim=-1)[0]
            pieces.append(processor.decode(ids).strip())
    transcribe_s = time.perf_counter() - t1
    return {
        "backend": "transformers-mms",
        "model_id": model_id,
        "device": device,
        "load_seconds": load_s,
        "transcribe_seconds": transcribe_s,
        "language": lang_used,
        "text": " ".join(p for p in pieces if p).strip(),
    }


def run_transcription(model_name: str, audio_path: Path, language: Optional[str]) -> dict[str, Any]:
    if model_name in FAST_WHISPER_MODELS:
        return transcribe_faster_whisper(model_name, audio_path, language)
    if model_name == "QwenCleo-ASR":
        return transcribe_qwencleo(audio_path, language)
    if model_name in QWEN_ASR_MODELS:
        return transcribe_qwen_asr(model_name, audio_path, language)
    if model_name == "Voxtral-Mini-3B":
        return transcribe_voxtral(audio_path, language)
    if model_name == "SeamlessM4T-v2-Large":
        return transcribe_seamless(audio_path, language)
    if model_name == "MMS-1B-all":
        return transcribe_mms(audio_path, language)
    raise KeyError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Worker + orchestration
# ---------------------------------------------------------------------------

def worker_main(
    model_name: str,
    audio_file: Path,
    reference_file: Optional[Path],
    meta_file: Path,
    transcript_file: Path,
    language: Optional[str],
) -> int:
    print(f"[worker] model={model_name} audio={audio_file} cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    for mod, pkg in (("psutil", "psutil"), ("pynvml", "nvidia-ml-py")):
        try:
            ensure_module(mod, pkg)
        except Exception:
            pass
    monitor = ResourceMonitor().start()
    try:
        reference = reference_file.read_text(encoding="utf-8").strip() if reference_file and reference_file.exists() else None
        duration = audio_duration_seconds(audio_file)
        meta = run_transcription(model_name, audio_file, language)
        text = str(meta.get("text", "")).strip()
        transcript_file.parent.mkdir(parents=True, exist_ok=True)
        transcript_file.write_text(text + "\n", encoding="utf-8")
        meta.update(
            {
                "status": "ok",
                "model": model_name,
                "audio": str(audio_file),
                "audio_name": audio_file.name,
                "audio_duration_seconds": duration,
                "transcript_file": str(transcript_file),
                "transcript_chars": len(text),
            }
        )
        if duration and meta.get("transcribe_seconds") is not None:
            meta["realtime_factor"] = float(meta["transcribe_seconds"]) / max(duration, 0.001)
        meta.update(score_transcript(reference, text))
        meta["resources"] = monitor.stop()
        write_json(meta_file, meta)
        print(f"[worker] OK {model_name} -> {transcript_file}")
        return 0
    except Exception as exc:
        payload = {
            "status": "error",
            "model": model_name,
            "audio": str(audio_file),
            "audio_name": audio_file.name,
            "error": repr(exc),
            "traceback": traceback.format_exc()[-6000:],
            "resources": monitor.stop(),
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
        dest = WORK_DIR / "_kaggle_arabic_asr_compare_worker.py"
        dest.write_text(src, encoding="utf-8")
        return dest.resolve()
    raise RuntimeError(
        "Cannot resolve script path for worker subprocess. Upload "
        "kaggle_arabic_asr_compare.py to /kaggle/working/ and run it with %run."
    )


def run_model_isolated(
    model_name: str,
    audio_path: Path,
    reference: Optional[str],
    gpu_index: int,
    language: Optional[str],
) -> dict[str, Any]:
    audio_key = safe_name(audio_path.stem)
    model_key = safe_name(model_name)
    meta_path = META_DIR / f"{model_key}__{audio_key}.json"
    transcript_path = TRANSCRIPT_DIR / f"{model_key}__{audio_key}.txt"
    reference_path = META_DIR / f"{audio_key}.reference.txt"
    if reference:
        reference_path.write_text(reference, encoding="utf-8")
    elif reference_path.exists():
        reference_path.unlink()
    if meta_path.exists():
        meta_path.unlink()

    env = os.environ.copy()
    if gpu_index >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env.setdefault("HF_HOME", str(CACHE_DIR / "hf"))
    env.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR / "hf"))

    cmd = [
        sys.executable,
        str(_this_script()),
        "--worker",
        "--model",
        model_name,
        "--audio-file",
        str(audio_path),
        "--meta",
        str(meta_path),
        "--transcript-file",
        str(transcript_path),
    ]
    if reference:
        cmd += ["--reference-file", str(reference_path)]
    if language:
        cmd += ["--language", language]

    print(f"\n{'=' * 72}\nASR {model_name} | {audio_path.name} | GPU {gpu_index if gpu_index >= 0 else 'CPU'}\n{'=' * 72}")
    started = time.perf_counter()
    proc = subprocess.run(cmd, env=env)
    wall_s = time.perf_counter() - started

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {"status": "error", "error": f"Worker exited {proc.returncode} without writing meta"}
    meta["wall_seconds"] = wall_s
    meta["gpu_index"] = gpu_index
    return meta


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


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model",
        "backend",
        "model_id",
        "audio_name",
        "status",
        "wer",
        "cer",
        "accuracy_percent",
        "char_accuracy_percent",
        "audio_duration_seconds",
        "load_seconds",
        "transcribe_seconds",
        "realtime_factor",
        "wall_seconds",
        "device",
        "compute_type",
        "language",
        "transcript_file",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Analytics reporting (printed tables + CSV/JSON/MD for model selection)
# ---------------------------------------------------------------------------

ANALYTICS_COLUMNS = [
    "model",
    "audio_name",
    "status",
    "gpu_index",
    "gpu_name",
    "backend",
    "device",
    "load_seconds",
    "transcribe_seconds",
    "wall_seconds",
    "audio_seconds",
    "rtf",
    "x_realtime",
    "wer",
    "cer",
    "accuracy_percent",
    "char_accuracy_percent",
    "transcript_chars",
    "peak_cpu_percent",
    "avg_cpu_percent",
    "peak_ram_mb",
    "avg_ram_mb",
    "peak_gpu_util_percent",
    "avg_gpu_util_percent",
    "peak_vram_mb",
    "avg_vram_mb",
    "model_vram_mb",
    "error",
]


def _safe_floats(vals: list[Any]) -> list[float]:
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
    """Min-max normalize to 0..100. Missing keys stay out of the map."""
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


def build_analytics_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for meta in runs:
        res = meta.get("resources", {}) or {}
        tr_s = float(meta.get("transcribe_seconds") or 0.0)
        audio_s = float(meta.get("audio_duration_seconds") or 0.0)
        rtf = round(tr_s / audio_s, 3) if audio_s > 0 and tr_s > 0 else ""
        x_rt = round(audio_s / tr_s, 2) if tr_s > 0 and audio_s > 0 else ""
        rows.append(
            {
                "model": meta.get("model", ""),
                "audio_name": meta.get("audio_name", ""),
                "status": meta.get("status", ""),
                "gpu_index": meta.get("gpu_index", ""),
                "gpu_name": res.get("gpu_name", ""),
                "backend": meta.get("backend", ""),
                "device": meta.get("device", ""),
                "load_seconds": round(float(meta.get("load_seconds") or 0.0), 2),
                "transcribe_seconds": round(tr_s, 2),
                "wall_seconds": round(float(meta.get("wall_seconds") or 0.0), 2),
                "audio_seconds": round(audio_s, 2),
                "rtf": rtf,
                "x_realtime": x_rt,
                "wer": round(float(meta["wer"]), 4) if "wer" in meta else "",
                "cer": round(float(meta["cer"]), 4) if "cer" in meta else "",
                "accuracy_percent": (
                    round(float(meta["accuracy_percent"]), 2) if "accuracy_percent" in meta else ""
                ),
                "char_accuracy_percent": (
                    round(float(meta["char_accuracy_percent"]), 2)
                    if "char_accuracy_percent" in meta
                    else ""
                ),
                "transcript_chars": meta.get("transcript_chars", ""),
                "peak_cpu_percent": res.get("peak_cpu_percent", ""),
                "avg_cpu_percent": res.get("avg_cpu_percent", ""),
                "peak_ram_mb": res.get("peak_ram_mb", ""),
                "avg_ram_mb": res.get("avg_ram_mb", ""),
                "peak_gpu_util_percent": res.get("peak_gpu_util_percent", ""),
                "avg_gpu_util_percent": res.get("avg_gpu_util_percent", ""),
                "peak_vram_mb": res.get("peak_vram_mb", ""),
                "avg_vram_mb": res.get("avg_vram_mb", ""),
                "model_vram_mb": res.get("model_vram_mb", ""),
                "error": meta.get("error", ""),
            }
        )
    return rows


def aggregate_by_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rich per-model aggregates across all audio files (ok runs for metrics)."""
    order: list[str] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        name = str(row.get("model") or "")
        if name not in grouped:
            grouped[name] = []
            order.append(name)
        grouped[name].append(row)

    agg_rows: list[dict[str, Any]] = []
    for name in order:
        model_rows = grouped[name]
        ok_rows = [r for r in model_rows if r.get("status") == "ok"]

        def col(key: str, source: Optional[list[dict[str, Any]]] = None) -> list[float]:
            src = source if source is not None else ok_rows
            return _safe_floats([r.get(key) for r in src])

        wer = col("wer")
        cer = col("cer")
        acc = col("accuracy_percent")
        char_acc = col("char_accuracy_percent")
        load = col("load_seconds")
        tr = col("transcribe_seconds")
        wall = col("wall_seconds")
        rtf = col("rtf")
        xrt = col("x_realtime")
        peak_cpu = col("peak_cpu_percent", model_rows)
        peak_ram = col("peak_ram_mb", model_rows)
        peak_gpu = col("peak_gpu_util_percent", model_rows)
        peak_vram = col("peak_vram_mb", model_rows)
        model_vram = col("model_vram_mb", model_rows)

        success_rate = round(100.0 * len(ok_rows) / len(model_rows), 1) if model_rows else 0.0
        realtime_ok = sum(1 for v in rtf if v < 1.0)
        realtime_pct = round(100.0 * realtime_ok / len(rtf), 1) if rtf else ""

        agg_rows.append(
            {
                "model": name,
                "runs": len(model_rows),
                "ok": len(ok_rows),
                "failed": len(model_rows) - len(ok_rows),
                "success_rate_percent": success_rate,
                "avg_wer": _stat_mean(wer, 4),
                "min_wer": _stat_min(wer, 4),
                "max_wer": _stat_max(wer, 4),
                "std_wer": _stat_std(wer, 4),
                "avg_cer": _stat_mean(cer, 4),
                "min_cer": _stat_min(cer, 4),
                "max_cer": _stat_max(cer, 4),
                "std_cer": _stat_std(cer, 4),
                "avg_accuracy_percent": _stat_mean(acc, 2),
                "min_accuracy_percent": _stat_min(acc, 2),
                "max_accuracy_percent": _stat_max(acc, 2),
                "std_accuracy_percent": _stat_std(acc, 2),
                "avg_char_accuracy_percent": _stat_mean(char_acc, 2),
                "avg_load_seconds": _stat_mean(load, 2),
                "min_load_seconds": _stat_min(load, 2),
                "max_load_seconds": _stat_max(load, 2),
                "avg_transcribe_seconds": _stat_mean(tr, 2),
                "min_transcribe_seconds": _stat_min(tr, 2),
                "max_transcribe_seconds": _stat_max(tr, 2),
                "avg_wall_seconds": _stat_mean(wall, 2),
                "avg_rtf": _stat_mean(rtf, 3),
                "min_rtf": _stat_min(rtf, 3),
                "max_rtf": _stat_max(rtf, 3),
                "std_rtf": _stat_std(rtf, 3),
                "avg_x_realtime": _stat_mean(xrt, 2),
                "realtime_capable_percent": realtime_pct,
                "peak_cpu_percent": _stat_max(peak_cpu, 1),
                "avg_peak_cpu_percent": _stat_mean(peak_cpu, 1),
                "peak_ram_mb": _stat_max(peak_ram, 1),
                "avg_peak_ram_mb": _stat_mean(peak_ram, 1),
                "peak_gpu_util_percent": _stat_max(peak_gpu, 1),
                "avg_peak_gpu_util_percent": _stat_mean(peak_gpu, 1),
                "peak_vram_mb": _stat_max(peak_vram, 1),
                "avg_peak_vram_mb": _stat_mean(peak_vram, 1),
                "model_vram_mb": _stat_max(model_vram, 1),
                "backend": next((r.get("backend") for r in ok_rows if r.get("backend")), ""),
                "device": next((r.get("device") for r in ok_rows if r.get("device")), ""),
            }
        )
    return agg_rows


def build_accuracy_ranking(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = [r for r in rows if r.get("status") == "ok" and r.get("accuracy_percent") not in ("", None)]
    ranked = sorted(
        scored,
        key=lambda r: (
            -float(r["accuracy_percent"]),
            float(r["wer"]) if r.get("wer") not in ("", None) else 1.0,
            float(r["rtf"]) if r.get("rtf") not in ("", None) else 999.0,
        ),
    )
    out: list[dict[str, Any]] = []
    for i, r in enumerate(ranked, 1):
        out.append(
            {
                "rank": i,
                "model": r.get("model", ""),
                "audio_name": r.get("audio_name", ""),
                "accuracy_percent": r.get("accuracy_percent", ""),
                "wer": r.get("wer", ""),
                "cer": r.get("cer", ""),
                "char_accuracy_percent": r.get("char_accuracy_percent", ""),
                "rtf": r.get("rtf", ""),
                "transcribe_seconds": r.get("transcribe_seconds", ""),
                "peak_vram_mb": r.get("peak_vram_mb", ""),
            }
        )
    return out


def build_asr_leaderboard(agg_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok = [r for r in agg_rows if int(r.get("ok") or 0) > 0]
    if not ok:
        return []

    has_acc = any(r.get("avg_accuracy_percent") not in ("", None) for r in ok)
    acc_map = {
        str(r["model"]): float(r["avg_accuracy_percent"])
        for r in ok
        if r.get("avg_accuracy_percent") not in ("", None)
    }
    rtf_map = {
        str(r["model"]): float(r["avg_rtf"]) for r in ok if r.get("avg_rtf") not in ("", None)
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

    acc_s = _norm_map(acc_map, lower_is_better=False) if has_acc else {str(r["model"]): 50.0 for r in ok}
    speed_s = _norm_map(rtf_map, lower_is_better=True)
    vram_s = _norm_map(vram_map, lower_is_better=True)
    load_s = _norm_map(load_map, lower_is_better=True)

    board: list[dict[str, Any]] = []
    for r in ok:
        name = str(r["model"])
        a = acc_s.get(name, 50.0)
        sp = speed_s.get(name, 50.0)
        vr = vram_s.get(name, 50.0)
        ld = load_s.get(name, 50.0)
        # Robot priority: accuracy first, then speed, then VRAM, then cold-load.
        if has_acc:
            robot = round(0.45 * a + 0.30 * sp + 0.15 * vr + 0.10 * ld, 2)
            balanced = round(0.35 * a + 0.35 * sp + 0.20 * vr + 0.10 * ld, 2)
        else:
            robot = round(0.55 * sp + 0.30 * vr + 0.15 * ld, 2)
            balanced = robot
        board.append(
            {
                "model": name,
                "ok_runs": r.get("ok", ""),
                "success_rate_percent": r.get("success_rate_percent", ""),
                "avg_accuracy_percent": r.get("avg_accuracy_percent", ""),
                "avg_wer": r.get("avg_wer", ""),
                "avg_cer": r.get("avg_cer", ""),
                "avg_rtf": r.get("avg_rtf", ""),
                "avg_x_realtime": r.get("avg_x_realtime", ""),
                "realtime_capable_percent": r.get("realtime_capable_percent", ""),
                "avg_load_seconds": r.get("avg_load_seconds", ""),
                "avg_transcribe_seconds": r.get("avg_transcribe_seconds", ""),
                "peak_vram_mb": r.get("peak_vram_mb", ""),
                "peak_ram_mb": r.get("peak_ram_mb", ""),
                "peak_cpu_percent": r.get("peak_cpu_percent", ""),
                "score_accuracy": a if has_acc else "",
                "score_speed": sp,
                "score_vram_efficiency": vr,
                "score_load": ld,
                "score_balanced": balanced,
                "score_robot_realtime": robot,
            }
        )
    board.sort(key=lambda x: (-float(x["score_robot_realtime"]), str(x["model"])))
    for i, row in enumerate(board, 1):
        row["rank_robot"] = i
    by_acc = sorted(
        board,
        key=lambda x: (
            -float(x["avg_accuracy_percent"]) if x.get("avg_accuracy_percent") not in ("", None) else 0.0,
            float(x["avg_rtf"]) if x.get("avg_rtf") not in ("", None) else 999.0,
        ),
    )
    for i, row in enumerate(by_acc, 1):
        row["rank_accuracy"] = i
    by_speed = sorted(
        board,
        key=lambda x: (
            float(x["avg_rtf"]) if x.get("avg_rtf") not in ("", None) else 999.0,
            -float(x["avg_accuracy_percent"]) if x.get("avg_accuracy_percent") not in ("", None) else 0.0,
        ),
    )
    for i, row in enumerate(by_speed, 1):
        row["rank_speed"] = i
    return board


def build_asr_recommendations(
    leaderboard: list[dict[str, Any]],
    agg_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not leaderboard:
        return {
            "status": "no_successful_models",
            "picks": {},
            "notes": ["No successful ASR runs; cannot recommend a model."],
        }

    def pick(key: str, reverse: bool = False, prefer_present: Optional[str] = None) -> Optional[dict[str, Any]]:
        eligible = [r for r in leaderboard if r.get(key) not in ("", None)]
        if prefer_present:
            eligible = [r for r in eligible if r.get(prefer_present) not in ("", None)] or eligible
        if not eligible:
            return None
        return sorted(eligible, key=lambda r: float(r[key]), reverse=reverse)[0]

    best_robot = leaderboard[0]
    best_acc = pick("avg_accuracy_percent", reverse=True)
    best_speed = pick("avg_rtf", reverse=False)
    best_vram = pick("peak_vram_mb", reverse=False)
    best_balanced = pick("score_balanced", reverse=True)

    realtime = [
        r
        for r in leaderboard
        if r.get("avg_rtf") not in ("", None) and float(r["avg_rtf"]) < 1.0
    ]
    realtime.sort(
        key=lambda r: (
            -float(r["avg_accuracy_percent"]) if r.get("avg_accuracy_percent") not in ("", None) else 0.0,
            float(r["avg_rtf"]),
        )
    )

    notes = [
        "score_robot_realtime weights accuracy (45%) + speed/RTF (30%) + low VRAM (15%) + fast load (10%).",
        "RTF < 1.0 means transcription finishes faster than audio duration (good for near-real-time).",
        "If WER/accuracy is missing, listen-quality still needs human review of transcripts.",
        "For ESP32 robot VPS: prefer low RTF + high accuracy + VRAM that fits your GPU with headroom.",
    ]
    if not best_acc or best_acc.get("avg_accuracy_percent") in ("", None):
        notes.append("No reference-based accuracy available; rankings are speed/resource dominated.")

    return {
        "status": "ok",
        "picks": {
            "best_for_robot_realtime": {
                "model": best_robot.get("model"),
                "why": "Highest composite robot score (accuracy + speed + VRAM + load).",
                "metrics": {
                    "score_robot_realtime": best_robot.get("score_robot_realtime"),
                    "avg_accuracy_percent": best_robot.get("avg_accuracy_percent"),
                    "avg_rtf": best_robot.get("avg_rtf"),
                    "peak_vram_mb": best_robot.get("peak_vram_mb"),
                },
            },
            "best_accuracy": {
                "model": (best_acc or {}).get("model"),
                "why": "Highest average word accuracy / lowest WER.",
                "metrics": {
                    "avg_accuracy_percent": (best_acc or {}).get("avg_accuracy_percent"),
                    "avg_wer": (best_acc or {}).get("avg_wer"),
                    "avg_cer": (best_acc or {}).get("avg_cer"),
                },
            },
            "best_speed": {
                "model": (best_speed or {}).get("model"),
                "why": "Lowest average RTF (fastest relative to audio length).",
                "metrics": {
                    "avg_rtf": (best_speed or {}).get("avg_rtf"),
                    "avg_x_realtime": (best_speed or {}).get("avg_x_realtime"),
                    "avg_transcribe_seconds": (best_speed or {}).get("avg_transcribe_seconds"),
                },
            },
            "lowest_vram": {
                "model": (best_vram or {}).get("model"),
                "why": "Lowest peak VRAM — useful for 6–16 GB GPUs or multi-model co-residency.",
                "metrics": {"peak_vram_mb": (best_vram or {}).get("peak_vram_mb")},
            },
            "best_balanced": {
                "model": (best_balanced or {}).get("model"),
                "why": "Balanced accuracy/speed/VRAM tradeoff.",
                "metrics": {"score_balanced": (best_balanced or {}).get("score_balanced")},
            },
            "best_realtime_with_accuracy": {
                "model": realtime[0].get("model") if realtime else None,
                "why": "RTF < 1 and best available accuracy among realtime-capable models.",
                "metrics": {
                    "avg_rtf": realtime[0].get("avg_rtf") if realtime else None,
                    "avg_accuracy_percent": realtime[0].get("avg_accuracy_percent") if realtime else None,
                },
            },
        },
        "leaderboard_top3": leaderboard[:3],
        "model_count_ok": len(leaderboard),
        "model_count_total": len(agg_rows),
        "notes": notes,
    }


def write_asr_selection_report(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    agg_rows: list[dict[str, Any]],
    leaderboard: list[dict[str, Any]],
    recommendations: dict[str, Any],
    accuracy_ranking: list[dict[str, Any]],
) -> Path:
    ok = [r for r in rows if r.get("status") == "ok"]
    bad = [r for r in rows if r.get("status") == "error"]
    lines: list[str] = [
        "# ASR Model Selection Report",
        "",
        "Auto-generated from the Kaggle Arabic ASR bake-off. Use this with the CSVs/JSON",
        "to pick the production ASR model for the ESP32 Arabic voice robot.",
        "",
        "## Run summary",
        "",
        f"- Total runs: **{len(rows)}**",
        f"- OK: **{len(ok)}**",
        f"- Failed: **{len(bad)}**",
        f"- Models with ≥1 OK run: **{len(leaderboard)}**",
        "",
        "## Recommended picks",
        "",
    ]
    picks = recommendations.get("picks") or {}
    for key, payload in picks.items():
        if not isinstance(payload, dict):
            continue
        model = payload.get("model") or "(none)"
        why = payload.get("why") or ""
        metrics = payload.get("metrics") or {}
        metric_txt = ", ".join(f"{k}={v}" for k, v in metrics.items() if v not in ("", None))
        lines.append(f"### `{key}`")
        lines.append("")
        lines.append(f"- **Model:** `{model}`")
        lines.append(f"- **Why:** {why}")
        if metric_txt:
            lines.append(f"- **Metrics:** {metric_txt}")
        lines.append("")

    lines.extend(["## Robot realtime leaderboard", ""])
    lines.append(
        "| Rank | Model | Robot score | Acc% | WER | RTF | xRT | VRAM pk MB | Success% |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in leaderboard:
        lines.append(
            "| {rank} | `{model}` | {robot} | {acc} | {wer} | {rtf} | {xrt} | {vram} | {ok}|".format(
                rank=row.get("rank_robot", ""),
                model=row.get("model", ""),
                robot=row.get("score_robot_realtime", ""),
                acc=row.get("avg_accuracy_percent", "-"),
                wer=row.get("avg_wer", "-"),
                rtf=row.get("avg_rtf", "-"),
                xrt=row.get("avg_x_realtime", "-"),
                vram=row.get("peak_vram_mb", "-"),
                ok=row.get("success_rate_percent", "-"),
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
            f"- Accuracy: avg={r.get('avg_accuracy_percent', '-')}% "
            f"(min={r.get('min_accuracy_percent', '-')}, max={r.get('max_accuracy_percent', '-')}, "
            f"std={r.get('std_accuracy_percent', '-')})"
        )
        lines.append(
            f"- WER/CER: avg_wer={r.get('avg_wer', '-')}, avg_cer={r.get('avg_cer', '-')}"
        )
        lines.append(
            f"- Speed: avg_rtf={r.get('avg_rtf', '-')}, "
            f"min_rtf={r.get('min_rtf', '-')}, max_rtf={r.get('max_rtf', '-')}, "
            f"realtime_capable={r.get('realtime_capable_percent', '-')}%"
        )
        lines.append(
            f"- Timing: load_avg={r.get('avg_load_seconds', '-')}s, "
            f"transcribe_avg={r.get('avg_transcribe_seconds', '-')}s"
        )
        lines.append(
            f"- Resources: CPU pk={r.get('peak_cpu_percent', '-')}%, "
            f"RAM pk={r.get('peak_ram_mb', '-')}MB, "
            f"GPU pk={r.get('peak_gpu_util_percent', '-')}%, "
            f"VRAM pk={r.get('peak_vram_mb', '-')}MB, "
            f"model VRAM={r.get('model_vram_mb', '-')}MB"
        )
        lines.append("")

    if accuracy_ranking:
        lines.extend(["## Per-run accuracy ranking (top 20)", ""])
        for row in accuracy_ranking[:20]:
            lines.append(
                f"{row.get('rank')}. `{row.get('model')}` / `{row.get('audio_name')}` — "
                f"Acc={row.get('accuracy_percent')}% WER={row.get('wer')} CER={row.get('cer')} "
                f"RTF={row.get('rtf')}"
            )
        lines.append("")

    if bad:
        lines.extend(["## Failures", ""])
        for r in bad:
            lines.append(f"- `{r.get('model')}` | `{r.get('audio_name')}`: {r.get('error')}")
        lines.append("")

    lines.extend(
        [
            "## How to use these files",
            "",
            "1. Open `asr_recommendations.json` for the primary pick.",
            "2. Confirm with `asr_leaderboard.csv` (sortable in Excel/Sheets).",
            "3. Drill into `asr_analytics.csv` for every audio×model run.",
            "4. Use `asr_analytics_by_model.csv` for min/max/std stability.",
            "5. If accuracy refs exist, also check `asr_accuracy_ranking.csv`.",
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
    headers = {
        "model": "Model",
        "audio_name": "Audio",
        "status": "Status",
        "gpu_index": "GPU",
        "load_seconds": "Load(s)",
        "transcribe_seconds": "Transcribe(s)",
        "wall_seconds": "Wall(s)",
        "audio_seconds": "Audio(s)",
        "rtf": "RTF",
        "x_realtime": "xRT",
        "wer": "WER",
        "cer": "CER",
        "accuracy_percent": "Acc%",
        "char_accuracy_percent": "CharAcc%",
        "runs": "Runs",
        "ok": "OK",
        "success_rate_percent": "OK%",
        "avg_wer": "WER avg",
        "avg_cer": "CER avg",
        "avg_accuracy_percent": "Acc% avg",
        "avg_char_accuracy_percent": "CharAcc% avg",
        "avg_load_seconds": "Load avg(s)",
        "avg_transcribe_seconds": "Transcribe avg(s)",
        "avg_rtf": "RTF avg",
        "peak_cpu_percent": "CPU pk%",
        "avg_cpu_percent": "CPU avg%",
        "peak_ram_mb": "RAM pk MB",
        "avg_ram_mb": "RAM avg MB",
        "peak_gpu_util_percent": "GPU pk%",
        "avg_gpu_util_percent": "GPU avg%",
        "peak_vram_mb": "VRAM pk MB",
        "model_vram_mb": "VRAM mdl MB",
        "rank_robot": "Rank",
        "score_robot_realtime": "Robot",
        "score_balanced": "Balanced",
        "score_accuracy": "AccScore",
        "score_speed": "SpeedScore",
    }
    labels = [headers.get(c, c) for c in columns]
    table = [labels] + [
        [str(row.get(c, "")) if row.get(c, "") != "" else "-" for c in columns] for row in rows
    ]
    widths = [max(len(r[i]) for r in table) for i in range(len(columns))]
    line = "  ".join("-" * w for w in widths)
    print(f"\n{title}")
    print(line)
    for i, r in enumerate(table):
        print("  ".join(val.ljust(w) for val, w in zip(r, widths)))
        if i == 0:
            print(line)
    print(line)


def print_analytics(
    rows: list[dict[str, Any]],
    agg_rows: list[dict[str, Any]],
    leaderboard: Optional[list[dict[str, Any]]] = None,
) -> None:
    print("\n" + "=" * 72)
    print("ASR MODEL ANALYTICS")
    print("=" * 72)
    _print_table(
        "Per run: timing / accuracy (RTF = transcribe time / audio duration; lower WER is better; Acc% = (1-WER)*100)",
        [
            "model",
            "audio_name",
            "status",
            "gpu_index",
            "load_seconds",
            "transcribe_seconds",
            "wall_seconds",
            "audio_seconds",
            "rtf",
            "x_realtime",
            "wer",
            "cer",
            "accuracy_percent",
            "char_accuracy_percent",
        ],
        rows,
    )
    _print_table(
        "Per run: resources (worker process tree + physical GPU, sampled every 0.5s)",
        [
            "model",
            "audio_name",
            "peak_cpu_percent",
            "avg_cpu_percent",
            "peak_ram_mb",
            "avg_ram_mb",
            "peak_gpu_util_percent",
            "avg_gpu_util_percent",
            "peak_vram_mb",
            "model_vram_mb",
        ],
        rows,
    )
    _print_table(
        "Per model: averages / stability across all audio files",
        [
            "model",
            "runs",
            "ok",
            "success_rate_percent",
            "avg_wer",
            "avg_cer",
            "avg_accuracy_percent",
            "avg_char_accuracy_percent",
            "avg_load_seconds",
            "avg_transcribe_seconds",
            "avg_rtf",
            "peak_cpu_percent",
            "peak_ram_mb",
            "peak_gpu_util_percent",
            "peak_vram_mb",
        ],
        agg_rows,
    )
    if leaderboard:
        _print_table(
            "Leaderboard: robot realtime composite (higher score is better)",
            [
                "rank_robot",
                "model",
                "score_robot_realtime",
                "score_balanced",
                "avg_accuracy_percent",
                "avg_rtf",
                "peak_vram_mb",
                "success_rate_percent",
            ],
            leaderboard,
        )
    failed = [r for r in rows if r["status"] == "error"]
    if failed:
        print("\nFailures:")
        for r in failed:
            print(f"  - {r['model']} | {r['audio_name']}: {r['error']}")


def write_analytics_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    return write_csv_rows(path, rows, ANALYTICS_COLUMNS)


def export_asr_analytics(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write all ASR analytics artifacts used for model selection."""
    agg_rows = aggregate_by_model(rows)
    leaderboard = build_asr_leaderboard(agg_rows)
    accuracy_ranking = build_accuracy_ranking(rows)
    recommendations = build_asr_recommendations(leaderboard, agg_rows)

    paths = {
        "analytics": write_analytics_csv(rows, output_dir / "asr_analytics.csv"),
        "by_model": write_csv_rows(output_dir / "asr_analytics_by_model.csv", agg_rows),
        "leaderboard": write_csv_rows(output_dir / "asr_leaderboard.csv", leaderboard),
        "accuracy_ranking": write_csv_rows(
            output_dir / "asr_accuracy_ranking.csv", accuracy_ranking
        ),
        "recommendations": output_dir / "asr_recommendations.json",
        "report": output_dir / "asr_selection_report.md",
    }
    write_json(paths["recommendations"], recommendations)
    write_asr_selection_report(
        paths["report"],
        rows=rows,
        agg_rows=agg_rows,
        leaderboard=leaderboard,
        recommendations=recommendations,
        accuracy_ranking=accuracy_ranking,
    )
    print_analytics(rows, agg_rows, leaderboard)

    print("\n" + "=" * 72)
    print("ASR SELECTION PICKS")
    print("=" * 72)
    for key, payload in (recommendations.get("picks") or {}).items():
        if isinstance(payload, dict) and payload.get("model"):
            print(f"  {key}: {payload['model']}")

    return {
        "agg_rows": agg_rows,
        "leaderboard": leaderboard,
        "recommendations": recommendations,
        "paths": paths,
    }


def main(
    audio_paths: Optional[list[Path]] = None,
    reference_json: Optional[Path] = None,
    reference_text: Optional[Path] = None,
    only: Optional[str] = None,
    skip: Optional[str] = None,
    language: Optional[str] = "ar",
    run_install: bool = False,
) -> dict[str, Any]:
    model_names = select_models(only, skip)
    if run_install:
        install_packages(model_names)

    # Remove cache leftovers written to /kaggle/working by older runs.
    if SCRATCH_DIR != WORK_DIR:
        leftover = WORK_DIR / "asr_cache"
        if leftover.exists() and leftover.is_dir():
            print(f"Removing old {leftover} from the limited working disk...")
            shutil.rmtree(leftover, ignore_errors=True)

    audios = discover_audio(audio_paths or [])
    if not audios:
        print("No audio files found.")
        print(f"Put WAV/MP3 files in {INPUT_DIR} or pass --audio /path/to/file_or_dir")
        return {"status": "no_audio", "audio_search_dir": str(INPUT_DIR), "models": model_names}

    reference_map = load_reference_map(reference_json)
    ref_path = resolve_default_reference_text(reference_text)
    default_reference = (
        ref_path.read_text(encoding="utf-8").strip()
        if ref_path
        else EMBEDDED_REFERENCE_TEXT
    )
    reference_source = str(ref_path) if ref_path else "embedded in script"
    print(
        f"Reference text for WER/accuracy: {reference_source} "
        f"({len(default_reference.split())} words)"
    )

    n_gpu = gpu_count()
    gpus = list_gpus()
    print(f"GPUs detected: {n_gpu}")
    print(json.dumps(gpus, indent=2))
    print(f"Output (persisted): {OUTPUT_DIR}")
    print(f"Scratch for model caches (temp disk): {SCRATCH_DIR}")
    print(f"Models: {model_names}")
    print(f"Audio files ({len(audios)}):")
    for p in audios:
        print(f"  - {p}")

    results: dict[str, Any] = {
        "output_dir": str(OUTPUT_DIR),
        "transcript_dir": str(TRANSCRIPT_DIR),
        "reference_text_file": str(ref_path) if ref_path else None,
        "reference_text_source": reference_source,
        "gpu_count": n_gpu,
        "gpus": gpus,
        "language": language,
        "models": model_names,
        "audio_files": [str(p) for p in audios],
        "runs": [],
        "research_notes": {
            "QwenCleo-ASR": "Best current open result found for Egyptian Arabic and Arabic-English code-switching.",
            "Audar-ASR-V1-Turbo": "Leaderboard-strong Arabic ASR, but GGUF/llama.cpp path is not enabled by default here.",
            "Arabic Whisper FT CT2": "Production-friendly CTranslate2 int8 Arabic dialect baselines.",
        },
    }

    flat_rows: list[dict[str, Any]] = []
    rr = 0
    for audio in audios:
        reference = reference_for_audio(audio, reference_map, default_reference)
        if not reference:
            print(f"WARNING: No reference for {audio.name} — WER/accuracy will be skipped for this file.")
        for model_name in model_names:
            gpu = -1 if n_gpu <= 0 else (rr % n_gpu)
            rr += 1
            meta = run_model_isolated(model_name, audio, reference, gpu, language)
            results["runs"].append(meta)
            flat_rows.append(meta)
            if meta.get("status") == "ok":
                res = meta.get("resources", {}) or {}
                if "wer" in meta:
                    acc = meta.get("accuracy_percent", max(0.0, (1.0 - float(meta["wer"])) * 100.0))
                    wer_part = (
                        f" WER={meta['wer']:.3f} Acc={acc:.1f}%"
                        f" CER={meta.get('cer', float('nan')):.3f}"
                    )
                else:
                    wer_part = " (no reference — accuracy not scored)"
                rtf_part = f" rtf={meta['realtime_factor']:.2f}" if "realtime_factor" in meta else ""
                print(f"OK {model_name} | {audio.name}{wer_part}{rtf_part}")
                print(
                    f"   cpu_peak={res.get('peak_cpu_percent', '-')}%  "
                    f"ram_peak={res.get('peak_ram_mb', '-')}MB  "
                    f"gpu_peak={res.get('peak_gpu_util_percent', '-')}%  "
                    f"vram_peak={res.get('peak_vram_mb', '-')}MB"
                )
            else:
                print(f"FAIL {model_name} | {audio.name}: {meta.get('error')}")

    summary_path = OUTPUT_DIR / "summary.json"
    csv_path = OUTPUT_DIR / "summary.csv"
    write_json(summary_path, results)
    write_summary_csv(csv_path, flat_rows)

    analytics_rows = build_analytics_rows(flat_rows)
    exported = export_asr_analytics(OUTPUT_DIR, analytics_rows)
    paths = exported["paths"]

    ok = [r for r in flat_rows if r.get("status") == "ok"]
    bad = [r for r in flat_rows if r.get("status") == "error"]
    print(f"\nDone. Summary JSON: {summary_path}")
    print(f"Summary CSV: {csv_path}")
    print(f"Analytics CSV: {paths['analytics']}")
    print(f"Per-model analytics CSV: {paths['by_model']}")
    print(f"Leaderboard CSV: {paths['leaderboard']}")
    print(f"Accuracy ranking CSV: {paths['accuracy_ranking']}")
    print(f"Recommendations JSON: {paths['recommendations']}")
    print(f"Selection report MD: {paths['report']}")
    print(f"OK runs: {len(ok)} / {len(flat_rows)}")
    if bad:
        print("Failed runs:")
        for row in bad:
            print(f"  - {row.get('model')} | {row.get('audio_name')}: {row.get('error')}")
    results["analytics"] = {
        "leaderboard": exported["leaderboard"],
        "recommendations": exported["recommendations"],
        "files": {k: str(v) for k, v in paths.items()},
    }
    write_json(summary_path, results)
    return results


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--audio", action="append", type=Path, default=[])
    parser.add_argument("--audio-file", type=Path)
    parser.add_argument("--reference-file", type=Path)
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument(
        "--reference-text",
        type=Path,
        help="Optional file overriding the embedded reference used for WER/accuracy",
    )
    parser.add_argument("--meta", type=Path)
    parser.add_argument("--transcript-file", type=Path)
    parser.add_argument("--language", default="ar")
    parser.add_argument("--only", help="Comma-separated model names to run")
    parser.add_argument("--skip", help="Comma-separated model names to skip")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--no-install", action="store_true")
    args, _unknown = parser.parse_known_args(argv)
    return args


if __name__ == "__main__":
    args = _parse_args()
    if args.worker:
        raise SystemExit(
            worker_main(
                model_name=args.model,
                audio_file=args.audio_file,
                reference_file=args.reference_file,
                meta_file=args.meta,
                transcript_file=args.transcript_file,
                language=args.language,
            )
        )
    on_kaggle = Path("/kaggle/working").exists()
    auto_install = on_kaggle and os.environ.get("AUTO_INSTALL", "1") != "0"
    main(
        audio_paths=args.audio,
        reference_json=args.reference_json,
        reference_text=args.reference_text,
        only=args.only,
        skip=args.skip,
        language=args.language,
        run_install=(args.install or os.environ.get("RUN_INSTALL", "0") == "1" or auto_install) and not args.no_install,
    )
