#!/usr/bin/env python3
"""
Kaggle Arabic TTS listening comparison (T4 / T4x2 safe)
=======================================================
Each model runs in its own subprocess with a dedicated GPU so a CUDA
assert in one model cannot poison the rest of the notebook session.

Usage on Kaggle:
  1) Runtime → Restart session (important if CUDA was already poisoned)
  2) Upload this .py to /kaggle/working/ (preferred), then:
       %run /kaggle/working/kaggle_arabic_tts_compare.py
     Or paste into a cell and call main() — workers auto-save the cell to disk.
  3) Run it. On Kaggle, plain %run auto-installs missing packages:
       %run /kaggle/working/kaggle_arabic_tts_compare.py
     For repeat runs after packages are installed:
       %run /kaggle/working/kaggle_arabic_tts_compare.py --no-install
  4) Download /kaggle/working/tts_outputs/*.wav
  5) Detailed analytics are printed and saved under /kaggle/working/tts_outputs/:
       tts_analytics.csv
       tts_analytics_by_model.csv
       tts_leaderboard.csv
       tts_recommendations.json
       tts_selection_report.md
     (load/gen time, RTF, throughput, CPU/RAM/GPU/VRAM, rankings, robot picks)
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
import urllib.request
import wave
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Paths / text
# ---------------------------------------------------------------------------

WORK_DIR = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
if not WORK_DIR.exists():
    WORK_DIR = Path.cwd() / "kaggle_working"
WORK_DIR.mkdir(parents=True, exist_ok=True)


def _scratch_dir() -> Path:
    """Big ephemeral disk for repos/model caches.

    /kaggle/working is limited to ~19 GB and persisted as notebook output, so
    only results should live there. Cloned repos and downloaded checkpoints go
    to the temp disk instead (not persisted, but much larger).
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

OUTPUT_DIR = WORK_DIR / "tts_outputs"       # results → persisted notebook output
META_DIR = WORK_DIR / "tts_meta"            # small JSON metadata → persisted
CACHE_DIR = SCRATCH_DIR / "tts_cache"       # HF checkpoints → big temp disk
REPOS_DIR = SCRATCH_DIR / "repos"           # git clones → big temp disk
for d in (OUTPUT_DIR, CACHE_DIR, REPOS_DIR, META_DIR):
    d.mkdir(parents=True, exist_ok=True)

DEFAULT_TEXT = (
    "السلام عليكم يا روبوت، صباح الخير. أنا محتاج أعمل اختبار لجودة تحويل "
    "النص إلى كلام باللهجة المصرية، عشان أقدر أقارن بين أكتر من موديل. "
    "فكرني إن عندي اجتماع بكرة الساعة عشرة ونص مع فريق التطوير، وبعدها "
    "Presentation الساعة اتنين، وبعدها Call مع العميل الساعة خمسة. "
    "كمان محتاجك تراجع معاي الـ dashboard، وتشوف الـ API response time، "
    "لازم يكون أقل من ميتين ملي ثانية. درجة الحرارة حوالي سبعة وثلاثين، "
    "وسعر الدولار حوالي خمسين جنيه. لو حصل delay في التقرير، ابعتلي "
    "notification قبل الساعة تلاتة ونص، وقولي لو في أي risk على الـ deadline. "
    "Please open the dashboard and send the report to the customer, "
    "and remind me again before the meeting by half an hour. "
    "وبالمناسبة، لو العميل سأل عن الـ pricing، قول له إن العرض سارٍ لمدة أسبوعين، "
    "وإن في خصم للعقد السنوي. شكراً ليك، وياريت الصوت يبقى واضح وطبيعي. "
    "كمان ممكن تضيف ملاحظة عن جدول الأعمال الإسبوعي، بحيث أعرف أولوية "
    "المهام الأساسيّة، وعايز قائمة تفصيلية بالـ KPIs المرتبطة بالمشروع الحالي. "
    "لو في أي changes في المواصفات، ابعتلي نسخة بالتحديثات فوراً على البريد الإلكتروني. "
    "عايز أعرف آخر status للـ deployment وتأكدلي إن الباك أب اتعمل الليلة الماضية. "
    "لو فيه مشاكل في التكامل مع الواجهات البرمجية، حاول تراجع اللوجات وترسللي ملخص بالأخطاء الشائعة. "
    "بالنسبة للعقود الجديدة، راجع مع قسم المالية لو فيه أي ملاحظات على العروض، وأكد مع العميل ميعاد توقيع العقد. "
    "لو احتاجنا نتواصل مع الدعم الفني، جهزلي كل التفاصيل، وسجل أي تذاكر technical تم فتحها مؤخراً. "
    "راجع أيضاً نسبة رضا العميل من خلال survey الأخير، ولو وصلت أي شكاوى لازم تبلغني على طول. "
    "أنا مهتم يكون الصوت طبيعي وثابت حتى مع جمل طويلة وزحام معلومات، ولو ممكن تضيف بعض intonation لتحسين السماع. "
    "وأخيراً، بعد انتهاء كل المهام المذكورة، ابعتلي تقرير نهائي يلخص العمل الأسبوعي وتوصيات للتحسين القادم."
)

ENABLE = {
    "VoiceTut-TTS": True,
    "NAMAA-Egyptian-TTS": True,
    "Chatterbox-Multilingual-V3": True,
    "SILMA-TTS": True,
    # "Kokoro-82M": True,
    # "Qwen3-TTS-0.6B": True,
    # "CosyVoice-0.5B": True,
    # "Fish-Speech": True,
}

# Order: Arabic refs first, then models that need a reference / can crash CUDA.
MODEL_ORDER = [
    "NAMAA-Egyptian-TTS",
    "Chatterbox-Multilingual-V3",
    "VoiceTut-TTS",
    "SILMA-TTS",
    # "Kokoro-82M",
    # "Qwen3-TTS-0.6B",
    # "CosyVoice-0.5B",
    # "Fish-Speech",
]

VOICETUT_REPO = "mohammedaly22/VoiceTut-TTS"
VOICETUT_SPEAKER = "Mohamed"
VOICETUT_CHUNK_CHARS = 180
QWEN3_BASE = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
SILMA_TTS_HF = "silma-ai/silma-tts"
SILMA_REF_URL = (
    "https://raw.githubusercontent.com/SILMA-AI/silma-tts/main/"
    "src/silma_tts/infer/ref_audio_samples/ar.ref.24k.wav"
)
SILMA_REF_TEXT = (
    "ويدقق النظر في القرآن الكريم وسائر الكتب السماوية "
    "ويتبع مسالك الرسل العظام عليهم الصلاة والسلام."
)
COSYVOICE_HF = "FunAudioLLM/CosyVoice2-0.5B"
FISH_SPEECH_HF = "fishaudio/fish-speech-1.5"


# ===========================================================================
# INSTALL
# ===========================================================================

INSTALL_BLOCK = r'''
import subprocess, sys

def _pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

subprocess.check_call(["apt-get", "update", "-qq"])
subprocess.check_call([
    "apt-get", "install", "-y", "-qq",
    "espeak-ng", "sox", "libsox-dev", "ffmpeg", "git", "git-lfs",
])

_pip("soundfile", "torchaudio", "librosa", "numpy", "huggingface_hub", "safetensors", "omegaconf", "einops")

# VoiceTut needs transformers>=5.3 for HiggsAudioV2TokenizerModel
_pip("-U", "transformers>=5.3.0")
_pip("omnivoice", "voicetut-tts")
_pip("-U", "transformers>=5.3.0")  # re-assert after omnivoice deps

_pip("--no-deps", "chatterbox-tts")
_pip(
    "resemble-perth", "s3tokenizer", "conformer==0.3.2", "diffusers==0.29.0",
    "safetensors>=0.5.3", "pykakasi==2.3.0", "spacy-pkuseg", "pyloudnorm",
    "omegaconf", "librosa==0.11.0",
)
_pip("-U", "qwen-tts")
_pip("kokoro>=0.9.4")
# Versions before 1.0.4 did not include the packaged Arabic reference WAV.
_pip("-U", "silma-tts>=1.0.4")
_pip("HyperPyYAML", "wetext", "modelscope", "pyarrow", "openai-whisper", "onnxruntime")
_pip("pyrootutils", "loguru", "lightning", "hydra-core", "tiktoken", "vector_quantize_pytorch")

# Analytics dependencies (CPU/RAM/GPU sampling)
_pip("psutil", "nvidia-ml-py")

# Keep transformers new enough for VoiceTut after all other installs.
_pip("-U", "transformers>=5.3.0")

def _import_ok(mod):
    return subprocess.run([sys.executable, "-c", f"import {mod}"], capture_output=True).returncode == 0

# Fix onnxruntime: Kaggle's onnxruntime-gpu wheel links libcudart.so.13 (CUDA 13),
# missing on T4 images (CUDA 12.x). Replace with the CPU wheel — TTS models only
# use ORT for small tokenizer / x-vector nets, so CPU is fine.
if not _import_ok("onnxruntime"):
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "onnxruntime", "onnxruntime-gpu"])
    try:
        _pip("--no-cache-dir", "onnxruntime==1.20.1")
    except Exception:
        _pip("--no-cache-dir", "onnxruntime")
    print("onnxruntime CPU reinstall:", "OK" if _import_ok("onnxruntime") else "STILL BROKEN")

# Fix pynini (SILMA/NeMo): corrupted bundled OpenFst .so ("cannot read file data").
if not _import_ok("pynini"):
    try:
        _pip("--force-reinstall", "--no-cache-dir", "pynini==2.1.6.post1")
    except Exception:
        _pip("--force-reinstall", "--no-cache-dir", "pynini")
    print("pynini force reinstall:", "OK" if _import_ok("pynini") else "STILL BROKEN")

print("Packages installed. transformers>=5.3.0 pinned for VoiceTut.")
'''


def install_packages() -> None:
    print("Installing packages for Kaggle Arabic TTS compare…")
    exec(INSTALL_BLOCK, {"__name__": "__main__"})  # noqa: S102


# ===========================================================================
# Helpers
# ===========================================================================

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


def pick_gpu(preferred: Optional[int] = None) -> int:
    """Pick a GPU index. On T4x2 prefers the freest card."""
    n = gpu_count()
    if n <= 0:
        return -1
    if preferred is not None and 0 <= preferred < n:
        return preferred
    best_i, best_free = 0, -1
    try:
        import torch

        for i in range(n):
            free, _total = torch.cuda.mem_get_info(i)
            if free > best_free:
                best_free = free
                best_i = i
    except Exception:
        best_i = 0
    return best_i


def cleanup_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def save_wav(path: Path, audio: Any, sample_rate: int) -> None:
    import numpy as np

    wav = np.asarray(audio)
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / peak
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf

        sf.write(str(path), wav, int(sample_rate))
        return
    except Exception:
        pass
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes((wav * 32767.0).astype("int16").tobytes())


def split_text_chunks(text: str, max_chars: int = VOICETUT_CHUNK_CHARS) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return [text]
    parts = re.split(r"(?<=[\.!\?؟،,;؛])\s+", text)
    chunks: list[str] = []
    buf = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if not buf:
            buf = part
        elif len(buf) + 1 + len(part) <= max_chars:
            buf = f"{buf} {part}"
        else:
            chunks.append(buf)
            buf = part
    if buf:
        chunks.append(buf)
    final: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final.append(chunk)
        else:
            for i in range(0, len(chunk), max_chars):
                final.append(chunk[i : i + max_chars])
    return final or [text]


def run_git_clone(url: str, dest: Path, recursive: bool = False, branch: Optional[str] = None) -> Path:
    if dest.exists() and any(dest.iterdir()):
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    cmd = ["git", "clone", "--depth", "1"]
    if branch:
        cmd += ["--branch", branch]
    if recursive:
        cmd.append("--recursive")
    cmd.extend([url, str(dest)])
    subprocess.check_call(cmd)
    return dest


def trim_wav_head(src: Path, dest: Path, max_seconds: float) -> Path:
    """Copy the first max_seconds of a WAV (models like CosyVoice cap ref length)."""
    import soundfile as sf

    data, sr = sf.read(str(src), dtype="float32")
    limit = int(max_seconds * sr)
    if data.shape[0] <= limit:
        return src
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), data[:limit], sr)
    return dest


def _pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])


def ensure_module(module_name: str, *packages: str, upgrade: bool = False, no_deps: bool = False) -> None:
    """Install packages lazily in Kaggle workers when the setup cell was skipped."""
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
        raise ModuleNotFoundError(
            f"No module named {module_name!r} after installing {' '.join(packages)}. "
            "Restart the Kaggle session and run main(run_install=True)."
        )


def _import_works(module_name: str) -> bool:
    """Check importability in a clean subprocess (broken .so files only fail at import)."""
    return (
        subprocess.run(
            [sys.executable, "-c", f"import {module_name}"],
            capture_output=True,
        ).returncode
        == 0
    )


def ensure_onnxruntime_cpu() -> None:
    """Kaggle ships onnxruntime-gpu linked against libcudart.so.13 (CUDA 13),
    which is absent on T4 images (CUDA 12.x). Swap in the CPU wheel."""
    if _import_works("onnxruntime"):
        return
    print("onnxruntime broken (libcudart.so.13) — reinstalling CPU wheel…")
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "onnxruntime", "onnxruntime-gpu"],
        capture_output=True,
    )
    try:
        _pip("--no-cache-dir", "onnxruntime==1.20.1")
    except Exception:
        _pip("--no-cache-dir", "onnxruntime")
    if not _import_works("onnxruntime"):
        raise ImportError("onnxruntime still broken after CPU wheel reinstall")


def ensure_pynini() -> None:
    """SILMA → NeMo text normalization needs pynini; the preinstalled wheel's
    bundled OpenFst .so can be corrupted ('cannot read file data')."""
    if _import_works("pynini"):
        return
    print("pynini broken — force reinstalling…")
    try:
        _pip("--force-reinstall", "--no-cache-dir", "pynini==2.1.6.post1")
    except Exception:
        _pip("--force-reinstall", "--no-cache-dir", "pynini")
    if not _import_works("pynini"):
        raise ImportError("pynini still broken after force reinstall")


def ensure_chatterbox() -> None:
    if importlib.util.find_spec("chatterbox") is not None:
        return
    try:
        ensure_module("chatterbox", "chatterbox-tts")
    except Exception:
        _pip("--no-deps", "chatterbox-tts")
        _pip(
            "resemble-perth",
            "s3tokenizer",
            "conformer==0.3.2",
            "diffusers==0.29.0",
            "safetensors>=0.5.3",
            "pykakasi==2.3.0",
            "spacy-pkuseg",
            "pyloudnorm",
            "omegaconf",
            "librosa==0.11.0",
        )
    if importlib.util.find_spec("chatterbox") is None:
        raise ModuleNotFoundError(
            "No module named 'chatterbox'. Restart Kaggle, run install_packages(), then main()."
        )


def ensure_transformers_for_voicetut() -> None:
    """omnivoice needs HiggsAudioV2TokenizerModel (transformers>=5.3)."""
    ensure_module("voicetut_tts", "omnivoice", "voicetut-tts")
    try:
        from transformers import HiggsAudioV2TokenizerModel  # noqa: F401

        return
    except Exception:
        pass
    print("Upgrading transformers>=5.3.0 for VoiceTut…")
    _pip("-U", "transformers>=5.3.0")
    from transformers import HiggsAudioV2TokenizerModel  # noqa: F401


def pick_ref_audio() -> Optional[Path]:
    for name in (
        "VoiceTut-TTS.wav",
        "NAMAA-Egyptian-TTS.wav",
        "Chatterbox-Multilingual-V3.wav",
        "SILMA-TTS.wav",
        "CosyVoice-0.5B.wav",
        "Fish-Speech.wav",
        "Kokoro-82M.wav",
    ):
        p = OUTPUT_DIR / name
        if p.exists() and p.stat().st_size > 44_000:
            return p
    return None


def get_silma_reference_audio() -> Path:
    """Resolve SILMA's official Arabic reference, downloading it if necessary."""
    spec = importlib.util.find_spec("silma_tts")
    if spec and spec.origin:
        package_dir = Path(spec.origin).resolve().parent
        exact = list(package_dir.rglob("ar.ref.24k.wav"))
        if exact:
            return exact[0]

    cached = CACHE_DIR / "ar.ref.24k.wav"
    if cached.exists() and cached.stat().st_size > 44_000:
        return cached

    print("SILMA reference WAV is absent from the installed wheel; downloading the official sample…")
    tmp = cached.with_suffix(".download")
    try:
        with urllib.request.urlopen(SILMA_REF_URL, timeout=60) as response, tmp.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        if tmp.stat().st_size <= 44_000:
            raise RuntimeError(f"downloaded file is unexpectedly small ({tmp.stat().st_size} bytes)")
        tmp.replace(cached)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return cached


def write_meta(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def wav_stats(path: Path) -> dict[str, Any]:
    """Duration / sample-rate / channels of a WAV (soundfile first, wave fallback)."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return {
            "audio_seconds": round(float(info.duration), 3),
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
        }
    except Exception:
        pass
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            return {
                "audio_seconds": round(frames / float(rate), 3) if rate else 0.0,
                "sample_rate": int(rate),
                "channels": int(handle.getnchannels()),
            }
    except Exception:
        return {"audio_seconds": 0.0, "sample_rate": 0, "channels": 0}


# ===========================================================================
# Resource monitoring (CPU / RAM / GPU sampled in a background thread)
# ===========================================================================

class ResourceMonitor:
    """Samples process-tree CPU%, RSS RAM, and GPU util/VRAM while a model runs.

    Runs inside the worker process. GPU stats come from NVML for the physical
    GPU selected via CUDA_VISIBLE_DEVICES, so nested subprocesses (e.g.
    fish-speech CLI calls) are captured too.
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


# ===========================================================================
# Model runners (executed inside isolated worker processes)
# ===========================================================================

def synth_voicetut(text: str, out_path: Path) -> dict[str, Any]:
    ensure_transformers_for_voicetut()
    from huggingface_hub import snapshot_download
    from voicetut_tts import VoiceTutTTS
    import numpy as np
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = "float16" if device.startswith("cuda") else "float32"
    t0 = time.perf_counter()
    local = snapshot_download(VOICETUT_REPO, cache_dir=str(CACHE_DIR / "hf"))
    refs = Path(local) / "reference_speakers" / "references.json"
    model = None
    last_err: Optional[Exception] = None
    for kwargs in (
        {"device": device, "dtype": dtype, "local_files_only": True, "attn_implementation": "sdpa"},
        {"device": device, "dtype": dtype, "local_files_only": True},
        {"device": device, "dtype": dtype, "local_files_only": False},
    ):
        try:
            load_kwargs = dict(kwargs)
            if refs.exists():
                load_kwargs["references"] = str(refs)
            try:
                model = VoiceTutTTS.from_pretrained(local, **load_kwargs)
            except TypeError:
                load_kwargs.pop("attn_implementation", None)
                model = VoiceTutTTS.from_pretrained(local, **load_kwargs)
            break
        except Exception as exc:
            last_err = exc
            model = None
    if model is None:
        model = VoiceTutTTS.from_pretrained(VOICETUT_REPO, device=device, dtype=dtype)
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    chunks = split_text_chunks(text, VOICETUT_CHUNK_CHARS)
    if len(chunks) == 1:
        model.synthesize(text, output=str(out_path), speaker=VOICETUT_SPEAKER, num_step=12, speed=1.0)
    else:
        import soundfile as sf

        pieces = []
        sr = 24000
        for i, chunk in enumerate(chunks):
            tmp = out_path.with_name(f"_vt_chunk_{i}.wav")
            model.synthesize(chunk, output=str(tmp), speaker=VOICETUT_SPEAKER, num_step=12, speed=1.0)
            audio, sr = sf.read(str(tmp), dtype="float32")
            pieces.append(np.asarray(audio, dtype=np.float32).reshape(-1))
            pieces.append(np.zeros(int(sr * 0.08), dtype=np.float32))
            tmp.unlink(missing_ok=True)
        save_wav(out_path, np.concatenate(pieces), sr)
    gen_s = time.perf_counter() - t1
    return {"load_seconds": load_s, "generation_seconds": gen_s, "note": f"speaker={VOICETUT_SPEAKER}; chunks={len(chunks)}"}


def synth_namaa(text: str, out_path: Path) -> dict[str, Any]:
    ensure_chatterbox()
    import torch
    from chatterbox import mtl_tts
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file as load_safetensors

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    ckpt_dir = snapshot_download(
        "NAMAA-Space/NAMAA-Egyptian-TTS",
        repo_type="model",
        cache_dir=str(CACHE_DIR / "hf"),
    )
    weights = Path(ckpt_dir) / "t3_mtl23ls_v2.safetensors"
    model = mtl_tts.ChatterboxMultilingualTTS.from_pretrained(device=device)
    map_device = "cuda:0" if device == "cuda" else "cpu"
    model.t3.load_state_dict(load_safetensors(str(weights), device=map_device))
    model.t3.to(device).eval()
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    chunks = split_text_chunks(text, 220)
    pieces = []
    with torch.inference_mode():
        for chunk in chunks:
            pieces.append(model.generate(chunk, language_id="ar").cpu())
    wav = torch.cat(pieces, dim=-1) if len(pieces) > 1 else pieces[0]
    try:
        import torchaudio as ta

        ta.save(str(out_path), wav, model.sr)
    except Exception:
        save_wav(out_path, wav, model.sr)
    gen_s = time.perf_counter() - t1
    return {"load_seconds": load_s, "generation_seconds": gen_s, "note": f"chunks={len(chunks)}"}


def synth_chatterbox_v3(text: str, out_path: Path) -> dict[str, Any]:
    ensure_chatterbox()
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    try:
        model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
        note = "t3_model=v3"
    except TypeError:
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        note = "default multilingual"
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    chunks = split_text_chunks(text, 220)
    pieces = []
    with torch.inference_mode():
        for chunk in chunks:
            pieces.append(model.generate(chunk, language_id="ar").cpu())
    wav = torch.cat(pieces, dim=-1) if len(pieces) > 1 else pieces[0]
    try:
        import torchaudio as ta

        ta.save(str(out_path), wav, model.sr)
    except Exception:
        save_wav(out_path, wav, model.sr)
    gen_s = time.perf_counter() - t1
    return {"load_seconds": load_s, "generation_seconds": gen_s, "note": f"{note}; chunks={len(chunks)}"}


def _purge_modules(*prefixes: str) -> None:
    for name in list(sys.modules):
        if any(name == p or name.startswith(p + ".") for p in prefixes):
            del sys.modules[name]


def _patch_qwen_tts_decorator() -> None:
    """qwen-tts uses `@check_model_inputs()` which breaks on transformers where
    check_model_inputs is a plain decorator (no factory call). Neutralize it."""
    spec = importlib.util.find_spec("qwen_tts")
    if not (spec and spec.origin):
        return
    pkg_dir = Path(spec.origin).resolve().parent
    for py in pkg_dir.rglob("modeling_*.py"):
        src = py.read_text(encoding="utf-8")
        patched = src.replace("@check_model_inputs()", "@check_model_inputs") if "@check_model_inputs()" in src else src
        if "from transformers.utils.generic import check_model_inputs" in patched:
            patched = patched.replace(
                "from transformers.utils.generic import check_model_inputs",
                "try:\n"
                "    from transformers.utils.generic import check_model_inputs\n"
                "except ImportError:\n"
                "    def check_model_inputs(func=None, **kwargs):\n"
                "        if func is None:\n"
                "            return lambda f: f\n"
                "        return func",
            )
        if patched != src:
            py.write_text(patched, encoding="utf-8")
            print(f"Patched qwen_tts decorator in {py.name}")


def synth_qwen3(text: str, out_path: Path, ref_audio: Optional[Path] = None) -> dict[str, Any]:
    """
    Skip CustomVoice for Arabic — it triggers CUDA device asserts.
    Use Base 0.6B voice-clone with greedy decoding + a prior Arabic ref clip.
    """
    ensure_onnxruntime_cpu()
    ensure_module("qwen_tts", "qwen-tts", upgrade=True)

    # qwen-tts 0.1.x pins transformers==4.57.3; the install block re-pins 5.x for
    # VoiceTut. Downgrade BEFORE the first transformers import in this worker —
    # a post-import pip install is useless because 5.x stays in sys.modules.
    try:
        import transformers  # noqa: F401  (only to read the version)

        needs_pin = not transformers.__version__.startswith("4.57.")
    except Exception:
        needs_pin = True
    if needs_pin:
        print("Pinning transformers==4.57.3 for qwen-tts…")
        _pip("transformers==4.57.3")
        _purge_modules("transformers", "qwen_tts")

    import torch
    import soundfile as sf
    from huggingface_hub import snapshot_download

    try:
        from qwen_tts import Qwen3TTSModel
    except Exception:
        # Last resort: neutralize the @check_model_inputs() decorator qwen-tts
        # uses, which is incompatible with some transformers releases.
        print("qwen_tts import failed — patching check_model_inputs decorator and retrying…")
        _patch_qwen_tts_decorator()
        _purge_modules("qwen_tts")
        from qwen_tts import Qwen3TTSModel

    using_silma_ref = ref_audio is None or not Path(ref_audio).exists()
    if using_silma_ref:
        ref_audio = get_silma_reference_audio()
    # Keep the clone ref short so it stays aligned with ref_text (first chunk).
    ref_audio = trim_wav_head(Path(ref_audio), out_path.with_name("_qwen_ref.wav"), 12.0)

    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    # The code predictor is a small AR module trained in bf16 and numerically
    # unstable in fp16 (NaN probs -> CUDA device assert in multinomial); Qwen
    # recommends bf16 on Ampere+ and fp32 everywhere else (e.g. Kaggle T4/P100).
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    # transformers 4.57.3 probes the optional remote
    # `additional_chat_templates/` directory while loading a processor. Recent
    # huggingface_hub versions raise on that directory's normal 404 instead of
    # treating it as empty. A local snapshot avoids the incompatible API path.
    model_dir = snapshot_download(
        repo_id=QWEN3_BASE,
        cache_dir=str(CACHE_DIR / "qwen3_tts"),
    )
    t0 = time.perf_counter()
    try:
        model = Qwen3TTSModel.from_pretrained(
            model_dir,
            device_map=device_map,
            dtype=dtype,
            attn_implementation="sdpa",
        )
    except Exception:
        model = Qwen3TTSModel.from_pretrained(
            model_dir,
            device_map=device_map,
            dtype=dtype,
        )
    load_s = time.perf_counter() - t0

    ref_text = SILMA_REF_TEXT if using_silma_ref else split_text_chunks(text, 120)[0]
    t1 = time.perf_counter()
    # Greedy / no sampling avoids the nan-probability CUDA assert on unsupported langs.
    wavs, sr = model.generate_voice_clone(
        text=text,
        language="Auto",
        ref_audio=str(ref_audio),
        ref_text=ref_text,
        max_new_tokens=4096,
        do_sample=False,
    )
    sf.write(str(out_path), wavs[0], sr)
    gen_s = time.perf_counter() - t1
    return {
        "load_seconds": load_s,
        "generation_seconds": gen_s,
        "note": f"Base voice-clone from {Path(ref_audio).name}; do_sample=False",
    }


def synth_kokoro(text: str, out_path: Path) -> dict[str, Any]:
    ensure_module("kokoro", "kokoro>=0.9.4")
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    t0 = time.perf_counter()
    pipeline = KPipeline(lang_code="a")
    load_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    chunks = [np.asarray(audio, dtype=np.float32) for _gs, _ps, audio in pipeline(text, voice="af_heart", speed=1.0)]
    if not chunks:
        raise RuntimeError("Kokoro produced no audio")
    sf.write(str(out_path), np.concatenate(chunks), 24000)
    gen_s = time.perf_counter() - t1
    return {
        "load_seconds": load_s,
        "generation_seconds": gen_s,
        "note": "speed baseline only — no native Arabic",
    }


def synth_silma(text: str, out_path: Path, ref_audio: Optional[Path] = None) -> dict[str, Any]:
    ensure_module("silma_tts", "silma-tts")
    ensure_pynini()  # NeMo text normalization dies on Kaggle's corrupted pynini .so
    import numpy as np
    import soundfile as sf
    from silma_tts.api import SilmaTTS

    # Follow SILMA's official model-card example exactly: use its Arabic sample
    # with the matching transcript. Reusing another model's generated WAV and
    # auto-transcribing it can condition SILMA on an incorrect/non-Arabic prompt.
    ref_file = get_silma_reference_audio()
    ref_text = SILMA_REF_TEXT

    t0 = time.perf_counter()
    # The current SILMA API multiplies speed by 1.3 once for every internal
    # Arabic chunk when force_tashkeel=True. That compounds badly for long text
    # (1.3 ** N) and turns speech into unintelligible audio. The model card says
    # unvowelled Arabic is supported, so disable that buggy optional path.
    silma_tts = SilmaTTS(force_tashkeel=False)
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    pieces = []
    sr = 24000
    chunks = split_text_chunks(text, 260)
    for i, chunk in enumerate(chunks):
        tmp = out_path if len(chunks) == 1 else out_path.with_name(f"_silma_chunk_{i}.wav")
        wav, sr, _spec = silma_tts.infer(
            ref_file=str(ref_file),
            ref_text=ref_text,
            gen_text=chunk,
            file_wave=str(tmp),
            seed=None,
            speed=1,
            force_tashkeel=False,
        )
        if len(chunks) > 1:
            audio, sr = sf.read(str(tmp), dtype="float32")
            pieces.append(np.asarray(audio if audio.size else wav, dtype=np.float32).reshape(-1))
            pieces.append(np.zeros(int(sr * 0.08), dtype=np.float32))
            tmp.unlink(missing_ok=True)
    if pieces:
        save_wav(out_path, np.concatenate(pieces), sr)
    gen_s = time.perf_counter() - t1
    return {
        "load_seconds": load_s,
        "generation_seconds": gen_s,
        "note": (
            f"{SILMA_TTS_HF}; official Arabic ref={ref_file.name}; "
            f"force_tashkeel=False; chunks={len(chunks)}"
        ),
    }


def synth_cosyvoice(text: str, out_path: Path, prompt_wav: Optional[Path] = None) -> dict[str, Any]:
    import torch
    import torchaudio
    from huggingface_hub import snapshot_download

    repo = run_git_clone(
        "https://github.com/FunAudioLLM/CosyVoice.git",
        REPOS_DIR / "CosyVoice",
        recursive=True,
    )
    # Avoid full requirements.txt (breaks transformers / builds pyworld from source).
    for pkg in (
        "HyperPyYAML",
        "onnxruntime",
        "openai-whisper",
        "wetext",
        "modelscope",
        "pyarrow",
        "gdown",
        "wget",
        "fastapi",
        "uvicorn",
        "gradio",
        "conformer",
        "diffusers",
        "lightning",
        "hydra-core",
    ):
        try:
            _pip(pkg)
        except Exception:
            pass
    try:
        _pip("pyworld")
    except Exception:
        print("pyworld optional install failed; continuing")
    ensure_onnxruntime_cpu()  # CosyVoice frontend imports onnxruntime

    model_dir = CACHE_DIR / "CosyVoice2-0.5B"
    if not model_dir.exists() or not any(model_dir.iterdir()):
        snapshot_download(COSYVOICE_HF, local_dir=str(model_dir))

    matcha = repo / "third_party" / "Matcha-TTS"
    if not matcha.exists():
        subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"], cwd=str(repo))

    sys.path = [str(repo), str(matcha)] + [p for p in sys.path if p not in (str(repo), str(matcha))]
    from cosyvoice.cli.cosyvoice import AutoModel

    default_prompt = repo / "asset" / "zero_shot_prompt.wav"
    prompt = Path(prompt_wav) if prompt_wav and Path(prompt_wav).exists() else default_prompt
    if not prompt.exists():
        raise FileNotFoundError(f"CosyVoice prompt missing: {prompt}")
    # CosyVoice asserts ref audio <= 30s at 16kHz; keep headroom.
    prompt = trim_wav_head(prompt, out_path.with_name("_cosy_prompt.wav"), 25.0)

    t0 = time.perf_counter()
    cosy = AutoModel(model_dir=str(model_dir))
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    pieces = []
    for chunk in split_text_chunks(text, 180):
        for item in cosy.inference_cross_lingual(chunk, str(prompt), stream=False):
            pieces.append(item["tts_speech"])
    if not pieces:
        raise RuntimeError("CosyVoice produced no audio")
    wav = torch.cat(pieces, dim=1) if len(pieces) > 1 else pieces[0]
    torchaudio.save(str(out_path), wav.cpu(), cosy.sample_rate)
    gen_s = time.perf_counter() - t1
    return {
        "load_seconds": load_s,
        "generation_seconds": gen_s,
        "note": f"CosyVoice2-0.5B cross_lingual prompt={prompt.name}",
    }


def synth_fish_speech(text: str, out_path: Path, ref_audio: Optional[Path] = None) -> dict[str, Any]:
    """fish-speech-1.5 only (openaudio-s1-mini is gated without HF access).

    The repo MUST be at the v1.5.0 tag: main-branch code ships the new DAC
    codec / dual_ar arch which cannot load the 1.5 firefly VQ-GAN checkpoint.
    """
    from huggingface_hub import snapshot_download

    repo = run_git_clone(
        "https://github.com/fishaudio/fish-speech.git",
        REPOS_DIR / "fish-speech-v1.5",
        recursive=False,
        branch="v1.5.0",
    )
    try:
        _pip("-e", str(repo), "--no-deps")
    except Exception:
        pass
    for pkg in (
        "pyrootutils",
        "einops",
        "tiktoken",
        "lightning",
        "hydra-core",
        "loguru",
        "loralib>=0.1.2",
        "kui>=1.6.0",
        "opencc-python-reimplemented==0.1.7",
        "resampy>=0.4.3",
        "silero-vad",
        "natsort",
        "vector_quantize_pytorch",
        "descript-audio-codec",
        "ormsgpack",
        "transformers<=4.57.3",
        "accelerate",
    ):
        try:
            _pip(pkg)
        except Exception:
            pass
    ensure_onnxruntime_cpu()  # torchmetrics (via pytorch_lightning) imports onnxruntime

    ckpt = CACHE_DIR / "fish-speech-1.5"
    if not ckpt.exists() or not any(ckpt.iterdir()):
        snapshot_download(FISH_SPEECH_HF, local_dir=str(ckpt))

    decoder = ckpt / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
    if not decoder.exists():
        decoder = ckpt / "codec.pth"
    if not decoder.exists():
        raise FileNotFoundError(f"Fish decoder missing in {ckpt}")

    # v1.5.0 ships CLI tools under tools/; older/newer trees use fish_speech/models/.
    codec_script = None
    for rel in (
        "tools/vqgan/inference.py",
        "fish_speech/models/vqgan/inference.py",
        "fish_speech/models/dac/inference.py",
    ):
        if (repo / rel).exists():
            codec_script = rel
            break
    if codec_script is None:
        raise FileNotFoundError("Fish codec inference script not found")

    t2s_script = None
    for rel in (
        "tools/llama/generate.py",
        "fish_speech/models/text2semantic/inference.py",
    ):
        if (repo / rel).exists():
            t2s_script = rel
            break
    if t2s_script is None:
        raise FileNotFoundError("Fish text2semantic script not found")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.perf_counter()

    prompt_tokens = None
    prompt_text = None
    if ref_audio and Path(ref_audio).exists():
        # Voice-clone prompt is optional — don't let a codec failure kill the run.
        try:
            ref_clip = trim_wav_head(Path(ref_audio), out_path.with_name("_fish_ref.wav"), 10.0)
            subprocess.check_call(
                [
                    sys.executable,
                    codec_script,
                    "-i",
                    str(ref_clip),
                    "--checkpoint-path",
                    str(decoder),
                ],
                cwd=str(repo),
                env=env,
            )
            found = sorted(repo.rglob("fake.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
            if found:
                prompt_tokens = found[0]
                prompt_text = split_text_chunks(text, 100)[0]
        except subprocess.CalledProcessError as exc:
            print(f"Fish-Speech ref encoding failed ({exc}); continuing without voice prompt")

    load_s = time.perf_counter() - t0
    t1 = time.perf_counter()

    cmd = [
        sys.executable,
        t2s_script,
        "--text",
        text,
        "--checkpoint-path",
        str(ckpt),
        "--half",
    ]
    if prompt_tokens is not None and prompt_text is not None:
        cmd += ["--prompt-text", prompt_text, "--prompt-tokens", str(prompt_tokens)]
    try:
        subprocess.check_call(cmd, cwd=str(repo), env=env)
    except subprocess.CalledProcessError:
        short = " ".join(split_text_chunks(text, 180)[:2])
        cmd[cmd.index("--text") + 1] = short
        # Drop prompts on retry
        if "--prompt-text" in cmd:
            i = cmd.index("--prompt-text")
            del cmd[i : i + 4]
        subprocess.check_call(cmd, cwd=str(repo), env=env)

    codes = sorted(repo.glob("codes_*.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not codes:
        codes = sorted(Path.cwd().glob("codes_*.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not codes:
        raise RuntimeError("Fish Speech did not write codes_*.npy")

    subprocess.check_call(
        [sys.executable, codec_script, "-i", str(codes[0]), "--checkpoint-path", str(decoder)],
        cwd=str(repo),
        env=env,
    )
    candidates = [p for p in list(repo.glob("fake.wav")) + list(repo.glob("*.wav")) if p.stat().st_size > 1000]
    if not candidates:
        candidates = [p for p in Path.cwd().glob("*.wav") if p.stat().st_size > 1000]
    if not candidates:
        raise RuntimeError("Fish Speech did not write a WAV")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    shutil.copy2(candidates[0], out_path)
    gen_s = time.perf_counter() - t1
    return {
        "load_seconds": load_s,
        "generation_seconds": gen_s,
        "note": "RESEARCH ONLY — fish-speech-1.5 (s1-mini skipped: gated)",
    }


RUNNERS = {
    "VoiceTut-TTS": lambda text, path, ref: synth_voicetut(text, path),
    "NAMAA-Egyptian-TTS": lambda text, path, ref: synth_namaa(text, path),
    "Chatterbox-Multilingual-V3": lambda text, path, ref: synth_chatterbox_v3(text, path),
    "Kokoro-82M": lambda text, path, ref: synth_kokoro(text, path),
    "SILMA-TTS": lambda text, path, ref: synth_silma(text, path, ref),
    "Qwen3-TTS-0.6B": lambda text, path, ref: synth_qwen3(text, path, ref),
    "CosyVoice-0.5B": lambda text, path, ref: synth_cosyvoice(text, path, ref),
    "Fish-Speech": lambda text, path, ref: synth_fish_speech(text, path, ref),
}


# ===========================================================================
# Subprocess worker + orchestrator
# ===========================================================================

def worker_main(model_name: str, text_file: Path, out_path: Path, ref_audio: Optional[Path], meta_path: Path) -> int:
    text = text_file.read_text(encoding="utf-8")
    print(f"[worker] model={model_name} cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')} gpus={gpu_count()}")
    for mod, pkg in (("psutil", "psutil"), ("pynvml", "nvidia-ml-py")):
        try:
            ensure_module(mod, pkg)
        except Exception:
            pass
    monitor = ResourceMonitor().start()
    try:
        meta = RUNNERS[model_name](text, out_path, ref_audio)
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise RuntimeError(f"No audio written to {out_path}")
        meta.update({"status": "ok", "output": str(out_path), "bytes": out_path.stat().st_size})
        meta.update(wav_stats(out_path))
        meta["resources"] = monitor.stop()
        write_meta(meta_path, meta)
        print(f"[worker] OK {model_name} → {out_path} ({out_path.stat().st_size} bytes)")
        return 0
    except Exception as exc:
        payload = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-4000:],
            "resources": monitor.stop(),
        }
        write_meta(meta_path, payload)
        print(f"[worker] FAIL {model_name}: {payload['error']}")
        traceback.print_exc()
        return 1
    finally:
        cleanup_gpu()


def _notebook_cell_source() -> Optional[str]:
    """Best-effort: recover this script when pasted into a Jupyter/Kaggle cell."""
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is None:
            return None
        cells = ip.user_ns.get("In") or []
        for cell in reversed(list(cells)):
            if not isinstance(cell, str):
                continue
            if "def run_model_isolated" in cell and "def worker_main" in cell:
                return cell
    except Exception:
        return None
    return None


def _this_script() -> Path:
    """Path to this file so worker subprocesses can re-exec it.

    Notebook cells do not define ``__file__``, so we fall back to an uploaded
    copy or write the pasted cell source under WORK_DIR.
    """
    try:
        path = Path(__file__).resolve()
        if path.is_file():
            return path
    except NameError:
        pass

    for candidate in (
        WORK_DIR / "kaggle_arabic_tts_compare.py",
        Path.cwd() / "kaggle_arabic_tts_compare.py",
        Path("/kaggle/working/kaggle_arabic_tts_compare.py"),
    ):
        if candidate.is_file():
            return candidate.resolve()

    src = _notebook_cell_source()
    if src:
        dest = WORK_DIR / "_kaggle_arabic_tts_compare_worker.py"
        dest.write_text(src, encoding="utf-8")
        return dest.resolve()

    raise RuntimeError(
        "Cannot resolve script path for worker subprocess (__file__ missing). "
        "Upload kaggle_arabic_tts_compare.py to /kaggle/working/ and run:\n"
        "  %run /kaggle/working/kaggle_arabic_tts_compare.py\n"
        "or:\n"
        "  !python /kaggle/working/kaggle_arabic_tts_compare.py"
    )


def run_model_isolated(model_name: str, text: str, gpu_index: int) -> dict[str, Any]:
    """Run one model in a fresh Python process bound to one GPU."""
    out_path = OUTPUT_DIR / f"{model_name}.wav"
    text_file = META_DIR / "input_text.txt"
    meta_path = META_DIR / f"{model_name}.json"
    text_file.write_text(text, encoding="utf-8")
    if meta_path.exists():
        meta_path.unlink()

    ref = pick_ref_audio()
    env = os.environ.copy()
    if gpu_index >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    # Avoid leftover assert state leaking if parent already poisoned (still restart session!).
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    cmd = [
        sys.executable,
        str(_this_script()),
        "--worker",
        "--model",
        model_name,
        "--text-file",
        str(text_file),
        "--out",
        str(out_path),
        "--meta",
        str(meta_path),
    ]
    if ref is not None:
        cmd += ["--ref", str(ref)]

    print(f"\n{'=' * 60}\n▶ {model_name} on GPU {gpu_index if gpu_index >= 0 else 'CPU'}\n{'=' * 60}")
    started = time.perf_counter()
    proc = subprocess.run(cmd, env=env)
    elapsed = time.perf_counter() - started

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {
            "status": "error",
            "error": f"Worker exited {proc.returncode} without writing meta",
        }
    meta["wall_seconds"] = elapsed
    meta["gpu_index"] = gpu_index
    return meta


# ===========================================================================
# Analytics reporting (printed tables + CSV/JSON/MD for model selection)
# ===========================================================================

ANALYTICS_COLUMNS = [
    "model",
    "status",
    "gpu_index",
    "gpu_name",
    "load_seconds",
    "generation_seconds",
    "wall_seconds",
    "audio_seconds",
    "rtf",
    "x_realtime",
    "chars_per_second",
    "audio_seconds_per_1k_chars",
    "sample_rate",
    "output_bytes",
    "bytes_per_second_audio",
    "peak_cpu_percent",
    "avg_cpu_percent",
    "peak_ram_mb",
    "avg_ram_mb",
    "peak_gpu_util_percent",
    "avg_gpu_util_percent",
    "peak_vram_mb",
    "avg_vram_mb",
    "model_vram_mb",
    "realtime_capable",
    "note",
    "error",
]


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


def build_analytics_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    text_chars = int(results.get("text_chars") or 0)
    rows: list[dict[str, Any]] = []
    for name, meta in results.get("models", {}).items():
        res = meta.get("resources", {}) or {}
        gen_s = float(meta.get("generation_seconds") or 0.0)
        audio_s = float(meta.get("audio_seconds") or 0.0)
        rtf = round(gen_s / audio_s, 3) if audio_s > 0 and gen_s > 0 else ""
        x_rt = round(audio_s / gen_s, 2) if gen_s > 0 and audio_s > 0 else ""
        cps = round(text_chars / gen_s, 1) if gen_s > 0 and text_chars > 0 else ""
        sec_per_1k = (
            round(1000.0 * gen_s / text_chars, 2) if gen_s > 0 and text_chars > 0 else ""
        )
        out_bytes = meta.get("bytes", "")
        bps = ""
        if audio_s > 0 and out_bytes not in ("", None):
            try:
                bps = round(float(out_bytes) / audio_s, 1)
            except Exception:
                bps = ""
        realtime = ""
        if rtf != "":
            realtime = "yes" if float(rtf) < 1.0 else "no"
        rows.append(
            {
                "model": name,
                "status": meta.get("status", ""),
                "gpu_index": meta.get("gpu_index", ""),
                "gpu_name": res.get("gpu_name", ""),
                "load_seconds": round(float(meta.get("load_seconds") or 0.0), 2),
                "generation_seconds": round(gen_s, 2),
                "wall_seconds": round(float(meta.get("wall_seconds") or 0.0), 2),
                "audio_seconds": round(audio_s, 2),
                "rtf": rtf,
                "x_realtime": x_rt,
                "chars_per_second": cps,
                "audio_seconds_per_1k_chars": sec_per_1k,
                "sample_rate": meta.get("sample_rate", ""),
                "output_bytes": out_bytes,
                "bytes_per_second_audio": bps,
                "peak_cpu_percent": res.get("peak_cpu_percent", ""),
                "avg_cpu_percent": res.get("avg_cpu_percent", ""),
                "peak_ram_mb": res.get("peak_ram_mb", ""),
                "avg_ram_mb": res.get("avg_ram_mb", ""),
                "peak_gpu_util_percent": res.get("peak_gpu_util_percent", ""),
                "avg_gpu_util_percent": res.get("avg_gpu_util_percent", ""),
                "peak_vram_mb": res.get("peak_vram_mb", ""),
                "avg_vram_mb": res.get("avg_vram_mb", ""),
                "model_vram_mb": res.get("model_vram_mb", ""),
                "realtime_capable": realtime,
                "note": meta.get("note", ""),
                "error": meta.get("error", ""),
            }
        )
    return rows


def aggregate_by_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per model with selection-oriented fields (TTS is usually one prompt)."""
    agg: list[dict[str, Any]] = []
    for r in rows:
        status = r.get("status", "")
        agg.append(
            {
                "model": r.get("model", ""),
                "status": status,
                "success": 1 if status == "ok" else 0,
                "load_seconds": r.get("load_seconds", ""),
                "generation_seconds": r.get("generation_seconds", ""),
                "wall_seconds": r.get("wall_seconds", ""),
                "audio_seconds": r.get("audio_seconds", ""),
                "rtf": r.get("rtf", ""),
                "x_realtime": r.get("x_realtime", ""),
                "chars_per_second": r.get("chars_per_second", ""),
                "audio_seconds_per_1k_chars": r.get("audio_seconds_per_1k_chars", ""),
                "sample_rate": r.get("sample_rate", ""),
                "output_bytes": r.get("output_bytes", ""),
                "peak_cpu_percent": r.get("peak_cpu_percent", ""),
                "peak_ram_mb": r.get("peak_ram_mb", ""),
                "peak_gpu_util_percent": r.get("peak_gpu_util_percent", ""),
                "peak_vram_mb": r.get("peak_vram_mb", ""),
                "model_vram_mb": r.get("model_vram_mb", ""),
                "realtime_capable": r.get("realtime_capable", ""),
                "gpu_name": r.get("gpu_name", ""),
                "note": r.get("note", ""),
                "error": r.get("error", ""),
            }
        )
    return agg


def build_tts_leaderboard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return []

    rtf_map = {
        str(r["model"]): float(r["rtf"]) for r in ok if r.get("rtf") not in ("", None)
    }
    cps_map = {
        str(r["model"]): float(r["chars_per_second"])
        for r in ok
        if r.get("chars_per_second") not in ("", None)
    }
    vram_map = {
        str(r["model"]): float(r["peak_vram_mb"])
        for r in ok
        if r.get("peak_vram_mb") not in ("", None)
    }
    load_map = {
        str(r["model"]): float(r["load_seconds"])
        for r in ok
        if r.get("load_seconds") not in ("", None)
    }
    gen_map = {
        str(r["model"]): float(r["generation_seconds"])
        for r in ok
        if r.get("generation_seconds") not in ("", None)
    }

    speed_s = _norm_map(rtf_map, lower_is_better=True)
    thr_s = _norm_map(cps_map, lower_is_better=False)
    vram_s = _norm_map(vram_map, lower_is_better=True)
    load_s = _norm_map(load_map, lower_is_better=True)
    gen_s = _norm_map(gen_map, lower_is_better=True)

    board: list[dict[str, Any]] = []
    for r in ok:
        name = str(r["model"])
        sp = speed_s.get(name, 50.0)
        th = thr_s.get(name, 50.0)
        vr = vram_s.get(name, 50.0)
        ld = load_s.get(name, 50.0)
        gn = gen_s.get(name, 50.0)
        # Listening quality is manual; scores are latency/resource oriented.
        robot = round(0.40 * sp + 0.25 * th + 0.20 * vr + 0.15 * ld, 2)
        balanced = round(0.30 * sp + 0.30 * th + 0.25 * vr + 0.15 * gn, 2)
        board.append(
            {
                "model": name,
                "status": r.get("status", ""),
                "rtf": r.get("rtf", ""),
                "x_realtime": r.get("x_realtime", ""),
                "chars_per_second": r.get("chars_per_second", ""),
                "generation_seconds": r.get("generation_seconds", ""),
                "load_seconds": r.get("load_seconds", ""),
                "audio_seconds": r.get("audio_seconds", ""),
                "peak_vram_mb": r.get("peak_vram_mb", ""),
                "peak_ram_mb": r.get("peak_ram_mb", ""),
                "peak_cpu_percent": r.get("peak_cpu_percent", ""),
                "realtime_capable": r.get("realtime_capable", ""),
                "score_speed": sp,
                "score_throughput": th,
                "score_vram_efficiency": vr,
                "score_load": ld,
                "score_balanced": balanced,
                "score_robot_realtime": robot,
                "wav_file": f"{name}.wav",
                "note": (
                    "Listen to WAV for Egyptian/code-switch quality; "
                    "scores cover speed/resources only."
                ),
            }
        )
    board.sort(key=lambda x: (-float(x["score_robot_realtime"]), str(x["model"])))
    for i, row in enumerate(board, 1):
        row["rank_robot"] = i
    by_speed = sorted(
        board,
        key=lambda x: float(x["rtf"]) if x.get("rtf") not in ("", None) else 999.0,
    )
    for i, row in enumerate(by_speed, 1):
        row["rank_speed"] = i
    by_vram = sorted(
        board,
        key=lambda x: float(x["peak_vram_mb"]) if x.get("peak_vram_mb") not in ("", None) else 999999.0,
    )
    for i, row in enumerate(by_vram, 1):
        row["rank_vram"] = i
    return board


def build_tts_recommendations(leaderboard: list[dict[str, Any]]) -> dict[str, Any]:
    if not leaderboard:
        return {
            "status": "no_successful_models",
            "picks": {},
            "notes": ["No successful TTS runs; cannot recommend a model."],
        }

    def pick(key: str, reverse: bool = False) -> Optional[dict[str, Any]]:
        eligible = [r for r in leaderboard if r.get(key) not in ("", None)]
        if not eligible:
            return None
        return sorted(eligible, key=lambda r: float(r[key]), reverse=reverse)[0]

    best_robot = leaderboard[0]
    best_speed = pick("rtf", reverse=False)
    best_thr = pick("chars_per_second", reverse=True)
    best_vram = pick("peak_vram_mb", reverse=False)
    best_balanced = pick("score_balanced", reverse=True)
    realtime = [r for r in leaderboard if r.get("realtime_capable") == "yes"]

    return {
        "status": "ok",
        "picks": {
            "best_for_robot_realtime": {
                "model": best_robot.get("model"),
                "why": "Best composite of RTF + throughput + low VRAM + fast load.",
                "metrics": {
                    "score_robot_realtime": best_robot.get("score_robot_realtime"),
                    "rtf": best_robot.get("rtf"),
                    "chars_per_second": best_robot.get("chars_per_second"),
                    "peak_vram_mb": best_robot.get("peak_vram_mb"),
                },
                "listen_file": f"{best_robot.get('model')}.wav",
            },
            "best_speed": {
                "model": (best_speed or {}).get("model"),
                "why": "Lowest RTF (generation faster relative to audio length).",
                "metrics": {
                    "rtf": (best_speed or {}).get("rtf"),
                    "x_realtime": (best_speed or {}).get("x_realtime"),
                    "generation_seconds": (best_speed or {}).get("generation_seconds"),
                },
            },
            "best_throughput": {
                "model": (best_thr or {}).get("model"),
                "why": "Highest characters synthesized per second.",
                "metrics": {"chars_per_second": (best_thr or {}).get("chars_per_second")},
            },
            "lowest_vram": {
                "model": (best_vram or {}).get("model"),
                "why": "Lowest peak VRAM — better for 6–16 GB GPUs.",
                "metrics": {"peak_vram_mb": (best_vram or {}).get("peak_vram_mb")},
            },
            "best_balanced": {
                "model": (best_balanced or {}).get("model"),
                "why": "Balanced speed/throughput/VRAM/generation time.",
                "metrics": {"score_balanced": (best_balanced or {}).get("score_balanced")},
            },
            "best_realtime_capable": {
                "model": realtime[0].get("model") if realtime else None,
                "why": "RTF < 1.0 (faster than real time).",
                "metrics": {"rtf": realtime[0].get("rtf") if realtime else None},
            },
        },
        "leaderboard_top3": leaderboard[:3],
        "model_count_ok": len(leaderboard),
        "notes": [
            "IMPORTANT: Naturalness / Egyptian dialect / code-switch quality must be judged by listening to WAVs.",
            "Automated scores only cover latency and resource fit for the robot pipeline.",
            "RTF < 1.0 is preferred for near-real-time synthesis; streaming TTFA is not measured here.",
            "For production, also measure time-to-first-audio with streaming APIs.",
        ],
    }


def write_tts_selection_report(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    leaderboard: list[dict[str, Any]],
    recommendations: dict[str, Any],
    text_chars: int,
) -> Path:
    ok = [r for r in rows if r.get("status") == "ok"]
    bad = [r for r in rows if r.get("status") == "error"]
    skipped = [r for r in rows if r.get("status") == "skipped"]
    lines = [
        "# TTS Model Selection Report",
        "",
        "Auto-generated from the Kaggle Arabic TTS bake-off.",
        "Combine this report with **listening** to each WAV before choosing a production voice.",
        "",
        "## Run summary",
        "",
        f"- Prompt chars: **{text_chars}**",
        f"- Models attempted: **{len(rows)}**",
        f"- OK: **{len(ok)}** | Failed: **{len(bad)}** | Skipped: **{len(skipped)}**",
        "",
        "## Recommended picks (speed/resources)",
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
        if payload.get("listen_file"):
            lines.append(f"- **Listen:** `{payload['listen_file']}`")
        lines.append("")

    lines.extend(
        [
            "## Robot realtime leaderboard",
            "",
            "| Rank | Model | Robot | RTF | xRT | Chars/s | Gen(s) | VRAM pk | Realtime |",
            "|---:|---|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in leaderboard:
        lines.append(
            "| {rank} | `{model}` | {robot} | {rtf} | {xrt} | {cps} | {gen} | {vram} | {rt} |".format(
                rank=row.get("rank_robot", ""),
                model=row.get("model", ""),
                robot=row.get("score_robot_realtime", ""),
                rtf=row.get("rtf", "-"),
                xrt=row.get("x_realtime", "-"),
                cps=row.get("chars_per_second", "-"),
                gen=row.get("generation_seconds", "-"),
                vram=row.get("peak_vram_mb", "-"),
                rt=row.get("realtime_capable", "-"),
            )
        )
    lines.append("")

    lines.extend(["## Per-model detail", ""])
    for r in rows:
        lines.append(f"### `{r.get('model')}` — {r.get('status')}")
        lines.append("")
        if r.get("status") == "ok":
            lines.append(
                f"- Timing: load={r.get('load_seconds')}s, gen={r.get('generation_seconds')}s, "
                f"wall={r.get('wall_seconds')}s, audio={r.get('audio_seconds')}s"
            )
            lines.append(
                f"- Speed: RTF={r.get('rtf')}, xRT={r.get('x_realtime')}, "
                f"chars/s={r.get('chars_per_second')}, "
                f"sec/1k chars={r.get('audio_seconds_per_1k_chars')}"
            )
            lines.append(
                f"- Resources: CPU pk={r.get('peak_cpu_percent')}%, "
                f"RAM pk={r.get('peak_ram_mb')}MB, "
                f"GPU pk={r.get('peak_gpu_util_percent')}%, "
                f"VRAM pk={r.get('peak_vram_mb')}MB, "
                f"model VRAM={r.get('model_vram_mb')}MB"
            )
            lines.append(f"- Output: `{r.get('model')}.wav` ({r.get('output_bytes')} bytes, sr={r.get('sample_rate')})")
        elif r.get("error"):
            lines.append(f"- Error: {r.get('error')}")
        if r.get("note"):
            lines.append(f"- Note: {r.get('note')}")
        lines.append("")

    lines.extend(
        [
            "## Listening checklist (manual quality)",
            "",
            "For each OK WAV, score 1–5:",
            "",
            "1. Egyptian dialect naturalness",
            "2. Arabic/English code-switching",
            "3. Numbers / dates / times",
            "4. Prosody / pauses / artifacts",
            "5. Overall robot-voice suitability",
            "",
            "## How to use these files",
            "",
            "1. Start with `tts_recommendations.json`.",
            "2. Sort `tts_leaderboard.csv` by robot/speed/VRAM.",
            "3. Listen to the shortlisted WAVs.",
            "4. Confirm resources in `tts_analytics.csv`.",
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
        "status": "Status",
        "gpu_index": "GPU",
        "load_seconds": "Load(s)",
        "generation_seconds": "Gen(s)",
        "wall_seconds": "Wall(s)",
        "audio_seconds": "Audio(s)",
        "rtf": "RTF",
        "x_realtime": "xRT",
        "chars_per_second": "Chars/s",
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
        "realtime_capable": "RT?",
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
    leaderboard: Optional[list[dict[str, Any]]] = None,
) -> None:
    print("\n" + "=" * 60)
    print("MODEL ANALYTICS")
    print("=" * 60)
    _print_table(
        "Timing / throughput (RTF = generation time / audio duration; lower is better)",
        [
            "model",
            "status",
            "gpu_index",
            "load_seconds",
            "generation_seconds",
            "wall_seconds",
            "audio_seconds",
            "rtf",
            "x_realtime",
            "chars_per_second",
            "realtime_capable",
        ],
        rows,
    )
    _print_table(
        "Resources (worker process tree + physical GPU, sampled every 0.5s)",
        [
            "model",
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
    if leaderboard:
        _print_table(
            "Leaderboard: robot realtime composite (higher is better; still listen to WAVs)",
            [
                "rank_robot",
                "model",
                "score_robot_realtime",
                "score_balanced",
                "rtf",
                "chars_per_second",
                "peak_vram_mb",
                "realtime_capable",
            ],
            leaderboard,
        )
    failed = [r for r in rows if r["status"] == "error"]
    if failed:
        print("\nFailures:")
        for r in failed:
            print(f"  - {r['model']}: {r['error']}")


def write_analytics_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    return write_csv_rows(path, rows, ANALYTICS_COLUMNS)


def export_tts_analytics(
    output_dir: Path,
    results: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    agg_rows = aggregate_by_model(rows)
    leaderboard = build_tts_leaderboard(rows)
    recommendations = build_tts_recommendations(leaderboard)
    paths = {
        "analytics": write_analytics_csv(rows, output_dir / "tts_analytics.csv"),
        "by_model": write_csv_rows(output_dir / "tts_analytics_by_model.csv", agg_rows),
        "leaderboard": write_csv_rows(output_dir / "tts_leaderboard.csv", leaderboard),
        "recommendations": output_dir / "tts_recommendations.json",
        "report": output_dir / "tts_selection_report.md",
    }
    write_meta(paths["recommendations"], recommendations)
    write_tts_selection_report(
        paths["report"],
        rows=rows,
        leaderboard=leaderboard,
        recommendations=recommendations,
        text_chars=int(results.get("text_chars") or 0),
    )
    print_analytics(rows, leaderboard)

    print("\n" + "=" * 60)
    print("TTS SELECTION PICKS (listen to WAVs to confirm quality)")
    print("=" * 60)
    for key, payload in (recommendations.get("picks") or {}).items():
        if isinstance(payload, dict) and payload.get("model"):
            print(f"  {key}: {payload['model']}")

    return {
        "agg_rows": agg_rows,
        "leaderboard": leaderboard,
        "recommendations": recommendations,
        "paths": paths,
    }


def _try_display(path: Path) -> None:
    try:
        from IPython.display import Audio, display

        print(f"▶ {path.name}")
        display(Audio(filename=str(path)))
    except Exception:
        pass


def _cleanup_working_scratch(names: tuple[str, ...]) -> None:
    """Remove repo/cache leftovers written to /kaggle/working by older runs."""
    if SCRATCH_DIR == WORK_DIR:
        return
    for name in names:
        leftover = WORK_DIR / name
        if leftover.exists() and leftover.is_dir():
            print(f"Removing old {leftover} from the limited working disk…")
            shutil.rmtree(leftover, ignore_errors=True)


def main(text: str = DEFAULT_TEXT, run_install: bool = False) -> dict[str, Any]:
    if run_install:
        install_packages()
    _cleanup_working_scratch(("repos", "tts_cache"))

    n_gpu = gpu_count()
    gpus = list_gpus()
    print(f"GPUs detected: {n_gpu}")
    print(json.dumps(gpus, indent=2))
    if n_gpu >= 2:
        print("T4x2 / multi-GPU mode: models alternate across cuda:0 and cuda:1 via CUDA_VISIBLE_DEVICES")
    print(f"Text ({len(text)} chars)")
    print(f"Output (persisted): {OUTPUT_DIR}")
    print(f"Scratch for repos/caches (temp disk): {SCRATCH_DIR}")

    results: dict[str, Any] = {
        "text": text,
        "text_chars": len(text),
        "output_dir": str(OUTPUT_DIR),
        "gpu_count": n_gpu,
        "gpus": gpus,
        "models": {},
    }

    # Round-robin across GPUs on T4x2.
    rr = 0
    for name in MODEL_ORDER:
        if not ENABLE.get(name, True):
            results["models"][name] = {"status": "skipped"}
            continue
        gpu = -1 if n_gpu <= 0 else (rr % n_gpu)
        rr += 1
        meta = run_model_isolated(name, text, gpu)
        results["models"][name] = meta
        out = OUTPUT_DIR / f"{name}.wav"
        if meta.get("status") == "ok" and out.exists():
            res = meta.get("resources", {}) or {}
            audio_s = float(meta.get("audio_seconds") or 0.0)
            gen_s = float(meta.get("generation_seconds") or 0.0)
            rtf = f"{gen_s / audio_s:.2f}" if audio_s > 0 else "n/a"
            print(
                f"✓ {name}: load={meta.get('load_seconds', 0):.1f}s  "
                f"gen={gen_s:.1f}s  audio={audio_s:.1f}s  rtf={rtf}  gpu={gpu}"
            )
            print(
                f"   cpu_peak={res.get('peak_cpu_percent', '-')}%  "
                f"ram_peak={res.get('peak_ram_mb', '-')}MB  "
                f"gpu_peak={res.get('peak_gpu_util_percent', '-')}%  "
                f"vram_peak={res.get('peak_vram_mb', '-')}MB"
            )
            _try_display(out)
        else:
            print(f"✗ {name}: {meta.get('error')}")

    summary_path = OUTPUT_DIR / "summary.json"
    rows = build_analytics_rows(results)
    exported = export_tts_analytics(OUTPUT_DIR, results, rows)
    paths = exported["paths"]
    results["analytics"] = {
        "leaderboard": exported["leaderboard"],
        "recommendations": exported["recommendations"],
        "files": {k: str(v) for k, v in paths.items()},
    }
    write_meta(summary_path, results)

    ok = [k for k, v in results["models"].items() if v.get("status") == "ok"]
    bad = [k for k, v in results["models"].items() if v.get("status") == "error"]
    print(f"\nDone. Summary → {summary_path}")
    print(f"Analytics CSV → {paths['analytics']}")
    print(f"Per-model analytics CSV → {paths['by_model']}")
    print(f"Leaderboard CSV → {paths['leaderboard']}")
    print(f"Recommendations JSON → {paths['recommendations']}")
    print(f"Selection report MD → {paths['report']}")
    print(f"OK ({len(ok)}): {ok}")
    if bad:
        print(f"FAILED ({len(bad)}): {bad}")
        for name in bad:
            print(f"  - {name}: {results['models'][name].get('error')}")
    print("WAV files:")
    for p in sorted(OUTPUT_DIR.glob("*.wav")):
        if p.name.startswith("_"):
            continue
        print(f"  - {p} ({p.stat().st_size} bytes)")
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--model")
    p.add_argument("--text-file", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--meta", type=Path)
    p.add_argument("--ref", type=Path, default=None)
    p.add_argument("--install", action="store_true")
    p.add_argument("--no-install", action="store_true")
    # parse_known_args: Jupyter/Colab injects "-f kernel-....json" into sys.argv
    args, _unknown = p.parse_known_args(argv)
    return args


if __name__ == "__main__":
    args = _parse_args()
    if args.worker:
        raise SystemExit(
            worker_main(args.model, args.text_file, args.out, args.ref, args.meta)
        )
    on_kaggle = Path("/kaggle/working").exists()
    auto_install = on_kaggle and os.environ.get("AUTO_INSTALL", "1") != "0"
    main(run_install=(args.install or os.environ.get("RUN_INSTALL", "0") == "1" or auto_install) and not args.no_install)
