from __future__ import annotations

import atexit
import gc
import json
import os
import threading
import time
import traceback
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

# Faster HF loads after the first download; quiet hub progress spam.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import psutil
import requests

try:
    import sounddevice as sd
    import soundfile as sf
except Exception:
    sd = None
    sf = None

try:
    import pynvml

    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False


def _shutdown_nvml() -> None:
    if not NVML_AVAILABLE:
        return
    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass


atexit.register(_shutdown_nvml)

APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Short text keeps interactive benchmarks fast on a 6 GB GPU.
DEFAULT_TEXT = (
    "إزيك يا روبوت؟ أنا عندي meeting الساعة تلاتة ونص، "
    "ومحتاجك تفكرني قبلها بنص ساعة."
)

# Optional stress sample — use the UI button; do not paste this by default.
LONG_SAMPLE_TEXT = (
    "السلام عليكم يا روبوت، صباح الخير. أنا محتاج أعمل اختبار لجودة تحويل "
    "النص إلى كلام باللهجة المصرية. فكرني إن عندي اجتماع بكرة الساعة عشرة ونص "
    "مع فريق التطوير، وبعدها Presentation الساعة اتنين، وبعدها Call مع العميل "
    "الساعة خمسة. درجة الحرارة حوالي سبعة وثلاثين، وسعر الدولار حوالي خمسين جنيه. "
    "Please open the dashboard and send the report to the customer. "
    "الـ API response time لازم يكون أقل من ميتين ملي ثانية."
)

VOICETUT_REPO = "mohammedaly22/VoiceTut-TTS"
VOICETUT_LONG_CHAR_THRESHOLD = 180

# ---------------------------------------------------------------------------
# Visual system
# ---------------------------------------------------------------------------

class Theme:
    bg = "#D9E3EA"
    bg_deep = "#C5D3DE"
    surface = "#F4F7FA"
    surface_raised = "#FFFFFF"
    ink = "#15202B"
    ink_soft = "#3D4F5F"
    ink_muted = "#6B7C8C"
    accent = "#0B6E6A"
    accent_hover = "#095E5A"
    accent_pressed = "#084E4B"
    accent_soft = "#D4ECEA"
    accent_line = "#7DBBB7"
    border = "#B7C5D1"
    border_soft = "#D5DEE6"
    success = "#1F7A45"
    success_soft = "#D9F0E3"
    warn = "#9A5B18"
    warn_soft = "#F5E6D3"
    danger = "#A13A3A"
    danger_soft = "#F5DADA"
    shadow = "#B8C5CF"

    font_brand = ("Bahnschrift", 28, "bold")
    font_brand_ar = ("Segoe UI", 16)
    font_title = ("Bahnschrift", 18, "bold")
    font_section = ("Bahnschrift", 12, "bold")
    font_body = ("Segoe UI", 10)
    font_body_bold = ("Segoe UI Semibold", 10)
    font_small = ("Segoe UI", 9)
    font_arabic = ("Segoe UI", 14)
    font_mono = ("Cascadia Mono", 10)
    font_metric = ("Bahnschrift", 22, "bold")
    font_metric_label = ("Segoe UI", 9)
    font_button = ("Segoe UI Semibold", 10)
    font_button_lg = ("Segoe UI Semibold", 11)


@dataclass(frozen=True)
class ModelInfo:
    key: str
    name: str
    backend: str
    description: str
    language: str
    license_name: str
    estimated_vram: str
    strengths: str
    limitations: str
    install_hint: str
    short_label: str
    blurb: str


MODELS: dict[str, ModelInfo] = {
    "voicetut": ModelInfo(
        key="voicetut",
        name="VoiceTut-TTS",
        short_label="VoiceTut",
        blurb="Egyptian dialect · local Python",
        backend="Local Python",
        description=(
            "Egyptian-Arabic TTS fine-tuned from OmniVoice. Includes built-in "
            "voices, Arabic/English code-switching, normalization, streaming, "
            "and optional zero-shot voice cloning."
        ),
        language="Egyptian Arabic + English code-switching",
        license_name="Apache-2.0",
        estimated_vram="~3 GB peak FP16 typical; test carefully on 6 GB GPUs",
        strengths="Egyptian dialect, built-in voices, numbers/dates normalization.",
        limitations=(
            "First model load downloads weights and can be slow. "
            "Needs omnivoice + voicetut-tts in the same Python that runs this GUI."
        ),
        install_hint=(
            "Already set up on this machine with:\n"
            "  pip install -r requirements-voicetut.txt\n\n"
            "Or manually:\n"
            "  pip install omnivoice voicetut-tts\n"
            "  python arabic_tts_benchmark_gui.py\n\n"
            "Default speaker: Mohamed"
        ),
    ),
    "chatterbox_server": ModelInfo(
        key="chatterbox_server",
        name="Chatterbox TTS Server",
        short_label="Chatterbox",
        blurb="HTTP API · multilingual Arabic",
        backend="HTTP API",
        description=(
            "Self-hosted Chatterbox family server with a Web UI, engine switching, "
            "Arabic through Chatterbox Multilingual, voice cloning and OpenAI-compatible APIs."
        ),
        language="Arabic via Chatterbox Multilingual",
        license_name="MIT server; verify underlying model license",
        estimated_vram="0.5B multilingual model; 6 GB GPU should be tested cautiously",
        strengths="Separate portable environment, API, engine hot-swap, unload endpoint.",
        limitations="For Arabic select the multilingual engine. Short-text streaming is chunk-level.",
        install_hint=(
            "Double-click start-chatterbox-server.bat in this project,\n"
            "or run:\n"
            "  cd Chatterbox-TTS-Server\n"
            "  python start.py --portable --nvidia-cu128\n\n"
            "Keep that window open. Default URL: http://127.0.0.1:8004\n"
            "In the server Web UI select Chatterbox Multilingual for Arabic."
        ),
    ),
    "namaa": ModelInfo(
        key="namaa",
        name="NAMAA Egyptian TTS",
        short_label="NAMAA",
        blurb="Egyptian phrasing · local Python",
        backend="Local Python",
        description=(
            "Egyptian Arabic configuration/checkpoint built on Chatterbox Multilingual. "
            "Supports optional reference audio for speaker and style transfer."
        ),
        language="Egyptian Arabic",
        license_name="MIT",
        estimated_vram="0.5B parameters, F32 checkpoint; GPU-memory pressure is possible",
        strengths="Egyptian phrasing and rhythm, reference-audio prompting.",
        limitations=(
            "Model card says the dialect behavior is based on configuration/prompting rather "
            "than a separately trained Egyptian acoustic model."
        ),
        install_hint=(
            "Install without downgrading CUDA PyTorch:\n"
            "  pip install --no-deps chatterbox-tts\n"
            "  pip install -r requirements-namaa.txt\n"
            "  python arabic_tts_benchmark_gui.py"
        ),
    ),
}


class Metrics:
    @staticmethod
    def system() -> dict[str, Any]:
        process = psutil.Process(os.getpid())
        data: dict[str, Any] = {
            "process_ram_mb": process.memory_info().rss / (1024**2),
            "system_ram_percent": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent(interval=None),
        }
        if NVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                data.update(
                    gpu_used_mb=mem.used / (1024**2),
                    gpu_total_mb=mem.total / (1024**2),
                    gpu_util_percent=util.gpu,
                )
            except Exception:
                pass
        return data


def _http_error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
    except Exception:
        text = (response.text or "").strip()
        return text[:500] if text else response.reason
    if isinstance(payload, dict):
        detail = payload.get("detail", payload)
        if isinstance(detail, str):
            return detail
        return json.dumps(detail, ensure_ascii=False)
    return str(payload)


def _raise_for_status(response: requests.Response) -> None:
    if response.ok:
        return
    detail = _http_error_detail(response)
    raise RuntimeError(f"HTTP {response.status_code}: {detail}")


def _looks_like_audio(data: bytes) -> bool:
    if len(data) < 12:
        return False
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return True
    if data[:3] == b"ID3" or data[:2] == b"\xff\xfb" or data[:2] == b"\xff\xf3":
        return True
    if data[:4] == b"OggS" or data[:4] == b"fLaC" or data[:4] == b"Opus":
        return True
    return False


class AdapterBase:
    def unload(self) -> None:
        pass

    def synthesize(
        self,
        text: str,
        output_path: Path,
        reference_audio: Optional[Path],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError


class VoiceTutAdapter(AdapterBase):
    def __init__(self) -> None:
        self.model = None
        self.device = "cpu"
        self.dtype = "float32"
        self.local_model_path: Optional[str] = None
        self._prompt_cache: dict[str, Any] = {}
        self._load_lock = threading.Lock()
        self._loading = False

    @staticmethod
    def _resolve_local_snapshot() -> str:
        """Resolve a local HF snapshot path (fast when already downloaded)."""
        from huggingface_hub import snapshot_download

        return snapshot_download(VOICETUT_REPO)

    def load(self) -> float:
        with self._load_lock:
            if self.model is not None:
                return 0.0
            self._loading = True
            started = time.perf_counter()
            try:
                from voicetut_tts import VoiceTutTTS
            except ImportError as exc:
                raise RuntimeError(
                    "VoiceTut-TTS is not installed in this Python environment.\n\n"
                    "Install with:\n"
                    "  pip install omnivoice voicetut-tts\n\n"
                    + MODELS["voicetut"].install_hint
                ) from exc

            try:
                device, dtype = _resolve_compute_device(prefer_cuda=True)
                _configure_cuda_runtime()
                self.device = device
                self.dtype = dtype

                local_path = self._resolve_local_snapshot()
                self.local_model_path = local_path
                refs = str(
                    Path(local_path) / "reference_speakers" / "references.json"
                )
                load_kwargs: dict[str, Any] = {
                    "device": device,
                    "dtype": dtype,
                    "local_files_only": True,
                    "attn_implementation": "sdpa",
                }

                try:
                    self.model = VoiceTutTTS.from_pretrained(
                        local_path,
                        references=refs if Path(refs).exists() else None,
                        **load_kwargs,
                    )
                except TypeError:
                    load_kwargs.pop("attn_implementation", None)
                    self.model = VoiceTutTTS.from_pretrained(
                        local_path,
                        references=refs if Path(refs).exists() else None,
                        **load_kwargs,
                    )
                except Exception:
                    load_kwargs["local_files_only"] = False
                    self.model = VoiceTutTTS.from_pretrained(
                        VOICETUT_REPO,
                        **load_kwargs,
                    )

                actual = _model_param_device(self.model)
                if device.startswith("cuda") and "cuda" not in actual:
                    raise RuntimeError(
                        "VoiceTut loaded on CPU even though CUDA was requested "
                        f"(wanted {device}, got {actual}). "
                        "Install a CUDA build of PyTorch and free GPU memory, then retry."
                    )
                return time.perf_counter() - started
            finally:
                self._loading = False

    def _voice_clone_prompt(
        self,
        speaker: str,
        reference_audio: Optional[Path],
        reference_text: str,
    ) -> Any:
        """Encode the speaker once and reuse across generations."""
        if reference_audio is not None:
            cache_key = f"ref:{reference_audio.resolve()}|{reference_text}"
            ref_path = str(reference_audio)
            ref_text = reference_text or None
        elif speaker:
            cache_key = f"spk:{speaker.lower()}"
            if not self.model.registry:
                return None
            spk = self.model.registry.get(speaker)
            ref_path = spk.audio_path
            ref_text = spk.reference_text
        else:
            return None

        cached = self._prompt_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            prompt = self.model.model.create_voice_clone_prompt(
                ref_audio=ref_path,
                ref_text=ref_text,
            )
        except Exception:
            return None
        self._prompt_cache[cache_key] = prompt
        return prompt

    def synthesize(self, text, output_path, reference_audio, options):
        load_seconds = self.load()
        started = time.perf_counter()

        num_step = int(options.get("num_step", 12))
        speed = float(options.get("speed", 1.0))
        speaker = str(options.get("speaker", "") or "").strip()
        ref_text = str(options.get("reference_text", "") or "").strip()
        use_chunks = bool(options.get("split_text", True)) and (
            len(text) >= VOICETUT_LONG_CHAR_THRESHOLD
        )

        try:
            import numpy as np
            import torch
            from voicetut_tts.engine import GenerationParams, split_sentences

            inference_ctx = (
                torch.inference_mode()
                if hasattr(torch, "inference_mode")
                else torch.no_grad()
            )
        except Exception:
            from contextlib import nullcontext

            inference_ctx = nullcontext()
            np = None  # type: ignore
            split_sentences = None  # type: ignore
            GenerationParams = None  # type: ignore

        params = GenerationParams(num_step=num_step, speed=speed)
        prompt = self._voice_clone_prompt(speaker, reference_audio, ref_text)

        with inference_ctx:
            if use_chunks and split_sentences is not None and np is not None:
                pieces: list[Any] = []
                gap = np.zeros(
                    int(self.model.sampling_rate * 0.08),
                    dtype=np.float32,
                )
                for sentence in split_sentences(text, max_chars=180):
                    if prompt is not None:
                        chunk = self.model.model.generate(
                            text=sentence,
                            language=self.model.language,
                            voice_clone_prompt=prompt,
                            **params.as_kwargs(),
                        )[0]
                    else:
                        chunk = self.model._generate_one(
                            sentence,
                            self.model.language,
                            None,
                            None,
                            None,
                            params,
                            True,
                        )
                    pieces.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
                    pieces.append(gap)
                wav = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
                self.model.save(wav, str(output_path))
            elif prompt is not None:
                wav = self.model.model.generate(
                    text=text,
                    language=self.model.language,
                    voice_clone_prompt=prompt,
                    **params.as_kwargs(),
                )[0]
                self.model.save(wav, str(output_path))
            else:
                kwargs: dict[str, Any] = {
                    "output": str(output_path),
                    "num_step": num_step,
                    "speed": speed,
                }
                if reference_audio:
                    kwargs["ref_audio"] = str(reference_audio)
                    if ref_text:
                        kwargs["ref_text"] = ref_text
                elif speaker:
                    kwargs["speaker"] = speaker
                self.model.synthesize(text, **kwargs)

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(
                f"VoiceTut did not write an audio file to {output_path}"
            )
        generation_seconds = time.perf_counter() - started
        return {
            "load_seconds": load_seconds,
            "generation_seconds": generation_seconds,
            "output_path": str(output_path),
            "device": self.device,
            "dtype": self.dtype,
            "param_device": _model_param_device(self.model),
            "chunked": use_chunks,
            "local_model_path": self.local_model_path,
        }

    def unload(self) -> None:
        with self._load_lock:
            self.model = None
            self._prompt_cache.clear()
            self.local_model_path = None
        _cleanup_cuda()


class NAMAAAdapter(AdapterBase):
    def __init__(self) -> None:
        self.model = None
        self.device = "cpu"

    def load(self) -> float:
        if self.model is not None:
            return 0.0
        started = time.perf_counter()
        try:
            import torch
            from chatterbox import mtl_tts
            from huggingface_hub import snapshot_download
            from safetensors.torch import load_file as load_safetensors
        except ImportError as exc:
            raise RuntimeError(
                "NAMAA dependencies are not installed in this Python environment.\n\n"
                "Install without downgrading CUDA PyTorch:\n"
                "  pip install --no-deps chatterbox-tts\n"
                "  pip install -r requirements-namaa.txt\n\n"
                "Then restart the GUI."
            ) from exc

        device, _dtype = _resolve_compute_device(prefer_cuda=True)
        _configure_cuda_runtime()
        # Chatterbox expects "cuda" / "cpu" rather than "cuda:0".
        device = "cuda" if device.startswith("cuda") else "cpu"
        self.device = device

        ckpt_dir = snapshot_download(
            repo_id="NAMAA-Space/NAMAA-Egyptian-TTS",
            repo_type="model",
            revision="main",
        )
        weights = Path(ckpt_dir) / "t3_mtl23ls_v2.safetensors"
        if not weights.exists():
            raise RuntimeError(
                f"NAMAA checkpoint missing expected file: {weights.name}"
            )
        model = mtl_tts.ChatterboxMultilingualTTS.from_pretrained(device=device)
        map_device = device if device == "cpu" else "cuda:0"
        t3_state = load_safetensors(str(weights), device=map_device)
        model.t3.load_state_dict(t3_state)
        model.t3.to(device).eval()
        # Keep weights on GPU; FP16 can break some Chatterbox ops, so stay FP32 here.
        if device == "cuda":
            torch.cuda.synchronize()
        self.model = model
        return time.perf_counter() - started

    def synthesize(self, text, output_path, reference_audio, options):
        load_seconds = self.load()
        started = time.perf_counter()

        kwargs: dict[str, Any] = {"language_id": "ar"}
        if reference_audio:
            kwargs["audio_prompt_path"] = str(reference_audio)

        try:
            import torch

            inference_ctx = (
                torch.inference_mode()
                if hasattr(torch, "inference_mode")
                else torch.no_grad()
            )
        except Exception:
            from contextlib import nullcontext

            inference_ctx = nullcontext()

        with inference_ctx:
            wav_tensor = self.model.generate(text, **kwargs)

        try:
            import torchaudio as ta

            ta.save(str(output_path), wav_tensor.cpu(), self.model.sr)
        except Exception:
            import numpy as np

            wav = wav_tensor.detach().cpu().numpy()
            if wav.ndim > 1:
                wav = wav[0]
            wav = np.clip(wav, -1.0, 1.0).astype("float32")
            if sf is not None:
                sf.write(str(output_path), wav, self.model.sr)
            else:
                with wave.open(str(output_path), "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(int(self.model.sr))
                    handle.writeframes((wav * 32767.0).astype("int16").tobytes())

        generation_seconds = time.perf_counter() - started
        return {
            "load_seconds": load_seconds,
            "generation_seconds": generation_seconds,
            "output_path": str(output_path),
            "device": self.device,
        }

    def unload(self) -> None:
        self.model = None
        _cleanup_cuda()


class ChatterboxServerAdapter(AdapterBase):
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/api/model-info", timeout=10)
        _raise_for_status(response)
        return response.json()

    def list_voices(self) -> list[str]:
        for path in ("/v1/audio/voices", "/get_predefined_voices"):
            try:
                response = requests.get(f"{self.base_url}{path}", timeout=15)
                _raise_for_status(response)
                return self._parse_voice_list(response.json())
            except Exception:
                continue
        try:
            info = self.health()
            return self._parse_voice_list(info.get("predefined_voices") or [])
        except Exception:
            return []

    @staticmethod
    def _parse_voice_list(payload: Any) -> list[str]:
        if isinstance(payload, dict):
            items = payload.get("data") or payload.get("voices") or []
        else:
            items = payload
        result: list[str] = []
        for item in items:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                value = (
                    item.get("id")
                    or item.get("filename")
                    or item.get("name")
                    or item.get("voice")
                    or item.get("display_name")
                    or ""
                )
                result.append(str(value))
        return [item for item in result if item]

    def synthesize(self, text, output_path, reference_audio, options):
        payload: dict[str, Any] = {
            "text": text,
            "voice_mode": "predefined",
            "output_format": "wav",
            "split_text": bool(options.get("split_text", True)),
            "chunk_size": int(options.get("chunk_size", 120)),
            "stream": False,
            "language": "ar",
            "temperature": float(options.get("temperature", 0.8)),
            "exaggeration": float(options.get("exaggeration", 0.5)),
            "cfg_weight": float(options.get("cfg_weight", 0.5)),
            "speed_factor": float(options.get("speed", 1.0)),
            "seed": int(options.get("seed", 0)),
        }

        voice = str(options.get("speaker", "") or "").strip()
        if reference_audio:
            if not reference_audio.exists():
                raise FileNotFoundError(f"Reference audio not found: {reference_audio}")
            with reference_audio.open("rb") as file_handle:
                upload = requests.post(
                    f"{self.base_url}/upload_reference",
                    files={
                        "files": (
                            reference_audio.name,
                            file_handle,
                            "audio/wav",
                        )
                    },
                    timeout=120,
                )
            _raise_for_status(upload)
            upload_payload = upload.json()
            uploaded = upload_payload.get("uploaded_files") or []
            errors = upload_payload.get("errors") or upload_payload.get("upload_errors") or []
            if errors and not uploaded:
                raise RuntimeError(
                    "Reference upload failed: "
                    + json.dumps(errors, ensure_ascii=False)
                )
            if not uploaded:
                raise RuntimeError(
                    "Reference upload returned no filenames. "
                    f"Server response: {json.dumps(upload_payload, ensure_ascii=False)}"
                )
            payload["voice_mode"] = "clone"
            payload["reference_audio_filename"] = uploaded[0]
        elif voice:
            payload["predefined_voice_id"] = voice

        started = time.perf_counter()
        try:
            response = requests.post(
                f"{self.base_url}/tts",
                json=payload,
                timeout=900,
            )
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot reach Chatterbox TTS Server at {self.base_url}.\n\n"
                "Start it first with:\n"
                "  start-chatterbox-server.bat\n\n"
                "Then press Test Server in the GUI before generating."
            ) from exc
        _raise_for_status(response)

        content_type = (response.headers.get("content-type") or "").lower()
        body = response.content
        if "application/json" in content_type or not _looks_like_audio(body):
            preview = body[:300].decode("utf-8", errors="replace")
            raise RuntimeError(
                "Chatterbox /tts did not return audio. "
                f"content-type={content_type or 'unknown'}; body={preview}"
            )

        output_path.write_bytes(body)
        generation_seconds = time.perf_counter() - started

        return {
            "load_seconds": 0.0,
            "generation_seconds": generation_seconds,
            "output_path": str(output_path),
            "server_content_type": content_type,
        }

    def unload(self) -> None:
        try:
            response = requests.post(f"{self.base_url}/api/unload", timeout=30)
            _raise_for_status(response)
        except Exception:
            pass


_CUDA_CONFIGURED = False
_CUDA_STATUS_CACHE: Optional[dict[str, Any]] = None


def _resolve_compute_device(prefer_cuda: bool = True) -> tuple[str, str]:
    """Return (device, dtype_name), preferring CUDA FP16 when available."""
    try:
        import torch
    except ImportError:
        return "cpu", "float32"

    if prefer_cuda and torch.cuda.is_available():
        # Explicit cuda:0 is more reliable with transformers device_map than bare "cuda".
        return "cuda:0", "float16"
    return "cpu", "float32"


def _configure_cuda_runtime() -> None:
    """Enable faster CUDA kernels once per process."""
    global _CUDA_CONFIGURED
    if _CUDA_CONFIGURED:
        return
    try:
        import torch

        if not torch.cuda.is_available():
            return

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        # Harmless on Turing (GTX 16xx); helps on Ampere+.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        # Warm the context so the first synth doesn't pay full init latency mid-run.
        torch.cuda.init()
        torch.cuda.synchronize()
        _CUDA_CONFIGURED = True
    except Exception:
        pass


def _cuda_status(force_refresh: bool = False) -> dict[str, Any]:
    """Describe the active compute device for UI and reports."""
    global _CUDA_STATUS_CACHE
    if _CUDA_STATUS_CACHE is not None and not force_refresh:
        return _CUDA_STATUS_CACHE
    try:
        import torch

        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            _CUDA_STATUS_CACHE = {
                "device": f"cuda:{index}",
                "name": props.name,
                "vram_total_mb": round(props.total_memory / (1024**2), 1),
                "cuda_version": torch.version.cuda,
                "torch_version": torch.__version__,
            }
        else:
            _CUDA_STATUS_CACHE = {
                "device": "cpu",
                "name": "CPU (CUDA not available in this Python)",
                "torch_version": torch.__version__,
            }
    except Exception:
        _CUDA_STATUS_CACHE = {
            "device": "cpu",
            "name": "CPU (PyTorch not installed)",
        }
    return _CUDA_STATUS_CACHE


def _model_param_device(model: Any) -> str:
    """Best-effort device string for a loaded model."""
    try:
        inner = getattr(model, "model", model)
        param = next(inner.parameters())
        return str(param.device)
    except Exception:
        return "unknown"


def _cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def audio_duration(path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Audio file is empty: {path}")

    if sf is not None:
        try:
            info = sf.info(str(path))
            return float(info.duration)
        except Exception:
            pass

    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            if rate <= 0:
                raise ValueError("Invalid WAV sample rate")
            return frames / float(rate)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read audio duration from {path.name}. "
            "The file may be corrupt or not a supported WAV."
        ) from exc


def _backend_ready(key: str) -> bool:
    if key == "voicetut":
        try:
            import voicetut_tts  # noqa: F401
            import omnivoice  # noqa: F401

            return True
        except Exception:
            return False
    if key == "namaa":
        try:
            import torch  # noqa: F401
            from chatterbox import mtl_tts  # noqa: F401
            from huggingface_hub import snapshot_download  # noqa: F401
            from safetensors.torch import load_file  # noqa: F401

            return True
        except Exception:
            return False
    if key == "chatterbox_server":
        return True
    return False


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# UI primitives
# ---------------------------------------------------------------------------

class AccentButton(tk.Canvas):
    def __init__(
        self,
        master: tk.Misc,
        text: str,
        command: Callable[[], None],
        *,
        primary: bool = True,
        width: int = 168,
        height: int = 40,
    ) -> None:
        super().__init__(
            master,
            width=width,
            height=height,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
            bg=Theme.surface_raised,
        )
        self._text = text
        self._command = command
        self._primary = primary
        self._enabled = True
        self._width = width
        self._height = height
        self._state = "idle"
        self.bind("<Enter>", lambda _e: self._set_state("hover"))
        self.bind("<Leave>", lambda _e: self._set_state("idle"))
        self.bind("<ButtonPress-1>", lambda _e: self._set_state("pressed"))
        self.bind("<ButtonRelease-1>", self._on_release)
        self._draw()

    def configure_text(self, text: str) -> None:
        self._text = text
        self._draw()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self.configure(cursor="hand2" if enabled else "arrow")
        self._draw()

    def _on_release(self, _event: tk.Event) -> None:
        if not self._enabled:
            return
        self._set_state("hover")
        self._command()

    def _set_state(self, state: str) -> None:
        if not self._enabled and state != "idle":
            return
        self._state = state
        self._draw()

    def _draw(self) -> None:
        self.delete("all")
        r = 8
        if not self._enabled:
            fill = Theme.border_soft
            outline = Theme.border
            text_color = Theme.ink_muted
        elif self._primary:
            fill = {
                "idle": Theme.accent,
                "hover": Theme.accent_hover,
                "pressed": Theme.accent_pressed,
            }[self._state]
            outline = fill
            text_color = "#FFFFFF"
        else:
            fill = {
                "idle": Theme.surface_raised,
                "hover": Theme.accent_soft,
                "pressed": Theme.border_soft,
            }[self._state]
            outline = Theme.accent if self._state != "idle" else Theme.border
            text_color = Theme.accent if self._state != "idle" else Theme.ink

        self._round_rect(1, 1, self._width - 2, self._height - 2, r, fill=fill, outline=outline)
        self.create_text(
            self._width / 2,
            self._height / 2,
            text=self._text,
            fill=text_color,
            font=Theme.font_button_lg if self._primary else Theme.font_button,
        )

    def _round_rect(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1,
        ]
        return self.create_polygon(points, smooth=True, **kwargs)


class ModelCard(tk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        info: ModelInfo,
        selected: bool,
        on_select: Callable[[str], None],
        ready: bool = False,
    ) -> None:
        super().__init__(master, bg=Theme.surface, highlightthickness=0)
        self.info = info
        self._selected = selected
        self._ready = ready
        self._on_select = on_select
        self._inner = tk.Frame(self, cursor="hand2")
        self._inner.pack(fill="x", padx=2, pady=2)
        self._title = tk.Label(self._inner, font=Theme.font_body_bold, anchor="w", cursor="hand2")
        self._title.pack(fill="x", padx=14, pady=(12, 2))
        self._blurb = tk.Label(self._inner, font=Theme.font_small, anchor="w", cursor="hand2")
        self._blurb.pack(fill="x", padx=14, pady=(0, 2))
        self._status = tk.Label(self._inner, font=Theme.font_small, anchor="w", cursor="hand2")
        self._status.pack(fill="x", padx=14, pady=(0, 12))
        for widget in (self, self._inner, self._title, self._blurb, self._status):
            widget.bind("<Button-1>", lambda _e: self._on_select(info.key))
        self.set_selected(selected)

    def set_ready(self, ready: bool) -> None:
        self._ready = ready
        self.set_selected(self._selected)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        if selected:
            bg = Theme.accent_soft
            title_color = Theme.accent
            border = Theme.accent
        else:
            bg = Theme.surface_raised
            title_color = Theme.ink
            border = Theme.border_soft
        self.configure(bg=border)
        for widget in (self._inner, self._title, self._blurb, self._status):
            widget.configure(bg=bg)
        self._title.configure(text=self.info.short_label, fg=title_color)
        self._blurb.configure(text=self.info.blurb, fg=Theme.ink_muted)
        if self.info.key == "chatterbox_server":
            status_text = "Uses HTTP server"
            status_color = Theme.ink_muted
        elif self._ready:
            status_text = "Ready in this Python"
            status_color = Theme.success
        else:
            status_text = "Not installed here"
            status_color = Theme.warn
        self._status.configure(text=status_text, fg=status_color)


class TabBar(tk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        tabs: list[tuple[str, str]],
        on_change: Callable[[str], None],
    ) -> None:
        super().__init__(master, bg=Theme.surface_raised)
        self._on_change = on_change
        self._buttons: dict[str, tk.Label] = {}
        self._active = tabs[0][0]
        for key, label in tabs:
            btn = tk.Label(
                self,
                text=label,
                font=Theme.font_body_bold,
                padx=16,
                pady=10,
                cursor="hand2",
                bg=Theme.surface_raised,
                fg=Theme.ink_muted,
            )
            btn.pack(side="left")
            btn.bind("<Button-1>", lambda _e, k=key: self.select(k))
            btn.bind("<Enter>", lambda _e, b=btn, k=key: self._hover(b, k, True))
            btn.bind("<Leave>", lambda _e, b=btn, k=key: self._hover(b, k, False))
            self._buttons[key] = btn
        self._underline = tk.Frame(self, height=2, bg=Theme.accent)
        self.select(self._active, notify=False)

    def _hover(self, button: tk.Label, key: str, entering: bool) -> None:
        if key == self._active:
            return
        button.configure(fg=Theme.ink if entering else Theme.ink_muted)

    def select(self, key: str, notify: bool = True) -> None:
        self._active = key
        for tab_key, button in self._buttons.items():
            active = tab_key == key
            button.configure(
                fg=Theme.accent if active else Theme.ink_muted,
                bg=Theme.surface_raised,
            )
        active_btn = self._buttons[key]
        self._underline.place_forget()
        self.update_idletasks()
        self._underline.place(
            x=active_btn.winfo_x() + 12,
            y=max(active_btn.winfo_height() - 2, 28),
            width=max(active_btn.winfo_width() - 24, 24),
            height=2,
        )
        if notify:
            self._on_change(key)


class MetricTile(tk.Frame):
    def __init__(self, master: tk.Misc, label: str) -> None:
        super().__init__(
            master,
            bg=Theme.surface_raised,
            highlightbackground=Theme.border_soft,
            highlightthickness=1,
        )
        self._value = tk.Label(
            self,
            text="—",
            font=Theme.font_metric,
            bg=Theme.surface_raised,
            fg=Theme.ink,
            anchor="w",
        )
        self._value.pack(fill="x", padx=16, pady=(14, 2))
        tk.Label(
            self,
            text=label,
            font=Theme.font_metric_label,
            bg=Theme.surface_raised,
            fg=Theme.ink_muted,
            anchor="w",
        ).pack(fill="x", padx=16, pady=(0, 14))

    def set_value(self, value: str) -> None:
        self._value.configure(text=value)


class ScrollableFrame(tk.Frame):
    def __init__(self, master: tk.Misc, bg: str = Theme.surface) -> None:
        super().__init__(master, bg=bg)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.body = tk.Frame(self.canvas, bg=bg)
        self._window = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.body.bind("<Configure>", self._on_configure)
        self.canvas.bind("<Configure>", self._on_canvas)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_configure(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self._window, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        if self.winfo_containing(event.x_root, event.y_root) is None:
            return
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class TTSBenchmarkApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Nabra · Arabic TTS Benchmark")
        self.geometry("1280x860")
        self.minsize(1100, 760)
        self.configure(bg=Theme.bg)

        self.adapters: dict[str, AdapterBase] = {}
        self.current_output: Optional[Path] = None
        self.reference_audio: Optional[Path] = None
        self._generating = False
        self._metrics_job: Optional[str] = None
        self._busy_job: Optional[str] = None
        self._busy_frame = 0
        self._model_cards: dict[str, ModelCard] = {}
        self._pages: dict[str, tk.Frame] = {}
        self._latest_report: Optional[dict[str, Any]] = None

        self.model_key = tk.StringVar(value="voicetut")
        self.server_url = tk.StringVar(value="http://127.0.0.1:8004")
        self.speaker = tk.StringVar(value="Mohamed")
        self.reference_text = tk.StringVar(value="")
        self.num_step = tk.StringVar(value="12")
        self.speed = tk.StringVar(value="1.0")
        self.temperature = tk.StringVar(value="0.8")
        self.exaggeration = tk.StringVar(value="0.5")
        self.cfg_weight = tk.StringVar(value="0.5")
        self.seed = tk.StringVar(value="0")
        self.chunk_size = tk.StringVar(value="120")
        self.split_text = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Ready to synthesize")
        self._preload_started = False

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._configure_ttk()
        self._build_ui()
        self._show_model_info()
        self._update_server_panel()
        self._metrics_job = self.after(800, self._refresh_live_metrics)
        self.after(80, self._animate_header_line)
        self.after(120, self._log_compute_device)
        # Warm VoiceTut on CUDA so the first Generate is mostly synthesis time.
        self.after(400, self._preload_model)

    def _configure_ttk(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(
            "Vertical.TScrollbar",
            background=Theme.border,
            troughcolor=Theme.surface,
            bordercolor=Theme.surface,
            arrowcolor=Theme.ink_muted,
        )
        style.configure(
            "Nabra.TCheckbutton",
            background=Theme.surface_raised,
            foreground=Theme.ink,
            font=Theme.font_body,
            focuscolor=Theme.surface_raised,
        )
        style.map(
            "Nabra.TCheckbutton",
            background=[("active", Theme.surface_raised)],
            foreground=[("active", Theme.ink)],
        )

    def _build_ui(self) -> None:
        shell = tk.Frame(self, bg=Theme.bg)
        shell.pack(fill="both", expand=True, padx=18, pady=18)

        self._build_header(shell)

        body = tk.Frame(shell, bg=Theme.bg)
        body.pack(fill="both", expand=True, pady=(16, 0))

        self._build_sidebar(body)

        main = tk.Frame(
            body,
            bg=Theme.surface_raised,
            highlightbackground=Theme.border,
            highlightthickness=1,
        )
        main.pack(side="left", fill="both", expand=True)

        self.tab_bar = TabBar(
            main,
            [
                ("synthesize", "Synthesize"),
                ("model", "Model Notes"),
                ("results", "Results"),
                ("log", "Activity"),
            ],
            self._show_page,
        )
        self.tab_bar.pack(fill="x", padx=8, pady=(8, 0))

        tk.Frame(main, height=1, bg=Theme.border_soft).pack(fill="x", padx=18)

        self.page_host = tk.Frame(main, bg=Theme.surface_raised)
        self.page_host.pack(fill="both", expand=True, padx=18, pady=18)

        self._pages["synthesize"] = self._build_synthesize_page(self.page_host)
        self._pages["model"] = self._build_model_page(self.page_host)
        self._pages["results"] = self._build_results_page(self.page_host)
        self._pages["log"] = self._build_log_page(self.page_host)
        self._show_page("synthesize")
        self.after(50, lambda: self.tab_bar.select("synthesize", notify=False))

        self._build_footer(shell)

    def _build_header(self, parent: tk.Misc) -> None:
        header = tk.Frame(
            parent,
            bg=Theme.surface_raised,
            highlightbackground=Theme.border,
            highlightthickness=1,
        )
        header.pack(fill="x")

        band = tk.Canvas(header, height=8, highlightthickness=0, bd=0, bg=Theme.surface_raised)
        band.pack(fill="x")
        self._header_band = band
        self._header_offset = 0

        content = tk.Frame(header, bg=Theme.surface_raised)
        content.pack(fill="x", padx=22, pady=(10, 16))

        brand = tk.Frame(content, bg=Theme.surface_raised)
        brand.pack(side="left", fill="y")

        title_row = tk.Frame(brand, bg=Theme.surface_raised)
        title_row.pack(anchor="w")
        tk.Label(
            title_row,
            text="Nabra",
            font=Theme.font_brand,
            bg=Theme.surface_raised,
            fg=Theme.ink,
        ).pack(side="left")
        tk.Label(
            title_row,
            text="  نَبْرة",
            font=Theme.font_brand_ar,
            bg=Theme.surface_raised,
            fg=Theme.accent,
        ).pack(side="left", pady=(8, 0))

        tk.Label(
            brand,
            text="Arabic speech synthesis lab — compare local and server voices with clear timing metrics.",
            font=Theme.font_body,
            bg=Theme.surface_raised,
            fg=Theme.ink_soft,
        ).pack(anchor="w", pady=(2, 0))

        metrics_box = tk.Frame(content, bg=Theme.accent_soft)
        metrics_box.pack(side="right")
        inner = tk.Frame(metrics_box, bg=Theme.accent_soft)
        inner.pack(padx=14, pady=10)
        tk.Label(
            inner,
            text="LIVE SYSTEM",
            font=Theme.font_small,
            bg=Theme.accent_soft,
            fg=Theme.accent,
        ).pack(anchor="e")
        self.live_metrics = tk.Label(
            inner,
            text="Collecting…",
            font=Theme.font_body_bold,
            bg=Theme.accent_soft,
            fg=Theme.ink,
        )
        self.live_metrics.pack(anchor="e")

    def _animate_header_line(self) -> None:
        band = self._header_band
        band.delete("all")
        width = max(band.winfo_width(), 200)
        self._header_offset = (self._header_offset + 3) % width
        for i in range(0, width + 40, 40):
            x = (i + self._header_offset) % (width + 40) - 20
            band.create_rectangle(x, 0, x + 28, 8, fill=Theme.accent, outline="")
            band.create_rectangle(x + 28, 0, x + 40, 8, fill=Theme.accent_line, outline="")
        self.after(40, self._animate_header_line)

    def _build_sidebar(self, parent: tk.Misc) -> None:
        side = tk.Frame(
            parent,
            width=280,
            bg=Theme.surface,
            highlightbackground=Theme.border,
            highlightthickness=1,
        )
        side.pack(side="left", fill="y", padx=(0, 16))
        side.pack_propagate(False)

        tk.Label(
            side,
            text="ENGINE",
            font=Theme.font_small,
            bg=Theme.surface,
            fg=Theme.ink_muted,
        ).pack(anchor="w", padx=18, pady=(18, 8))

        cards = tk.Frame(side, bg=Theme.surface)
        cards.pack(fill="x", padx=12)
        for key, info in MODELS.items():
            card = ModelCard(
                cards,
                info,
                selected=(key == self.model_key.get()),
                on_select=self._select_model,
                ready=_backend_ready(key),
            )
            card.pack(fill="x", pady=5)
            self._model_cards[key] = card

        self.server_panel = tk.Frame(side, bg=Theme.surface)
        self.server_panel.pack(fill="x", padx=18, pady=(18, 0))
        tk.Label(
            self.server_panel,
            text="SERVER URL",
            font=Theme.font_small,
            bg=Theme.surface,
            fg=Theme.ink_muted,
        ).pack(anchor="w")
        entry_wrap = tk.Frame(
            self.server_panel,
            bg=Theme.surface_raised,
            highlightbackground=Theme.border,
            highlightthickness=1,
        )
        entry_wrap.pack(fill="x", pady=(6, 8))
        self.server_entry = tk.Entry(
            entry_wrap,
            textvariable=self.server_url,
            font=Theme.font_body,
            bd=0,
            relief="flat",
            bg=Theme.surface_raised,
            fg=Theme.ink,
            insertbackground=Theme.ink,
        )
        self.server_entry.pack(fill="x", padx=10, pady=8)
        AccentButton(
            self.server_panel,
            "Test connection",
            self._test_server,
            primary=False,
            width=244,
            height=36,
        ).pack(anchor="w")

        tip = tk.Frame(side, bg=Theme.accent_soft)
        tip.pack(side="bottom", fill="x", padx=14, pady=14)
        tk.Label(
            tip,
            text="Tip",
            font=Theme.font_body_bold,
            bg=Theme.accent_soft,
            fg=Theme.accent,
            anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 0))
        tk.Label(
            tip,
            text="Run each heavy model in its own environment. Use Chatterbox Server when you want isolation without reloading this app.",
            font=Theme.font_small,
            bg=Theme.accent_soft,
            fg=Theme.ink_soft,
            wraplength=230,
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=12, pady=(4, 12))

    def _build_synthesize_page(self, parent: tk.Misc) -> tk.Frame:
        page = tk.Frame(parent, bg=Theme.surface_raised)

        scroll = ScrollableFrame(page, bg=Theme.surface_raised)
        scroll.pack(fill="both", expand=True)
        body = scroll.body

        tk.Label(
            body,
            text="Script",
            font=Theme.font_section,
            bg=Theme.surface_raised,
            fg=Theme.ink,
        ).pack(anchor="w")
        tk.Label(
            body,
            text="Write Egyptian Arabic, English, or mixed code-switching text.",
            font=Theme.font_small,
            bg=Theme.surface_raised,
            fg=Theme.ink_muted,
        ).pack(anchor="w", pady=(0, 8))

        text_shell = tk.Frame(
            body,
            bg=Theme.surface,
            highlightbackground=Theme.border,
            highlightthickness=1,
        )
        text_shell.pack(fill="x")
        self.text_widget = tk.Text(
            text_shell,
            height=6,
            wrap="word",
            font=Theme.font_arabic,
            bd=0,
            padx=14,
            pady=12,
            bg=Theme.surface,
            fg=Theme.ink,
            insertbackground=Theme.accent,
            relief="flat",
        )
        self.text_widget.pack(fill="x")
        self.text_widget.insert("1.0", DEFAULT_TEXT)
        self.text_widget.tag_configure("rtl", justify="right")
        self.text_widget.tag_add("rtl", "1.0", "end")

        sample_row = tk.Frame(body, bg=Theme.surface_raised)
        sample_row.pack(fill="x", pady=(8, 0))
        AccentButton(
            sample_row,
            "Short sample",
            lambda: self._set_sample_text(DEFAULT_TEXT),
            primary=False,
            width=120,
            height=32,
        ).pack(side="left")
        AccentButton(
            sample_row,
            "Long sample",
            lambda: self._set_sample_text(LONG_SAMPLE_TEXT),
            primary=False,
            width=120,
            height=32,
        ).pack(side="left", padx=(8, 0))
        tk.Label(
            sample_row,
            text="Short text is much faster. Long text auto-chunks on VoiceTut.",
            font=Theme.font_small,
            bg=Theme.surface_raised,
            fg=Theme.ink_muted,
        ).pack(side="left", padx=(12, 0))

        tk.Label(
            body,
            text="Voice & generation",
            font=Theme.font_section,
            bg=Theme.surface_raised,
            fg=Theme.ink,
        ).pack(anchor="w", pady=(18, 8))

        params = tk.Frame(body, bg=Theme.surface_raised)
        params.pack(fill="x")
        rows = [
            ("Speaker / voice ID", self.speaker),
            ("Reference transcript", self.reference_text),
            ("VoiceTut steps", self.num_step),
            ("Speed", self.speed),
            ("Temperature", self.temperature),
            ("Exaggeration", self.exaggeration),
            ("CFG weight", self.cfg_weight),
            ("Seed", self.seed),
            ("Chunk size", self.chunk_size),
        ]
        for index, (label, variable) in enumerate(rows):
            cell = tk.Frame(params, bg=Theme.surface_raised)
            cell.grid(row=index // 3, column=index % 3, sticky="ew", padx=(0, 12), pady=6)
            tk.Label(
                cell,
                text=label,
                font=Theme.font_small,
                bg=Theme.surface_raised,
                fg=Theme.ink_muted,
                anchor="w",
            ).pack(fill="x")
            entry_wrap = tk.Frame(
                cell,
                bg=Theme.surface,
                highlightbackground=Theme.border_soft,
                highlightthickness=1,
            )
            entry_wrap.pack(fill="x", pady=(4, 0))
            tk.Entry(
                entry_wrap,
                textvariable=variable,
                font=Theme.font_body,
                bd=0,
                relief="flat",
                bg=Theme.surface,
                fg=Theme.ink,
                insertbackground=Theme.ink,
            ).pack(fill="x", padx=10, pady=8)
        for column in range(3):
            params.columnconfigure(column, weight=1)

        check_row = tk.Frame(body, bg=Theme.surface_raised)
        check_row.pack(fill="x", pady=(4, 0))
        ttk.Checkbutton(
            check_row,
            text="Split long text into chunks",
            variable=self.split_text,
            style="Nabra.TCheckbutton",
        ).pack(anchor="w")

        tk.Label(
            body,
            text="Reference audio",
            font=Theme.font_section,
            bg=Theme.surface_raised,
            fg=Theme.ink,
        ).pack(anchor="w", pady=(18, 8))

        ref = tk.Frame(
            body,
            bg=Theme.surface,
            highlightbackground=Theme.border_soft,
            highlightthickness=1,
        )
        ref.pack(fill="x")
        ref_inner = tk.Frame(ref, bg=Theme.surface)
        ref_inner.pack(fill="x", padx=14, pady=12)
        self.ref_label = tk.Label(
            ref_inner,
            text="No reference selected",
            font=Theme.font_body,
            bg=Theme.surface,
            fg=Theme.ink_muted,
            anchor="w",
        )
        self.ref_label.pack(side="left", fill="x", expand=True)
        AccentButton(
            ref_inner,
            "Clear",
            self._clear_reference,
            primary=False,
            width=84,
            height=34,
        ).pack(side="right", padx=(8, 0))
        AccentButton(
            ref_inner,
            "Browse",
            self._select_reference,
            primary=False,
            width=96,
            height=34,
        ).pack(side="right")

        actions = tk.Frame(body, bg=Theme.surface_raised)
        actions.pack(fill="x", pady=(22, 8))
        self.generate_button = AccentButton(
            actions,
            "Generate & benchmark",
            self._generate,
            primary=True,
            width=200,
            height=44,
        )
        self.generate_button.pack(side="left")
        AccentButton(
            actions,
            "Preload GPU",
            self._preload_model,
            primary=False,
            width=120,
            height=44,
        ).pack(side="left", padx=(10, 0))
        AccentButton(
            actions,
            "Play output",
            self._play,
            primary=False,
            width=120,
            height=44,
        ).pack(side="left", padx=(10, 0))
        AccentButton(
            actions,
            "Open folder",
            self._open_outputs,
            primary=False,
            width=120,
            height=44,
        ).pack(side="left", padx=(10, 0))
        AccentButton(
            actions,
            "Unload VRAM",
            self._unload,
            primary=False,
            width=130,
            height=44,
        ).pack(side="right")

        return page

    def _build_model_page(self, parent: tk.Misc) -> tk.Frame:
        page = tk.Frame(parent, bg=Theme.surface_raised)
        self.model_summary = tk.Frame(page, bg=Theme.surface_raised)
        self.model_summary.pack(fill="x", pady=(0, 12))
        shell = tk.Frame(
            page,
            bg=Theme.surface,
            highlightbackground=Theme.border_soft,
            highlightthickness=1,
        )
        shell.pack(fill="both", expand=True)
        self.info_text = tk.Text(
            shell,
            wrap="word",
            state="disabled",
            font=Theme.font_mono,
            bd=0,
            padx=16,
            pady=16,
            bg=Theme.surface,
            fg=Theme.ink_soft,
            relief="flat",
        )
        self.info_text.pack(fill="both", expand=True)
        return page

    def _build_results_page(self, parent: tk.Misc) -> tk.Frame:
        page = tk.Frame(parent, bg=Theme.surface_raised)

        tiles = tk.Frame(page, bg=Theme.surface_raised)
        tiles.pack(fill="x")
        self.metric_tiles = {
            "load": MetricTile(tiles, "LOAD TIME"),
            "generation": MetricTile(tiles, "GENERATION"),
            "duration": MetricTile(tiles, "AUDIO LENGTH"),
            "rtf": MetricTile(tiles, "REALTIME FACTOR"),
        }
        for index, tile in enumerate(self.metric_tiles.values()):
            tile.grid(row=0, column=index, sticky="ew", padx=(0 if index == 0 else 10, 0))
            tiles.columnconfigure(index, weight=1)

        self.result_model = tk.Label(
            page,
            text="Run a synthesis to populate benchmark results.",
            font=Theme.font_body,
            bg=Theme.surface_raised,
            fg=Theme.ink_muted,
            anchor="w",
        )
        self.result_model.pack(fill="x", pady=(16, 8))

        shell = tk.Frame(
            page,
            bg=Theme.surface,
            highlightbackground=Theme.border_soft,
            highlightthickness=1,
        )
        shell.pack(fill="both", expand=True)
        self.metrics_text = tk.Text(
            shell,
            wrap="word",
            state="disabled",
            font=Theme.font_mono,
            bd=0,
            padx=16,
            pady=16,
            bg=Theme.surface,
            fg=Theme.ink_soft,
            relief="flat",
        )
        self.metrics_text.pack(fill="both", expand=True)
        return page

    def _build_log_page(self, parent: tk.Misc) -> tk.Frame:
        page = tk.Frame(parent, bg=Theme.surface_raised)
        shell = tk.Frame(
            page,
            bg="#14202A",
            highlightbackground=Theme.border,
            highlightthickness=1,
        )
        shell.pack(fill="both", expand=True)
        self.log_text = tk.Text(
            shell,
            wrap="word",
            state="disabled",
            font=Theme.font_mono,
            bd=0,
            padx=16,
            pady=16,
            bg="#14202A",
            fg="#D7E6DF",
            insertbackground="#D7E6DF",
            relief="flat",
        )
        self.log_text.pack(fill="both", expand=True)
        return page

    def _build_footer(self, parent: tk.Misc) -> None:
        footer = tk.Frame(
            parent,
            bg=Theme.surface_raised,
            highlightbackground=Theme.border,
            highlightthickness=1,
        )
        footer.pack(fill="x", pady=(14, 0))
        inner = tk.Frame(footer, bg=Theme.surface_raised)
        inner.pack(fill="x", padx=16, pady=10)

        self.status_dot = tk.Canvas(
            inner, width=10, height=10, bg=Theme.surface_raised, highlightthickness=0
        )
        self.status_dot.pack(side="left", padx=(0, 8))
        self._draw_status_dot(Theme.success)

        self.status_label = tk.Label(
            inner,
            textvariable=self.status,
            font=Theme.font_body,
            bg=Theme.surface_raised,
            fg=Theme.ink_soft,
        )
        self.status_label.pack(side="left")

        tk.Label(
            inner,
            text="outputs/",
            font=Theme.font_small,
            bg=Theme.surface_raised,
            fg=Theme.ink_muted,
        ).pack(side="right")

    def _draw_status_dot(self, color: str) -> None:
        self.status_dot.delete("all")
        self.status_dot.create_oval(1, 1, 9, 9, fill=color, outline=color)

    def _show_page(self, key: str) -> None:
        page = self._pages.get(key)
        if page is None:
            return
        for other in self._pages.values():
            other.pack_forget()
        page.pack(fill="both", expand=True)

    def _select_model(self, key: str) -> None:
        if key not in MODELS:
            return
        self.model_key.set(key)
        for card_key, card in self._model_cards.items():
            card.set_selected(card_key == key)
        self._show_model_info()
        self._update_server_panel()
        self.status.set(f"Selected {MODELS[key].short_label}")
        if key == "voicetut":
            self.after(250, self._preload_model)

    def _set_sample_text(self, content: str) -> None:
        self.text_widget.delete("1.0", "end")
        self.text_widget.insert("1.0", content.strip())
        self.text_widget.tag_add("rtl", "1.0", "end")

    def _preload_model(self) -> None:
        key = self.model_key.get()
        if key != "voicetut":
            self._set_status(
                "Preload is for VoiceTut (local GPU model)",
                Theme.warn,
            )
            return
        if not _backend_ready("voicetut"):
            messagebox.showwarning(
                "VoiceTut missing",
                "Install VoiceTut first:\npip install -r requirements-voicetut.txt",
            )
            return

        adapter = self._get_adapter("voicetut", self.server_url.get().strip())
        if getattr(adapter, "model", None) is not None:
            self._set_status("VoiceTut already loaded on GPU", Theme.success)
            return
        if getattr(adapter, "_loading", False):
            self._set_status("VoiceTut is already loading…", Theme.warn)
            return

        self._set_status("Preloading VoiceTut onto CUDA…", Theme.warn)
        self._log("Preloading VoiceTut onto GPU in the background…")

        def worker() -> None:
            try:
                seconds = adapter.load()
                self._log(
                    f"VoiceTut ready on {getattr(adapter, 'device', '?')} "
                    f"({getattr(adapter, 'dtype', '?')}) in {seconds:.1f}s"
                )
                self.after(
                    0,
                    lambda: self._set_status(
                        f"VoiceTut loaded · {getattr(adapter, 'device', 'GPU')}",
                        Theme.success,
                    ),
                )
            except Exception as exc:
                self._log(f"Preload failed: {exc}\n{traceback.format_exc()}")
                self.after(
                    0,
                    lambda: messagebox.showerror("Preload failed", str(exc)),
                )
                self.after(
                    0,
                    lambda: self._set_status("Preload failed", Theme.danger),
                )

        threading.Thread(target=worker, daemon=True).start()

    def _update_server_panel(self) -> None:
        if self.model_key.get() == "chatterbox_server":
            self.server_panel.pack(fill="x", padx=18, pady=(18, 0))
        else:
            self.server_panel.pack_forget()

    def _show_model_info(self) -> None:
        info = MODELS[self.model_key.get()]
        for child in self.model_summary.winfo_children():
            child.destroy()

        badge = tk.Frame(self.model_summary, bg=Theme.accent_soft)
        badge.pack(anchor="w")
        tk.Label(
            badge,
            text=info.backend.upper(),
            font=Theme.font_small,
            bg=Theme.accent_soft,
            fg=Theme.accent,
            padx=10,
            pady=4,
        ).pack()

        tk.Label(
            self.model_summary,
            text=info.name,
            font=Theme.font_title,
            bg=Theme.surface_raised,
            fg=Theme.ink,
            anchor="w",
        ).pack(fill="x", pady=(10, 2))
        tk.Label(
            self.model_summary,
            text=f"{info.language}  ·  {info.license_name}",
            font=Theme.font_body,
            bg=Theme.surface_raised,
            fg=Theme.ink_muted,
            anchor="w",
        ).pack(fill="x")

        content = (
            f"Estimated VRAM\n{info.estimated_vram}\n\n"
            f"Description\n{info.description}\n\n"
            f"Strengths\n{info.strengths}\n\n"
            f"Limitations\n{info.limitations}\n\n"
            f"Installation\n{info.install_hint}\n"
        )
        self._set_text(self.info_text, content)

    def _get_adapter(self, key: str, server_url: str) -> AdapterBase:
        if key == "chatterbox_server":
            adapter = ChatterboxServerAdapter(server_url)
            self.adapters[key] = adapter
            return adapter
        if key not in self.adapters:
            if key == "voicetut":
                self.adapters[key] = VoiceTutAdapter()
            elif key == "namaa":
                self.adapters[key] = NAMAAAdapter()
            else:
                raise ValueError(f"Unknown model: {key}")
        return self.adapters[key]

    def _select_reference(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose reference audio",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.ogg"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.reference_audio = Path(selected)
            self.ref_label.config(
                text=self.reference_audio.name,
                fg=Theme.ink,
            )

    def _clear_reference(self) -> None:
        self.reference_audio = None
        self.ref_label.config(text="No reference selected", fg=Theme.ink_muted)

    def _collect_synth_options(self) -> dict[str, Any]:
        try:
            num_step = int(float(self.num_step.get().strip() or "12"))
            speed = float(self.speed.get().strip() or "1.0")
            temperature = float(self.temperature.get().strip() or "0.8")
            exaggeration = float(self.exaggeration.get().strip() or "0.5")
            cfg_weight = float(self.cfg_weight.get().strip() or "0.5")
            seed = int(float(self.seed.get().strip() or "0"))
            chunk_size = int(float(self.chunk_size.get().strip() or "120"))
        except ValueError as exc:
            raise ValueError(
                "One or more parameters are not valid numbers. "
                "Check steps, speed, temperature, exaggeration, "
                "CFG weight, seed, and chunk size."
            ) from exc

        return {
            "speaker": self.speaker.get(),
            "reference_text": self.reference_text.get(),
            "num_step": max(1, num_step),
            "speed": _clamp(speed, 0.25, 4.0),
            "temperature": _clamp(temperature, 0.0, 1.5),
            "exaggeration": _clamp(exaggeration, 0.25, 2.0),
            "cfg_weight": _clamp(cfg_weight, 0.2, 1.0),
            "seed": max(0, seed),
            "chunk_size": int(_clamp(chunk_size, 50, 500)),
            "split_text": bool(self.split_text.get()),
        }

    def _generate(self) -> None:
        if self._generating:
            return

        text = self.text_widget.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("No text", "Enter text to synthesize.")
            return

        try:
            options = self._collect_synth_options()
        except ValueError as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return

        key = self.model_key.get()
        if key not in MODELS:
            messagebox.showerror("Unknown model", f"Unrecognized model key: {key}")
            return

        server_url = self.server_url.get().strip()
        reference_audio = self.reference_audio

        self._generating = True
        self.generate_button.set_enabled(False)
        self.generate_button.configure_text("Generating…")
        self.status.set("Generating speech and collecting metrics…")
        self._draw_status_dot(Theme.warn)
        self._start_busy_animation()
        threading.Thread(
            target=self._generate_worker,
            args=(text, key, server_url, reference_audio, options),
            daemon=True,
        ).start()

    def _start_busy_animation(self) -> None:
        self._busy_frame = 0

        def tick() -> None:
            if not self._generating:
                self._busy_job = None
                return
            dots = "." * ((self._busy_frame % 3) + 1)
            self.generate_button.configure_text(f"Generating{dots}")
            self._busy_frame += 1
            self._busy_job = self.after(400, tick)

        tick()

    def _generate_worker(
        self,
        text: str,
        key: str,
        server_url: str,
        reference_audio: Optional[Path],
        options: dict[str, Any],
    ) -> None:
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = OUTPUT_DIR / f"{key}-{timestamp}.wav"
            adapter = self._get_adapter(key, server_url)

            before = Metrics.system()
            wall_start = time.perf_counter()
            result = adapter.synthesize(
                text,
                output_path,
                reference_audio,
                options,
            )
            wall_seconds = time.perf_counter() - wall_start
            after = Metrics.system()

            duration = audio_duration(output_path)
            generation_seconds = float(result["generation_seconds"])
            rtf = generation_seconds / duration if duration else None

            report = {
                "timestamp": timestamp,
                "model": MODELS[key].name,
                "text": text,
                "reference_audio": str(reference_audio) if reference_audio else None,
                "load_seconds": round(float(result.get("load_seconds", 0.0)), 3),
                "generation_seconds": round(generation_seconds, 3),
                "wall_seconds": round(wall_seconds, 3),
                "audio_duration_seconds": round(duration, 3),
                "realtime_factor": round(rtf, 3) if rtf is not None else None,
                "compute": _cuda_status(),
                "device": result.get("device") or _cuda_status().get("device"),
                "dtype": result.get("dtype"),
                "before": before,
                "after": after,
                "output_path": str(output_path),
            }
            report_path = output_path.with_suffix(".json")
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.current_output = output_path

            self.after(0, lambda report=report: self._show_result(report))
            self._log(json.dumps(report, ensure_ascii=False, indent=2))
        except Exception as exc:
            error = f"{exc}\n\n{traceback.format_exc()}"
            self._log(error)
            message = str(exc)
            self.after(
                0,
                lambda message=message: messagebox.showerror(
                    "Generation failed", message
                ),
            )
            self.after(0, lambda: self._set_status("Generation failed", Theme.danger))
        finally:
            self.after(0, self._generation_finished)

    def _generation_finished(self) -> None:
        self._generating = False
        if self._busy_job is not None:
            try:
                self.after_cancel(self._busy_job)
            except Exception:
                pass
            self._busy_job = None
        self.generate_button.configure_text("Generate & benchmark")
        self.generate_button.set_enabled(True)

    def _set_status(self, text: str, color: str) -> None:
        self.status.set(text)
        self._draw_status_dot(color)

    def _show_result(self, report: dict[str, Any]) -> None:
        self._latest_report = report
        self.metric_tiles["load"].set_value(f"{report['load_seconds']}s")
        self.metric_tiles["generation"].set_value(f"{report['generation_seconds']}s")
        self.metric_tiles["duration"].set_value(f"{report['audio_duration_seconds']}s")
        rtf = report["realtime_factor"]
        self.metric_tiles["rtf"].set_value("—" if rtf is None else f"{rtf}")

        self.result_model.configure(
            text=f"{report['model']}  ·  saved to {Path(report['output_path']).name}",
            fg=Theme.ink,
        )

        compute = report.get("compute") or {}
        device_line = report.get("device") or compute.get("device") or "unknown"
        dtype_line = report.get("dtype")
        device_detail = f"Device: {device_line}"
        if dtype_line:
            device_detail += f" ({dtype_line})"
        if compute.get("name"):
            device_detail += f" · {compute['name']}"

        lines = [
            f"Model: {report['model']}",
            device_detail,
            f"Load time: {report['load_seconds']} s",
            f"Generation time: {report['generation_seconds']} s",
            f"Wall time: {report['wall_seconds']} s",
            f"Output duration: {report['audio_duration_seconds']} s",
            f"Realtime factor: {report['realtime_factor']}",
            f"Output: {report['output_path']}",
            "",
            "Memory before:",
            json.dumps(report["before"], indent=2),
            "",
            "Memory after:",
            json.dumps(report["after"], indent=2),
        ]
        self._set_text(self.metrics_text, "\n".join(lines))
        self._set_status("Benchmark complete", Theme.success)
        self.tab_bar.select("results")

    def _play(self) -> None:
        if not self.current_output or not self.current_output.exists():
            messagebox.showinfo("No output", "Generate audio first.")
            return
        if sd is None or sf is None:
            os.startfile(str(self.current_output))
            return
        try:
            audio, rate = sf.read(str(self.current_output), dtype="float32")
            sd.stop()
            sd.play(audio, rate)
            self._set_status("Playing last output", Theme.accent)
        except Exception as exc:
            messagebox.showerror("Playback error", str(exc))

    def _open_outputs(self) -> None:
        OUTPUT_DIR.mkdir(exist_ok=True)
        os.startfile(str(OUTPUT_DIR))

    def _unload(self) -> None:
        try:
            adapter = self.adapters.get(self.model_key.get())
            if adapter:
                adapter.unload()
            _cleanup_cuda()
            self._set_status("Model unloaded and cache cleared", Theme.success)
            self._log("Model unloaded and CUDA cache cleanup requested.")
        except Exception as exc:
            messagebox.showerror("Unload error", str(exc))

    def _test_server(self) -> None:
        url = self.server_url.get().strip()
        self._set_status("Testing Chatterbox server…", Theme.warn)

        def worker() -> None:
            try:
                adapter = ChatterboxServerAdapter(url)
                info = adapter.health()
                voices = adapter.list_voices()
                result = (
                    "Chatterbox server is reachable.\n\n"
                    + json.dumps(info, ensure_ascii=False, indent=2)
                    + ("\n\nVoices:\n" + "\n".join(voices) if voices else "")
                )
                self.after(0, lambda result=result: self._set_text(self.metrics_text, result))
                self.after(0, lambda: self._set_status("Server connected", Theme.success))
                self.after(0, lambda: self.tab_bar.select("results"))
            except Exception as exc:
                message = str(exc)
                self.after(
                    0,
                    lambda message=message: messagebox.showerror(
                        "Server connection failed",
                        f"{message}\n\nStart Chatterbox-TTS-Server first.",
                    ),
                )
                self.after(
                    0,
                    lambda: self._set_status("Server connection failed", Theme.danger),
                )

        threading.Thread(target=worker, daemon=True).start()

    def _log_compute_device(self) -> None:
        def worker() -> None:
            status = _cuda_status(force_refresh=True)
            if str(status.get("device", "")).startswith("cuda"):
                _configure_cuda_runtime()
                message = (
                    "CUDA ready: "
                    f"{status.get('name')} · {status.get('device')} · "
                    f"{status.get('vram_total_mb')} MB · "
                    f"torch {status.get('torch_version')} · "
                    f"cuda {status.get('cuda_version')}"
                )
                self.after(
                    0,
                    lambda: self._set_status(
                        f"CUDA · {status.get('name', 'GPU')}",
                        Theme.success,
                    ),
                )
            else:
                message = (
                    "CUDA not available in this Python — models will run on CPU "
                    "(much slower). Install PyTorch with CUDA for your GPU."
                )
                self.after(
                    0,
                    lambda: self._set_status(
                        "CPU only · CUDA unavailable",
                        Theme.warn,
                    ),
                )
            self._log(message)

        threading.Thread(target=worker, daemon=True).start()

    def _refresh_live_metrics(self) -> None:
        try:
            metrics = Metrics.system()
            compute = _CUDA_STATUS_CACHE or {}
            device_label = (
                "CUDA"
                if str(compute.get("device", "")).startswith("cuda")
                else ("…" if _CUDA_STATUS_CACHE is None else "CPU")
            )
            text = (
                f"{device_label}   "
                f"RAM {metrics['process_ram_mb']:.0f} MB   "
                f"CPU {metrics['cpu_percent']:.0f}%"
            )
            if "gpu_used_mb" in metrics:
                text += (
                    f"   GPU {metrics['gpu_used_mb']:.0f}/"
                    f"{metrics['gpu_total_mb']:.0f} MB "
                    f"({metrics['gpu_util_percent']}%)"
                )
            self.live_metrics.config(text=text)
        except Exception:
            pass
        self._metrics_job = self.after(1000, self._refresh_live_metrics)

    def _log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")

        def append() -> None:
            self.log_text.config(state="normal")
            self.log_text.insert("end", f"[{stamp}]\n{message}\n\n")
            self.log_text.see("end")
            self.log_text.config(state="disabled")

        self.after(0, append)

    def _on_close(self) -> None:
        for job in (self._metrics_job, self._busy_job):
            if job is not None:
                try:
                    self.after_cancel(job)
                except Exception:
                    pass
        self._metrics_job = None
        self._busy_job = None
        if sd is not None:
            try:
                sd.stop()
            except Exception:
                pass
        try:
            self.unbind_all("<MouseWheel>")
        except Exception:
            pass
        self.destroy()

    @staticmethod
    def _set_text(widget: tk.Text, content: str) -> None:
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.config(state="disabled")


if __name__ == "__main__":
    app = TTSBenchmarkApp()
    app.mainloop()
