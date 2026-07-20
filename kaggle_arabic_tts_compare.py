#!/usr/bin/env python3
"""
Kaggle Arabic TTS listening comparison (T4 / T4x2 safe)
=======================================================
Each model runs in its own subprocess with a dedicated GPU so a CUDA
assert in one model cannot poison the rest of the notebook session.

Usage on Kaggle:
  1) Runtime → Restart session (important if CUDA was already poisoned)
  2) install_packages()
  3) main()
  4) Download /kaggle/working/tts_outputs/*.wav
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
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

OUTPUT_DIR = WORK_DIR / "tts_outputs"
CACHE_DIR = WORK_DIR / "tts_cache"
REPOS_DIR = WORK_DIR / "repos"
META_DIR = WORK_DIR / "tts_meta"
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
    "وإن في خصم للعقد السنوي. شكراً ليك، وياريت الصوت يبقى واضح وطبيعي."
)

ENABLE = {
    "VoiceTut-TTS": True,
    "NAMAA-Egyptian-TTS": True,
    "Chatterbox-Multilingual-V3": True,
    "Qwen3-TTS-0.6B": True,
    "Kokoro-82M": True,
    "CosyVoice-0.5B": True,
    "Fish-Speech": True,
}

# Order: Arabic refs first, then models that need a reference / can crash CUDA.
MODEL_ORDER = [
    "NAMAA-Egyptian-TTS",
    "Chatterbox-Multilingual-V3",
    "VoiceTut-TTS",
    "Kokoro-82M",
    "Qwen3-TTS-0.6B",
    "CosyVoice-0.5B",
    "Fish-Speech",
]

VOICETUT_REPO = "mohammedaly22/VoiceTut-TTS"
VOICETUT_SPEAKER = "Mohamed"
VOICETUT_CHUNK_CHARS = 180
QWEN3_BASE = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
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
_pip("HyperPyYAML", "wetext", "modelscope", "pyarrow", "openai-whisper", "onnxruntime")
_pip("pyrootutils", "loguru", "lightning", "hydra-core", "tiktoken", "vector_quantize_pytorch")

# Keep transformers new enough for VoiceTut after all other installs.
_pip("-U", "transformers>=5.3.0")
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


def run_git_clone(url: str, dest: Path, recursive: bool = False) -> Path:
    if dest.exists() and any(dest.iterdir()):
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    cmd = ["git", "clone", "--depth", "1"]
    if recursive:
        cmd.append("--recursive")
    cmd.extend([url, str(dest)])
    subprocess.check_call(cmd)
    return dest


def _pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])


def ensure_transformers_for_voicetut() -> None:
    """omnivoice needs HiggsAudioV2TokenizerModel (transformers>=5.3)."""
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
    ):
        p = OUTPUT_DIR / name
        if p.exists() and p.stat().st_size > 44_000:
            return p
    return None


def write_meta(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def synth_qwen3(text: str, out_path: Path, ref_audio: Optional[Path] = None) -> dict[str, Any]:
    """
    Skip CustomVoice for Arabic — it triggers CUDA device asserts.
    Use Base 0.6B voice-clone with greedy decoding + a prior Arabic ref clip.
    """
    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    if ref_audio is None or not Path(ref_audio).exists():
        raise RuntimeError(
            "Qwen3 Base voice-clone needs a reference WAV "
            "(run NAMAA/Chatterbox/VoiceTut first)."
        )

    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    t0 = time.perf_counter()
    try:
        model = Qwen3TTSModel.from_pretrained(
            QWEN3_BASE,
            device_map=device_map,
            dtype=dtype,
            attn_implementation="sdpa",
        )
    except Exception:
        model = Qwen3TTSModel.from_pretrained(
            QWEN3_BASE,
            device_map=device_map,
            dtype=dtype,
        )
    load_s = time.perf_counter() - t0

    ref_text = split_text_chunks(text, 120)[0]
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
    ):
        try:
            _pip(pkg)
        except Exception:
            pass
    try:
        _pip("pyworld")
    except Exception:
        print("pyworld optional install failed; continuing")

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
    """fish-speech-1.5 only (openaudio-s1-mini is gated without HF access)."""
    from huggingface_hub import snapshot_download

    repo = run_git_clone(
        "https://github.com/fishaudio/fish-speech.git",
        REPOS_DIR / "fish-speech",
        recursive=False,
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
        "natsort",
        "vector_quantize_pytorch",
        "descript-audio-codec",
        "ormsgpack",
        "transformers",
        "accelerate",
    ):
        try:
            _pip(pkg)
        except Exception:
            pass

    ckpt = CACHE_DIR / "fish-speech-1.5"
    if not ckpt.exists() or not any(ckpt.iterdir()):
        snapshot_download(FISH_SPEECH_HF, local_dir=str(ckpt))

    decoder = ckpt / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
    if not decoder.exists():
        decoder = ckpt / "codec.pth"
    if not decoder.exists():
        raise FileNotFoundError(f"Fish decoder missing in {ckpt}")

    # 1.5 uses vqgan path; newer repos may only ship dac.
    codec_script = None
    for rel in (
        "fish_speech/models/vqgan/inference.py",
        "fish_speech/models/dac/inference.py",
    ):
        if (repo / rel).exists():
            codec_script = rel
            break
    if codec_script is None:
        raise FileNotFoundError("Fish codec inference script not found")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.perf_counter()

    prompt_tokens = None
    prompt_text = None
    if ref_audio and Path(ref_audio).exists():
        subprocess.check_call(
            [
                sys.executable,
                codec_script,
                "-i",
                str(ref_audio),
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

    load_s = time.perf_counter() - t0
    t1 = time.perf_counter()

    cmd = [
        sys.executable,
        "fish_speech/models/text2semantic/inference.py",
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
    try:
        meta = RUNNERS[model_name](text, out_path, ref_audio)
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise RuntimeError(f"No audio written to {out_path}")
        meta.update({"status": "ok", "output": str(out_path), "bytes": out_path.stat().st_size})
        write_meta(meta_path, meta)
        print(f"[worker] OK {model_name} → {out_path} ({out_path.stat().st_size} bytes)")
        return 0
    except Exception as exc:
        payload = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-4000:],
        }
        write_meta(meta_path, payload)
        print(f"[worker] FAIL {model_name}: {payload['error']}")
        traceback.print_exc()
        return 1
    finally:
        cleanup_gpu()


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
        str(Path(__file__).resolve()),
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


def _try_display(path: Path) -> None:
    try:
        from IPython.display import Audio, display

        print(f"▶ {path.name}")
        display(Audio(filename=str(path)))
    except Exception:
        pass


def main(text: str = DEFAULT_TEXT, run_install: bool = False) -> dict[str, Any]:
    if run_install:
        install_packages()

    n_gpu = gpu_count()
    gpus = list_gpus()
    print(f"GPUs detected: {n_gpu}")
    print(json.dumps(gpus, indent=2))
    if n_gpu >= 2:
        print("T4x2 / multi-GPU mode: models alternate across cuda:0 and cuda:1 via CUDA_VISIBLE_DEVICES")
    print(f"Text ({len(text)} chars)")
    print(f"Output: {OUTPUT_DIR}")

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
            print(
                f"✓ {name}: load={meta.get('load_seconds', 0):.1f}s  "
                f"gen={meta.get('generation_seconds', 0):.1f}s  gpu={gpu}"
            )
            _try_display(out)
        else:
            print(f"✗ {name}: {meta.get('error')}")

    summary_path = OUTPUT_DIR / "summary.json"
    write_meta(summary_path, results)
    ok = [k for k, v in results["models"].items() if v.get("status") == "ok"]
    bad = [k for k, v in results["models"].items() if v.get("status") == "error"]
    print(f"\nDone. Summary → {summary_path}")
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--model")
    p.add_argument("--text-file", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--meta", type=Path)
    p.add_argument("--ref", type=Path, default=None)
    p.add_argument("--install", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.worker:
        raise SystemExit(
            worker_main(args.model, args.text_file, args.out, args.ref, args.meta)
        )
    main(run_install=args.install or os.environ.get("RUN_INSTALL", "0") == "1")
