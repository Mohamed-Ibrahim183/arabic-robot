from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_FILE = PROJECT_ROOT / "audio" / "arabic_test.wav"
LOCAL_MODEL_DIR = PROJECT_ROOT / "models" / "faster-whisper-small"


def _pick_whisper_device() -> tuple[str, str]:
    """Prefer CUDA float16; fall back to CPU int8."""
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass

    return "cpu", "int8"


def main() -> None:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit(
            "faster-whisper is not installed.\n"
            "Install with:\n"
            "  pip install -r requirements-scripts.txt"
        ) from exc

    if not AUDIO_FILE.exists():
        raise FileNotFoundError(
            f"Missing audio file: {AUDIO_FILE}\n"
            "Record one first with: python scripts/record_audio.py"
        )

    if LOCAL_MODEL_DIR.exists() and (LOCAL_MODEL_DIR / "model.bin").exists():
        model_path: str | Path = LOCAL_MODEL_DIR
        print(f"Loading local model: {LOCAL_MODEL_DIR}")
    else:
        model_path = "small"
        print(
            "Local model not found at models/faster-whisper-small; "
            "falling back to Hugging Face 'small'."
        )

    device, compute_type = _pick_whisper_device()
    print(f"Compute: device={device}, compute_type={compute_type}")

    load_start = time.perf_counter()
    try:
        model = WhisperModel(
            str(model_path),
            device=device,
            compute_type=compute_type,
            cpu_threads=8 if device == "cpu" else 0,
        )
    except Exception as exc:
        if device != "cpu":
            print(
                f"CUDA load failed ({exc}); falling back to CPU int8.",
                file=sys.stderr,
            )
            device, compute_type = "cpu", "int8"
            model = WhisperModel(
                str(model_path),
                device=device,
                compute_type=compute_type,
                cpu_threads=8,
            )
        else:
            raise
    load_time = time.perf_counter() - load_start

    transcribe_start = time.perf_counter()
    segments, info = model.transcribe(
        str(AUDIO_FILE),
        language="ar",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    segments = list(segments)
    transcribe_time = time.perf_counter() - transcribe_start

    transcript = " ".join(
        segment.text.strip() for segment in segments if segment.text.strip()
    )

    print(f"\nModel loading time: {load_time:.2f} seconds")
    print(f"Transcription time: {transcribe_time:.2f} seconds")
    print(f"Detected language: {info.language}")
    print(f"Language probability: {info.language_probability:.3f}")

    print("\nTranscript:")
    print(transcript or "[No speech detected]")

    print("\nSegments:")
    for segment in segments:
        print(
            f"[{segment.start:.2f}s -> {segment.end:.2f}s] "
            f"{segment.text.strip()}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
