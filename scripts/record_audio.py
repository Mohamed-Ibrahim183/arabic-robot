from __future__ import annotations

import sys
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16000
DURATION_SECONDS = 10
OUTPUT_FILE = PROJECT_ROOT / "audio" / "arabic_test.wav"


def main() -> None:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit(
            "sounddevice is not installed.\n"
            "Install with:\n"
            "  pip install -r requirements-scripts.txt"
        ) from exc

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Recording {DURATION_SECONDS}s at {SAMPLE_RATE} Hz. Speak Arabic.")
    audio = sd.rec(
        int(DURATION_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    sd.wait()

    with wave.open(str(OUTPUT_FILE), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(audio.tobytes())

    print(f"Saved: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
