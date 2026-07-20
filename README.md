# Nabra · Arabic TTS Benchmark

Desktop lab UI for comparing Arabic TTS engines with timing and memory metrics.

This GUI compares:

1. VoiceTut-TTS through its Python API.
2. Chatterbox-TTS-Server through HTTP.
3. NAMAA Egyptian TTS through its Python API.

## Important architecture

Do not install every TTS project into the existing `robot` environment. Their
PyTorch and audio dependencies may conflict. The safest setup is:

- Run this GUI from a light environment.
- Run Chatterbox Server from its own portable installation.
- For VoiceTut or NAMAA direct-local testing, copy the GUI into that model's
  environment and launch it there.

A later version can use a small local service for each model so the same GUI
controls all three environments simultaneously.

## Install the GUI

```bat
cd C:\Users\pc\arabic-test
conda create -n tts-gui python=3.11 -y
conda activate tts-gui
pip install -r requirements-gui.txt
python arabic_tts_benchmark_gui.py
```

Tkinter normally ships with Conda's Windows Python.

## Chatterbox-TTS-Server: recommended first test

```bat
cd C:\Users\pc\arabic-test
git clone https://github.com/devnen/Chatterbox-TTS-Server.git
cd Chatterbox-TTS-Server
start.bat
```

On first launch:

- Choose Portable Mode.
- Choose NVIDIA GPU / CUDA 12.1.
- In the web UI select Chatterbox Multilingual for Arabic.
- Keep the server running.
- In the benchmark GUI select `Chatterbox TTS Server`.
- The default server address is often `http://127.0.0.1:8004`.
- Press `Test Server`.

Your GTX 1660 Super has 6 GB VRAM. Close Rocket League, browsers with hardware
acceleration, Lively Wallpaper and other GPU-heavy programs before testing.

## VoiceTut-TTS

If PyTorch CUDA is already installed in this Python, install VoiceTut with:

```bat
cd C:\Users\pc\arabic-test
pip install -r requirements-voicetut.txt
python arabic_tts_benchmark_gui.py
```

Select `VoiceTut`. Default speaker is `Mohamed`. Other built-in IDs include
`Asmaa`. First run downloads model weights from Hugging Face and can take a while.

Begin with `num_step=16` on a 6 GB GPU. Higher values may improve quality but
increase latency.

## NAMAA environment

NAMAA is based on Chatterbox Multilingual. Install it in a dedicated Python
3.10 environment according to the current Chatterbox package instructions.
The GUI adapter follows the exact NAMAA model-card loading method:

- Download the NAMAA checkpoint with `snapshot_download`.
- Load `ChatterboxMultilingualTTS`.
- Replace its T3 state with `t3_mtl23ls_v2.safetensors`.
- Generate with `language_id="ar"`.

When the required `chatterbox`, `torch`, `torchaudio`, `huggingface_hub`, and
`safetensors` packages are installed, launch the GUI from that environment and
select `NAMAA Egyptian TTS`.

## Whisper scripts (optional)

The `scripts/` folder has a separate Arabic STT smoke test that uses the local
`models/faster-whisper-small` checkpoint when present:

```bat
cd C:\Users\pc\arabic-test
pip install -r requirements-scripts.txt
python scripts/record_audio.py
python scripts/test_whisper.py
```

Paths are resolved from the project root, so these work from any working directory.

## Metrics

Each generation writes:

- WAV output.
- JSON report.
- Model-load time.
- Generation time.
- Audio duration.
- Realtime factor.
- Process RAM.
- System RAM.
- GPU VRAM and utilization when NVML is available.

Outputs are saved to the `outputs` folder.

Realtime factor interpretation:

- `< 1.0`: faster than realtime.
- `1.0`: five seconds of audio take five seconds to generate.
- `> 1.0`: slower than realtime.

For a robot, also manually measure time to first audio. This GUI currently
measures complete-file generation. Chatterbox's server streaming is chunk-level,
and VoiceTut has a separate stream API that can be added in a subsequent test.
