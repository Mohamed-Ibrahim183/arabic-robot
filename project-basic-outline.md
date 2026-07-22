# Complete Chat Transfer Summary — Arabic Voice Robot Project

> Scope note: This document captures every recoverable detail available in the current conversation context, including the summarized earlier discussion and the visible question/answer sequence. It is not a verbatim export of hidden or unavailable chat messages.

---

## 1. Project Goal

The user has a customer project to build a real-time Arabic-speaking personal robot using an ESP32.

The current customer implementation simply calls the ChatGPT API, but the user considers it:
- slow
- not good enough for real-time interaction
- potentially unsuitable for scaling to many robots

The goal is to research and test a better architecture with:
- high-quality Arabic ASR
- high-quality Arabic TTS
- Egyptian Arabic support
- Arabic-English code-switching
- very low latency
- real-time turn-taking
- scalability to many ESP32 robots
- possible deployment on a VPS
- local testing first before deployment

---

## 2. Main Architectural Questions Discussed

### Question: Will the robot record everything and send everything to the server?

Answer:
No. The better design is not to continuously record and upload endless full audio files.

Recommended behavior:
- use wake word detection, push-to-talk, or voice activity detection
- start sending when speech begins
- send small audio chunks over a persistent connection
- stop sending when speech ends
- avoid sending silence
- optionally keep some processing on the ESP32, such as:
  - wake word
  - VAD
  - microphone capture
  - echo cancellation
  - audio compression
  - playback

Recommended streaming formats/protocols:
- WebSocket
- WebRTC
- compressed audio such as Opus where appropriate

The robot can send:
- one whole recorded audio file, which is simpler but slower
- multiple streaming audio chunks, which is preferred for real-time conversation

### Question: What does the server send back to the robot?

Answer:
The server can return:
- transcription text
- LLM response text
- synthesized speech audio
- preferably streaming audio chunks so playback starts before the complete response is generated

Typical pipeline:

ESP32 microphone  
→ streaming audio chunks  
→ server-side ASR  
→ transcript  
→ LLM  
→ response text  
→ TTS  
→ streaming audio chunks  
→ ESP32 speaker

### Question: Why is an LLM needed?

Answer:
An LLM is not mandatory for fixed commands, but is useful for:
- natural conversation
- understanding flexible user intent
- multi-turn context
- reasoning
- answering open-ended questions
- remembering conversation state
- tool use
- scheduling
- smart-home control
- dialog management

For a command-only robot, a rules engine or intent classifier may be enough.

---

## 3. Recommended Production Architecture

Recommended high-level flow:

ESP32  
→ wake word or push-to-talk  
→ VAD  
→ audio chunk streaming  
→ backend gateway  
→ streaming ASR  
→ dialog/LLM layer  
→ streaming TTS  
→ audio streamed back to ESP32

ESP32 responsibilities:
- microphone input
- speaker output
- wake word
- VAD
- echo cancellation where possible
- audio buffering
- Opus or PCM streaming
- playback
- network connection management

Server responsibilities:
- ASR
- LLM
- TTS
- session management
- authentication
- rate limiting
- per-robot state
- caching
- observability
- scaling

Possible frameworks/services discussed:
- LiveKit
- Pipecat
- Azure Voice Live
- Gemini Live
- ElevenLabs Agents
- Hume EVI
- Deepgram
- Google Cloud Speech
- AWS speech services
- Mistral Voxtral

Recommended self-hosted/VPS pipeline:

ESP32  
→ WebSocket  
→ Streaming STT  
→ LLM  
→ Streaming TTS  
→ ESP32

---

## 4. Local Testing Machine

User machine:
- Operating system: Windows
- GPU: NVIDIA GTX 1660 Super
- GPU memory: 6 GB
- CPU: Intel Core i5-10400F
- RAM: 16 GB

Implications:
- suitable for smaller or quantized models
- can test Whisper Small and some medium-sized models
- 6 GB VRAM requires careful unloading and separate environments
- large TTS and LLM models may need CPU offload, quantization, or VPS testing
- do not load multiple large models simultaneously

---

## 5. Local Environment Setup

Miniconda was installed.

Base environment:
- Python 3.14
- Conda 26.5.3
- installed under:
  `C:\ProgramData\miniconda3`

An update of the base environment failed because `ProgramData` was not writable for the current user. This was considered non-blocking.

A dedicated Conda environment was created:

```bat
conda create -n robot python=3.11 -y
```

Environment location:

```text
C:\Users\pc\.conda\envs\robot
```

Activation:

```bat
conda activate robot
```

Project directory:

```text
C:\Users\pc\arabic-test
```

---

## 6. Faster-Whisper Installation

Installed packages included:
- faster-whisper
- sounddevice
- scipy
- numpy

Command used:

```bat
pip install faster-whisper sounddevice scipy numpy
```

A problem was observed:
some packages were being imported from the user-level Python folder:

```text
C:\Users\pc\AppData\Roaming\Python\Python311\site-packages
```

This caused dependency leakage into the Conda environment.

Recommended fix:

```bat
conda env config vars set PYTHONNOUSERSITE=1
conda deactivate
conda activate robot
```

Verification command:

```bat
python -c "import site; print('User site enabled:', site.ENABLE_USER_SITE); print(site.getusersitepackages())"
```

Observed result:

```text
User site enabled: False
C:\Users\pc\AppData\Roaming\Python\Python311\site-packages
```

Interpretation:
- user site path still exists as a known path
- but `ENABLE_USER_SITE` is `False`
- therefore the Conda environment is correctly isolated

---

## 7. Reinstalling SoundDevice

The user ran:

```bat
python -m pip install --force-reinstall sounddevice
```

Output showed successful reinstall of:
- sounddevice 0.5.5
- cffi 2.1.0
- pycparser 3.0

Then the user ran:

```bat
python -c "import faster_whisper, sounddevice, numpy, scipy; print('faster-whisper:', faster_whisper.__file__); print('sounddevice:', sounddevice.__file__); print('numpy:', numpy.__file__); print('scipy:', scipy.__file__)"
```

Error:

```text
ModuleNotFoundError: No module named 'tqdm'
```

Explanation:
`tqdm` had previously been coming from the user-site packages. Once isolation was enabled, it was no longer available inside the `robot` environment.

Recommended fix:

```bat
python -m pip install tqdm flatbuffers
```

Then re-test imports:

```bat
python -c "import faster_whisper, sounddevice, numpy, scipy; print('faster-whisper:', faster_whisper.__file__); print('sounddevice:', sounddevice.__file__); print('numpy:', numpy.__file__); print('scipy:', scipy.__file__)"
```

Expected all paths to start with:

```text
C:\Users\pc\.conda\envs\robot\
```

Additional checks:

```bat
python -m pip check
```

Expected:

```text
No broken requirements found.
```

And:

```bat
python -c "import tqdm, flatbuffers, faster_whisper; print('All imports successful')"
```

Expected:

```text
All imports successful
```

---

## 8. Suggested Project Structure

Recommended structure:

```text
arabic-test
│
├── audio
├── scripts
├── output
└── results
```

Commands:

```bat
mkdir audio
mkdir scripts
mkdir results
```

---

## 9. Microphone Recording Script

A script named:

```text
scripts\record_audio.py
```

was proposed.

Purpose:
- record 10 seconds
- 16 kHz
- mono
- int16
- save to:
  `audio/arabic_test.wav`

Code:

```python
from pathlib import Path

import sounddevice as sd
from scipy.io.wavfile import write


SAMPLE_RATE = 16000
DURATION_SECONDS = 10
OUTPUT_FILE = Path("audio/arabic_test.wav")


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("Recording starts now. Speak Arabic.")

    audio = sd.rec(
        int(DURATION_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )

    sd.wait()
    write(OUTPUT_FILE, SAMPLE_RATE, audio)

    print(f"Saved: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
```

Run:

```bat
python scripts\record_audio.py
```

Verify:

```bat
dir audio
```

Expected:

```text
arabic_test.wav
```

---

## 10. Faster-Whisper Benchmark Script

A script named:

```text
scripts\test_whisper.py
```

was proposed.

Initial model:
- Whisper Small
- CPU
- int8
- Arabic forced
- VAD enabled
- beam size 5
- 8 CPU threads

Code:

```python
import time
from pathlib import Path

from faster_whisper import WhisperModel


AUDIO_FILE = Path("audio/arabic_test.wav")
MODEL_NAME = "small"


def main() -> None:
    if not AUDIO_FILE.exists():
        raise FileNotFoundError(f"Missing audio file: {AUDIO_FILE.resolve()}")

    print(f"Loading model: {MODEL_NAME}")
    load_start = time.perf_counter()

    model = WhisperModel(
        MODEL_NAME,
        device="cpu",
        compute_type="int8",
        cpu_threads=8,
    )

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
        segment.text.strip()
        for segment in segments
        if segment.text.strip()
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
    main()
```

Run:

```bat
python scripts\test_whisper.py
```

---

## 11. Initial ASR Test Phrases

Suggested phrases:

```text
إزيك يا روبوت، ممكن تساعدني أرتب مواعيدي النهارده؟
```

```text
فكرني بالاجتماع الساعة تلاتة ونص العصر.
```

```text
أنا محتاج أشغل النور في أوضة المعيشة.
```

```text
بكرة عندي meeting مع العميل الساعة ten thirty.
```

Things to record:
- incorrect words
- numbers and times
- Egyptian dialect quality
- mixed Arabic-English quality
- transcription time

---

## 12. Hugging Face Download Warnings

The user ran:

```bat
python scripts\test_whisper.py
```

Output:

```text
Loading model: small
Warning: You are sending unauthenticated requests to the HF Hub.
```

Also a Windows symlink warning.

Explanation:
- unauthenticated downloads still work
- authentication only improves rate limits and reliability
- symlink warning is not fatal
- Windows can cache without symlinks, but may use more disk space
- Developer Mode or Administrator rights can enable symlinks

The first model download is approximately hundreds of MB and may appear silent.

---

## 13. Hugging Face Authentication

The user tried:

```bat
huggingface-cli login
```

Output stated that it is deprecated.

Correct command:

```bat
hf auth login
```

The user chose browser login.

Authentication succeeded.

Token details:
- saved under Hugging Face cache
- token name:
  `oauth-Mohamed1-1`

Observed success messages:
- Token is valid
- Login successful
- Current active token set
- token refreshes automatically

---

## 14. Whisper Appeared Stuck

After authentication, the user ran:

```bat
python scripts\test_whisper.py
```

It printed:

```text
Loading model: small
```

and appeared stuck.

Recommended diagnostic approach:
- model may be downloading silently
- verify cache folder size
- inspect network/disk usage
- use explicit Hugging Face CLI download for visible progress

Recommended command:

```bat
hf download Systran/faster-whisper-small --local-dir models\faster-whisper-small
```

Then change:

```python
MODEL_NAME = "small"
```

to:

```python
MODEL_NAME = "models/faster-whisper-small"
```

Then run:

```bat
python scripts\test_whisper.py
```

This avoids network checks at model startup.

Suggested cache check:

```bat
dir "%USERPROFILE%\.cache\huggingface\hub\models--Systran--faster-whisper-small" /s
```

If `hf download` also freezes:

```bat
set HF_HUB_DISABLE_XET=1
hf download Systran/faster-whisper-small --local-dir models\faster-whisper-small
```

Optional persistent setting:

```bat
conda env config vars set HF_HUB_DISABLE_XET=1
conda deactivate
conda activate robot
```

Connectivity tests:

```bat
hf auth whoami
hf models info Systran/faster-whisper-small
```

---

## 15. ASR Models Planned for Testing

Models discussed:
- Faster-Whisper Small
- Whisper Medium
- Whisper Large-v3 Turbo
- Qwen3-ASR
- Voxtral Realtime

Purpose:
- establish a baseline
- compare Arabic accuracy
- compare Egyptian dialect accuracy
- compare code-switching
- compare speed and hardware usage

Note:
Earlier discussion suggested Qwen3-ASR might be promising for Egyptian Arabic and mixed Arabic-English, but this remains to be benchmarked rather than assumed.

---

## 16. LLM Models Planned for Testing

Candidates:
- Jais-2 8B
- Qwen
- Gemma
- Llama variants

Metrics:
- Arabic quality
- Egyptian conversational ability
- first-token latency
- tokens per second
- RAM
- VRAM
- instruction following
- reasoning
- suitability for a robot

---

## 17. TTS Models the User Requested to Test

The user specifically requested testing these three:

### VoiceTut-TTS

Repository:

```text
https://github.com/MohammedAly22/VoiceTuT-TTS
```

Key points discussed:
- Egyptian Arabic
- Arabic-English code-switching
- built-in voices
- text normalization
- streaming
- voice cloning
- separate Python environment recommended
- reported peak VRAM around 2.93 GB FP16 on an NVIDIA T4 according to the project information reviewed
- likely realistic for a 6 GB GPU, but must be benchmarked

### Chatterbox-TTS-Server

Repository:

```text
https://github.com/devnen/Chatterbox-TTS-Server
```

Key points:
- self-hosted server
- Web UI
- HTTP API
- OpenAI-compatible APIs
- model/engine switching
- voice cloning
- portable Windows installation
- unload endpoint
- Arabic through Chatterbox Multilingual
- recommended first because it is isolated and controlled over HTTP

### NAMAA Egyptian TTS

Model:

```text
https://huggingface.co/NAMAA-Space/NAMAA-Egyptian-TTS
```

Key points:
- Egyptian Arabic configuration/checkpoint
- built on Chatterbox Multilingual
- optional reference audio
- speaker/style transfer
- MIT license stated in prior review
- 0.5B model family
- may require significant VRAM because of checkpoint precision
- should be installed separately

---

## 18. TTS Environment Strategy

Strong recommendation:
Do not install all TTS models into the same Python environment.

Suggested separation:
- `robot`: ASR testing
- `tts-gui`: lightweight GUI
- `voicetut`: VoiceTut dependencies
- separate Chatterbox/NAMAA environment
- Chatterbox Server portable install

Reason:
- different PyTorch versions
- audio dependency conflicts
- package conflicts
- limited 6 GB VRAM
- easier debugging
- easier model unloading

Preferred control model:
- lightweight GUI
- communicates with model-specific servers over HTTP where possible

---

## 19. TTS Benchmark GUI Request

The user asked for a GUI script that:
- allows model selection
- shows model information
- allows synthesis
- compares models
- measures performance
- supports the three requested models

A package was created named:

```text
arabic_tts_benchmark_gui.zip
```

It contained:
- `arabic_tts_benchmark_gui.py`
- `requirements-gui.txt`
- `README.md`

The GUI was designed with:
- Tkinter
- model selector
- Chatterbox server URL
- Arabic text input
- speaker/voice ID
- reference transcript
- reference WAV selection
- synthesis parameters
- playback
- open output folder
- unload model
- live RAM/CPU/GPU display
- detailed model information
- benchmark results
- log tab

---

## 20. GUI Model Adapters

### VoiceTut Adapter

Expected behavior:
- import `VoiceTutTTS`
- load:
  `mohammedaly22/VoiceTut-TTS`
- support:
  - speaker
  - reference audio
  - reference text
  - num_step
  - speed
- save output WAV
- record load and generation time
- unload and clear CUDA cache

### NAMAA Adapter

Expected behavior:
- import Chatterbox multilingual components
- use `snapshot_download`
- download:
  `NAMAA-Space/NAMAA-Egyptian-TTS`
- load the base Chatterbox Multilingual model
- load:
  `t3_mtl23ls_v2.safetensors`
- generate with:
  `language_id="ar"`
- optionally use:
  `audio_prompt_path`
- save using torchaudio or scipy fallback
- unload and clear CUDA cache

### Chatterbox Server Adapter

Expected behavior:
- connect over HTTP
- model info endpoint
- voice listing endpoint
- TTS endpoint
- reference audio upload
- unload endpoint

Default URL used:

```text
http://127.0.0.1:8004
```

The exact endpoint compatibility must be verified against the currently installed server version.

---

## 21. GUI Benchmark Metrics

For every generation:
- model name
- timestamp
- input text
- reference audio path
- load time
- generation time
- wall time
- output audio duration
- realtime factor
- process RAM
- system RAM percentage
- CPU usage
- GPU VRAM used
- GPU VRAM total
- GPU utilization
- output WAV path
- JSON report path

Realtime factor interpretation:
- `< 1.0`: faster than real time
- `1.0`: generation duration equals playback duration
- `> 1.0`: slower than real time

Important:
For robot deployment, complete-file generation time is not enough. Also measure:
- time to first audio
- streaming chunk delay
- interruption responsiveness
- end-to-end response latency

---

## 22. GUI Installation Instructions

Suggested:

```bat
cd C:\Users\pc\arabic-test
conda create -n tts-gui python=3.11 -y
conda activate tts-gui
pip install -r requirements-gui.txt
python arabic_tts_benchmark_gui.py
```

GUI dependencies:
- requests
- psutil
- sounddevice
- soundfile
- nvidia-ml-py

---

## 23. Chatterbox Installation Recommendation

Suggested:

```bat
cd C:\Users\pc\arabic-test
git clone https://github.com/devnen/Chatterbox-TTS-Server.git
cd Chatterbox-TTS-Server
start.bat
```

Recommended choices:
- Portable Mode
- NVIDIA GPU
- CUDA 12.1
- Chatterbox Multilingual engine for Arabic

Then:
- leave server running
- open GUI
- select Chatterbox TTS Server
- set URL:
  `http://127.0.0.1:8004`
- click Test Server
- synthesize Arabic

---

## 24. VoiceTut Installation Recommendation

Suggested separate environment:

```bat
conda create -n voicetut python=3.10 -y
conda activate voicetut
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/k2-fsa/OmniVoice.git
pip install "voicetut-tts[web]"
pip install requests psutil sounddevice soundfile nvidia-ml-py
python arabic_tts_benchmark_gui.py
```

Voice listing command discussed:

```bat
voicetut --list-speakers
```

Suggested initial VoiceTut parameters:
- `num_step = 16`
- `speed = 1.0`

Possible built-in speaker names mentioned:
- Mohamed
- Asmaa

These names should be verified against the installed version.

---

## 25. NAMAA Installation Note

NAMAA should use:
- its own Python 3.10 environment
- compatible Chatterbox package
- torch
- torchaudio
- huggingface_hub
- safetensors

The GUI adapter followed the model card loading approach:
- download model repo
- load Chatterbox Multilingual
- replace/load T3 state
- generate Arabic audio

Exact package versions still need validation.

---

## 26. Original Short Default TTS Text

The initial GUI text was:

```python
DEFAULT_TEXT = (
    "إزيك يا روبوت؟ أنا عندي meeting الساعة تلاتة ونص، "
    "ومحتاجك تفكرني قبلها بنص ساعة."
)
```

The user asked to make it bigger and include more text.

---

## 27. Expanded Default Benchmark Text

A much larger benchmark was proposed, covering:
- Egyptian Arabic
- Modern Standard Arabic
- English
- Arabic-English code-switching
- meetings
- times
- reminders
- weather
- percentages
- currency
- API terminology
- CPU/GPU/RAM terminology
- numbers
- dates
- order numbers
- invoice numbers
- phone numbers
- email
- URL
- punctuation
- questions
- exclamations
- natural pauses

Proposed code:

```python
DEFAULT_TEXT = """
السلام عليكم يا روبوت، صباح الخير. أنا محتاج أعمل اختبار كامل لجودة تحويل النص إلى كلام باللغة العربية، وخصوصًا باللهجة المصرية.

إزيك؟ عامل إيه النهارده؟ يارب تكون جاهز لأن عندنا شوية مهام كتير.

أول حاجة، فكرني إن عندي اجتماع مهم بكرة الساعة عشرة ونص صباحًا مع فريق التطوير، وبعد الاجتماع عندي Presentation الساعة اتنين إلا ربع، وبعدها Call مع العميل الساعة خمسة مساءً.

ممكن كمان تضيف Reminder يوم الجمعة الساعة سبعة مساءً إني أتصل بأحمد، ومتنساش تفكرني قبلها بنص ساعة.

درجة الحرارة النهارده حوالي سبعة وثلاثين درجة مئوية، ونسبة الرطوبة خمسة وستين في المية، واحتمال سقوط الأمطار عشرة في المية فقط.

بالمناسبة، سعر الدولار النهارده حوالي خمسين جنيه مصري، وسعر اليورو حوالي سبعة وخمسين جنيه، أما الميزانية المتاحة للمشروع فهي مية خمسة وعشرين ألف جنيه.

دلوقتي هنجرب شوية جمل فيها عربي وإنجليزي مع بعض.

Please open the dashboard, check the latest production report, then send an email to the customer confirming the shipment schedule.

عايز كمان تعمل Refresh للبيانات، وبعد كده Export للنتائج في ملف PDF، وبعدها Upload على Google Drive.

الـ API response time لازم يكون أقل من ميتين ملي ثانية، ولو زاد عن كده اعرض Warning في الـ Dashboard.

الـ CPU usage حاليًا حوالي خمسة وأربعين في المية، والـ GPU memory المستخدمة حوالي أربعة جيجابايت ونص، بينما الرام المستخدمة حوالي اتناشر جيجابايت.

خلينا نجرب شوية أرقام.

واحد، اتنين، تلاتة، أربعة، خمسة، ستة، سبعة، تمانية، تسعة، عشرة.

أحد عشر، اثنا عشر، ثلاثة عشر، أربعة عشر، خمسة عشر، عشرون، خمسون، مائة، مائتان، ألف، عشرة آلاف، مائة ألف، مليون.

دلوقتي هنجرب قراءة تاريخ.

اليوم هو الاثنين الموافق عشرين يوليو ألفين وستة وعشرين.

وده رقم طلب: 25437891.

وده رقم فاتورة: INV-2026-000875.

وده رقم هاتف: صفر واحد صفر، واحد اتنين تلاتة، أربعة خمسة ستة، سبعة تمانية تسعة صفر.

وده بريد إلكتروني.

support@example.com

وده عنوان موقع إلكتروني.

https://example.com

دلوقتي هنجرب شوية علامات ترقيم.

هل تستطيع مساعدتي؟

بالتأكيد!

رائع.

ممتاز...

لكن انتظر قليلًا.

هل أنت متأكد؟

نعم، أنا متأكد بنسبة مائة في المائة.

دلوقتي عايز أسمع الكلام بسرعة طبيعية، وبعدها بسرعة أعلى قليلًا، وبعدها بسرعة أبطأ شوية، علشان أقدر أقارن جودة النطق بين النماذج المختلفة.

في النهاية، الهدف من الاختبار ده هو قياس جودة النطق باللهجة المصرية، وسرعة توليد الصوت، ومدى طبيعية الأداء، والتعامل مع الأرقام، والتواريخ، والكلمات الإنجليزية داخل الجملة العربية، وكمان مقارنة استهلاك الذاكرة وسرعة التنفيذ بين جميع نماذج تحويل النص إلى كلام.
"""
```

A recommendation was also made to later add a benchmark suite with multiple shorter categorized test cases rather than relying only on one very long paragraph.

---

## 28. Suggested TTS Benchmark Categories

Recommended categories:
1. Greeting
2. Egyptian conversation
3. MSA
4. Numbers
5. Dates
6. Times
7. Currency
8. Percentages
9. Arabic-English code switching
10. Technical terms
11. Questions
12. Emotional/prosodic variation
13. Long paragraph
14. Commands
15. Voice cloning

Manual evaluation:
- naturalness
- pronunciation
- accent
- code-switching
- numbers
- dates
- pause handling
- emotional tone
- noise/artifacts
- speaker consistency

---

## 29. VPS Discussion

The user asked whether hosting AI models on a VPS while all robots communicate with that server is a good solution.

Answer:
It can be a good architecture if designed correctly.

Advantages:
- centralized model updates
- one powerful inference server
- easier monitoring
- ESP32 stays simple
- all robots can use the same AI stack
- easier improvements without firmware changes

Risks:
- GPU cost
- latency
- internet dependency
- scaling
- concurrency
- bandwidth
- queueing
- privacy
- single point of failure

Recommended production approach:
- persistent WebSocket/WebRTC sessions
- VAD on device
- only send active speech
- streaming ASR
- streaming LLM
- streaming TTS
- per-session queues
- rate limiting
- autoscaling
- model batching where compatible
- multiple worker processes
- metrics and logging
- fallback responses if the server is unavailable

---

## 30. Local-First Testing Strategy

The user wanted to test all models on the PC first.

Agreed strategy:
1. test ASR locally
2. test TTS locally
3. test LLMs locally
4. compare quality and speed
5. select the most promising stack
6. deploy selected models/services to VPS
7. connect ESP32
8. test end-to-end latency
9. optimize streaming and concurrency

---

## 31. Benchmark Phases

### Phase 1: ASR
- Faster-Whisper
- Whisper variants
- Qwen3-ASR
- Voxtral if feasible

### Phase 2: TTS
- VoiceTut
- Chatterbox
- NAMAA
- possibly Piper
- XTTS
- StyleTTS2

### Phase 3: LLM
- Jais-2
- Qwen
- Gemma
- Llama

### Phase 4: End-to-end
ESP32  
→ WebSocket  
→ ASR  
→ LLM  
→ TTS  
→ ESP32

---

## 32. Current Status at End of Chat

Completed:
- clarified architecture
- discussed why streaming is better
- clarified what server sends back
- explained LLM role
- researched several ASR/TTS/LLM options
- set up Conda
- created Python 3.11 `robot` environment
- disabled user-site package leakage
- installed Faster-Whisper dependencies
- fixed missing tqdm/flatbuffers issue
- authenticated with Hugging Face
- prepared recorder script
- prepared Whisper benchmark script
- diagnosed model download appearing stuck
- planned explicit local model download
- selected three TTS models for testing
- created a GUI package
- defined metrics
- expanded default benchmark text
- recommended separate environments

Pending:
- confirm Faster-Whisper Small fully downloads
- complete first ASR transcription
- switch Faster-Whisper to GPU and compare CPU vs GPU
- install and run Chatterbox Server
- verify GUI endpoints against installed Chatterbox version
- run VoiceTut
- validate VoiceTut speakers
- install NAMAA dependencies
- benchmark all TTS models
- measure time-to-first-audio
- benchmark LLMs
- create full end-to-end streaming prototype
- evaluate VPS sizing and concurrency
- integrate with ESP32

---

## 33. Recommended Immediate Next Steps

1. Complete Whisper download:

```bat
set HF_HUB_DISABLE_XET=1
hf download Systran/faster-whisper-small --local-dir models\faster-whisper-small
```

2. Change in `test_whisper.py`:

```python
MODEL_NAME = "models/faster-whisper-small"
```

3. Run:

```bat
python scripts\test_whisper.py
```

4. Confirm CPU baseline.

5. Then configure GPU Faster-Whisper.

6. Install Chatterbox Server first because it is easiest to isolate.

7. Run the GUI against Chatterbox.

8. Record:
- text
- generated WAV
- load time
- generation time
- output duration
- RTF
- RAM
- VRAM
- quality score

9. Repeat with VoiceTut and NAMAA.

---

## 34. Important Cautions

- Do not install all model stacks in one environment.
- Do not assume a model supports Egyptian Arabic well until tested.
- Do not judge real-time suitability only by total synthesis time.
- Measure time to first partial transcript and first audio.
- Keep only one large GPU model loaded at a time on the GTX 1660 Super.
- Close GPU-heavy applications before testing.
- Record exact versions of all packages.
- Save benchmark results as JSON/CSV.
- Use the same test audio/text for fair comparisons.
- Separate objective metrics from human quality ratings.
- Treat previously mentioned endpoint names and package commands as version-sensitive and verify against the current repositories when installing.

---

## 35. Minimal Context Prompt for a New Chat

Use this prompt in a new conversation:

> I am building a real-time Egyptian Arabic conversational robot using ESP32. My PC is Windows with GTX 1660 Super 6 GB, i5-10400F, and 16 GB RAM. I created a Conda Python 3.11 environment called `robot`, disabled user-site packages, installed Faster-Whisper, and authenticated with Hugging Face. I am testing `Systran/faster-whisper-small` first. I also want to benchmark VoiceTut-TTS, Chatterbox-TTS-Server, and NAMAA-Egyptian-TTS using separate environments and a GUI that records load time, generation time, output duration, realtime factor, CPU, RAM, GPU, VRAM, WAV output, and JSON results. The final architecture is ESP32 → VAD/wake word → streaming audio → server ASR → LLM → streaming TTS → ESP32. Continue from the pending steps in the attached transfer document.
