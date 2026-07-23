Arabic Robot Research — Client Package
======================================

Prepared: July 2026
Topic: Egyptian Arabic voice robot — model stack + hosting

How to read (recommended order)
-------------------------------
1. Arabic_Robot_Research_Summary.pdf
   One-page overview of all decisions. Start here.

2. Combined_Families_Summary.pdf
   How ASR + LLM + TTS fit together on one GPU (VRAM budget).

3. Family bake-off reports (any order):
   - ASR_Model_Selection_Report.pdf   → speech recognition
   - LLM_Model_Selection_Report.pdf   → conversational brain
   - TTS_Model_Selection_Report.pdf   → robot voice / synthesis

4. VPS_Provider_Research_Report.pdf
   GPU cloud choice, capacity, and cost estimates.

5. TTS Voices/  (listen before freezing TTS)
   - VoiceTut-TTS.wav              ← recommended default
   - SILMA-TTS.wav                 ← realtime alternative
   - Chatterbox-Multilingual-V3.wav
   - NAMAA-Egyptian-TTS.wav

Production default (from the bake-offs)
---------------------------------------
ASR  Whisper-Large-v3-Turbo-CT2
LLM  Nile-Chat-4B  (quality alt: Qwen3-8B int4)
TTS  VoiceTut-TTS  (confirm by listening)
Host RunPod Secure Cloud — RTX 4090 24 GB

Notes
-----
- TTS automated scores cover speed/resources only. Please listen to the WAVs.
- GPU prices change; re-check provider dashboards before purchase.

Research by Eng. Mohamed Soltan.
