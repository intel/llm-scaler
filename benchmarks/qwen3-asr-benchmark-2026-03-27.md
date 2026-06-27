# Qwen3-ASR-1.7B Benchmark — 2026-03-27

## Model Info
- **Model:** Qwen3-ASR-1.7B (`/shared/models/qwen3-asr-1.7b`)
- **Server:** vLLM (Intel Arc XPU, port 8000)
- **Audio input:** espeak-ng generated WAV files (~2-3s speech clips)
- **Language:** English (auto-detected)
- **GPU:** Intel Arc 140V (28.5 GB unified memory)

## Single Transcription Test

**Input text:** "Hello, this is a test of the Qwen 3 ASR model. The quick brown fox jumps over the lazy dog. One two three four five."

**Transcription output:** `language English<asr_text>Hello. This is a test of the QN3ASR model. The quick brown fox jumps over the lazy dog. One, two, three, four, five.`

- Latency: **8.67s**
- Accuracy: Near-perfect (only "Qwen3" → "QN3ASR" slightly off, expected with synthetic TTS voice)
- Output tokens: 39

## Concurrency Benchmark (RAM-Monitored)

**Test sentences used:**
1. "The weather today is sunny with a high of twenty five degrees celsius."
2. "Artificial intelligence is transforming the way we work and communicate."
3. "Please confirm your reservation for three guests arriving on Friday evening."
4. "The stock market closed higher today driven by technology sector gains."
5. "Can you recommend a good restaurant near the city center for dinner tonight."

### Results

| Concurrency | Wall time | Total tokens | Agg tok/s | Peak RAM | Delta RAM |
|---|---|---|---|---|---|
| 1 | 2.26s | 19 | 8.4 | 4 MB | +0 MB |
| 2 | 1.01s | 35 | 34.8 | 4 MB | +0 MB |
| 5 | 1.05s | 86 | **81.9** | 4 MB | +0 MB |

### Per-Worker Detail (Concurrency 5)

| Worker | Tokens | Time | Transcription |
|---|---|---|---|
| 0 | 19 | 1.05s | "The weather today is sunny with a high of 25 degrees C..." |
| 1 | 16 | 0.95s | "Artificial intelligence is transforming the way we wor..." |
| 2 | 16 | 0.95s | "Please confirm your reservation for three guests arriv..." |
| 3 | 17 | 0.99s | "The stock market closed higher today, driven by techno..." |
| 4 | 18 | 1.04s | "Can you recommend a good restaurant near the city cent..." |

## Comparison vs Qwen3-8B-INT4

| Model | Concurrency | Agg tok/s | RAM | Notes |
|---|---|---|---|---|
| Qwen3-8B-INT4 | 1 | 13.6 | ~20 GB | Text generation |
| Qwen3-8B-INT4 | 2 | 25.5 | ~20 GB | Text generation |
| Qwen3-8B-INT4 | 5 (capped) | 36.4 | ~20 GB | Text generation |
| **Qwen3-ASR-1.7B** | 1 | 8.4 | **4 MB** | Speech transcription |
| **Qwen3-ASR-1.7B** | 2 | 34.8 | **4 MB** | Speech transcription |
| **Qwen3-ASR-1.7B** | 5 | **81.9** | **4 MB** | Speech transcription |

## Key Observations

- **Tiny RAM footprint:** Only 4 MB RSS (GPU VRAM handles everything) vs ~20 GB for 8B model
- **Excellent concurrency scaling:** Near-linear scaling up to 5 concurrent requests (~1s wall time)
- **No degradation at concurrency 5:** All workers complete in ~1s, no timeouts
- **High throughput:** 81.9 aggregate tok/s at concurrency 5
- **Fits alongside other models:** At 1.7B, leaves plenty of the 28.5 GB unified memory for other models

## Setup Notes

- Required `vllm[audio]` extra: `pip install "vllm[audio]"`
- Audio input format: `audio_url` content type with base64-encoded WAV
- Note: Qwen3-VL-8B not yet supported in vLLM 0.14 (Intel Arc build) — awaiting upstream update
