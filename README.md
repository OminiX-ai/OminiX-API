# OminiX-API

OpenAI-compatible API server for OminiX-MLX models on Apple Silicon.

## Features

- **OpenAI-compatible endpoints** - Drop-in replacement for OpenAI API
- **LLM chat** - Chat completions with Qwen3, Mistral, GLM models
- **Speech-to-text** - Audio transcription with Paraformer ASR
- **Text-to-speech** - Voice cloning with GPT-SoVITS (named voices, few-shot, pre-computed codes)
- **Image generation** - Text-to-image with FLUX.2-klein and Z-Image-Turbo
- **WebSocket streaming TTS** - MiniMax T2A compatible protocol with per-message voice switching
- **Voice registry** - Named voices with aliases, configurable via `voices.json`
- **Dynamic model loading** - Switch models at runtime without server restart
- **Memory efficient** - One model per category, automatic unloading when switching
- **Pure Rust** - No Python dependencies at runtime

## Prerequisites

- macOS 14.0+ (Sonoma)
- Apple Silicon (M1/M2/M3/M4)
- Rust 1.82+
- Xcode Command Line Tools

## Build

```bash
# Clone repository
git clone https://github.com/anthropics/OminiX-API.git
cd OminiX-API

# Build (requires OminiX-MLX in sibling directory)
cargo build --release

# The binary is at target/release/ominix-api
```

## Quick Start

```bash
# Run with LLM only (downloads model automatically)
LLM_MODEL=mlx-community/Qwen3-4B-bf16 cargo run --release

# Run with all models
PORT=8080 \
LLM_MODEL=mlx-community/Qwen3-4B-bf16 \
ASR_MODEL_DIR=./models/paraformer \
TTS_REF_AUDIO=./audio/reference.wav \
IMAGE_MODEL=zimage \
cargo run --release
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | HTTP server port |
| `LLM_MODEL` | `mlx-community/Mistral-7B-Instruct-v0.2-4bit` | HuggingFace model ID |
| `ASR_MODEL_DIR` | (empty) | Path to Paraformer model directory |
| `TTS_REF_AUDIO` | (empty) | Path to reference audio for voice cloning |
| `IMAGE_MODEL` | (empty) | Image model: `zimage` or `flux` |
| `FLUX_MODEL_DIR` | (auto-download) | Custom path to FLUX.2-klein model |
| `ZIMAGE_MODEL_DIR` | (auto-download) | Custom path to Z-Image-Turbo model |
| `VOICES_CONFIG` | `~/.dora/models/primespeech/voices.json` | Path to voice registry file |
| `TTS_VOICES_DIR` | (none) | Allowed directory for voice file path references |

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/models` | GET | List loaded models |
| `/v1/models/status` | GET | Get current model status for each category |
| `/v1/models/load` | POST | Load/switch model dynamically |
| `/v1/models/unload` | POST | Unload model to free memory |
| `/v1/chat/completions` | POST | Chat completions (LLM) |
| `/v1/audio/transcriptions` | POST | Speech-to-text (ASR) |
| `/v1/audio/speech` | POST | Text-to-speech (TTS) |
| `/v1/images/generations` | POST | Image generation |
| `/ws/v1/tts` | WebSocket | Streaming TTS with per-message voice switching |

---

## Text-to-Speech (TTS)

The TTS engine uses GPT-SoVITS for voice cloning. It supports three quality tiers, selected automatically based on what's available for each voice:

```
┌─────────────────────────────────────────────────────┐
│              Voice Quality Tiers                     │
├─────────────────────────────────────────────────────┤
│                                                      │
│  1. Pre-computed codes  (best quality)               │
│     ref_audio + ref_text + codes_path                │
│                                                      │
│  2. Few-shot / HuBERT   (good quality)               │
│     ref_audio + ref_text  (HuBERT extracts codes)    │
│                                                      │
│  3. Zero-shot            (baseline)                  │
│     ref_audio only  (mel spectrogram matching)       │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### Voice Registry (`voices.json`)

Named voices are configured in a JSON file. The server loads this at startup from `$VOICES_CONFIG` or the default path `~/.dora/models/primespeech/voices.json`.

```json
{
  "default_voice": "doubao",
  "models_base_path": "~/.dora/models/primespeech",
  "voices": {
    "doubao": {
      "ref_audio": "moyoyo/ref_audios/doubao_ref_mix_new.wav",
      "ref_text": "这家resturant的steak很有名",
      "codes_path": "gpt-sovits-mlx/doubao_mixed_codes.bin",
      "speed_factor": 1.1,
      "aliases": ["default"]
    },
    "luoxiang": {
      "ref_audio": "moyoyo/ref_audios/luoxiang_ref.wav",
      "ref_text": "复杂的问题背后也许没有统一的答案...",
      "codes_path": "gpt-sovits-mlx/codes/luoxiang_codes.bin",
      "speed_factor": 1.1,
      "aliases": ["luo"]
    }
  }
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `default_voice` | No | Default voice name (used when no voice specified) |
| `models_base_path` | No | Base directory for resolving relative paths |
| `voices.<name>.ref_audio` | Yes | Reference audio file (WAV, relative to base path) |
| `voices.<name>.ref_text` | Yes | Transcript of the reference audio |
| `voices.<name>.codes_path` | No | Pre-computed semantic codes (`.bin`, best quality) |
| `voices.<name>.speed_factor` | No | Speaking speed multiplier |
| `voices.<name>.aliases` | No | Alternative names for this voice |

Paths can be absolute or relative to `models_base_path`. Tilde (`~`) is expanded to `$HOME`.

### Voice Resolution

When a voice name is provided (via HTTP `voice` field or WebSocket `voice_id`), it resolves in this order:

```
voice: "luo"
    │
    ├─ 1. Registry lookup (case-insensitive)
    │     ├─ Direct name match ("luo"?)  → no
    │     └─ Alias match ("luo" in luoxiang.aliases?)  → yes
    │           │
    │           ├─ Has codes_path?  → pre-computed codes mode
    │           ├─ HuBERT available?  → few-shot mode
    │           └─ Otherwise  → zero-shot mode
    │
    └─ 2. File path fallback (if not in registry)
          └─ Validated for path traversal, then used as zero-shot reference
```

Voice switching is on-demand. The base models (BERT, T2S, VITS, HuBERT) stay loaded; only the reference conditioning changes per voice. Switching overhead is ~100-700ms depending on the voice.

---

### HTTP TTS API

**`POST /v1/audio/speech`**

Synthesize speech from text. Supports dynamic voice switching per request.

**Request:**

```json
{
  "input": "Hello, this is a test.",
  "voice": "luoxiang",
  "model": "gpt-sovits",
  "response_format": "wav",
  "speed": 1.0
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input` | string | (required) | Text to synthesize |
| `voice` | string | (none) | Voice name, alias, or file path |
| `model` | string | (ignored) | Accepted for OpenAI compatibility |
| `response_format` | string | `"wav"` | Audio format |
| `speed` | float | `1.0` | Speaking speed (0.25 - 4.0) |

**Response:** Raw WAV audio bytes with `Content-Type: audio/wav`.

**Examples:**

```bash
# Use a named voice from the registry
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "你好世界", "voice": "doubao"}' \
  -o doubao.wav

# Use a voice alias
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "你好世界", "voice": "luo"}' \
  -o luoxiang.wav

# Switch voices between requests (no reconnection needed)
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "主持人说话", "voice": "marc"}' -o host.wav

curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "嘉宾回应", "voice": "luoxiang"}' -o guest.wav
```

**Python example with dynamic voice switching:**

```python
import requests

API = "http://localhost:8080/v1/audio/speech"

dialogue = [
    ("marc", "大家好，欢迎来到今天的节目。"),
    ("luoxiang", "谢谢主持人，很高兴来到这里。"),
    ("yangmi", "我也很开心参加今天的讨论。"),
]

for i, (voice, text) in enumerate(dialogue):
    resp = requests.post(API, json={"input": text, "voice": voice})
    with open(f"line_{i+1}_{voice}.wav", "wb") as f:
        f.write(resp.content)
    print(f"[{voice}] {len(resp.content)} bytes")
```

---

### WebSocket TTS API

**`GET /ws/v1/tts`** (upgrade to WebSocket)

Streaming TTS over a persistent WebSocket connection. Follows the MiniMax T2A protocol with an extension for per-message voice switching.

#### Protocol Flow

```
Client                                    Server
  │                                         │
  │  ──── WebSocket Connect ────────────►   │
  │  ◄──── connected_success ───────────    │
  │                                         │
  │  ──── task_start ───────────────────►   │
  │  ◄──── task_started ───────────────     │
  │                                         │
  │  ──── task_continue (text) ─────────►   │
  │  ◄──── task_progress (audio chunk) ─    │
  │  ◄──── task_progress (audio chunk) ─    │
  │  ◄──── task_progress (final chunk) ─    │
  │                                         │
  │  ──── task_continue (text) ─────────►   │  (repeat)
  │  ◄──── task_progress ... ───────────    │
  │                                         │
  │  ──── task_finish ──────────────────►   │
  │  ◄──── connection closed ───────────    │
```

#### 1. Connection

Connect to the WebSocket endpoint. The server sends a confirmation:

```json
{"event": "connected_success"}
```

#### 2. Task Start

Configure voice and audio settings for the session:

```json
{
  "event": "task_start",
  "voice_setting": {
    "voice_id": "marc",
    "speed": 1.0
  },
  "audio_setting": {
    "format": "wav"
  }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `voice_setting.voice_id` | string | (none) | Default voice for this session |
| `voice_setting.speed` | float | `1.0` | Speaking speed |
| `audio_setting.format` | string | `"wav"` | Audio format |

Server responds:

```json
{"event": "task_started"}
```

#### 3. Task Continue (Synthesize)

Send text to synthesize. Optionally override the voice per message:

```json
{
  "event": "task_continue",
  "text": "要合成的文本",
  "voice_id": "luoxiang"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | Text to synthesize |
| `voice_id` | string | No | Override voice for this message (falls back to `task_start` voice) |

The server synthesizes the audio and streams it back as hex-encoded 8KB chunks:

```json
{"event": "task_progress", "data": {"audio": "52494646..."}, "is_final": false}
{"event": "task_progress", "data": {"audio": "a1b2c3d4..."}, "is_final": false}
{"event": "task_progress", "data": {"audio": "e5f6a7b8..."}, "is_final": true}
```

The `data.audio` field contains hex-encoded raw audio bytes. The `is_final` flag indicates the last chunk for this text segment. Decode with `bytes.fromhex(audio_hex)`.

Processing is sequential: the next `task_continue` is not processed until the current one finishes streaming all chunks.

#### 4. Task Finish

Close the session:

```json
{"event": "task_finish"}
```

#### Error Handling

Errors are sent as:

```json
{"event": "error", "message": "Description of the error"}
```

Common errors:
- `"Missing 'text' field"` - `task_continue` without `text`
- `"TTS error: ..."` - Synthesis failure
- `"Inference unavailable"` - TTS model not loaded
- `"TTS timed out after 60s"` - Synthesis exceeded timeout
- `"Invalid JSON"` - Malformed message
- `"Unknown event: ..."` - Unrecognized event type

#### Multi-Participant Example (Python)

Use per-message `voice_id` to switch speakers within a single connection:

```python
import asyncio
import json
import struct
import websockets

async def multi_voice_dialogue():
    dialogue = [
        ("marc",     "大家好，欢迎来到今天的节目。"),
        ("luoxiang", "谢谢主持人。我来谈谈法律方面的问题。"),
        ("yangmi",   "我觉得AI在创意领域也很有潜力。"),
        ("marc",     "两位说得都有道理，感谢参与讨论！"),
    ]

    async with websockets.connect("ws://localhost:8080/ws/v1/tts") as ws:
        await ws.recv()  # connected_success

        # Configure session (default voice, can be overridden per message)
        await ws.send(json.dumps({
            "event": "task_start",
            "voice_setting": {"voice_id": "marc", "speed": 1.0},
            "audio_setting": {"format": "wav"}
        }))
        await ws.recv()  # task_started

        all_pcm = b""

        for speaker, text in dialogue:
            # Send text with per-message voice override
            await ws.send(json.dumps({
                "event": "task_continue",
                "text": text,
                "voice_id": speaker
            }))

            # Collect audio chunks
            audio = b""
            while True:
                resp = json.loads(await ws.recv())
                if resp["event"] == "error":
                    print(f"Error: {resp['message']}")
                    break
                hex_data = resp.get("data", {}).get("audio", "")
                if hex_data:
                    audio += bytes.fromhex(hex_data)
                if resp.get("is_final"):
                    break

            print(f"[{speaker:8s}] {len(audio):>7d} bytes | {text}")

            # Save individual line
            with open(f"{speaker}_{text[:10]}.wav", "wb") as f:
                f.write(audio)

            # Accumulate PCM (skip 44-byte WAV header)
            if len(audio) > 44:
                all_pcm += audio[44:]

        # Write combined WAV
        with open("full_dialogue.wav", "wb") as f:
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36 + len(all_pcm)))
            f.write(b"WAVEfmt ")
            f.write(struct.pack("<IHHIIHH", 16, 1, 1, 32000, 64000, 2, 16))
            f.write(b"data")
            f.write(struct.pack("<I", len(all_pcm)))
            f.write(all_pcm)

        print(f"\nSaved full_dialogue.wav ({len(all_pcm) / 2 / 32000:.1f}s)")

        await ws.send(json.dumps({"event": "task_finish"}))

asyncio.run(multi_voice_dialogue())
```

#### Single Voice Example (Python)

```python
import asyncio
import json
import websockets

async def simple_tts():
    async with websockets.connect("ws://localhost:8080/ws/v1/tts") as ws:
        await ws.recv()  # connected_success

        await ws.send(json.dumps({
            "event": "task_start",
            "voice_setting": {"voice_id": "doubao", "speed": 1.0},
            "audio_setting": {"format": "wav"}
        }))
        await ws.recv()  # task_started

        # Send multiple sentences (processed sequentially)
        for text in ["第一句话。", "第二句话。", "第三句话。"]:
            await ws.send(json.dumps({
                "event": "task_continue",
                "text": text
            }))

            audio = b""
            while True:
                resp = json.loads(await ws.recv())
                hex_data = resp.get("data", {}).get("audio", "")
                if hex_data:
                    audio += bytes.fromhex(hex_data)
                if resp.get("is_final"):
                    break

            print(f"{len(audio)} bytes for: {text}")

        await ws.send(json.dumps({"event": "task_finish"}))

asyncio.run(simple_tts())
```

#### JavaScript/Node.js Example

```javascript
const WebSocket = require('ws');

const ws = new WebSocket('ws://localhost:8080/ws/v1/tts');

ws.on('open', () => {
  console.log('Connected');
});

ws.on('message', (data) => {
  const msg = JSON.parse(data);

  switch (msg.event) {
    case 'connected_success':
      // Configure voice
      ws.send(JSON.stringify({
        event: 'task_start',
        voice_setting: { voice_id: 'doubao', speed: 1.0 },
        audio_setting: { format: 'wav' }
      }));
      break;

    case 'task_started':
      // Start synthesizing
      ws.send(JSON.stringify({
        event: 'task_continue',
        text: '你好，这是一个测试。'
      }));
      break;

    case 'task_progress':
      const audioHex = msg.data.audio;
      const audioBuffer = Buffer.from(audioHex, 'hex');
      // Process audio chunk (e.g., write to file, play back)
      console.log(`Received ${audioBuffer.length} bytes (final: ${msg.is_final})`);

      if (msg.is_final) {
        ws.send(JSON.stringify({ event: 'task_finish' }));
      }
      break;

    case 'error':
      console.error('Error:', msg.message);
      break;
  }
});
```

---

## Chat Completions (LLM)

**`POST /v1/chat/completions`**

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

**Response:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1706000000,
  "model": "qwen3",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "The capital of France is Paris."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 8,
    "total_tokens": 33
  }
}
```

---

## Audio Transcription (ASR)

**`POST /v1/audio/transcriptions`**

```bash
# Encode audio as base64
AUDIO_B64=$(base64 -i audio.wav)

curl http://localhost:8080/v1/audio/transcriptions \
  -H "Content-Type: application/json" \
  -d "{
    \"file\": \"$AUDIO_B64\",
    \"model\": \"paraformer\"
  }"
```

**Response:**
```json
{
  "text": "Hello, this is a test.",
  "language": "zh",
  "duration": 2.5
}
```

---

## Image Generation

**`POST /v1/images/generations`**

```bash
# Text-to-image with Z-Image (faster)
curl http://localhost:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over the ocean, digital art",
    "model": "zimage",
    "size": "512x512",
    "n": 1
  }'

# Text-to-image with FLUX.2-klein (higher quality)
curl http://localhost:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat sitting on a windowsill",
    "model": "flux",
    "size": "1024x1024"
  }'
```

**Response:**
```json
{
  "created": 1706000000,
  "data": [{
    "b64_json": "/9j/4AAQSkZJRg...",
    "revised_prompt": "A beautiful sunset over the ocean, digital art"
  }]
}
```

### Image-to-Image (img2img)

```bash
# Encode reference image as base64
IMG_B64=$(base64 -i reference.png)

curl http://localhost:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"Make it look like a watercolor painting\",
    \"model\": \"zimage\",
    \"size\": \"512x512\",
    \"image\": \"$IMG_B64\",
    \"strength\": 0.75
  }"
```

---

## Dynamic Model Management

The server supports **one model per category** (LLM, ASR, TTS, Image). Models can be loaded, switched, and unloaded at runtime without restarting the server.

```
┌─────────────────────────────────────────────────────────┐
│                   Model Slots                            │
├─────────────┬─────────────┬─────────────┬───────────────┤
│     LLM     │     ASR     │     TTS     │    Image      │
│  (1 slot)   │  (1 slot)   │  (1 slot)   │   (1 slot)    │
├─────────────┼─────────────┼─────────────┼───────────────┤
│ Qwen/Mistral│  Paraformer │  GPT-SoVITS │  FLUX/Z-Image │
└─────────────┴─────────────┴─────────────┴───────────────┘
```

### Check Model Status

```bash
curl http://localhost:8080/v1/models/status
```

**Response:**
```json
{
  "status": "success",
  "models": {
    "llm": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "asr": null,
    "tts": null,
    "image": "zimage"
  }
}
```

### Load/Switch Models

```bash
# Load LLM
curl http://localhost:8080/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/Qwen2.5-7B-Instruct-4bit", "model_type": "llm"}'

# Load ASR
curl http://localhost:8080/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "/path/to/paraformer/model", "model_type": "asr"}'

# Load TTS (provide reference audio)
curl http://localhost:8080/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "/path/to/reference/audio.wav", "model_type": "tts"}'

# Load Image model
curl http://localhost:8080/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "zimage", "model_type": "image"}'
```

### Unload Models (Free Memory)

```bash
# Unload one model type
curl http://localhost:8080/v1/models/unload \
  -H "Content-Type: application/json" \
  -d '{"model_type": "llm"}'

# Unload all models
curl http://localhost:8080/v1/models/unload \
  -H "Content-Type: application/json" \
  -d '{"model_type": "all"}'
```

### Model Type Reference

| Type | Load Parameter | Description |
|------|----------------|-------------|
| `llm` | HuggingFace model ID | e.g., `mlx-community/Qwen2.5-7B-Instruct-4bit` |
| `asr` | Path to model directory | Directory with `paraformer.safetensors` |
| `tts` | Path to reference audio | WAV file for voice cloning |
| `image` | `zimage` or `flux` | Image generation model |

---

## Model Setup

### LLM Models

Models download automatically from HuggingFace. Recommended:

| Model | HuggingFace ID | Memory |
|-------|----------------|--------|
| Qwen3-4B | `mlx-community/Qwen3-4B-bf16` | 8 GB |
| Qwen3-8B | `mlx-community/Qwen3-8B-bf16` | 16 GB |
| Mistral-7B | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | 4 GB |

### ASR Model (Paraformer)

```bash
# Download model files
huggingface-cli download funaudiollm/paraformer-large-mlx --local-dir ./models/paraformer
```

Required files:
```
models/paraformer/
├── paraformer.safetensors   # Model weights
├── am.mvn                   # CMVN normalization
└── tokens.txt               # Vocabulary (8404 tokens)
```

### TTS Model (GPT-SoVITS)

GPT-SoVITS uses few-shot voice cloning. Provide a reference audio file:

```bash
# Reference audio requirements:
# - WAV format, 16kHz or higher
# - 3-10 seconds of clean speech
# - Single speaker, minimal background noise

TTS_REF_AUDIO=./audio/reference.wav cargo run --release
```

The voice registry (`voices.json`) is loaded automatically if present. See the [Voice Registry](#voice-registry-voicesjson) section for configuration.

### Image Models

Image models download automatically from HuggingFace. Choose between:

| Model | ID | Steps | Memory | Speed |
|-------|----|-------|--------|-------|
| Z-Image-Turbo | `zimage` | 9 | ~12 GB | ~3s |
| FLUX.2-klein | `flux` | 4 | ~13 GB | ~5s |

#### Model Download URLs

**FLUX.2-klein (MLX format):**
| Source | URL |
|--------|-----|
| HuggingFace | https://huggingface.co/black-forest-labs/FLUX.2-klein-4B |
| ModelScope | https://modelscope.cn/models/black-forest-labs/FLUX.2-klein-4B |

**Z-Image-Turbo (MLX format):**
| Source | URL |
|--------|-----|
| HuggingFace | https://huggingface.co/uqer1244/MLX-z-image |

**Original Models (for reference):**
| Model | Original Source |
|-------|-----------------|
| FLUX.2-klein | https://huggingface.co/black-forest-labs/FLUX.1-schnell |
| Z-Image-Turbo | https://huggingface.co/Zheng-Peng-Fei/Z-Image |

#### Environment Variables for Image Models

```bash
# Use specific model (auto-downloads if not present)
IMAGE_MODEL=flux cargo run --release    # Use FLUX.2-klein
IMAGE_MODEL=zimage cargo run --release  # Use Z-Image-Turbo

# Use custom local model path (optional)
FLUX_MODEL_DIR=/path/to/flux-model cargo run --release
ZIMAGE_MODEL_DIR=/path/to/zimage-model cargo run --release
```

#### Manual Download

```bash
# Download FLUX.2-klein
huggingface-cli download black-forest-labs/FLUX.2-klein-4B --local-dir ./models/flux

# Download Z-Image-Turbo
huggingface-cli download uqer1244/MLX-z-image --local-dir ./models/zimage

# Or using git lfs
git lfs install
git clone https://huggingface.co/black-forest-labs/FLUX.2-klein-4B ./models/flux
git clone https://huggingface.co/uqer1244/MLX-z-image ./models/zimage
```

#### Model Directory Structure

```
models/flux/                          # FLUX.2-klein
├── transformer/
│   └── diffusion_pytorch_model.safetensors
├── text_encoder/
│   ├── model-00001-of-00002.safetensors
│   └── model-00002-of-00002.safetensors
├── vae/
│   └── diffusion_pytorch_model.safetensors
└── tokenizer/
    └── tokenizer.json

models/zimage/                        # Z-Image-Turbo
├── transformer/
│   └── model.safetensors
├── text_encoder/
│   └── model.safetensors
├── vae/
│   └── diffusion_pytorch_model.safetensors
└── tokenizer/
    └── tokenizer.json
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   HTTP/WS Server (Salvo)                      │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐│
│  │  /chat  │ │  /asr   │ │  /tts   │ │ /images │ │ ws/tts ││
│  │  (REST) │ │  (REST) │ │  (REST) │ │  (REST) │ │  (WS)  ││
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └───┬────┘│
│       │           │           │            │          │      │
│       └───────────┴───────────┴────────────┴──────────┘      │
│                               │                              │
│                   mpsc::channel (bounded: 32)                │
│                               │                              │
└───────────────────────────────┼──────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────┐
│              Inference Thread (owns all models)               │
│                                                               │
│  ┌──────────────┐ ┌──────────────┐ ┌───────────────────────┐ │
│  │  LLM Slot    │ │  ASR Slot    │ │      TTS Slot         │ │
│  │  Qwen3 /     │ │  Paraformer  │ │  GPT-SoVITS engine    │ │
│  │  Mistral /   │ │      /       │ │  + voice registry     │ │
│  │  (empty)     │ │   (empty)    │ │  + on-demand switching│ │
│  └──────────────┘ └──────────────┘ └───────────────────────┘ │
│                                                               │
│  ┌──────────────┐                                             │
│  │ Image Slot   │                                             │
│  │ Z-Image /    │                                             │
│  │ FLUX /       │                                             │
│  │ (empty)      │                                             │
│  └──────────────┘                                             │
└──────────────────────────────────────────────────────────────┘
```

**Key Design Points:**

- **Actor Model**: MLX models don't implement `Send`/`Sync`, so all models run on a dedicated inference thread. HTTP and WebSocket handlers communicate via bounded async channels.

- **One Model Per Slot**: Each category has exactly one model slot. Loading a new model automatically unloads the previous one to free GPU memory.

- **Shared Inference Path**: Both the REST `POST /v1/audio/speech` and WebSocket `ws/v1/tts` endpoints use the same `InferenceRequest::Speech` channel to the inference thread. Voice switching works identically for both.

- **Dynamic Loading**: Models can be loaded, switched, and unloaded at runtime via `/v1/models/load` and `/v1/models/unload` endpoints.

- **Memory Efficient**: Unused model slots remain empty. Unload models you're not using to free GPU memory for other tasks.

- **Request Timeouts**: All inference operations have configurable timeouts to prevent hanging:
  - Chat: 5 minutes
  - Transcription: 2 minutes
  - TTS: 1 minute
  - Image: 10 minutes
  - Model loading: 5 minutes

## Performance

Benchmarks on Apple M3 Max (128GB):

| Task | Model | Throughput | Memory |
|------|-------|------------|--------|
| LLM | Qwen3-4B | 45 tok/s | 8 GB |
| ASR | Paraformer | 18x real-time | 500 MB |
| TTS | GPT-SoVITS | ~3x real-time | 2 GB |
| TTS voice switch | - | ~100-700ms overhead | - |
| Image | Z-Image-Turbo | ~3s/image | 12 GB |
| Image | FLUX.2-klein | ~5s/image | 13 GB |

## Python Client Example

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"  # No auth required
)

# Chat completion
response = client.chat.completions.create(
    model="qwen3",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)

# Image generation
response = client.images.generate(
    model="zimage",
    prompt="A cat in space",
    size="512x512"
)
print(response.data[0].b64_json[:50] + "...")
```

## License

MIT OR Apache-2.0
