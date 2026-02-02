# OminiX-API

OpenAI-compatible API server for OminiX-MLX models on Apple Silicon.

## Features

- **OpenAI-compatible endpoints** - Drop-in replacement for OpenAI API
- **LLM chat** - Chat completions with Qwen3, Mistral, GLM models
- **Speech-to-text** - Audio transcription with Paraformer ASR
- **Text-to-speech** - Voice cloning with GPT-SoVITS
- **Image generation** - Text-to-image with FLUX.2-klein and Z-Image-Turbo
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

---

## API Examples

### Chat Completions (LLM)

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

### Audio Transcription (ASR)

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

### Text-to-Speech (TTS)

```bash
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, world! This is a voice cloning test.",
    "model": "gpt-sovits",
    "voice": "/path/to/different/reference.wav"
  }' \
  --output output.wav
```

Returns WAV audio file directly.

### Image Generation

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

### Dynamic Model Management

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

#### Check Model Status

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

#### Load/Switch LLM Model

```bash
# Load Qwen model
curl http://localhost:8080/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "model_type": "llm"
  }'

# Switch to Mistral (Qwen is automatically unloaded first)
curl http://localhost:8080/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "model_type": "llm"
  }'
```

#### Load ASR Model

```bash
curl http://localhost:8080/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/paraformer/model",
    "model_type": "asr"
  }'
```

#### Load TTS Model

```bash
curl http://localhost:8080/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/reference/audio.wav",
    "model_type": "tts"
  }'
```

#### Load/Switch Image Model

```bash
# Load Z-Image (faster, 9 steps)
curl http://localhost:8080/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zimage",
    "model_type": "image"
  }'

# Switch to FLUX (higher quality, 4 steps)
curl http://localhost:8080/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model": "flux",
    "model_type": "image"
  }'
```

#### Unload Model (Free Memory)

```bash
# Unload LLM to free GPU memory
curl http://localhost:8080/v1/models/unload \
  -H "Content-Type: application/json" \
  -d '{"model_type": "llm"}'

# Unload all models
curl http://localhost:8080/v1/models/unload \
  -H "Content-Type: application/json" \
  -d '{"model_type": "all"}'
```

#### Model Type Reference

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
┌─────────────────────────────────────────────────────────┐
│                   HTTP Server (Salvo)                    │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐    │
│  │  /chat  │ │  /asr   │ │  /tts   │ │   /images   │    │
│  └────┬────┘ └────┬────┘ └────┬────┘ └──────┬──────┘    │
│       │           │           │             │            │
│  ┌────┴───────────┴───────────┴─────────────┴────┐      │
│  │          /v1/models/load & /unload            │      │
│  └───────────────────────┬───────────────────────┘      │
│                          │                               │
│              mpsc::channel (bounded: 32)                 │
│                          │                               │
└──────────────────────────┼───────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Inference Thread (owns models)              │
│                                                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐     │
│  │  LLM Slot    │ │  ASR Slot    │ │  TTS Slot    │     │
│  │ ┌──────────┐ │ │ ┌──────────┐ │ │ ┌──────────┐ │     │
│  │ │  Qwen3   │ │ │ │Paraformer│ │ │ │GPT-SoVITS│ │     │
│  │ │    OR    │ │ │ │    OR    │ │ │ │    OR    │ │     │
│  │ │ Mistral  │ │ │ │  (empty) │ │ │ │  (empty) │ │     │
│  │ │    OR    │ │ │ └──────────┘ │ │ └──────────┘ │     │
│  │ │  (empty) │ │ └──────────────┘ └──────────────┘     │
│  │ └──────────┘ │                                       │
│  └──────────────┘ ┌──────────────┐                      │
│                   │ Image Slot   │                      │
│                   │ ┌──────────┐ │                      │
│                   │ │  Z-Image │ │                      │
│                   │ │    OR    │ │                      │
│                   │ │   FLUX   │ │                      │
│                   │ │    OR    │ │                      │
│                   │ │  (empty) │ │                      │
│                   │ └──────────┘ │                      │
│                   └──────────────┘                      │
└─────────────────────────────────────────────────────────┘
```

**Key Design Points:**

- **Actor Model**: MLX models don't implement `Send`/`Sync`, so all models run on a dedicated inference thread. HTTP handlers communicate via bounded async channels.

- **One Model Per Slot**: Each category has exactly one model slot. Loading a new model automatically unloads the previous one to free GPU memory.

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
| TTS | GPT-SoVITS | 4x real-time | 2 GB |
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
