# OminiX-API

OpenAI-compatible API server for OminiX-MLX models on Apple Silicon.

## Features

- **OpenAI-compatible endpoints** - Drop-in replacement for OpenAI API
- **LLM inference** - Chat completions with Mistral, Qwen, GLM models
- **Speech-to-text** - Audio transcription with Paraformer ASR
- **Text-to-speech** - Voice cloning with GPT-SoVITS
- **Image generation** - Text-to-image with FLUX.2-klein (placeholder)

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completions (LLM) |
| `/v1/audio/transcriptions` | POST | Speech-to-text (ASR) |
| `/v1/audio/speech` | POST | Text-to-speech (TTS) |
| `/v1/images/generations` | POST | Image generation |

## Quick Start

```bash
# Run with default settings (LLM only)
cargo run --release

# Run with specific models
LLM_MODEL=mlx-community/Mistral-7B-Instruct-v0.2-4bit \
ASR_MODEL_DIR=/path/to/paraformer \
TTS_REF_AUDIO=/path/to/reference.wav \
cargo run --release
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | HTTP server port |
| `LLM_MODEL` | `mlx-community/Mistral-7B-Instruct-v0.2-4bit` | HuggingFace model ID for LLM |
| `ASR_MODEL_DIR` | (empty) | Path to Paraformer model directory |
| `TTS_REF_AUDIO` | (empty) | Path to TTS reference audio file |
| `IMAGE_MODEL` | (empty) | Image model ID (placeholder) |

## API Usage

### Chat Completions

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral",
    "messages": [
      {"role": "user", "content": "Hello, who are you?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

### Audio Transcription

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

### Text-to-Speech

```bash
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, world!",
    "model": "gpt-sovits"
  }' \
  --output output.wav
```

### Image Generation

```bash
curl http://localhost:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over the ocean",
    "size": "512x512",
    "n": 1
  }'
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   HTTP Server (Salvo)               │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │  /chat  │ │  /asr   │ │  /tts   │ │ /images │   │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │
│       │           │           │           │         │
│       └───────────┴───────────┴───────────┘         │
│                       │                             │
│              mpsc::channel                          │
│                       │                             │
└───────────────────────┼─────────────────────────────┘
                        │
┌───────────────────────┼─────────────────────────────┐
│            Inference Thread (owns models)           │
│                       │                             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │   LLM   │ │   ASR   │ │   TTS   │ │  Image  │   │
│  │(Mistral)│ │(Parafor)│ │(SoVITS) │ │ (FLUX)  │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
└─────────────────────────────────────────────────────┘
```

Note: MLX models don't implement `Send`/`Sync`, so all models run on a dedicated inference thread. HTTP handlers communicate via channels.

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Rust 1.82+
- ~4GB+ RAM for LLM (model dependent)

## Model Setup

### LLM (Chat Completions)

Models are automatically downloaded from HuggingFace. Supported models:
- `mlx-community/Mistral-7B-Instruct-v0.2-4bit` (default)
- `mlx-community/Qwen3-4B-bf16`
- `mlx-community/GLM-4-9B-Chat-4bit`

### ASR (Transcriptions)

Download Paraformer model files to a directory:
```
paraformer/
├── paraformer.safetensors
├── am.mvn
└── tokens.txt
```

### TTS (Speech)

Provide a reference audio file (WAV format) for voice cloning.

## License

MIT OR Apache-2.0
