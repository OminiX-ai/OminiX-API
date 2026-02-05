//! OminiX-API: OpenAI-compatible API server for OminiX-MLX models
//!
//! Provides endpoints for:
//! - POST /v1/chat/completions - LLM chat completions
//! - POST /v1/audio/transcriptions - Speech-to-text (ASR)
//! - POST /v1/audio/speech - Text-to-speech (TTS)
//! - POST /v1/images/generations - Image generation
//! - WS   /ws/v1/tts - WebSocket streaming TTS with per-message voice switching
//! - POST /v1/voices/train - Voice cloning training (quick mode)
//! - GET  /v1/voices/train/{id}/progress - SSE training progress stream
//! - GET  /v1/voices - List registered voices
//!
//! Note: MLX models don't implement Send/Sync, so we use channels to
//! communicate with dedicated inference and training threads.

use std::time::Duration;
use eyre::Context;
use salvo::cors::*;
use salvo::prelude::*;
use salvo::sse::{SseEvent, SseKeepAlive};
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio::time::timeout;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;
use salvo::websocket::{Message, WebSocket, WebSocketUpgrade};

/// Timeout for LLM chat completions (can be long for large responses)
const CHAT_TIMEOUT: Duration = Duration::from_secs(300); // 5 minutes
/// Timeout for audio transcription
const TRANSCRIPTION_TIMEOUT: Duration = Duration::from_secs(120); // 2 minutes
/// Timeout for text-to-speech
const TTS_TIMEOUT: Duration = Duration::from_secs(60); // 1 minute
/// Timeout for image generation (can be slow depending on model)
const IMAGE_TIMEOUT: Duration = Duration::from_secs(600); // 10 minutes
/// Timeout for model loading
const MODEL_LOAD_TIMEOUT: Duration = Duration::from_secs(300); // 5 minutes

mod types;
mod utils;
mod llm;
mod asr;
mod tts;
mod image;
mod model_config;
mod training;

use types::*;

/// Render a standardized error response with proper HTTP status code
fn render_error(res: &mut Response, status: salvo::http::StatusCode, message: &str, error_type: &str) {
    res.status_code(status);
    res.render(Json(ApiError {
        error: ApiErrorDetail {
            message: message.to_string(),
            r#type: error_type.to_string(),
            code: None,
        },
    }));
}

/// Request sent to the inference thread
enum InferenceRequest {
    Chat {
        request: ChatCompletionRequest,
        response_tx: oneshot::Sender<eyre::Result<ChatCompletionResponse>>,
    },
    Transcribe {
        request: TranscriptionRequest,
        response_tx: oneshot::Sender<eyre::Result<TranscriptionResponse>>,
    },
    Speech {
        request: SpeechRequest,
        response_tx: oneshot::Sender<eyre::Result<Vec<u8>>>,
    },
    Image {
        request: ImageGenerationRequest,
        response_tx: oneshot::Sender<eyre::Result<ImageGenerationResponse>>,
    },
    /// Load/switch LLM model dynamically
    LoadLlmModel {
        model_id: String,
        response_tx: oneshot::Sender<eyre::Result<String>>,
    },
    /// Load/switch ASR model dynamically
    LoadAsrModel {
        model_dir: String,
        response_tx: oneshot::Sender<eyre::Result<String>>,
    },
    /// Load/switch TTS model dynamically
    LoadTtsModel {
        ref_audio: String,
        response_tx: oneshot::Sender<eyre::Result<String>>,
    },
    /// Load/switch image generation model dynamically
    LoadImageModel {
        model_id: String,
        response_tx: oneshot::Sender<eyre::Result<String>>,
    },
    /// Unload a model to free memory
    UnloadModel {
        model_type: String,
        response_tx: oneshot::Sender<eyre::Result<String>>,
    },
    /// Get current model status
    GetModelStatus {
        response_tx: oneshot::Sender<ModelStatus>,
    },
    /// Reload voices.json after training completes
    ReloadVoices {
        response_tx: oneshot::Sender<eyre::Result<()>>,
    },
}

/// Current status of all models
#[derive(Clone, serde::Serialize)]
struct ModelStatus {
    llm: Option<String>,
    asr: Option<String>,
    tts: Option<String>,
    image: Option<String>,
}

/// Application state shared across HTTP handlers
#[derive(Clone)]
pub struct AppState {
    /// Channel to send inference requests
    inference_tx: mpsc::Sender<InferenceRequest>,
    /// Channel to send training requests
    training_tx: mpsc::Sender<training::TrainingRequest>,
    /// Broadcast channel for training progress events (Sender is Clone)
    progress_tx: broadcast::Sender<TrainingProgressEvent>,
    /// Shared cancel flag for the active training task
    cancel_flag: training::CancelFlag,
}

/// Configuration from environment
#[derive(Debug, Clone)]
pub struct Config {
    pub port: u16,
    pub llm_model: String,
    pub asr_model_dir: String,
    pub tts_ref_audio: String,
    pub image_model: String,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            port: std::env::var("PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8080),
            llm_model: std::env::var("LLM_MODEL")
                .unwrap_or_else(|_| "mlx-community/Mistral-7B-Instruct-v0.2-4bit".to_string()),
            asr_model_dir: std::env::var("ASR_MODEL_DIR")
                .unwrap_or_else(|_| "".to_string()),
            tts_ref_audio: std::env::var("TTS_REF_AUDIO")
                .unwrap_or_else(|_| "".to_string()),
            image_model: std::env::var("IMAGE_MODEL")
                .unwrap_or_else(|_| "".to_string()),
        }
    }
}

/// Inference thread that owns all models (models are not Send/Sync)
fn inference_thread(
    config: Config,
    mut rx: mpsc::Receiver<InferenceRequest>,
    ready_tx: oneshot::Sender<()>,
) {
    // Track current model IDs for status reporting
    let mut current_llm_model: Option<String> = None;
    let mut current_asr_model: Option<String> = None;
    let mut current_tts_model: Option<String> = None;
    let mut current_image_model: Option<String> = None;

    // Load LLM (mutable so it can be swapped)
    let mut llm: Option<llm::LlmEngine> = if !config.llm_model.is_empty() {
        tracing::info!("Loading LLM model: {}", config.llm_model);
        match llm::LlmEngine::new(&config.llm_model) {
            Ok(engine) => {
                tracing::info!("LLM model loaded successfully");
                current_llm_model = Some(config.llm_model.clone());
                Some(engine)
            }
            Err(e) => {
                tracing::warn!("Failed to load LLM model: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Load ASR (mutable so it can be swapped)
    let mut asr: Option<asr::AsrEngine> = if !config.asr_model_dir.is_empty() {
        tracing::info!("Loading ASR model from: {}", config.asr_model_dir);
        match asr::AsrEngine::new(&config.asr_model_dir) {
            Ok(engine) => {
                tracing::info!("ASR model loaded successfully");
                current_asr_model = Some(config.asr_model_dir.clone());
                Some(engine)
            }
            Err(e) => {
                tracing::warn!("Failed to load ASR model: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Load TTS (mutable so it can be swapped)
    let mut tts: Option<tts::TtsEngine> = if !config.tts_ref_audio.is_empty() {
        tracing::info!("Loading TTS model with ref audio: {}", config.tts_ref_audio);
        match tts::TtsEngine::new(&config.tts_ref_audio) {
            Ok(engine) => {
                tracing::info!("TTS model loaded successfully");
                current_tts_model = Some(config.tts_ref_audio.clone());
                Some(engine)
            }
            Err(e) => {
                tracing::warn!("Failed to load TTS model: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Load Image model (mutable so it can be swapped)
    let mut image_engine: Option<image::ImageEngine> = if !config.image_model.is_empty() {
        tracing::info!("Loading image model: {}", config.image_model);
        match image::ImageEngine::new(&config.image_model) {
            Ok(engine) => {
                tracing::info!("Image model loaded successfully");
                let normalized = normalize_image_model(&config.image_model);
                current_image_model = Some(normalized.to_string());
                Some(engine)
            }
            Err(e) => {
                tracing::warn!("Failed to load image model: {}", e);
                None
            }
        }
    } else {
        None
    };

    fn normalize_image_model(model: &str) -> &'static str {
        let lower = model.to_lowercase();
        if lower.contains("flux") {
            "flux"
        } else if lower.contains("qwen") {
            "qwen-image"
        } else {
            "zimage"
        }
    }

    // Signal that models are loaded
    let _ = ready_tx.send(());

    tracing::info!("Inference thread ready, processing requests...");
    tracing::info!("Dynamic model loading enabled - use POST /v1/models/load to switch models");

    // Process requests
    while let Some(request) = rx.blocking_recv() {
        match request {
            InferenceRequest::Chat { request, response_tx } => {
                let result = if let Some(ref engine) = llm {
                    engine.generate(&request)
                } else {
                    Err(eyre::eyre!("LLM model not loaded"))
                };
                let _ = response_tx.send(result);
            }
            InferenceRequest::Transcribe { request, response_tx } => {
                let result = if let Some(ref engine) = asr {
                    engine.transcribe(&request)
                } else {
                    Err(eyre::eyre!("ASR model not loaded"))
                };
                let _ = response_tx.send(result);
            }
            InferenceRequest::Speech { request, response_tx } => {
                let result = if let Some(ref mut engine) = tts {
                    engine.synthesize(&request)
                } else {
                    Err(eyre::eyre!("TTS model not loaded"))
                };
                let _ = response_tx.send(result);
            }
            InferenceRequest::Image { request, response_tx } => {
                // Check if we need to switch models based on request
                let requested_model = request.model.as_deref().unwrap_or("");
                if !requested_model.is_empty() {
                    let normalized = normalize_image_model(requested_model);
                    let current_normalized = current_image_model.as_deref().unwrap_or("");

                    if normalized != current_normalized || image_engine.is_none() {
                        tracing::info!("Switching image model: {:?} -> {}", current_image_model, normalized);
                        // Drop old engine to free memory
                        image_engine = None;

                        match image::ImageEngine::new(normalized) {
                            Ok(engine) => {
                                tracing::info!("Image model {} loaded successfully", normalized);
                                current_image_model = Some(normalized.to_string());
                                image_engine = Some(engine);
                            }
                            Err(e) => {
                                tracing::error!("Failed to load image model {}: {}", normalized, e);
                            }
                        }
                    }
                }

                let result = if let Some(ref mut engine) = image_engine {
                    engine.generate(&request)
                } else {
                    Err(eyre::eyre!("Image model not loaded. Specify 'model': 'zimage' or 'flux' in your request."))
                };
                let _ = response_tx.send(result);
            }

            // === Dynamic Model Loading ===

            InferenceRequest::LoadLlmModel { model_id, response_tx } => {
                tracing::info!("Loading LLM model: {}", model_id);

                // Drop old engine to free memory before loading new one
                llm = None;
                current_llm_model = None;

                let result = match llm::LlmEngine::new(&model_id) {
                    Ok(engine) => {
                        tracing::info!("LLM model {} loaded successfully", model_id);
                        current_llm_model = Some(model_id.clone());
                        llm = Some(engine);
                        Ok(model_id)
                    }
                    Err(e) => {
                        tracing::error!("Failed to load LLM model {}: {}", model_id, e);
                        Err(e)
                    }
                };
                let _ = response_tx.send(result);
            }

            InferenceRequest::LoadAsrModel { model_dir, response_tx } => {
                tracing::info!("Loading ASR model from: {}", model_dir);

                // Drop old engine to free memory
                asr = None;
                current_asr_model = None;

                let result = match asr::AsrEngine::new(&model_dir) {
                    Ok(engine) => {
                        tracing::info!("ASR model loaded successfully from {}", model_dir);
                        current_asr_model = Some(model_dir.clone());
                        asr = Some(engine);
                        Ok(model_dir)
                    }
                    Err(e) => {
                        tracing::error!("Failed to load ASR model from {}: {}", model_dir, e);
                        Err(e)
                    }
                };
                let _ = response_tx.send(result);
            }

            InferenceRequest::LoadTtsModel { ref_audio, response_tx } => {
                tracing::info!("Loading TTS model with ref audio: {}", ref_audio);

                // Drop old engine to free memory
                tts = None;
                current_tts_model = None;

                let result = match tts::TtsEngine::new(&ref_audio) {
                    Ok(engine) => {
                        tracing::info!("TTS model loaded successfully with ref: {}", ref_audio);
                        current_tts_model = Some(ref_audio.clone());
                        tts = Some(engine);
                        Ok(ref_audio)
                    }
                    Err(e) => {
                        tracing::error!("Failed to load TTS model with {}: {}", ref_audio, e);
                        Err(e)
                    }
                };
                let _ = response_tx.send(result);
            }

            InferenceRequest::LoadImageModel { model_id, response_tx } => {
                let normalized = normalize_image_model(&model_id);
                tracing::info!("Loading image model: {} (normalized: {})", model_id, normalized);

                // Drop old engine to free memory
                image_engine = None;
                current_image_model = None;

                let result = match image::ImageEngine::new(normalized) {
                    Ok(engine) => {
                        tracing::info!("Image model {} loaded successfully", normalized);
                        current_image_model = Some(normalized.to_string());
                        image_engine = Some(engine);
                        Ok(normalized.to_string())
                    }
                    Err(e) => {
                        tracing::error!("Failed to load image model {}: {}", normalized, e);
                        Err(e)
                    }
                };
                let _ = response_tx.send(result);
            }

            InferenceRequest::UnloadModel { model_type, response_tx } => {
                tracing::info!("Unloading model type: {}", model_type);

                let result = match model_type.as_str() {
                    "llm" => {
                        llm = None;
                        let prev = current_llm_model.take();
                        Ok(format!("Unloaded LLM model: {:?}", prev))
                    }
                    "asr" => {
                        asr = None;
                        let prev = current_asr_model.take();
                        Ok(format!("Unloaded ASR model: {:?}", prev))
                    }
                    "tts" => {
                        tts = None;
                        let prev = current_tts_model.take();
                        Ok(format!("Unloaded TTS model: {:?}", prev))
                    }
                    "image" => {
                        image_engine = None;
                        let prev = current_image_model.take();
                        Ok(format!("Unloaded image model: {:?}", prev))
                    }
                    "all" => {
                        llm = None;
                        asr = None;
                        tts = None;
                        image_engine = None;
                        current_llm_model = None;
                        current_asr_model = None;
                        current_tts_model = None;
                        current_image_model = None;
                        Ok("Unloaded all models".to_string())
                    }
                    _ => Err(eyre::eyre!("Unknown model type: {}. Use: llm, asr, tts, image, or all", model_type)),
                };
                let _ = response_tx.send(result);
            }

            InferenceRequest::GetModelStatus { response_tx } => {
                let status = ModelStatus {
                    llm: current_llm_model.clone(),
                    asr: current_asr_model.clone(),
                    tts: current_tts_model.clone(),
                    image: current_image_model.clone(),
                };
                let _ = response_tx.send(status);
            }

            InferenceRequest::ReloadVoices { response_tx } => {
                tracing::info!("Reloading voice registry");
                if let Some(ref mut engine) = tts {
                    engine.reload_voices();
                    let _ = response_tx.send(Ok(()));
                } else {
                    let _ = response_tx.send(Ok(()));
                }
            }
        }
    }

    tracing::info!("Inference thread shutting down");
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ominix_api=info".into()),
        )
        .init();

    let config = Config::from_env();
    tracing::info!("Starting OminiX-API server on port {}", config.port);

    // Print model configuration status report
    model_config::print_startup_report();

    // Create channel for inference requests
    let (inference_tx, inference_rx) = mpsc::channel::<InferenceRequest>(32);
    let (ready_tx, ready_rx) = oneshot::channel();

    // Create channels for training
    let (training_tx, training_rx) = mpsc::channel::<training::TrainingRequest>(8);
    let (progress_tx, _) = broadcast::channel::<TrainingProgressEvent>(256);

    // Spawn inference thread (owns all models)
    let config_clone = config.clone();
    std::thread::spawn(move || {
        inference_thread(config_clone, inference_rx, ready_tx);
    });

    // Spawn training thread (owns training models, separate from inference)
    let progress_tx_clone = progress_tx.clone();
    let inference_tx_for_training = inference_tx.clone();
    let cancel_flag: training::CancelFlag = Default::default();
    let cancel_flag_clone = cancel_flag.clone();
    std::thread::spawn(move || {
        training::training_thread(training_rx, progress_tx_clone, inference_tx_for_training, cancel_flag_clone);
    });

    // Wait for models to load
    ready_rx.await.context("Failed to receive ready signal from inference thread")?;
    tracing::info!("Inference thread ready");

    let state = AppState {
        inference_tx,
        training_tx,
        progress_tx,
        cancel_flag,
    };

    // Build Salvo router
    let router = Router::new()
        .hoop(affix_state::inject(state))
        .hoop(
            Cors::new()
                .allow_origin(AllowOrigin::any())
                .allow_methods(AllowMethods::any())
                .allow_headers(AllowHeaders::any())
                .into_handler(),
        )
        // Health & Models
        .push(Router::with_path("health").get(health))
        .push(Router::with_path("v1/models").get(list_models))
        .push(Router::with_path("v1/models/status").get(model_status))
        .push(Router::with_path("v1/models/load").post(load_model))
        .push(Router::with_path("v1/models/unload").post(unload_model))
        // Chat completions
        .push(Router::with_path("v1/chat/completions").post(chat_completions))
        // Audio endpoints
        .push(Router::with_path("v1/audio/transcriptions").post(audio_transcriptions))
        .push(Router::with_path("v1/audio/speech").post(audio_speech))
        // Image generation
        .push(Router::with_path("v1/images/generations").post(images_generations))
        // WebSocket TTS
        .push(Router::with_path("ws/v1/tts").get(ws_tts))
        // Voice cloning training
        .push(Router::with_path("v1/voices").get(list_voices))
        .push(Router::with_path("v1/voices/train").post(start_voice_training))
        .push(Router::with_path("v1/voices/train/status").get(get_training_status))
        .push(Router::with_path("v1/voices/train/progress").get(training_progress_sse))
        .push(Router::with_path("v1/voices/train/cancel").post(cancel_training));

    let listen_addr = format!("0.0.0.0:{}", config.port);
    let acceptor = TcpListener::new(&listen_addr).bind().await;

    tracing::info!("HTTP server listening on http://{}", listen_addr);
    tracing::info!("Endpoints:");
    tracing::info!("  GET  /health");
    tracing::info!("  GET  /v1/models");
    tracing::info!("  GET  /v1/models/status       - Get current model status");
    tracing::info!("  POST /v1/models/load         - Load model dynamically");
    tracing::info!("  POST /v1/models/unload       - Unload model to free memory");
    tracing::info!("  POST /v1/chat/completions");
    tracing::info!("  POST /v1/audio/transcriptions");
    tracing::info!("  POST /v1/audio/speech");
    tracing::info!("  POST /v1/images/generations");
    tracing::info!("  WS   /ws/v1/tts              - WebSocket streaming TTS");
    tracing::info!("  GET  /v1/voices              - List registered voices");
    tracing::info!("  POST /v1/voices/train        - Start voice cloning training");
    tracing::info!("  GET  /v1/voices/train/{{id}}    - Get training task status");
    tracing::info!("  GET  /v1/voices/train/{{id}}/progress - SSE training progress");
    tracing::info!("");
    tracing::info!("Dynamic model loading examples:");
    tracing::info!("  curl -X POST http://localhost:{}/v1/models/load -H 'Content-Type: application/json' -d '{{\"model\": \"mlx-community/Qwen2.5-7B-Instruct-4bit\", \"model_type\": \"llm\"}}'", config.port);
    tracing::info!("  curl -X POST http://localhost:{}/v1/models/unload -H 'Content-Type: application/json' -d '{{\"model_type\": \"llm\"}}'", config.port);

    Server::new(acceptor).serve(router).await;

    Ok(())
}

/// GET /health - Health check
#[handler]
async fn health(res: &mut Response) {
    res.render(Json(serde_json::json!({
        "status": "healthy",
        "service": "ominix-api"
    })));
}

/// GET /v1/models - List available models (queries live status)
#[handler]
async fn list_models(depot: &mut Depot, res: &mut Response) -> Result<(), StatusError> {
    let state = depot
        .obtain::<AppState>()
        .map_err(|_| StatusError::internal_server_error())?;

    let (response_tx, response_rx) = oneshot::channel();
    state.inference_tx.send(InferenceRequest::GetModelStatus {
        response_tx,
    }).await.map_err(|_| StatusError::internal_server_error())?;

    let status = response_rx.await
        .map_err(|_| StatusError::internal_server_error())?;

    let now = chrono::Utc::now().timestamp();
    let mut data = Vec::new();

    if let Some(ref id) = status.llm {
        data.push(serde_json::json!({
            "id": id, "object": "model", "created": now,
            "owned_by": "local", "type": "llm"
        }));
    }
    if let Some(ref id) = status.asr {
        data.push(serde_json::json!({
            "id": "paraformer", "object": "model", "created": now,
            "owned_by": "local", "type": "asr"
        }));
        let _ = id; // asr stores path, not display name
    }
    if let Some(ref _id) = status.tts {
        data.push(serde_json::json!({
            "id": "gpt-sovits", "object": "model", "created": now,
            "owned_by": "local", "type": "tts"
        }));
    }
    if let Some(ref id) = status.image {
        data.push(serde_json::json!({
            "id": id, "object": "model", "created": now,
            "owned_by": "local", "type": "image"
        }));
    }

    res.render(Json(serde_json::json!({
        "object": "list",
        "data": data
    })));
    Ok(())
}

/// GET /v1/models/status - Get current model status
#[handler]
async fn model_status(
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = depot
        .obtain::<AppState>()
        .map_err(|_| StatusError::internal_server_error())?;

    let (response_tx, response_rx) = oneshot::channel();
    state.inference_tx.send(InferenceRequest::GetModelStatus {
        response_tx,
    }).await.map_err(|_| StatusError::internal_server_error())?;

    let status = response_rx.await
        .map_err(|_| StatusError::internal_server_error())?;

    res.render(Json(serde_json::json!({
        "status": "success",
        "models": status
    })));
    Ok(())
}

/// POST /v1/models/load - Load a model dynamically
///
/// Request body:
/// - model: Model ID or path (required)
/// - model_type: "llm", "asr", "tts", or "image" (default: "llm")
///
/// For LLM: model is a HuggingFace model ID (e.g., "mlx-community/Qwen2.5-7B-Instruct-4bit")
/// For ASR: model is a path to the model directory
/// For TTS: model is a path to the reference audio file
/// For Image: model is "zimage" or "flux"
#[handler]
async fn load_model(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = depot
        .obtain::<AppState>()
        .map_err(|_| StatusError::internal_server_error())?;

    #[derive(serde::Deserialize)]
    struct LoadModelRequest {
        model: String,
        #[serde(default = "default_model_type")]
        model_type: String,
    }

    fn default_model_type() -> String {
        "llm".to_string()
    }

    let request: LoadModelRequest = req.parse_json().await.map_err(|e| {
        tracing::error!("Failed to parse request: {}", e);
        StatusError::bad_request()
    })?;

    let model_type = request.model_type.as_str();
    tracing::info!("Loading model: {} (type: {})", request.model, model_type);

    // Validate model paths to prevent directory traversal
    if !utils::is_safe_path(&request.model) {
        render_error(res, salvo::http::StatusCode::BAD_REQUEST,
            "Model path contains invalid directory traversal", "invalid_request_error");
        return Ok(());
    }

    let inference_request = match model_type {
        "llm" => {
            let (response_tx, response_rx) = oneshot::channel();
            state.inference_tx.send(InferenceRequest::LoadLlmModel {
                model_id: request.model.clone(),
                response_tx,
            }).await.map_err(|_| StatusError::internal_server_error())?;
            response_rx
        }
        "asr" => {
            let (response_tx, response_rx) = oneshot::channel();
            state.inference_tx.send(InferenceRequest::LoadAsrModel {
                model_dir: request.model.clone(),
                response_tx,
            }).await.map_err(|_| StatusError::internal_server_error())?;
            response_rx
        }
        "tts" => {
            let (response_tx, response_rx) = oneshot::channel();
            state.inference_tx.send(InferenceRequest::LoadTtsModel {
                ref_audio: request.model.clone(),
                response_tx,
            }).await.map_err(|_| StatusError::internal_server_error())?;
            response_rx
        }
        "image" => {
            let (response_tx, response_rx) = oneshot::channel();
            state.inference_tx.send(InferenceRequest::LoadImageModel {
                model_id: request.model.clone(),
                response_tx,
            }).await.map_err(|_| StatusError::internal_server_error())?;
            response_rx
        }
        _ => {
            render_error(res, salvo::http::StatusCode::BAD_REQUEST,
                &format!("Unknown model_type: {}. Use: llm, asr, tts, or image", model_type),
                "invalid_request_error");
            return Ok(());
        }
    };

    let result = timeout(MODEL_LOAD_TIMEOUT, inference_request).await
        .map_err(|_| {
            tracing::error!("Model loading timed out after {:?}", MODEL_LOAD_TIMEOUT);
            StatusError::gateway_timeout()
        })?
        .map_err(|_| StatusError::internal_server_error())?;

    match result {
        Ok(loaded_model) => {
            res.render(Json(serde_json::json!({
                "status": "success",
                "model": loaded_model,
                "model_type": model_type
            })));
        }
        Err(e) => {
            render_error(res, salvo::http::StatusCode::INTERNAL_SERVER_ERROR,
                &e.to_string(), "server_error");
        }
    }

    Ok(())
}

/// POST /v1/models/unload - Unload a model to free memory
///
/// Request body:
/// - model_type: "llm", "asr", "tts", "image", or "all"
#[handler]
async fn unload_model(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = depot
        .obtain::<AppState>()
        .map_err(|_| StatusError::internal_server_error())?;

    #[derive(serde::Deserialize)]
    struct UnloadModelRequest {
        model_type: String,
    }

    let request: UnloadModelRequest = req.parse_json().await.map_err(|e| {
        tracing::error!("Failed to parse request: {}", e);
        StatusError::bad_request()
    })?;

    tracing::info!("Unloading model type: {}", request.model_type);

    let (response_tx, response_rx) = oneshot::channel();
    state.inference_tx.send(InferenceRequest::UnloadModel {
        model_type: request.model_type.clone(),
        response_tx,
    }).await.map_err(|_| StatusError::internal_server_error())?;

    let result = timeout(Duration::from_secs(30), response_rx).await
        .map_err(|_| {
            tracing::error!("Model unloading timed out");
            StatusError::gateway_timeout()
        })?
        .map_err(|_| StatusError::internal_server_error())?;

    match result {
        Ok(message) => {
            res.render(Json(serde_json::json!({
                "status": "success",
                "message": message
            })));
        }
        Err(e) => {
            render_error(res, salvo::http::StatusCode::BAD_REQUEST,
                &e.to_string(), "invalid_request_error");
        }
    }

    Ok(())
}

/// POST /v1/chat/completions - OpenAI-compatible chat completions
#[handler]
async fn chat_completions(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = depot
        .obtain::<AppState>()
        .map_err(|_| StatusError::internal_server_error())?;

    let request: ChatCompletionRequest = req.parse_json().await.map_err(|e| {
        tracing::error!("Failed to parse request: {}", e);
        StatusError::bad_request()
    })?;

    // Check if streaming is requested
    if request.stream.unwrap_or(false) {
        tracing::warn!("Streaming not yet implemented, falling back to non-streaming");
    }

    // Send to inference thread
    let (response_tx, response_rx) = oneshot::channel();
    state.inference_tx.send(InferenceRequest::Chat {
        request,
        response_tx,
    }).await.map_err(|_| StatusError::internal_server_error())?;

    // Wait for response with timeout
    let response = timeout(CHAT_TIMEOUT, response_rx).await
        .map_err(|_| {
            tracing::error!("Chat completion timed out after {:?}", CHAT_TIMEOUT);
            StatusError::gateway_timeout()
        })?
        .map_err(|_| StatusError::internal_server_error())?
        .map_err(|e| {
            tracing::error!("Generation error: {}", e);
            StatusError::internal_server_error()
        })?;

    res.render(Json(response));
    Ok(())
}

/// POST /v1/audio/transcriptions - Speech-to-text
#[handler]
async fn audio_transcriptions(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = depot
        .obtain::<AppState>()
        .map_err(|_| StatusError::internal_server_error())?;

    // Use larger body size limit for audio uploads (10MB)
    let request: TranscriptionRequest = req.parse_json_with_max_size(10 * 1024 * 1024).await.map_err(|e| {
        tracing::error!("Failed to parse request: {}", e);
        StatusError::bad_request()
    })?;

    let (response_tx, response_rx) = oneshot::channel();
    state.inference_tx.send(InferenceRequest::Transcribe {
        request,
        response_tx,
    }).await.map_err(|_| StatusError::internal_server_error())?;

    let response = timeout(TRANSCRIPTION_TIMEOUT, response_rx).await
        .map_err(|_| {
            tracing::error!("Transcription timed out after {:?}", TRANSCRIPTION_TIMEOUT);
            StatusError::gateway_timeout()
        })?
        .map_err(|_| StatusError::internal_server_error())?
        .map_err(|e| {
            tracing::error!("Transcription error: {}", e);
            StatusError::internal_server_error()
        })?;

    res.render(Json(response));
    Ok(())
}

/// POST /v1/audio/speech - Text-to-speech
#[handler]
async fn audio_speech(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = depot
        .obtain::<AppState>()
        .map_err(|_| StatusError::internal_server_error())?;

    let request: SpeechRequest = req.parse_json().await.map_err(|e| {
        tracing::error!("Failed to parse request: {}", e);
        StatusError::bad_request()
    })?;

    let (response_tx, response_rx) = oneshot::channel();
    state.inference_tx.send(InferenceRequest::Speech {
        request,
        response_tx,
    }).await.map_err(|_| StatusError::internal_server_error())?;

    let audio_data = timeout(TTS_TIMEOUT, response_rx).await
        .map_err(|_| {
            tracing::error!("TTS timed out after {:?}", TTS_TIMEOUT);
            StatusError::gateway_timeout()
        })?
        .map_err(|_| StatusError::internal_server_error())?
        .map_err(|e| {
            tracing::error!("TTS error: {}", e);
            StatusError::internal_server_error()
        })?;

    res.headers_mut()
        .insert("Content-Type", "audio/wav".parse().unwrap());
    res.write_body(audio_data).ok();
    Ok(())
}

/// POST /v1/images/generations - Image generation
#[handler]
async fn images_generations(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = depot
        .obtain::<AppState>()
        .map_err(|_| StatusError::internal_server_error())?;

    // Use larger body size limit for image uploads (10MB for img2img)
    let request: ImageGenerationRequest = req.parse_json_with_max_size(10 * 1024 * 1024).await.map_err(|e| {
        tracing::error!("Failed to parse request: {}", e);
        StatusError::bad_request()
    })?;

    let (response_tx, response_rx) = oneshot::channel();
    state.inference_tx.send(InferenceRequest::Image {
        request,
        response_tx,
    }).await.map_err(|_| StatusError::internal_server_error())?;

    let response = timeout(IMAGE_TIMEOUT, response_rx).await
        .map_err(|_| {
            tracing::error!("Image generation timed out after {:?}", IMAGE_TIMEOUT);
            StatusError::gateway_timeout()
        })?
        .map_err(|_| StatusError::internal_server_error())?
        .map_err(|e| {
            tracing::error!("Image generation error: {}", e);
            StatusError::internal_server_error()
        })?;

    res.render(Json(response));
    Ok(())
}

// ============================================================================
// Voice Cloning Training Endpoints
// ============================================================================

/// GET /v1/voices - List all registered voices
#[handler]
async fn list_voices(res: &mut Response) {
    // Read voices.json directly
    let voices_path = utils::expand_tilde("~/.dora/models/primespeech/voices.json");
    let voices = match std::fs::read_to_string(&voices_path) {
        Ok(content) => {
            match serde_json::from_str::<serde_json::Value>(&content) {
                Ok(config) => {
                    let mut voice_list = Vec::new();
                    if let Some(voices) = config.get("voices").and_then(|v| v.as_object()) {
                        for (name, voice) in voices {
                            let aliases = voice.get("aliases")
                                .and_then(|a| a.as_array())
                                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                                .unwrap_or_default();
                            voice_list.push(VoiceInfo {
                                name: name.clone(),
                                aliases,
                            });
                        }
                    }
                    voice_list
                }
                Err(_) => Vec::new(),
            }
        }
        Err(_) => Vec::new(),
    };

    res.render(Json(VoiceListResponse { voices }));
}

/// POST /v1/voices/train - Start voice cloning training
#[handler]
async fn start_voice_training(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = depot.obtain::<AppState>()
        .map_err(|_| StatusError::internal_server_error())?;

    // Parse request (50MB limit for audio data)
    let request: VoiceTrainRequest = req.parse_json_with_max_size(50 * 1024 * 1024)
        .await.map_err(|e| {
            tracing::error!("Failed to parse training request: {}", e);
            StatusError::bad_request()
        })?;

    // Validate and sanitize voice name to prevent path traversal
    let voice_name = match utils::sanitize_voice_name(&request.voice_name) {
        Ok(name) => name,
        Err(e) => {
            render_error(res, salvo::http::StatusCode::BAD_REQUEST,
                &e.to_string(), "invalid_request_error");
            return Ok(());
        }
    };

    if request.transcript.is_empty() {
        render_error(res, salvo::http::StatusCode::BAD_REQUEST,
            "transcript is required", "invalid_request_error");
        return Ok(());
    }

    // Generate task ID
    let task_id = format!("train-{}", uuid::Uuid::new_v4().simple());

    // Decode and save audio to a temporary directory
    let base_dir = std::env::var("TRAINING_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .map(|h| h.join(".dora/training"))
                .unwrap_or_else(|| std::path::PathBuf::from("/tmp/ominix-training"))
        });
    let work_dir = base_dir.join(&task_id);
    std::fs::create_dir_all(&work_dir)
        .map_err(|_| StatusError::internal_server_error())?;

    let audio_bytes = base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD,
        &request.audio,
    ).map_err(|e| {
        tracing::error!("Base64 decode failed: {}", e);
        StatusError::bad_request()
    })?;

    let audio_path = work_dir.join("ref_audio.wav");
    std::fs::write(&audio_path, &audio_bytes)
        .map_err(|_| StatusError::internal_server_error())?;

    // Send training request to training thread
    let (response_tx, response_rx) = oneshot::channel();
    state.training_tx.send(training::TrainingRequest::StartTraining {
        task_id: task_id.clone(),
        voice_name: voice_name.clone(),
        audio_path,
        transcript: request.transcript,
        quality: request.quality,
        language: request.language,
        denoise: request.denoise,
        response_tx,
    }).await.map_err(|_| StatusError::internal_server_error())?;

    // Wait for acknowledgement (not completion)
    match response_rx.await {
        Ok(Ok(())) => {
            res.render(Json(VoiceTrainResponse {
                task_id,
                status: "accepted".to_string(),
                message: format!("Training started for voice '{}'", voice_name),
            }));
        }
        Ok(Err(e)) => {
            render_error(res, salvo::http::StatusCode::CONFLICT,
                &e.to_string(), "server_error");
        }
        Err(_) => {
            render_error(res, salvo::http::StatusCode::SERVICE_UNAVAILABLE,
                "Training thread unavailable", "server_error");
        }
    }

    Ok(())
}

/// GET /v1/voices/train/{task_id} - Get training task status
#[handler]
async fn get_training_status(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = depot.obtain::<AppState>()
        .map_err(|_| StatusError::internal_server_error())?;

    let task_id: String = req.query::<String>("task_id")
        .unwrap_or_default();
    if task_id.is_empty() {
        render_error(res, salvo::http::StatusCode::BAD_REQUEST,
            "task_id query parameter required", "invalid_request_error");
        return Ok(());
    }

    let (response_tx, response_rx) = oneshot::channel();
    state.training_tx.send(training::TrainingRequest::GetStatus {
        task_id: task_id.clone(),
        response_tx,
    }).await.map_err(|_| StatusError::internal_server_error())?;

    match timeout(Duration::from_secs(5), response_rx).await {
        Ok(Ok(Some(status))) => {
            res.render(Json(status));
        }
        Ok(Ok(None)) => {
            render_error(res, salvo::http::StatusCode::NOT_FOUND,
                &format!("Task not found: {}", task_id), "not_found_error");
        }
        _ => {
            render_error(res, salvo::http::StatusCode::SERVICE_UNAVAILABLE,
                "Training thread unavailable", "server_error");
        }
    }

    Ok(())
}

/// GET /v1/voices/train/{task_id}/progress - SSE streaming training progress
#[handler]
async fn training_progress_sse(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = depot.obtain::<AppState>()
        .map_err(|_| StatusError::internal_server_error())?;

    let task_id: String = req.query::<String>("task_id")
        .unwrap_or_default();
    if task_id.is_empty() {
        render_error(res, salvo::http::StatusCode::BAD_REQUEST,
            "task_id query parameter required", "invalid_request_error");
        return Ok(());
    }

    // Subscribe to progress events
    let rx = state.progress_tx.subscribe();

    // Filter for this task's events and convert to SSE
    let task_id_clone = task_id.clone();
    let stream = BroadcastStream::new(rx)
        .filter_map(move |result| {
            match result {
                Ok(event) if event.task_id == task_id_clone => {
                    let data = serde_json::to_string(&event).ok()?;
                    let sse_event = SseEvent::default().text(data);
                    Some(Ok::<_, std::convert::Infallible>(sse_event))
                }
                _ => None,
            }
        });

    SseKeepAlive::new(stream).stream(res);
    Ok(())
}

/// POST /v1/voices/train/{task_id}/cancel - Cancel training
#[handler]
async fn cancel_training(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = depot.obtain::<AppState>()
        .map_err(|_| StatusError::internal_server_error())?;

    let task_id: String = req.query::<String>("task_id")
        .unwrap_or_default();
    if task_id.is_empty() {
        render_error(res, salvo::http::StatusCode::BAD_REQUEST,
            "task_id query parameter required", "invalid_request_error");
        return Ok(());
    }

    match training::cancel_training_task(&state.cancel_flag, &task_id) {
        Ok(()) => {
            res.render(Json(serde_json::json!({
                "status": "success",
                "message": format!("Training {} cancelled", task_id)
            })));
        }
        Err(e) => {
            render_error(res, salvo::http::StatusCode::NOT_FOUND,
                &e.to_string(), "not_found_error");
        }
    }

    Ok(())
}


// ============================================================================
// WebSocket TTS
// ============================================================================

/// GET /ws/v1/tts - WebSocket streaming TTS
///
/// Protocol (MiniMax T2A compatible):
/// 1. Connect -> Server sends {"event": "connected_success"}
/// 2. Client sends {"event": "task_start", "voice_setting": {...}, "audio_setting": {...}}
///    -> Server sends {"event": "task_started"}
/// 3. Client sends {"event": "task_continue", "text": "..."}
///    -> Server streams {"event": "task_progress", "data": {"audio": "<hex>"}, "is_final": bool}
/// 4. Client sends {"event": "task_finish"} -> connection closes
#[handler]
async fn ws_tts(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = depot
        .obtain::<AppState>()
        .map_err(|_| StatusError::internal_server_error())?
        .clone();

    WebSocketUpgrade::new()
        .upgrade(req, res, |ws| handle_tts_websocket(ws, state))
        .await
}

async fn handle_tts_websocket(mut ws: WebSocket, state: AppState) {
    // Send connection success
    let msg = serde_json::json!({"event": "connected_success"});
    if ws.send(Message::text(msg.to_string())).await.is_err() {
        return;
    }

    // Session state from task_start
    let mut voice: Option<String> = None;
    let mut speed: f32 = 1.0;
    let mut audio_format = "wav".to_string();
    const CHUNK_SIZE: usize = 8192;

    while let Some(msg) = ws.recv().await {
        let msg = match msg {
            Ok(m) => m,
            Err(_) => break,
        };

        if msg.is_close() {
            break;
        }

        let text = match msg.as_str() {
            Ok(t) => t.to_string(),
            Err(_) => continue, // skip non-text (ping/pong/binary)
        };

        let event: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(_) => {
                let err = serde_json::json!({"event": "error", "message": "Invalid JSON"});
                let _ = ws.send(Message::text(err.to_string())).await;
                continue;
            }
        };

        match event.get("event").and_then(|e| e.as_str()) {
            Some("task_start") => {
                if let Some(vs) = event.get("voice_setting") {
                    voice = vs.get("voice_id")
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    speed = vs.get("speed")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1.0) as f32;
                }
                if let Some(audio_s) = event.get("audio_setting") {
                    audio_format = audio_s.get("format")
                        .and_then(|v| v.as_str())
                        .unwrap_or("wav")
                        .to_string();
                }

                let resp = serde_json::json!({"event": "task_started"});
                if ws.send(Message::text(resp.to_string())).await.is_err() {
                    break;
                }
            }

            Some("task_continue") => {
                let input_text = match event.get("text").and_then(|t| t.as_str()) {
                    Some(t) => t.to_string(),
                    None => {
                        let err = serde_json::json!({"event": "error", "message": "Missing 'text' field"});
                        let _ = ws.send(Message::text(err.to_string())).await;
                        continue;
                    }
                };

                // Per-message voice_id override, falls back to task_start voice
                let msg_voice = event.get("voice_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .or_else(|| voice.clone());

                let speech_req = SpeechRequest {
                    model: None,
                    input: input_text,
                    voice: msg_voice,
                    response_format: audio_format.clone(),
                    speed,
                };

                let (response_tx, response_rx) = oneshot::channel();
                if state.inference_tx.send(InferenceRequest::Speech {
                    request: speech_req,
                    response_tx,
                }).await.is_err() {
                    let err = serde_json::json!({"event": "error", "message": "Inference unavailable"});
                    let _ = ws.send(Message::text(err.to_string())).await;
                    break;
                }

                match timeout(TTS_TIMEOUT, response_rx).await {
                    Ok(Ok(Ok(audio_data))) => {
                        if audio_data.is_empty() {
                            let resp = serde_json::json!({
                                "event": "task_progress",
                                "data": {"audio": ""},
                                "is_final": true
                            });
                            let _ = ws.send(Message::text(resp.to_string())).await;
                        } else {
                            let total = audio_data.len();
                            let mut sent = 0;

                            for chunk in audio_data.chunks(CHUNK_SIZE) {
                                sent += chunk.len();
                                let is_final = sent >= total;
                                let hex_audio = hex::encode(chunk);

                                let resp = serde_json::json!({
                                    "event": "task_progress",
                                    "data": {"audio": hex_audio},
                                    "is_final": is_final
                                });

                                if ws.send(Message::text(resp.to_string())).await.is_err() {
                                    return;
                                }
                            }
                        }
                    }
                    Ok(Ok(Err(e))) => {
                        let err = serde_json::json!({
                            "event": "error",
                            "message": format!("TTS error: {}", e)
                        });
                        let _ = ws.send(Message::text(err.to_string())).await;
                    }
                    Ok(Err(_)) => {
                        let err = serde_json::json!({
                            "event": "error",
                            "message": "Inference channel dropped"
                        });
                        let _ = ws.send(Message::text(err.to_string())).await;
                    }
                    Err(_) => {
                        let err = serde_json::json!({
                            "event": "error",
                            "message": format!("TTS timed out after {:?}", TTS_TIMEOUT)
                        });
                        let _ = ws.send(Message::text(err.to_string())).await;
                    }
                }
            }

            Some("task_finish") => {
                break;
            }

            Some(unknown) => {
                let err = serde_json::json!({
                    "event": "error",
                    "message": format!("Unknown event: {}", unknown)
                });
                let _ = ws.send(Message::text(err.to_string())).await;
            }

            None => {
                let err = serde_json::json!({
                    "event": "error",
                    "message": "Missing 'event' field"
                });
                let _ = ws.send(Message::text(err.to_string())).await;
            }
        }
    }

    let _ = ws.close().await;
}
