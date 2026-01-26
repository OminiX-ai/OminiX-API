//! OminiX-API: OpenAI-compatible API server for OminiX-MLX models
//!
//! Provides endpoints for:
//! - POST /v1/chat/completions - LLM chat completions
//! - POST /v1/audio/transcriptions - Speech-to-text (ASR)
//! - POST /v1/audio/speech - Text-to-speech (TTS)
//! - POST /v1/images/generations - Image generation
//!
//! Note: MLX models don't implement Send/Sync, so we use channels to
//! communicate with a dedicated inference thread.

use eyre::Context;
use salvo::cors::*;
use salvo::prelude::*;
use tokio::sync::{mpsc, oneshot};

mod types;
mod llm;
mod asr;
mod tts;
mod image;

use types::*;

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
    /// Load/switch image generation model dynamically
    LoadImageModel {
        model_id: String,
        response_tx: oneshot::Sender<eyre::Result<String>>,
    },
}

/// Application state shared across HTTP handlers
#[derive(Clone)]
pub struct AppState {
    /// Channel to send inference requests
    inference_tx: mpsc::Sender<InferenceRequest>,
    /// Available models info
    models: Vec<ModelInfo>,
}

#[derive(Clone)]
struct ModelInfo {
    id: String,
    model_type: String,
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
    models_tx: oneshot::Sender<Vec<ModelInfo>>,
) {
    let mut models = Vec::new();

    // Load LLM
    let llm = if !config.llm_model.is_empty() {
        tracing::info!("Loading LLM model: {}", config.llm_model);
        match llm::LlmEngine::new(&config.llm_model) {
            Ok(engine) => {
                tracing::info!("LLM model loaded successfully");
                models.push(ModelInfo {
                    id: config.llm_model.clone(),
                    model_type: "llm".to_string(),
                });
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

    // Load ASR
    let asr = if !config.asr_model_dir.is_empty() {
        tracing::info!("Loading ASR model from: {}", config.asr_model_dir);
        match asr::AsrEngine::new(&config.asr_model_dir) {
            Ok(engine) => {
                tracing::info!("ASR model loaded successfully");
                models.push(ModelInfo {
                    id: "paraformer".to_string(),
                    model_type: "asr".to_string(),
                });
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

    // Load TTS
    let mut tts = if !config.tts_ref_audio.is_empty() {
        tracing::info!("Loading TTS model with ref audio: {}", config.tts_ref_audio);
        match tts::TtsEngine::new(&config.tts_ref_audio) {
            Ok(engine) => {
                tracing::info!("TTS model loaded successfully");
                models.push(ModelInfo {
                    id: "gpt-sovits".to_string(),
                    model_type: "tts".to_string(),
                });
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

    // Load Image model (can be changed dynamically)
    let mut image_engine: Option<image::ImageEngine> = None;
    let mut current_image_model: String = String::new();

    if !config.image_model.is_empty() {
        tracing::info!("Loading image model: {}", config.image_model);
        match image::ImageEngine::new(&config.image_model) {
            Ok(engine) => {
                tracing::info!("Image model loaded successfully");
                current_image_model = config.image_model.clone();
                models.push(ModelInfo {
                    id: current_image_model.clone(),
                    model_type: "image".to_string(),
                });
                image_engine = Some(engine);
            }
            Err(e) => {
                tracing::warn!("Failed to load image model: {}", e);
            }
        }
    }

    // Helper to determine canonical model ID from user input
    fn normalize_image_model(model: &str) -> &'static str {
        let lower = model.to_lowercase();
        if lower.contains("zimage") || lower.contains("z-image") {
            "zimage"
        } else if lower.contains("flux") {
            "flux"
        } else if lower == "zimage" || lower == "z-image-turbo" {
            "zimage"
        } else {
            // Default to zimage for unknown models
            "zimage"
        }
    }

    // Send models info back to main thread
    let _ = models_tx.send(models);

    tracing::info!("Inference thread ready, processing requests...");

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
                    let current_normalized = normalize_image_model(&current_image_model);

                    if normalized != current_normalized || image_engine.is_none() {
                        tracing::info!("Switching image model: {} -> {}", current_image_model, normalized);
                        // Drop old engine to free memory
                        image_engine = None;

                        match image::ImageEngine::new(normalized) {
                            Ok(engine) => {
                                tracing::info!("Image model {} loaded successfully", normalized);
                                current_image_model = normalized.to_string();
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
            InferenceRequest::LoadImageModel { model_id, response_tx } => {
                let normalized = normalize_image_model(&model_id);
                tracing::info!("Loading image model: {} (normalized: {})", model_id, normalized);

                // Drop old engine to free memory
                image_engine = None;

                let result = match image::ImageEngine::new(normalized) {
                    Ok(engine) => {
                        tracing::info!("Image model {} loaded successfully", normalized);
                        current_image_model = normalized.to_string();
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

    // Create channel for inference requests
    let (inference_tx, inference_rx) = mpsc::channel::<InferenceRequest>(32);
    let (models_tx, models_rx) = oneshot::channel();

    // Spawn inference thread (owns all models)
    let config_clone = config.clone();
    std::thread::spawn(move || {
        inference_thread(config_clone, inference_rx, models_tx);
    });

    // Wait for models to load
    let models = models_rx.await.context("Failed to receive models info")?;
    tracing::info!("Loaded {} models", models.len());

    let state = AppState {
        inference_tx,
        models,
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
        .push(Router::with_path("v1/models/load").post(load_model))
        // Chat completions
        .push(Router::with_path("v1/chat/completions").post(chat_completions))
        // Audio endpoints
        .push(Router::with_path("v1/audio/transcriptions").post(audio_transcriptions))
        .push(Router::with_path("v1/audio/speech").post(audio_speech))
        // Image generation
        .push(Router::with_path("v1/images/generations").post(images_generations));

    let listen_addr = format!("0.0.0.0:{}", config.port);
    let acceptor = TcpListener::new(&listen_addr).bind().await;

    tracing::info!("HTTP server listening on http://{}", listen_addr);
    tracing::info!("Endpoints:");
    tracing::info!("  GET  /health");
    tracing::info!("  GET  /v1/models");
    tracing::info!("  POST /v1/models/load         - Load model dynamically");
    tracing::info!("  POST /v1/chat/completions");
    tracing::info!("  POST /v1/audio/transcriptions");
    tracing::info!("  POST /v1/audio/speech");
    tracing::info!("  POST /v1/images/generations  - Supports 'model': 'zimage' or 'flux'");

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

/// GET /v1/models - List available models
#[handler]
async fn list_models(depot: &mut Depot, res: &mut Response) -> Result<(), StatusError> {
    let state = depot
        .obtain::<AppState>()
        .map_err(|_| StatusError::internal_server_error())?;

    let now = chrono::Utc::now().timestamp();
    let data: Vec<_> = state.models.iter().map(|m| {
        serde_json::json!({
            "id": m.id,
            "object": "model",
            "created": now,
            "owned_by": "local",
            "type": m.model_type
        })
    }).collect();

    res.render(Json(serde_json::json!({
        "object": "list",
        "data": data
    })));
    Ok(())
}

/// POST /v1/models/load - Load a model dynamically
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
        #[serde(default)]
        model_type: Option<String>,
    }

    let request: LoadModelRequest = req.parse_json().await.map_err(|e| {
        tracing::error!("Failed to parse request: {}", e);
        StatusError::bad_request()
    })?;

    let model_type = request.model_type.as_deref().unwrap_or("image");

    match model_type {
        "image" => {
            let (response_tx, response_rx) = oneshot::channel();
            state.inference_tx.send(InferenceRequest::LoadImageModel {
                model_id: request.model.clone(),
                response_tx,
            }).await.map_err(|_| StatusError::internal_server_error())?;

            let result = response_rx.await
                .map_err(|_| StatusError::internal_server_error())?;

            match result {
                Ok(loaded_model) => {
                    res.render(Json(serde_json::json!({
                        "status": "success",
                        "model": loaded_model,
                        "model_type": "image"
                    })));
                }
                Err(e) => {
                    res.render(Json(serde_json::json!({
                        "status": "error",
                        "error": e.to_string()
                    })));
                }
            }
        }
        _ => {
            res.render(Json(serde_json::json!({
                "status": "error",
                "error": format!("Dynamic loading not supported for model_type: {}", model_type)
            })));
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

    // Wait for response
    let response = response_rx.await
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

    let response = response_rx.await
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

    let audio_data = response_rx.await
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

    let response = response_rx.await
        .map_err(|_| StatusError::internal_server_error())?
        .map_err(|e| {
            tracing::error!("Image generation error: {}", e);
            StatusError::internal_server_error()
        })?;

    res.render(Json(response));
    Ok(())
}
