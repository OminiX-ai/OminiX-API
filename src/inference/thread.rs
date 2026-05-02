use tokio::sync::{mpsc, oneshot};

use crate::config::Config;
use crate::engines::{asr, image, llm, mflux, pymlx_cosmos, pymlx_flux, pymlx_image_edit, pymlx_wan22, tts, video, vlm};
use crate::inference::tts_pool::{Qwen3TtsEngines, TtsPoolConfig};

use super::{InferenceRequest, ModelStatus};

/// Helper to load a model into a slot, freeing the old one first.
fn load_model_slot<E, F>(
    slot: &mut Option<E>,
    name: &mut Option<String>,
    model_id: &str,
    loader: F,
) -> eyre::Result<String>
where
    F: FnOnce(&str) -> eyre::Result<E>,
{
    // Drop old engine to free memory, then clear MLX cache
    *slot = None;
    *name = None;
    unsafe { mlx_sys::mlx_clear_cache(); }

    let engine = loader(model_id)?;
    tracing::info!("Model loaded successfully: {}", model_id);
    *name = Some(model_id.to_string());
    *slot = Some(engine);
    Ok(model_id.to_string())
}

fn normalize_image_model(model: &str) -> &'static str {
    image::ImageModelType::from_model_id(model).normalized_name()
}

/// Single inference thread that owns ALL models (models are not Send/Sync).
///
/// All GPU work — LLM, ASR, TTS, image, VLM — is serialized through this
/// thread's queue. This prevents Metal command buffer conflicts that crash
/// the process when MLX inference runs concurrently on separate threads.
pub fn inference_thread(
    config: Config,
    tts_pool_config: TtsPoolConfig,
    mut rx: mpsc::Receiver<InferenceRequest>,
    ready_tx: oneshot::Sender<()>,
) {
    // Model slots — ALL models owned by this thread
    let mut llm_engine: Option<llm::LlmEngine> = None;
    let mut asr_engine: Option<asr::AsrEngine> = None;
    let mut tts_engine: Option<tts::TtsEngine> = None;
    let mut image_engine: Option<image::ImageEngine> = None;
    let mut mflux_engine: Option<mflux::MfluxEngine> = None;
    let mut video_engine: Option<video::VideoEngine> = None;
    let mut vlm_engine: Option<vlm::VlmEngine> = None;
    let mut qwen3_tts = Qwen3TtsEngines::new(tts_pool_config.eager_load);

    // Python MLX subprocess engines (lazy-initialized)
    let mut pymlx_image_edit_engine: Option<pymlx_image_edit::PymlxImageEditEngine> = None;
    let mut pymlx_cosmos_engine: Option<pymlx_cosmos::PymlxCosmosEngine> = None;
    let mut pymlx_flux_engine: Option<pymlx_flux::PymlxFluxEngine> = None;
    let mut pymlx_wan22_engine: Option<pymlx_wan22::PymlxWan22Engine> = None;

    // Name tracking
    let mut current_llm_model: Option<String> = None;
    let mut current_asr_model: Option<String> = None;
    let mut current_tts_model: Option<String> = None;
    let mut current_image_model: Option<String> = None;
    let mut current_video_model: Option<String> = None;
    let mut current_vlm_model: Option<String> = None;

    // Startup loading
    if !config.llm_model.is_empty() {
        tracing::info!("Loading LLM model: {}", config.llm_model);
        if let Err(e) = load_model_slot(&mut llm_engine, &mut current_llm_model, &config.llm_model, llm::LlmEngine::new) {
            tracing::warn!("Failed to load LLM model: {}", e);
        }
    }
    if !config.asr_model_dir.is_empty() {
        tracing::info!("Loading ASR model from: {}", config.asr_model_dir);
        match load_model_slot(&mut asr_engine, &mut current_asr_model, &config.asr_model_dir, asr::AsrEngine::new) {
            Ok(_) => {
                // Store the detected backend name instead of the raw path
                if let Some(ref engine) = asr_engine {
                    current_asr_model = Some(engine.backend_name().to_string());
                }
            }
            Err(e) => tracing::warn!("Failed to load ASR model: {}", e),
        }
    }
    if !config.tts_ref_audio.is_empty() {
        tracing::info!("Loading TTS model with ref audio: {}", config.tts_ref_audio);
        if let Err(e) = load_model_slot(&mut tts_engine, &mut current_tts_model, &config.tts_ref_audio, tts::TtsEngine::new) {
            tracing::warn!("Failed to load TTS model: {}", e);
        }
    }
    if !config.image_model.is_empty() {
        tracing::info!("Loading image model: {}", config.image_model);
        match load_model_slot(&mut image_engine, &mut current_image_model, &config.image_model, image::ImageEngine::new) {
            Ok(_) => {
                let normalized = normalize_image_model(&config.image_model);
                current_image_model = Some(normalized.to_string());
            }
            Err(e) => tracing::warn!("Failed to load image model: {}", e),
        }
    }
    if !config.vlm_model.is_empty() {
        tracing::info!("Loading VLM model: {}", config.vlm_model);
        if let Err(e) = load_model_slot(&mut vlm_engine, &mut current_vlm_model, &config.vlm_model, vlm::VlmEngine::new) {
            tracing::warn!("Failed to load VLM model: {}", e);
        }
    }
    // Qwen3-TTS engines are eager-loaded above via Qwen3TtsEngines::new().

    // Signal that models are loaded
    let _ = ready_tx.send(());

    tracing::info!("Inference thread ready, processing requests...");
    tracing::info!("Dynamic model loading enabled - use POST /v1/models/load to switch models");

    // Process requests
    while let Some(request) = rx.blocking_recv() {
        match request {
            InferenceRequest::Chat { request, response_tx } => {
                let result = if let Some(ref mut engine) = llm_engine {
                    engine.generate(&request)
                } else {
                    Err(eyre::eyre!("LLM model not loaded"))
                };
                let _ = response_tx.send(result);
            }
            InferenceRequest::Transcribe { request, expected_backend, response_tx } => {
                // Validate backend if a model-specific endpoint was used
                if let Some(ref expected) = expected_backend {
                    if let Some(ref engine) = asr_engine {
                        let actual = engine.backend_name();
                        if actual != expected.as_str() {
                            let _ = response_tx.send(Err(eyre::eyre!(
                                "Expected {} ASR but {} is loaded. Use POST /v1/models/load with model_type=asr to switch.",
                                expected, actual
                            )));
                            continue;
                        }
                    }
                }
                let result = if let Some(ref mut engine) = asr_engine {
                    engine.transcribe(&request)
                } else {
                    Err(eyre::eyre!("ASR model not loaded. Use POST /v1/models/load with model_type=asr"))
                };
                let _ = response_tx.send(result);
            }
            // Speech via inference thread: only for GPT-SoVITS (tts_engine).
            // Qwen3-TTS Speech requests go to the TTS pool instead.
            InferenceRequest::Speech { request, response_tx } => {
                let result = if let Some(ref mut engine) = tts_engine {
                    engine.synthesize(&request)
                } else {
                    Err(eyre::eyre!(
                        "GPT-SoVITS model not loaded. Use POST /v1/models/load with model_type=tts, \
                         or use /v1/audio/tts/qwen3 for Qwen3-TTS."
                    ))
                };
                let _ = response_tx.send(result);
            }
            InferenceRequest::SpeechStream { request, chunk_tx } => {
                // GPT-SoVITS doesn't support streaming — synthesize full then send
                if let Some(ref mut engine) = tts_engine {
                    match engine.synthesize(&request) {
                        Ok(wav_data) => {
                            let _ = chunk_tx.blocking_send(super::AudioChunk::Pcm(wav_data));
                            let _ = chunk_tx.blocking_send(super::AudioChunk::Done {
                                total_samples: 0,
                                duration_secs: 0.0,
                            });
                        }
                        Err(e) => {
                            let _ = chunk_tx.blocking_send(super::AudioChunk::Error(e.to_string()));
                        }
                    }
                } else {
                    let _ = chunk_tx.blocking_send(super::AudioChunk::Error(
                        "GPT-SoVITS model not loaded".to_string(),
                    ));
                }
            }
            InferenceRequest::SpeechClone { response_tx, .. } => {
                let _ = response_tx.send(Err(eyre::eyre!(
                    "Voice cloning is only available via Qwen3-TTS. Use /v1/audio/tts/clone"
                )));
            }
            InferenceRequest::Image { request, response_tx } => {
                let requested_model = request.model.as_deref().unwrap_or("");
                let model_type = image::ImageModelType::from_model_id(requested_model);

                let result = match model_type {
                    image::ImageModelType::QwenImage => {
                        // Qwen-Image → use mflux subprocess
                        if mflux_engine.is_none() {
                            match mflux::MfluxEngine::new(requested_model) {
                                Ok(engine) => { mflux_engine = Some(engine); }
                                Err(e) => { tracing::error!("Failed to init mflux: {}", e); }
                            }
                        }
                        if let Some(ref engine) = mflux_engine {
                            engine.generate(&request)
                        } else {
                            Err(eyre::eyre!("mflux engine not available. Install mflux: pip install mflux"))
                        }
                    }
                    image::ImageModelType::QwenImageEdit => {
                        // Qwen-Image-Edit → Python MLX subprocess
                        if pymlx_image_edit_engine.is_none() {
                            match pymlx_image_edit::PymlxImageEditEngine::new(requested_model) {
                                Ok(engine) => { pymlx_image_edit_engine = Some(engine); }
                                Err(e) => { tracing::error!("Failed to init image edit engine: {}", e); }
                            }
                        }
                        if let Some(ref engine) = pymlx_image_edit_engine {
                            engine.generate(&request)
                        } else {
                            Err(eyre::eyre!("Image edit engine not available. Check Python environment."))
                        }
                    }
                    image::ImageModelType::CosmosT2I => {
                        // Cosmos Predict2 T2I → Python MLX subprocess
                        if pymlx_cosmos_engine.is_none() {
                            match pymlx_cosmos::PymlxCosmosEngine::new(requested_model) {
                                Ok(engine) => { pymlx_cosmos_engine = Some(engine); }
                                Err(e) => { tracing::error!("Failed to init Cosmos engine: {}", e); }
                            }
                        }
                        if let Some(ref engine) = pymlx_cosmos_engine {
                            engine.generate_image(&request)
                        } else {
                            Err(eyre::eyre!("Cosmos engine not available. Check Python environment."))
                        }
                    }
                    image::ImageModelType::FluxKleinGguf => {
                        // FLUX.2-klein GGUF → Python MLX subprocess
                        if pymlx_flux_engine.is_none() {
                            match pymlx_flux::PymlxFluxEngine::new(requested_model) {
                                Ok(engine) => { pymlx_flux_engine = Some(engine); }
                                Err(e) => { tracing::error!("Failed to init FLUX GGUF engine: {}", e); }
                            }
                        }
                        if let Some(ref engine) = pymlx_flux_engine {
                            engine.generate(&request)
                        } else {
                            Err(eyre::eyre!("FLUX GGUF engine not available. Check Python environment."))
                        }
                    }
                    _ => {
                        // FLUX / Z-Image → existing Rust engine
                        if !requested_model.is_empty() {
                            let normalized = normalize_image_model(requested_model);
                            let current_normalized = current_image_model.as_deref().unwrap_or("");

                            if normalized != current_normalized || image_engine.is_none() {
                                tracing::info!("Switching image model: {:?} -> {}", current_image_model, requested_model);
                                image_engine = None;
                                unsafe { mlx_sys::mlx_clear_cache(); }

                                match image::ImageEngine::new(requested_model) {
                                    Ok(engine) => {
                                        tracing::info!("Image model {} loaded successfully", requested_model);
                                        current_image_model = Some(normalized.to_string());
                                        image_engine = Some(engine);
                                    }
                                    Err(e) => {
                                        tracing::error!("Failed to load image model {}: {}", requested_model, e);
                                    }
                                }
                            }
                        }

                        if let Some(ref mut engine) = image_engine {
                            engine.generate(&request)
                        } else {
                            Err(eyre::eyre!("Image model not loaded. Specify 'model': 'zimage' or 'flux' in your request."))
                        }
                    }
                };
                let _ = response_tx.send(result);
            }

            InferenceRequest::Video { request, response_tx } => {
                let requested_model = request.model.as_deref().unwrap_or("");
                let lower = requested_model.to_lowercase();

                let result = if lower.contains("wan22-gguf") || lower.contains("wan2.2-gguf") || lower.contains("wan22_gguf") {
                    // Wan 2.2 GGUF → Python MLX subprocess
                    if pymlx_wan22_engine.is_none() {
                        match pymlx_wan22::PymlxWan22Engine::new(requested_model) {
                            Ok(engine) => { pymlx_wan22_engine = Some(engine); }
                            Err(e) => { tracing::error!("Failed to init Wan2.2 GGUF engine: {}", e); }
                        }
                    }
                    if let Some(ref engine) = pymlx_wan22_engine {
                        engine.generate(&request)
                    } else {
                        Err(eyre::eyre!("Wan2.2 GGUF engine not available. Check Python environment."))
                    }
                } else if lower.contains("cosmos") && lower.contains("v2w") {
                    // Cosmos V2W → Python MLX subprocess
                    if pymlx_cosmos_engine.is_none() {
                        match pymlx_cosmos::PymlxCosmosEngine::new(requested_model) {
                            Ok(engine) => { pymlx_cosmos_engine = Some(engine); }
                            Err(e) => { tracing::error!("Failed to init Cosmos engine: {}", e); }
                        }
                    }
                    if let Some(ref engine) = pymlx_cosmos_engine {
                        engine.generate_video(&request)
                    } else {
                        Err(eyre::eyre!("Cosmos V2W engine not available. Check Python environment."))
                    }
                } else if let Some(ref mut engine) = video_engine {
                    engine.generate(&request)
                } else {
                    Err(eyre::eyre!("Video model not loaded. Use POST /v1/models/load with model_type=video"))
                };
                let _ = response_tx.send(result);
            }

            InferenceRequest::VlmCompletion { request, response_tx } => {
                let result = if let Some(ref mut engine) = vlm_engine {
                    engine.describe(&request)
                } else {
                    Err(eyre::eyre!("VLM model not loaded"))
                };
                let _ = response_tx.send(result);
            }

            // === Dynamic Model Loading ===

            InferenceRequest::LoadLlmModel { model_id, response_tx } => {
                tracing::info!("Loading LLM model: {}", model_id);
                let result = load_model_slot(&mut llm_engine, &mut current_llm_model, &model_id, llm::LlmEngine::new);
                let _ = response_tx.send(result);
            }
            InferenceRequest::LoadAsrModel { model_dir, response_tx } => {
                tracing::info!("Loading ASR model from: {}", model_dir);
                let result = load_model_slot(&mut asr_engine, &mut current_asr_model, &model_dir, asr::AsrEngine::new);
                // Store backend name instead of raw path for better observability
                if result.is_ok() {
                    if let Some(ref engine) = asr_engine {
                        current_asr_model = Some(engine.backend_name().to_string());
                    }
                }
                let _ = response_tx.send(result);
            }
            InferenceRequest::LoadTtsModel { ref_audio, response_tx } => {
                tracing::info!("Loading TTS model with ref audio: {}", ref_audio);
                let result = load_model_slot(&mut tts_engine, &mut current_tts_model, &ref_audio, tts::TtsEngine::new);
                let _ = response_tx.send(result);
            }
            InferenceRequest::LoadVlmModel { model_id, response_tx } => {
                tracing::info!("Loading VLM model: {}", model_id);
                let result = load_model_slot(&mut vlm_engine, &mut current_vlm_model, &model_id, vlm::VlmEngine::new);
                let _ = response_tx.send(result);
            }
            InferenceRequest::LoadQwen3TtsModel { model_dir, response_tx } => {
                tracing::info!("Loading Qwen3-TTS model: {}", model_dir);
                let result = qwen3_tts.load_model(&model_dir);
                let _ = response_tx.send(result);
            }
            InferenceRequest::LoadVideoModel { model_id, response_tx } => {
                tracing::info!("Loading video model: {}", model_id);
                let result = load_model_slot(
                    &mut video_engine,
                    &mut current_video_model,
                    &model_id,
                    video::VideoEngine::new,
                );
                let _ = response_tx.send(result);
            }
            InferenceRequest::LoadImageModel { model_id, response_tx } => {
                let model_type = image::ImageModelType::from_model_id(&model_id);
                let normalized = normalize_image_model(&model_id);
                tracing::info!("Loading image model: {} (normalized: {}, type: {:?})", model_id, normalized, model_type);

                let result = match model_type {
                    image::ImageModelType::QwenImage => {
                        match mflux::MfluxEngine::new(&model_id) {
                            Ok(engine) => {
                                mflux_engine = Some(engine);
                                current_image_model = Some(normalized.to_string());
                                Ok(model_id)
                            }
                            Err(e) => {
                                tracing::error!("Failed to init mflux for {}: {}", model_id, e);
                                Err(e)
                            }
                        }
                    }
                    image::ImageModelType::QwenImageEdit => {
                        match pymlx_image_edit::PymlxImageEditEngine::new(&model_id) {
                            Ok(engine) => {
                                pymlx_image_edit_engine = Some(engine);
                                current_image_model = Some(normalized.to_string());
                                Ok(model_id)
                            }
                            Err(e) => {
                                tracing::error!("Failed to init image edit for {}: {}", model_id, e);
                                Err(e)
                            }
                        }
                    }
                    image::ImageModelType::CosmosT2I => {
                        match pymlx_cosmos::PymlxCosmosEngine::new(&model_id) {
                            Ok(engine) => {
                                pymlx_cosmos_engine = Some(engine);
                                current_image_model = Some(normalized.to_string());
                                Ok(model_id)
                            }
                            Err(e) => {
                                tracing::error!("Failed to init Cosmos for {}: {}", model_id, e);
                                Err(e)
                            }
                        }
                    }
                    image::ImageModelType::FluxKleinGguf => {
                        match pymlx_flux::PymlxFluxEngine::new(&model_id) {
                            Ok(engine) => {
                                pymlx_flux_engine = Some(engine);
                                current_image_model = Some(normalized.to_string());
                                Ok(model_id)
                            }
                            Err(e) => {
                                tracing::error!("Failed to init FLUX GGUF for {}: {}", model_id, e);
                                Err(e)
                            }
                        }
                    }
                    _ => {
                        image_engine = None;
                        current_image_model = None;
                        unsafe { mlx_sys::mlx_clear_cache(); }
                        match image::ImageEngine::new(&model_id) {
                            Ok(engine) => {
                                tracing::info!("Image model {} loaded successfully", model_id);
                                current_image_model = Some(normalized.to_string());
                                image_engine = Some(engine);
                                Ok(model_id)
                            }
                            Err(e) => {
                                tracing::error!("Failed to load image model {}: {}", model_id, e);
                                Err(e)
                            }
                        }
                    }
                };
                let _ = response_tx.send(result);
            }

            InferenceRequest::UnloadModel { model_type, response_tx } => {
                tracing::info!("Unloading model type: {}", model_type);
                let result = match model_type.as_str() {
                    "llm" => {
                        llm_engine = None;
                        let prev = current_llm_model.take();
                        Ok(format!("Unloaded LLM model: {:?}", prev))
                    }
                    "asr" => {
                        asr_engine = None;
                        let prev = current_asr_model.take();
                        Ok(format!("Unloaded ASR model: {:?}", prev))
                    }
                    "tts" => {
                        tts_engine = None;
                        let prev = current_tts_model.take();
                        let prev_qwen3 = qwen3_tts.unload();
                        Ok(format!(
                            "Unloaded TTS model: {:?}; Qwen3-TTS: {:?}",
                            prev, prev_qwen3
                        ))
                    }
                    "image" => {
                        image_engine = None;
                        let prev = current_image_model.take();
                        Ok(format!("Unloaded image model: {:?}", prev))
                    }
                    "video" => {
                        video_engine = None;
                        let prev = current_video_model.take();
                        Ok(format!("Unloaded video model: {:?}", prev))
                    }
                    "vlm" => {
                        vlm_engine = None;
                        let prev = current_vlm_model.take();
                        Ok(format!("Unloaded VLM model: {:?}", prev))
                    }
                    "qwen3_tts" => {
                        let prev = qwen3_tts.unload();
                        Ok(format!("Unloaded Qwen3-TTS model: {:?}", prev))
                    }
                    "all" => {
                        llm_engine = None;
                        asr_engine = None;
                        tts_engine = None;
                        let prev_qwen3_tts = qwen3_tts.unload();
                        image_engine = None;
                        video_engine = None;
                        vlm_engine = None;
                        current_llm_model = None;
                        current_asr_model = None;
                        current_tts_model = None;
                        current_image_model = None;
                        current_video_model = None;
                        current_vlm_model = None;
                        Ok(format!("Unloaded all models; Qwen3-TTS: {:?}", prev_qwen3_tts))
                    }
                    _ => Err(eyre::eyre!("Unknown model type: {}. Use: llm, asr, tts, qwen3_tts, image, video, vlm, or all", model_type)),
                };
                // Free MLX GPU memory cache after dropping model weights
                if result.is_ok() {
                    unsafe { mlx_sys::mlx_clear_cache(); }
                    tracing::info!("Cleared MLX cache after unload");
                }
                let _ = response_tx.send(result);
            }

            InferenceRequest::GetModelStatus { response_tx } => {
                let status = ModelStatus {
                    llm: current_llm_model.clone(),
                    asr: current_asr_model.clone(),
                    tts: current_tts_model.clone(),
                    qwen3_tts: qwen3_tts.current_variant_name().map(|s| s.to_string()),
                    image: current_image_model.clone(),
                    video: current_video_model.clone(),
                    vlm: current_vlm_model.clone(),
                    ascend: None, // Populated by handler from AppState
                };
                let _ = response_tx.send(status);
            }

            InferenceRequest::ReloadVoices { response_tx } => {
                tracing::info!("Reloading voice registry");
                if let Some(ref mut engine) = tts_engine {
                    engine.reload_voices();
                }
                let _ = response_tx.send(Ok(()));
            }

            // Qwen3-TTS (serialized through the same queue as ASR/LLM)
            InferenceRequest::Qwen3Tts(tts_request) => {
                qwen3_tts.handle(tts_request);
            }
        }
    }

    tracing::info!("Inference thread shutting down");
}
