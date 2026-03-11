use tokio::sync::{mpsc, oneshot};

use crate::config::Config;
use crate::engines::{asr, image, llm, qwen3_tts, tts, vlm};

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
    // Drop old engine to free memory
    *slot = None;
    *name = None;

    let engine = loader(model_id)?;
    tracing::info!("Model loaded successfully: {}", model_id);
    *name = Some(model_id.to_string());
    *slot = Some(engine);
    Ok(model_id.to_string())
}

fn normalize_image_model(model: &str) -> &'static str {
    image::ImageModelType::from_model_id(model).normalized_name()
}

/// Search standard model directories for a TTS model variant.
/// Looks in `~/.OminiX/models/` and `~/.ominix/models/` for dirs matching the pattern.
fn find_tts_model(variant: &str) -> Option<String> {
    let home = dirs::home_dir()?;
    let search_dirs = [
        home.join(".OminiX").join("models"),
        home.join(".ominix").join("models"),
    ];
    for dir in &search_dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy().to_lowercase();
                if entry.path().is_dir() && name_str.contains("tts") && name_str.contains(variant) {
                    return Some(entry.path().to_string_lossy().to_string());
                }
            }
        }
    }
    None
}

/// Inference thread that owns all models (models are not Send/Sync)
pub fn inference_thread(
    config: Config,
    mut rx: mpsc::Receiver<InferenceRequest>,
    ready_tx: oneshot::Sender<()>,
) {
    // Model slots
    let mut llm_engine: Option<llm::LlmEngine> = None;
    let mut asr_engine: Option<asr::AsrEngine> = None;
    let mut tts_engine: Option<tts::TtsEngine> = None;
    let mut image_engine: Option<image::ImageEngine> = None;
    let mut vlm_engine: Option<vlm::VlmEngine> = None;
    let mut qwen3_tts_engine: Option<qwen3_tts::Qwen3TtsEngine> = None;

    // Name tracking
    let mut current_llm_model: Option<String> = None;
    let mut current_asr_model: Option<String> = None;
    let mut current_tts_model: Option<String> = None;
    let mut current_image_model: Option<String> = None;
    let mut current_vlm_model: Option<String> = None;
    let mut current_qwen3_tts_model: Option<String> = None;

    // Startup loading
    if !config.llm_model.is_empty() {
        tracing::info!("Loading LLM model: {}", config.llm_model);
        if let Err(e) = load_model_slot(&mut llm_engine, &mut current_llm_model, &config.llm_model, llm::LlmEngine::new) {
            tracing::warn!("Failed to load LLM model: {}", e);
        }
    }
    if !config.asr_model_dir.is_empty() {
        tracing::info!("Loading ASR model from: {}", config.asr_model_dir);
        if let Err(e) = load_model_slot(&mut asr_engine, &mut current_asr_model, &config.asr_model_dir, asr::AsrEngine::new) {
            tracing::warn!("Failed to load ASR model: {}", e);
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
    if !config.qwen3_tts_model_dir.is_empty() {
        tracing::info!("Loading Qwen3-TTS model: {}", config.qwen3_tts_model_dir);
        if let Err(e) = load_model_slot(&mut qwen3_tts_engine, &mut current_qwen3_tts_model, &config.qwen3_tts_model_dir, qwen3_tts::Qwen3TtsEngine::new) {
            tracing::warn!("Failed to load Qwen3-TTS model: {}", e);
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
                let result = if let Some(ref mut engine) = llm_engine {
                    engine.generate(&request)
                } else {
                    Err(eyre::eyre!("LLM model not loaded"))
                };
                let _ = response_tx.send(result);
            }
            InferenceRequest::Transcribe { request, response_tx } => {
                let result = if let Some(ref mut engine) = asr_engine {
                    engine.transcribe(&request)
                } else {
                    Err(eyre::eyre!("ASR model not loaded"))
                };
                let _ = response_tx.send(result);
            }
            InferenceRequest::Speech { request, response_tx } => {
                let needs_cloning = request.reference_audio.is_some();

                // Auto-switch TTS model variant if needed
                if needs_cloning {
                    // Voice cloning needs Base model
                    if qwen3_tts_engine.as_ref().is_some_and(|e| !e.supports_voice_cloning())
                        || qwen3_tts_engine.is_none()
                    {
                        if let Some(base_path) = find_tts_model("base") {
                            tracing::info!("Auto-switching to Base TTS model: {base_path}");
                            if let Err(e) = load_model_slot(
                                &mut qwen3_tts_engine,
                                &mut current_qwen3_tts_model,
                                &base_path,
                                qwen3_tts::Qwen3TtsEngine::new,
                            ) {
                                tracing::warn!("Failed to auto-load Base TTS: {e}");
                            }
                        } else {
                            tracing::warn!("Base TTS model not found on disk — download qwen3-tts-base");
                        }
                    }
                } else {
                    // Preset speakers need CustomVoice model
                    if qwen3_tts_engine.as_ref().is_some_and(|e| !e.supports_preset_speakers()) {
                        if let Some(cv_path) = find_tts_model("customvoice") {
                            tracing::info!("Auto-switching to CustomVoice TTS model: {cv_path}");
                            if let Err(e) = load_model_slot(
                                &mut qwen3_tts_engine,
                                &mut current_qwen3_tts_model,
                                &cv_path,
                                qwen3_tts::Qwen3TtsEngine::new,
                            ) {
                                tracing::warn!("Failed to auto-load CustomVoice TTS: {e}");
                            }
                        } else {
                            tracing::warn!("CustomVoice TTS model not found on disk — download qwen3-tts");
                        }
                    }
                }

                let result = if needs_cloning {
                    if let Some(ref mut engine) = qwen3_tts_engine {
                        let ref_audio = request.reference_audio.as_deref().unwrap();
                        let language = request.language.as_deref().unwrap_or("chinese");
                        engine.synthesize_clone(&request.input, ref_audio, language, request.speed)
                    } else {
                        Err(eyre::eyre!(
                            "Base TTS model not available. Download it first: \
                             download_model(model_id='qwen3-tts-base')"
                        ))
                    }
                } else if let Some(ref mut engine) = qwen3_tts_engine {
                    engine.synthesize(&request)
                } else if let Some(ref mut engine) = tts_engine {
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
                        tracing::info!("Switching image model: {:?} -> {}", current_image_model, requested_model);
                        image_engine = None;

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

                let result = if let Some(ref mut engine) = image_engine {
                    engine.generate(&request)
                } else {
                    Err(eyre::eyre!("Image model not loaded. Specify 'model': 'zimage' or 'flux' in your request."))
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
                let result = load_model_slot(&mut qwen3_tts_engine, &mut current_qwen3_tts_model, &model_dir, qwen3_tts::Qwen3TtsEngine::new);
                let _ = response_tx.send(result);
            }
            InferenceRequest::LoadImageModel { model_id, response_tx } => {
                let normalized = normalize_image_model(&model_id);
                tracing::info!("Loading image model: {} (normalized: {})", model_id, normalized);
                image_engine = None;
                current_image_model = None;
                let result = match image::ImageEngine::new(&model_id) {
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
                        Ok(format!("Unloaded TTS model: {:?}", prev))
                    }
                    "image" => {
                        image_engine = None;
                        let prev = current_image_model.take();
                        Ok(format!("Unloaded image model: {:?}", prev))
                    }
                    "vlm" => {
                        vlm_engine = None;
                        let prev = current_vlm_model.take();
                        Ok(format!("Unloaded VLM model: {:?}", prev))
                    }
                    "qwen3_tts" => {
                        qwen3_tts_engine = None;
                        let prev = current_qwen3_tts_model.take();
                        Ok(format!("Unloaded Qwen3-TTS model: {:?}", prev))
                    }
                    "all" => {
                        llm_engine = None;
                        asr_engine = None;
                        tts_engine = None;
                        qwen3_tts_engine = None;
                        image_engine = None;
                        vlm_engine = None;
                        current_llm_model = None;
                        current_asr_model = None;
                        current_tts_model = None;
                        current_qwen3_tts_model = None;
                        current_image_model = None;
                        current_vlm_model = None;
                        Ok("Unloaded all models".to_string())
                    }
                    _ => Err(eyre::eyre!("Unknown model type: {}. Use: llm, asr, tts, qwen3_tts, image, vlm, or all", model_type)),
                };
                let _ = response_tx.send(result);
            }

            InferenceRequest::GetModelStatus { response_tx } => {
                let status = ModelStatus {
                    llm: current_llm_model.clone(),
                    asr: current_asr_model.clone(),
                    tts: current_tts_model.clone(),
                    qwen3_tts: current_qwen3_tts_model.clone(),
                    image: current_image_model.clone(),
                    vlm: current_vlm_model.clone(),
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
        }
    }

    tracing::info!("Inference thread shutting down");
}
