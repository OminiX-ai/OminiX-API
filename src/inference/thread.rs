use std::collections::VecDeque;
use std::time::Instant;

use tokio::sync::mpsc::error::TryRecvError;
use tokio::sync::{mpsc, oneshot};

use crate::config::Config;
use crate::engines::{asr, image, llm, tts, vlm};
use crate::inference::tts_pool::{Qwen3TtsEngines, TtsPoolConfig};
use crate::types::{TranscriptionRequest, TranscriptionResponse};

use super::{InferenceRequest, ModelStatus};

/// One transcription waiting to be served, with the caller to reply to.
struct PendingTranscribe {
    request: TranscriptionRequest,
    response_tx: oneshot::Sender<eyre::Result<TranscriptionResponse>>,
}

/// Run a coalesced group of transcriptions and reply to each caller.
///
/// The engine returns one result per request in order, so a failure — a bad
/// audio file, a language that could not share the batch — is delivered only to
/// the caller it belongs to.
fn serve_transcribe_batch(
    asr_engine: &mut Option<asr::AsrEngine>,
    batch: Vec<PendingTranscribe>,
) {
    let Some(engine) = asr_engine.as_mut() else {
        for item in batch {
            let _ = item.response_tx.send(Err(eyre::eyre!(
                "ASR model not loaded. Use POST /v1/models/load with model_type=asr"
            )));
        }
        return;
    };

    if batch.len() > 1 {
        tracing::debug!("ASR batch of {}", batch.len());
    }

    let (requests, senders): (Vec<_>, Vec<_>) =
        batch.into_iter().map(|p| (p.request, p.response_tx)).unzip();
    let results = engine.transcribe_batch(&requests);

    for (tx, result) in senders.into_iter().zip(results) {
        let _ = tx.send(result);
    }
}

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
    let mut vlm_engine: Option<vlm::VlmEngine> = None;
    let mut qwen3_tts = Qwen3TtsEngines::new(tts_pool_config.eager_load);

    // Name tracking
    let mut current_llm_model: Option<String> = None;
    let mut current_asr_model: Option<String> = None;
    let mut current_tts_model: Option<String> = None;
    let mut current_image_model: Option<String> = None;
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
                // Warm up so the first real request doesn't pay MLX graph
                // compilation (~5s for Qwen3-ASR).
                if let Some(ref mut engine) = asr_engine {
                    engine.warmup();
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

    let max_batch = config.asr_max_batch();
    let batch_window = config.asr_batch_window();
    tracing::info!("Host: {}", config.describe_host());
    tracing::info!(
        "ASR batching: mode={} max_batch={} window={}ms",
        config.asr_mode.as_str(),
        max_batch,
        batch_window.as_millis()
    );
    if !config.asr_use_ane() && config.chip.generation() > 0 && config.chip.generation() < 5 {
        tracing::info!(
            "ASR Neural Engine offload not used on {} — GPU-only. \
             The ANE only becomes a useful second unit from M5 onward.",
            config.chip.as_str()
        );
    }

    // Requests pulled off the channel while forming a batch but not part of it.
    // They are served, in arrival order, before the channel is read again.
    let mut deferred: VecDeque<InferenceRequest> = VecDeque::new();

    // Process requests
    loop {
        let request = match deferred.pop_front() {
            Some(r) => r,
            None => match rx.blocking_recv() {
                Some(r) => r,
                None => break,
            },
        };
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

                let mut batch = vec![PendingTranscribe { request, response_tx }];

                // Coalesce. The rule is opportunistic: take what has already
                // arrived, and only wait for more once there is evidence of
                // load. A lone request at low load is therefore never delayed,
                // which is what usually makes batching hurt latency.
                //
                // This needs no tuning to find the right batch size. At a given
                // offered load the queue depth settles at arrival-rate times
                // service-time, so the batch that forms is the one the load
                // actually justifies.
                if max_batch > 1 {
                    let can_batch =
                        asr_engine.as_ref().is_some_and(|e| e.supports_batching());
                    if can_batch {
                        let deadline = Instant::now() + batch_window;
                        loop {
                            if batch.len() >= max_batch {
                                break;
                            }
                            match rx.try_recv() {
                                Ok(InferenceRequest::Transcribe {
                                    request,
                                    expected_backend: exp,
                                    response_tx,
                                }) => {
                                    // A mismatched backend can't join the batch;
                                    // answer it here rather than dropping it.
                                    let mismatch = exp.as_deref().and_then(|want| {
                                        asr_engine.as_ref().map(|e| e.backend_name()).and_then(
                                            |actual| (actual != want).then(|| (want.to_string(), actual)),
                                        )
                                    });
                                    match mismatch {
                                        Some((want, actual)) => {
                                            let _ = response_tx.send(Err(eyre::eyre!(
                                                "Expected {} ASR but {} is loaded. Use POST /v1/models/load with model_type=asr to switch.",
                                                want, actual
                                            )));
                                        }
                                        None => batch.push(PendingTranscribe {
                                            request,
                                            response_tx,
                                        }),
                                    }
                                }
                                Ok(other) => deferred.push_back(other),
                                Err(TryRecvError::Empty) => {
                                    // Nothing queued. Only linger if a batch is
                                    // already forming and there is time left.
                                    if batch.len() < 2
                                        || batch_window.is_zero()
                                        || Instant::now() >= deadline
                                    {
                                        break;
                                    }
                                    std::thread::sleep(std::time::Duration::from_micros(200));
                                }
                                Err(TryRecvError::Disconnected) => break,
                            }
                        }
                    }
                }

                serve_transcribe_batch(&mut asr_engine, batch);
                continue;
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
                        vlm_engine = None;
                        current_llm_model = None;
                        current_asr_model = None;
                        current_tts_model = None;
                        current_image_model = None;
                        current_vlm_model = None;
                        Ok(format!("Unloaded all models; Qwen3-TTS: {:?}", prev_qwen3_tts))
                    }
                    _ => Err(eyre::eyre!("Unknown model type: {}. Use: llm, asr, tts, qwen3_tts, image, vlm, or all", model_type)),
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
            // Periodic keep-warm so the first ASR after idle isn't cold.
            InferenceRequest::KeepWarmAsr => {
                if let Some(ref mut engine) = asr_engine {
                    engine.warmup();
                }
            }
        }
    }

    tracing::info!("Inference thread shutting down");
}
