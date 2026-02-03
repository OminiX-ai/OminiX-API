//! Voice cloning training engine
//!
//! Runs on a dedicated thread (MLX models are not Send/Sync).
//! Pipeline: audio slicing → denoising → ASR → feature extraction → VITS training → voice registration

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use eyre::{Context, Result};
use tokio::sync::{broadcast, mpsc, oneshot};

use gpt_sovits_mlx::{
    AudioSlicer, SlicerConfig, Denoiser, DenoiseConfig,
    preprocess_text, Language,
};
use gpt_sovits_mlx::training::{
    VITSTrainer, VITSTrainingConfig, VITSDataset,
};
use gpt_sovits_mlx::models::hubert::load_hubert_model;
use gpt_sovits_mlx::audio::{
    load_wav, save_wav, resample,
};
use mlx_rs::module::Module;

use crate::types::*;

/// Shared cancel flag: holds (task_id, flag) while training is active.
/// The HTTP handler can set the flag directly without going through the channel.
pub type CancelFlag = Arc<std::sync::Mutex<Option<(String, Arc<AtomicBool>)>>>;

/// Cancel a training task by task_id. Returns Ok if the flag was set, Err if
/// no matching task is currently running.
pub fn cancel_training_task(cancel_state: &CancelFlag, task_id: &str) -> Result<()> {
    let guard = cancel_state.lock().unwrap();
    if let Some((ref active_id, ref flag)) = *guard {
        if active_id == task_id {
            flag.store(true, Ordering::Relaxed);
            return Ok(());
        }
    }
    Err(eyre::eyre!("No active training task with id: {}", task_id))
}

// ============================================================================
// Constants
// ============================================================================

/// Base path for GPT-SoVITS models
const MODELS_BASE: &str = "~/.dora/models/primespeech/gpt-sovits-mlx";
/// VITS pretrained weights (v2)
const VITS_PRETRAINED: &str = "vits_pretrained_v2.safetensors";
/// HuBERT model for SSL feature extraction
const HUBERT_WEIGHTS: &str = "hubert.safetensors";
/// Default voices config path
const VOICES_CONFIG_PATH: &str = "~/.dora/models/primespeech/voices.json";
/// Training data sample rate
const TRAINING_SAMPLE_RATE: u32 = 32000;
/// HuBERT input sample rate
const HUBERT_SAMPLE_RATE: u32 = 16000;
/// STFT hop length (must match VITS config)
const HOP_LENGTH: i32 = 640;

// ============================================================================
// Channel Types
// ============================================================================

/// Messages sent to the training thread
pub enum TrainingRequest {
    /// Start a new voice cloning training job
    StartTraining {
        task_id: String,
        voice_name: String,
        audio_path: PathBuf,
        transcript: String,
        quality: TrainingQuality,
        language: String,
        denoise: bool,
        response_tx: oneshot::Sender<Result<()>>,
    },
    /// Get current training task status
    GetStatus {
        task_id: String,
        response_tx: oneshot::Sender<Option<TrainingTaskStatus>>,
    },
}

// ============================================================================
// Training Thread
// ============================================================================

/// Training thread entry point — processes jobs sequentially
pub fn training_thread(
    mut rx: mpsc::Receiver<TrainingRequest>,
    progress_tx: broadcast::Sender<TrainingProgressEvent>,
    inference_tx: mpsc::Sender<crate::InferenceRequest>,
    cancel_state: CancelFlag,
) {
    tracing::info!("Training thread started");

    // Track current task state
    let mut current_task: Option<TrainingTaskStatus> = None;

    while let Some(request) = rx.blocking_recv() {
        match request {
            TrainingRequest::StartTraining {
                task_id,
                voice_name,
                audio_path,
                transcript,
                quality,
                language,
                denoise,
                response_tx,
            } => {
                // Check if already training
                if let Some(ref task) = current_task {
                    if task.status != TrainingStage::Complete && task.status != TrainingStage::Failed {
                        let _ = response_tx.send(Err(eyre::eyre!(
                            "Training already in progress: {} ({})",
                            task.voice_name, task.task_id
                        )));
                        continue;
                    }
                }

                // Set up task status
                current_task = Some(TrainingTaskStatus {
                    task_id: task_id.clone(),
                    voice_name: voice_name.clone(),
                    status: TrainingStage::Queued,
                    progress: 0.0,
                    created_at: chrono::Utc::now().to_rfc3339(),
                    completed_at: None,
                    error: None,
                });

                // Acknowledge receipt
                let _ = response_tx.send(Ok(()));

                // Create cancel flag for this task and store in shared state
                let cancel_flag = Arc::new(AtomicBool::new(false));
                {
                    let mut guard = cancel_state.lock().unwrap();
                    *guard = Some((task_id.clone(), cancel_flag.clone()));
                }

                // Run the full pipeline
                let result = run_training_pipeline(
                    &task_id,
                    &voice_name,
                    &audio_path,
                    &transcript,
                    &quality,
                    &language,
                    denoise,
                    &progress_tx,
                    &inference_tx,
                    &cancel_flag,
                );

                // Clear cancel flag
                {
                    let mut guard = cancel_state.lock().unwrap();
                    *guard = None;
                }

                match result {
                    Ok(()) => {
                        send_progress(
                            &progress_tx, &task_id, TrainingStage::Complete,
                            1.0, 1.0, "Voice cloning complete", true, None, None,
                        );
                        if let Some(ref mut task) = current_task {
                            task.status = TrainingStage::Complete;
                            task.progress = 1.0;
                            task.completed_at = Some(chrono::Utc::now().to_rfc3339());
                        }
                    }
                    Err(e) => {
                        let err_msg = format!("{:#}", e);
                        let is_cancelled = cancel_flag.load(Ordering::Relaxed);
                        if is_cancelled {
                            tracing::info!("Training cancelled by user: {}", task_id);
                            send_progress(
                                &progress_tx, &task_id, TrainingStage::Failed,
                                0.0, 0.0, "Training cancelled by user",
                                true, Some("Cancelled by user".to_string()), None,
                            );
                            if let Some(ref mut task) = current_task {
                                task.status = TrainingStage::Failed;
                                task.error = Some("Cancelled by user".to_string());
                                task.completed_at = Some(chrono::Utc::now().to_rfc3339());
                            }
                        } else {
                            tracing::error!("Training failed: {}", err_msg);
                            send_progress(
                                &progress_tx, &task_id, TrainingStage::Failed,
                                0.0, 0.0, &format!("Training failed: {}", err_msg),
                                true, Some(err_msg.clone()), None,
                            );
                            if let Some(ref mut task) = current_task {
                                task.status = TrainingStage::Failed;
                                task.error = Some(err_msg);
                                task.completed_at = Some(chrono::Utc::now().to_rfc3339());
                            }
                        }
                    }
                }
            }
            TrainingRequest::GetStatus { task_id, response_tx } => {
                let status = current_task
                    .as_ref()
                    .filter(|t| t.task_id == task_id)
                    .cloned();
                let _ = response_tx.send(status);
            }
        }
    }

    tracing::info!("Training thread shutting down");
}

// ============================================================================
// Training Pipeline
// ============================================================================

/// Check if cancellation was requested, returning Err if so.
fn check_cancelled(flag: &AtomicBool) -> Result<()> {
    if flag.load(Ordering::Relaxed) {
        Err(eyre::eyre!("Cancelled by user"))
    } else {
        Ok(())
    }
}

fn run_training_pipeline(
    task_id: &str,
    voice_name: &str,
    audio_path: &Path,
    transcript: &str,
    quality: &TrainingQuality,
    _language: &str,
    denoise: bool,
    progress_tx: &broadcast::Sender<TrainingProgressEvent>,
    inference_tx: &mpsc::Sender<crate::InferenceRequest>,
    cancel_flag: &AtomicBool,
) -> Result<()> {
    let base_dir = training_base_dir();
    let work_dir = base_dir.join(task_id);
    let sliced_dir = work_dir.join("sliced");
    let dataset_dir = work_dir.join("vits_data");
    std::fs::create_dir_all(&sliced_dir)?;
    std::fs::create_dir_all(&dataset_dir)?;

    // === Stage 1: Audio Slicing ===
    send_progress(
        progress_tx, task_id, TrainingStage::AudioSlicing,
        0.0, 0.0, "Slicing reference audio...", false, None, None,
    );
    let chunk_paths = stage_audio_slicing(audio_path, &sliced_dir)?;
    tracing::info!("Audio slicing: {} chunks produced", chunk_paths.len());
    if chunk_paths.is_empty() {
        return Err(eyre::eyre!("No audio chunks produced from slicing. Audio may be too short."));
    }
    check_cancelled(cancel_flag)?;

    // === Stage 2: Denoising (optional) ===
    if denoise {
        send_progress(
            progress_tx, task_id, TrainingStage::Denoising,
            0.10, 0.0, "Denoising audio chunks...", false, None, None,
        );
        stage_denoise(&chunk_paths)?;
        check_cancelled(cancel_flag)?;
    }

    // All chunks share the same transcript from the client
    let transcripts: Vec<String> = vec![transcript.to_string(); chunk_paths.len()];

    // === Stage 3: Feature Extraction ===
    check_cancelled(cancel_flag)?;
    send_progress(
        progress_tx, task_id, TrainingStage::FeatureExtraction,
        0.20, 0.0, "Extracting HuBERT and phoneme features...", false, None, None,
    );
    let num_samples = stage_feature_extraction(&chunk_paths, &transcripts, &dataset_dir, progress_tx, task_id)?;
    tracing::info!("Feature extraction: {} samples prepared", num_samples);
    check_cancelled(cancel_flag)?;

    // === Stage 4: VITS Fewshot Training ===
    send_progress(
        progress_tx, task_id, TrainingStage::VitsTraining,
        0.40, 0.0, "Starting VITS fewshot training...", false, None, None,
    );
    let vits_output = work_dir.join("vits_finetuned.safetensors");
    stage_vits_training(&dataset_dir, &vits_output, quality, progress_tx, task_id, cancel_flag)?;
    tracing::info!("VITS training complete: {:?}", vits_output);
    check_cancelled(cancel_flag)?;

    // === Stage 5: Register Voice ===
    send_progress(
        progress_tx, task_id, TrainingStage::RegisteringVoice,
        0.95, 0.0, "Registering trained voice...", false, None, None,
    );
    register_voice(voice_name, &vits_output, audio_path, transcript)?;

    // Tell inference thread to reload voices
    let (reload_tx, reload_rx) = oneshot::channel();
    if inference_tx.blocking_send(crate::InferenceRequest::ReloadVoices { response_tx: reload_tx }).is_ok() {
        let _ = reload_rx.blocking_recv();
    }
    tracing::info!("Voice '{}' registered and inference reloaded", voice_name);

    Ok(())
}

// ============================================================================
// Stage 1: Audio Slicing
// ============================================================================

fn stage_audio_slicing(audio_path: &Path, output_dir: &Path) -> Result<Vec<PathBuf>> {
    let slicer = AudioSlicer::new(SlicerConfig {
        sample_rate: TRAINING_SAMPLE_RATE,
        threshold_db: -34.0,
        min_length_ms: 4000,
        min_interval_ms: 300,
        hop_size_ms: 20,
        max_sil_kept_ms: 1000,
        max_amplitude: 0.9,
        alpha: 0.25,
    });

    tracing::info!("Slicing: {:?}", audio_path);

    // Load audio
    let (samples, sample_rate) = load_wav(audio_path)
        .map_err(|e| eyre::eyre!("Failed to load {:?}: {}", audio_path, e))?;

    // Resample to training sample rate if needed
    let samples = if sample_rate != TRAINING_SAMPLE_RATE {
        resample(&samples, sample_rate, TRAINING_SAMPLE_RATE)
    } else {
        samples
    };

    // Slice into chunks
    let chunks = slicer.slice(&samples);

    let mut output_files = Vec::new();
    for (i, (chunk_samples, _start_ms, _end_ms)) in chunks.into_iter().enumerate() {
        let out_path = output_dir.join(format!("chunk_{:04}.wav", i));
        save_wav(&chunk_samples, TRAINING_SAMPLE_RATE, &out_path)
            .map_err(|e| eyre::eyre!("Failed to save chunk: {}", e))?;
        output_files.push(out_path);
    }

    Ok(output_files)
}

// ============================================================================
// Stage 2: Denoising
// ============================================================================

fn stage_denoise(chunk_paths: &[PathBuf]) -> Result<()> {
    let denoiser = Denoiser::new(DenoiseConfig {
        sample_rate: TRAINING_SAMPLE_RATE,
        ..DenoiseConfig::default()
    }).map_err(|e| eyre::eyre!("Failed to create denoiser: {}", e))?;

    for path in chunk_paths {
        denoiser.process_file(path, path)
            .map_err(|e| eyre::eyre!("Denoise failed for {:?}: {}", path, e))?;
    }

    Ok(())
}

// ============================================================================
// Stage 3: Feature Extraction
// ============================================================================

fn stage_feature_extraction(
    audio_files: &[PathBuf],
    transcripts: &[String],
    dataset_dir: &Path,
    progress_tx: &broadcast::Sender<TrainingProgressEvent>,
    task_id: &str,
) -> Result<usize> {
    // Create subdirectories
    let ssl_dir = dataset_dir.join("ssl");
    let audio_out_dir = dataset_dir.join("audio");
    let phoneme_dir = dataset_dir.join("phonemes");
    std::fs::create_dir_all(&ssl_dir)?;
    std::fs::create_dir_all(&audio_out_dir)?;
    std::fs::create_dir_all(&phoneme_dir)?;

    // Load HuBERT model for SSL feature extraction
    let hubert_path = expand_tilde(&format!("{}/{}", MODELS_BASE, HUBERT_WEIGHTS));
    tracing::info!("Loading HuBERT model from: {}", hubert_path);
    let mut hubert = load_hubert_model(&hubert_path)
        .map_err(|e| eyre::eyre!("Failed to load HuBERT: {}", e))?;

    let mut metadata_samples = Vec::new();
    let total = audio_files.len();

    for (i, (audio_path, transcript)) in audio_files.iter().zip(transcripts.iter()).enumerate() {
        let sample_id = format!("sample_{:04}", i);

        // Progress
        let stage_progress = i as f32 / total as f32;
        send_progress(
            progress_tx, task_id, TrainingStage::FeatureExtraction,
            0.25 + 0.15 * stage_progress, stage_progress,
            &format!("Extracting features {}/{}", i + 1, total),
            false, None, None,
        );

        // 1. Load audio at training sample rate
        let (samples, sr) = load_wav(audio_path)
            .map_err(|e| eyre::eyre!("Load audio failed: {}", e))?;
        let samples_32k = if sr != TRAINING_SAMPLE_RATE {
            resample(&samples, sr, TRAINING_SAMPLE_RATE)
        } else {
            samples
        };

        // 2. Load audio at 16kHz for HuBERT
        let samples_16k = resample(&samples_32k, TRAINING_SAMPLE_RATE, HUBERT_SAMPLE_RATE);

        // 3. Extract HuBERT SSL features
        let audio_arr = mlx_rs::Array::from_slice(&samples_16k, &[1, samples_16k.len() as i32]);
        mlx_rs::transforms::eval([&audio_arr])
            .map_err(|e| eyre::eyre!("Eval failed: {:?}", e))?;
        let ssl_features = hubert.forward(&audio_arr)
            .map_err(|e| eyre::eyre!("HuBERT forward failed: {:?}", e))?;
        mlx_rs::transforms::eval([&ssl_features])
            .map_err(|e| eyre::eyre!("Eval failed: {:?}", e))?;

        // ssl_features: [1, T, 768] → [768, T] for npy
        let ssl_shape = ssl_features.shape().to_vec();
        let ssl_t = ssl_shape[1] as usize;
        let ssl_dim = ssl_shape[2] as usize;
        let ssl_data: Vec<f32> = (0..ssl_features.size())
            .map(|j| ssl_features.as_slice::<f32>()[j])
            .collect();
        // Transpose [1, T, 768] → [768, T]
        let mut ssl_transposed = vec![0.0f32; ssl_dim * ssl_t];
        for t in 0..ssl_t {
            for d in 0..ssl_dim {
                ssl_transposed[d * ssl_t + t] = ssl_data[t * ssl_dim + d];
            }
        }
        write_npy_f32(&ssl_dir.join(format!("{}.npy", sample_id)), &ssl_transposed, &[ssl_dim, ssl_t])?;

        // 4. Save audio as f32 npy
        write_npy_f32(&audio_out_dir.join(format!("{}.npy", sample_id)), &samples_32k, &[samples_32k.len()])?;

        // 5. Extract phoneme IDs from transcript
        let preproc = preprocess_text(transcript, Some(Language::Mixed));
        let phoneme_ids: Vec<i32> = preproc.phoneme_ids.clone();
        write_npy_i32(&phoneme_dir.join(format!("{}.npy", sample_id)), &phoneme_ids)?;

        metadata_samples.push(serde_json::json!({
            "id": sample_id,
            "ssl_len": ssl_t,
            "audio_len": samples_32k.len(),
            "phoneme_len": phoneme_ids.len(),
        }));
    }

    // Write metadata.json
    let metadata = serde_json::json!({
        "num_samples": metadata_samples.len(),
        "sample_rate": TRAINING_SAMPLE_RATE,
        "ssl_dim": 768,
        "samples": metadata_samples,
    });
    std::fs::write(
        dataset_dir.join("metadata.json"),
        serde_json::to_string_pretty(&metadata)?,
    )?;

    Ok(metadata_samples.len())
}

// ============================================================================
// Stage 5: VITS Fewshot Training
// ============================================================================

fn stage_vits_training(
    dataset_dir: &Path,
    output_path: &Path,
    quality: &TrainingQuality,
    progress_tx: &broadcast::Sender<TrainingProgressEvent>,
    task_id: &str,
    cancel_flag: &AtomicBool,
) -> Result<()> {
    let epochs = match quality {
        TrainingQuality::Fast => 4,
        TrainingQuality::Standard => 8,
        TrainingQuality::High => 16,
    };

    let config = VITSTrainingConfig {
        learning_rate_g: 1e-5,
        learning_rate_d: 1e-5,
        batch_size: 2,
        segment_size: 20480,
        log_every: 1,
        save_every: 10000, // We save manually at the end
        ..VITSTrainingConfig::default()
    };

    // Create trainer
    let mut trainer = VITSTrainer::new(config)
        .map_err(|e| eyre::eyre!("Failed to create VITS trainer: {}", e))?;

    // Load pretrained weights with regularization
    let pretrained_path = expand_tilde(&format!("{}/{}", MODELS_BASE, VITS_PRETRAINED));
    tracing::info!("Loading pretrained VITS: {}", pretrained_path);
    trainer.load_generator_weights_with_regularization(&pretrained_path)
        .map_err(|e| eyre::eyre!("Failed to load pretrained VITS: {}", e))?;

    // Freeze non-decoder layers (fewshot mode)
    trainer.freeze_non_decoder_layers();
    tracing::info!("Fewshot mode: frozen encoder/flow, training decoder/ref_enc/ssl_proj");

    // Load dataset
    let mut dataset = VITSDataset::load(dataset_dir)
        .map_err(|e| eyre::eyre!("Failed to load VITS dataset: {}", e))?;
    tracing::info!("VITS dataset: {} samples", dataset.len());

    if dataset.is_empty() {
        return Err(eyre::eyre!("VITS dataset is empty"));
    }

    // Training loop
    let hop_length = HOP_LENGTH;
    let segment_size = 20480;
    let total_steps = epochs * dataset.len().max(1);
    let mut global_step = 0;

    for epoch in 0..epochs {
        dataset.shuffle(Some(epoch as u64 + 42));

        for batch_result in dataset.iter_batches(2, segment_size, hop_length) {
            check_cancelled(cancel_flag)?;

            let batch = batch_result
                .map_err(|e| eyre::eyre!("Batch loading failed: {}", e))?;

            let losses = trainer.train_step(&batch)
                .map_err(|e| eyre::eyre!("Training step failed: {}", e))?;

            global_step += 1;

            // Progress
            let stage_progress = global_step as f32 / total_steps as f32;
            let overall_progress = 0.40 + 0.50 * stage_progress;

            send_progress(
                progress_tx, task_id, TrainingStage::VitsTraining,
                overall_progress, stage_progress,
                &format!("Epoch {}/{}, step {}: total_loss={:.4}", epoch + 1, epochs, global_step, losses.loss_total),
                false, None,
                Some(TrainingLossInfo {
                    epoch: epoch + 1,
                    step: global_step,
                    loss_total: Some(losses.loss_total),
                    loss_mel: Some(losses.loss_mel),
                    loss_kl: Some(losses.loss_kl),
                }),
            );
        }

        tracing::info!("Epoch {}/{} complete", epoch + 1, epochs);
    }

    // Save finetuned generator
    trainer.save_generator(output_path)
        .map_err(|e| eyre::eyre!("Failed to save generator: {}", e))?;
    tracing::info!("Saved finetuned VITS to {:?}", output_path);

    Ok(())
}

// ============================================================================
// Stage 5: Voice Registration
// ============================================================================

fn register_voice(
    voice_name: &str,
    vits_weights: &Path,
    ref_audio: &Path,
    transcript: &str,
) -> Result<()> {
    // Copy trained weights and reference audio to persistent directory
    let trained_dir = PathBuf::from(expand_tilde("~/.dora/models/primespeech/trained"));
    let voice_dir = trained_dir.join(voice_name);
    std::fs::create_dir_all(&voice_dir)?;

    std::fs::copy(vits_weights, voice_dir.join("vits_finetuned.safetensors"))?;
    std::fs::copy(ref_audio, voice_dir.join("ref.wav"))?;

    // Read existing voices.json
    let voices_path = PathBuf::from(expand_tilde(VOICES_CONFIG_PATH));
    let mut voices_json: serde_json::Value = if voices_path.exists() {
        let content = std::fs::read_to_string(&voices_path)
            .context("Failed to read voices.json")?;
        serde_json::from_str(&content)
            .context("Failed to parse voices.json")?
    } else {
        serde_json::json!({
            "default_voice": "doubao",
            "models_base_path": "~/.dora/models/primespeech",
            "voices": {}
        })
    };

    // Add new voice entry
    if let Some(voices) = voices_json.get_mut("voices").and_then(|v| v.as_object_mut()) {
        voices.insert(voice_name.to_string(), serde_json::json!({
            "ref_audio": format!("trained/{}/ref.wav", voice_name),
            "ref_text": transcript,
            "codes_path": null,
            "aliases": [],
            "speed_factor": 1.0,
            "vits_weights": format!("trained/{}/vits_finetuned.safetensors", voice_name)
        }));
    }

    // Write updated voices.json
    std::fs::write(&voices_path, serde_json::to_string_pretty(&voices_json)?)?;
    tracing::info!("Registered voice '{}' in {:?}", voice_name, voices_path);

    Ok(())
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Base directory for training work files
fn training_base_dir() -> PathBuf {
    std::env::var("TRAINING_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .map(|h| h.join(".dora/training"))
                .unwrap_or_else(|| PathBuf::from("/tmp/ominix-training"))
        })
}

/// Expand ~ to home directory
fn expand_tilde(path: &str) -> String {
    crate::utils::expand_tilde(path)
}



/// Send a training progress event
fn send_progress(
    tx: &broadcast::Sender<TrainingProgressEvent>,
    task_id: &str,
    stage: TrainingStage,
    progress: f32,
    stage_progress: f32,
    message: &str,
    is_complete: bool,
    error: Option<String>,
    losses: Option<TrainingLossInfo>,
) {
    let event = TrainingProgressEvent {
        task_id: task_id.to_string(),
        stage,
        progress,
        stage_progress,
        message: message.to_string(),
        losses,
        is_complete,
        error,
    };
    let _ = tx.send(event);
}

// ============================================================================
// NPY File I/O
// ============================================================================

/// Write a f32 array to NPY format
fn write_npy_f32(path: &Path, data: &[f32], shape: &[usize]) -> Result<()> {
    let descr = "<f4";
    write_npy(path, data, shape, descr, 4)
}

/// Write an i32 array to NPY format
fn write_npy_i32(path: &Path, data: &[i32]) -> Result<()> {
    let descr = "<i4";
    write_npy(path, data, &[data.len()], descr, 4)
}

/// Generic NPY writer
fn write_npy<T>(path: &Path, data: &[T], shape: &[usize], descr: &str, elem_size: usize) -> Result<()> {
    let shape_str = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!("({})", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", "))
    };

    let header_str = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        descr, shape_str
    );

    // Pad header to 64-byte alignment (including magic + version + header_len)
    let prefix_len = 10; // magic(6) + version(2) + header_len(2)
    let total_header = prefix_len + header_str.len() + 1; // +1 for \n
    let padding = (64 - (total_header % 64)) % 64;
    let padded_header = format!("{}{}\n", header_str, " ".repeat(padding));
    let header_len = padded_header.len() as u16;

    let mut buf = Vec::new();
    buf.extend_from_slice(b"\x93NUMPY"); // magic
    buf.push(1); // major version
    buf.push(0); // minor version
    buf.extend_from_slice(&header_len.to_le_bytes());
    buf.extend_from_slice(padded_header.as_bytes());

    // Write raw data
    let data_bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * elem_size)
    };
    buf.extend_from_slice(data_bytes);

    std::fs::write(path, &buf).context("Failed to write NPY file")?;
    Ok(())
}
