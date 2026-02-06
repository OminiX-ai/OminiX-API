//! OminiX-API: OpenAI-compatible API server for OminiX-MLX models
//!
//! Provides endpoints for:
//! - POST /v1/chat/completions - LLM chat completions
//! - POST /v1/audio/transcriptions - Speech-to-text (ASR)
//! - POST /v1/audio/speech - Text-to-speech (TTS)
//! - POST /v1/images/generations - Image generation
//! - WS   /ws/v1/tts - WebSocket streaming TTS with per-message voice switching
//! - POST /v1/voices/train - Voice cloning training (quick mode)
//! - GET  /v1/voices/train/progress - SSE training progress stream
//! - GET  /v1/voices - List registered voices
//!
//! Note: MLX models don't implement Send/Sync, so we use channels to
//! communicate with dedicated inference and training threads.

use eyre::Context;
use salvo::prelude::*;
use tokio::sync::{broadcast, mpsc, oneshot};

mod config;
mod error;
mod state;

mod engines;
mod handlers;
mod inference;
mod router;

mod model_config;
mod training;
mod types;
mod utils;

use config::Config;
use inference::InferenceRequest;
use state::AppState;
use types::TrainingProgressEvent;

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

    // Scan hub caches and register discovered models
    model_config::scan_hub_caches();
    model_config::print_startup_report();

    // Create channels for inference and training
    let (inference_tx, inference_rx) = mpsc::channel::<InferenceRequest>(32);
    let (training_tx, training_rx) = mpsc::channel::<training::TrainingRequest>(8);
    let (progress_tx, _) = broadcast::channel::<TrainingProgressEvent>(256);
    let (ready_tx, ready_rx) = oneshot::channel();

    // Spawn inference thread (owns all models)
    let config_clone = config.clone();
    std::thread::spawn(move || {
        inference::inference_thread(config_clone, inference_rx, ready_tx);
    });

    // Spawn training thread (owns training models, separate from inference)
    let progress_tx_clone = progress_tx.clone();
    let inference_tx_for_training = inference_tx.clone();
    let cancel_flag: training::CancelFlag = Default::default();
    let cancel_flag_clone = cancel_flag.clone();
    std::thread::spawn(move || {
        training::training_thread(
            training_rx,
            progress_tx_clone,
            inference_tx_for_training,
            cancel_flag_clone,
        );
    });

    // Wait for models to load
    ready_rx
        .await
        .context("Failed to receive ready signal from inference thread")?;
    tracing::info!("Inference thread ready");

    let state = AppState {
        inference_tx,
        training_tx,
        progress_tx,
        cancel_flag,
    };

    let router = router::build_router(state);

    let listen_addr = format!("0.0.0.0:{}", config.port);
    let acceptor = TcpListener::new(&listen_addr).bind().await;

    tracing::info!("HTTP server listening on http://{}", listen_addr);
    tracing::info!("Endpoints:");
    tracing::info!("  GET  /health");
    tracing::info!("  GET  /v1/models");
    tracing::info!("  GET  /v1/models/status");
    tracing::info!("  GET  /v1/models/report");
    tracing::info!("  POST /v1/models/load");
    tracing::info!("  POST /v1/models/unload");
    tracing::info!("  POST /v1/models/quantize");
    tracing::info!("  POST /v1/chat/completions");
    tracing::info!("  POST /v1/audio/transcriptions");
    tracing::info!("  POST /v1/audio/speech");
    tracing::info!("  POST /v1/images/generations");
    tracing::info!("  WS   /ws/v1/tts");
    tracing::info!("  GET  /v1/voices");
    tracing::info!("  POST /v1/voices/train");
    tracing::info!("  GET  /v1/voices/train/status");
    tracing::info!("  GET  /v1/voices/train/progress");
    tracing::info!("  POST /v1/voices/train/cancel");

    Server::new(acceptor).serve(router).await;

    Ok(())
}
