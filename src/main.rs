//! OminiX-API: OpenAI-compatible API server for OminiX-MLX models
//!
//! Provides endpoints for:
//! - POST /v1/chat/completions - LLM chat completions
//! - POST /v1/audio/transcriptions - Speech-to-text (ASR)
//! - POST /v1/audio/speech - Text-to-speech with preset voices (TTS)
//! - POST /v1/audio/speech/clone - Voice cloning TTS with reference audio
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

mod chip;
mod config;
mod error;
mod server_config;
mod state;

mod download;
mod engines;
mod handlers;
mod inference;
mod model_registry;
mod router;

mod model_config;
mod training;
mod types;
mod utils;
mod version;
mod voice_registry;

use config::Config;
use inference::{InferenceRequest, TtsPoolConfig};
use state::AppState;
use types::{DownloadProgressEvent, TrainingProgressEvent};

fn host_port(host: &str, port: u16) -> String {
    let host = host
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(host);
    if host.contains(':') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    }
}

fn discovery_url(listen_host: &str, port: u16) -> String {
    let host = listen_host
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(listen_host);
    let reachable_host = match host {
        "0.0.0.0" | "::" => "127.0.0.1",
        host => host,
    };
    format!("http://{}", host_port(reachable_host, port))
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ominix_api=info,qwen3_tts_mlx=info".into()),
        )
        .init();

    let mut config = Config::from_env();
    let server_config = std::sync::Arc::new(server_config::ServerConfig::load());
    let sglang_router = engines::sglang::SglangRouter::from_env()?;
    tracing::info!(
        "Starting OminiX-API v{} on port {}",
        version::full_version(),
        config.port
    );

    // Apply the gatekeeper before collision checks so a blocked local model is
    // cleared rather than mistaken for an active backend.
    config.apply_server_config(&server_config);

    if let Some(router) = &sglang_router {
        for model in router.model_ids() {
            if !server_config.is_model_allowed("llm", &model) {
                eyre::bail!(
                    "OminiX-SGLang route {:?} is blocked by allowed_models.llm",
                    model
                );
            }
        }
        if !server_config.is_model_allowed("llm", router.served_model()) {
            eyre::bail!(
                "OminiX-SGLang served model {:?} is blocked by allowed_models.llm",
                router.served_model()
            );
        }
        if !config.llm_model.is_empty() && router.routes_model(&config.llm_model) {
            eyre::bail!(
                "LLM_MODEL collides with an exact OminiX-SGLang route; configure one backend for model {:?}",
                config.llm_model
            );
        }
        tracing::info!(
            models = ?router.model_ids(),
            "Configured explicit OminiX-SGLang CUDA model routing"
        );
    }
    let has_sglang_router = sglang_router.is_some();

    // Validate app manifest if provided
    if let Some(ref manifest_path) = config.app_manifest {
        version::validate_manifest(manifest_path)?;
    }

    // Scan hub caches and register discovered models
    model_config::scan_hub_caches();
    model_config::print_startup_report();

    // Create channels for inference and training
    let (inference_tx, inference_rx) = mpsc::channel::<InferenceRequest>(32);
    let (training_tx, training_rx) = mpsc::channel::<training::TrainingRequest>(8);
    let (progress_tx, _) = broadcast::channel::<TrainingProgressEvent>(256);
    let (ready_tx, ready_rx) = oneshot::channel();

    // Spawn inference thread (owns ALL models — single thread serializes GPU access)
    let config_clone = config.clone();
    let tts_pool_config = TtsPoolConfig::from_env();
    std::thread::spawn(move || {
        inference::inference_thread(config_clone, tts_pool_config, inference_rx, ready_tx);
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

    // Qwen3-TTS is now handled inline by the inference thread (no separate pool).
    // This serializes all GPU access through a single thread, preventing Metal
    // command buffer crashes when ASR and TTS run concurrently.

    // Create download channels and thread
    let (download_tx, download_rx) = mpsc::channel::<download::DownloadRequest>(16);
    let (download_progress_tx, _) = broadcast::channel::<DownloadProgressEvent>(256);
    let download_cancel_flags: download::DownloadCancelFlags = Default::default();
    let download_progress_tx_clone = download_progress_tx.clone();
    let download_cancel_flags_clone = download_cancel_flags.clone();
    std::thread::spawn(move || {
        download::download_thread(download_rx, download_progress_tx_clone, download_cancel_flags_clone);
    });

    // Wait for models to load
    ready_rx
        .await
        .context("Failed to receive ready signal from inference thread")?;
    tracing::info!("Inference thread ready");

    // Keep ASR warm: periodically re-run a short inference so the MLX graph /
    // model pages aren't released after idle (first-after-idle cold start is
    // otherwise ~5s). No-op when no ASR model is configured.
    if !config.asr_model_dir.is_empty() {
        let warm_tx = inference_tx.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(std::time::Duration::from_secs(180));
            tick.tick().await; // consume the immediate first tick (startup already warmed)
            loop {
                tick.tick().await;
                if warm_tx.send(InferenceRequest::KeepWarmAsr).await.is_err() {
                    break; // inference thread shut down
                }
            }
        });
    }

    // // Initialize Ascend backend if configured via environment
    // let ascend_config = engines::ascend::AscendConfig::from_env().map(|cfg| {
    //     tracing::info!("Ascend backend configured: bin_dir={}", cfg.bin_dir.display());
    //     if cfg.has_llm() { tracing::info!("  LLM: {:?}", cfg.llm_model); }
    //     if cfg.has_vlm() { tracing::info!("  VLM: {:?} + {:?}", cfg.vlm_model, cfg.vlm_mmproj); }
    //     if cfg.has_asr() { tracing::info!("  ASR: {:?}", cfg.asr_model_dir); }
    //     if cfg.has_tts() { tracing::info!("  TTS: {:?}", cfg.tts_model_dir); }
    //     if cfg.has_outetts() { tracing::info!("  OuteTTS: {:?}", cfg.outetts_model); }
    //     if cfg.has_diffusion() { tracing::info!("  Diffusion: {:?}", cfg.diffusion_model); }
    //     std::sync::Arc::new(cfg)
    // });
    let ascend_config: Option<std::sync::Arc<engines::ascend::AscendConfig>> = None;

    // B3 §5.3.3: select Ascend TTS backend once at startup (ASCEND_TTS_TRANSPORT).
    // When `ascend_config` is None we don't build a backend — handlers that
    // need it already check `state.ascend_config` first and 503 if absent.
    let ascend_tts_backend = ascend_config
        .as_ref()
        .map(|cfg| engines::tts_backends::build_ascend_tts_backend(cfg.clone()));

    let state = AppState {
        inference_tx,
        training_tx,
        progress_tx,
        cancel_flag,
        download_tx,
        download_progress_tx,
        download_cancel_flags,
        server_config,
        sglang_router: sglang_router.map(std::sync::Arc::new),
        ascend_config,
        ascend_tts_backend,
    };

    let router = router::build_router(state);

    // Preserve the existing bind behavior for local-only MLX deployments. A
    // configured CUDA route defaults to loopback because this API currently
    // delegates public authentication/TLS to a reverse proxy.
    let default_host = if has_sglang_router {
        "127.0.0.1"
    } else {
        "0.0.0.0"
    };
    let listen_host = std::env::var("OMINIX_API_HOST")
        .ok()
        .map(|host| host.trim().to_string())
        .filter(|host| !host.is_empty())
        .unwrap_or_else(|| default_host.to_string());
    let listen_addr = host_port(&listen_host, config.port);
    let acceptor = TcpListener::new(&listen_addr).bind().await;

    tracing::info!("HTTP server listening on http://{}", listen_addr);

    // Write a reachable address for the configured bind. Wildcard listeners
    // advertise IPv4 loopback; concrete IPv4, IPv6, and hostname binds retain
    // their selected host.
    let api_url = discovery_url(&listen_host, config.port);
    if let Some(home) = std::env::var_os("HOME") {
        let discovery_dir = std::path::Path::new(&home).join(".ominix");
        let _ = std::fs::create_dir_all(&discovery_dir);
        let discovery_file = discovery_dir.join("api_url");
        if let Err(e) = std::fs::write(&discovery_file, &api_url) {
            tracing::warn!(
                "Failed to write discovery file {}: {e}",
                discovery_file.display()
            );
        } else {
            tracing::info!("Discovery file: {}", discovery_file.display());
        }
    }

    tracing::info!("Endpoints:");
    tracing::info!("  GET  /health");
    tracing::info!("  GET  /readyz");
    tracing::info!("  GET  /v1/version");
    tracing::info!("  --- Models ---");
    tracing::info!("  GET  /v1/models");
    tracing::info!("  GET  /v1/models/status");
    tracing::info!("  GET  /v1/models/report");
    tracing::info!("  POST /v1/models/load");
    tracing::info!("  POST /v1/models/unload");
    tracing::info!("  POST /v1/models/quantize");
    tracing::info!("  GET  /v1/models/catalog");
    tracing::info!("  POST /v1/models/download");
    tracing::info!("  GET  /v1/models/download/progress");
    tracing::info!("  POST /v1/models/download/cancel");
    tracing::info!("  POST /v1/models/remove");
    tracing::info!("  POST /v1/models/scan");
    tracing::info!("  --- LLM ---");
    tracing::info!("  POST /v1/chat/completions");
    tracing::info!("  --- TTS (model-specific) ---");
    tracing::info!("  POST /v1/audio/tts/qwen3       (Qwen3-TTS preset voices)");
    tracing::info!("  POST /v1/audio/tts/clone        (Qwen3-TTS voice cloning)");
    tracing::info!("  POST /v1/audio/tts/sovits       (GPT-SoVITS)");
    tracing::info!("  --- ASR (model-specific) ---");
    tracing::info!("  POST /v1/audio/asr/qwen3        (Qwen3-ASR)");
    tracing::info!("  POST /v1/audio/asr/paraformer    (Paraformer)");
    tracing::info!("  --- Legacy (auto-routes) ---");
    tracing::info!("  POST /v1/audio/speech            (legacy TTS)");
    tracing::info!("  POST /v1/audio/speech/clone      (legacy clone)");
    tracing::info!("  POST /v1/audio/transcriptions    (legacy ASR)");
    // tracing::info!("  --- Ascend NPU ---");
    // tracing::info!("  POST /v1/chat/completions/ascend (LLM on Ascend)");
    // tracing::info!("  POST /v1/vlm/completions/ascend  (VLM on Ascend)");
    // tracing::info!("  POST /v1/audio/asr/ascend        (Qwen3-ASR on Ascend)");
    // tracing::info!("  POST /v1/audio/tts/ascend        (Qwen3-TTS on Ascend)");
    // tracing::info!("  POST /v1/audio/tts/ascend/clone  (Voice clone on Ascend)");
    // tracing::info!("  POST /v1/audio/tts/outetts       (OuteTTS on Ascend)");
    // tracing::info!("  POST /v1/images/generations/ascend (Image gen on Ascend)");
    tracing::info!("  --- Other ---");
    tracing::info!("  POST /v1/images/generations");
    tracing::info!("  POST /v1/vlm/completions");
    tracing::info!("  WS   /ws/v1/tts");
    tracing::info!("  GET  /v1/voices");
    tracing::info!("  POST /v1/voices/train");
    tracing::info!("  GET  /v1/voices/train/status");
    tracing::info!("  GET  /v1/voices/train/progress");
    tracing::info!("  POST /v1/voices/train/cancel");

    Server::new(acceptor).serve(router).await;

    Ok(())
}

#[cfg(test)]
mod routing_address_tests {
    use super::*;

    #[test]
    fn formats_ipv4_and_ipv6_listener_addresses() {
        assert_eq!(host_port("127.0.0.1", 8080), "127.0.0.1:8080");
        assert_eq!(host_port("::1", 8080), "[::1]:8080");
        assert_eq!(host_port("[::1]", 8080), "[::1]:8080");
    }

    #[test]
    fn discovery_tracks_concrete_hosts_and_maps_wildcards_to_loopback() {
        assert_eq!(discovery_url("0.0.0.0", 8080), "http://127.0.0.1:8080");
        assert_eq!(discovery_url("::", 8080), "http://127.0.0.1:8080");
        assert_eq!(discovery_url("::1", 8080), "http://[::1]:8080");
        assert_eq!(discovery_url("192.0.2.10", 8080), "http://192.0.2.10:8080");
    }
}
