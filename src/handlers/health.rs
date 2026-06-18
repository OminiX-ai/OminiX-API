use salvo::prelude::*;
use tokio::sync::oneshot;

use crate::inference::{InferenceRequest, ModelStatus};

use super::helpers::get_state;

/// GET /health - Health check (includes version for quick identification)
#[handler]
pub async fn health(res: &mut Response) {
    res.render(Json(serde_json::json!({
        "status": "healthy",
        "service": "ominix-api",
        "version": crate::version::full_version()
    })));
}

/// GET /v1/models - List available models (queries live status + TTS pool)
#[handler]
pub async fn list_models(depot: &mut Depot, res: &mut Response) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    let (response_tx, response_rx) = oneshot::channel();
    state
        .inference_tx
        .send(InferenceRequest::GetModelStatus { response_tx })
        .await
        .map_err(|_| StatusError::internal_server_error())?;

    let status: ModelStatus = response_rx
        .await
        .map_err(|_| StatusError::internal_server_error())?;

    let now = chrono::Utc::now().timestamp();
    let mut data = Vec::new();

    if let Some(ref id) = status.llm {
        data.push(serde_json::json!({
            "id": id, "object": "model", "created": now,
            "owned_by": "local", "type": "llm",
            "endpoints": ["/v1/chat/completions"]
        }));
    }
    if let Some(ref id) = status.asr {
        let endpoint = match id.as_str() {
            "qwen3-asr" => "/v1/audio/asr/qwen3",
            "paraformer" => "/v1/audio/asr/paraformer",
            _ => "/v1/audio/transcriptions",
        };
        data.push(serde_json::json!({
            "id": id, "object": "model", "created": now,
            "owned_by": "local", "type": "asr",
            "endpoints": [endpoint, "/v1/audio/transcriptions"]
        }));
    }
    if let Some(ref id) = status.tts {
        data.push(serde_json::json!({
            "id": id, "object": "model", "created": now,
            "owned_by": "local", "type": "tts",
            "endpoints": ["/v1/audio/tts/sovits"]
        }));
    }
    // TTS pool models (Qwen3-TTS): scan disk for available models
    // The pool auto-loads these on demand, so report whatever is on disk.
    for (variant, endpoint) in [
        ("customvoice", "/v1/audio/tts/qwen3"),
        ("base", "/v1/audio/tts/clone"),
    ] {
        if let Some(name) = find_tts_model_name(variant) {
            data.push(serde_json::json!({
                "id": name, "object": "model", "created": now,
                "owned_by": "local", "type": "qwen3_tts",
                "endpoints": [endpoint]
            }));
        }
    }
    if let Some(ref id) = status.image {
        data.push(serde_json::json!({
            "id": id, "object": "model", "created": now,
            "owned_by": "local", "type": "image",
            "endpoints": ["/v1/images/generations"]
        }));
    }
    if let Some(ref id) = status.video {
        data.push(serde_json::json!({
            "id": id, "object": "model", "created": now,
            "owned_by": "local", "type": "video",
            "endpoints": ["/v1/videos/generations"]
        }));
    }
    if let Some(ref id) = status.vlm {
        data.push(serde_json::json!({
            "id": id, "object": "model", "created": now,
            "owned_by": "local", "type": "vlm",
            "endpoints": ["/v1/vlm/completions"]
        }));
    }

    res.render(Json(serde_json::json!({
        "object": "list",
        "data": data
    })));
    Ok(())
}

/// Scan standard TTS model directories for a specific variant.
/// Returns the directory name if found, None otherwise.
fn find_tts_model_name(variant: &str) -> Option<String> {
    fn has_qwen3_tts_weights(path: &std::path::Path) -> bool {
        path.join("model.safetensors").is_file()
            || path.join("model.safetensors.index.json").is_file()
    }

    let home = dirs::home_dir()?;
    for subdir in &[".OminiX/models", ".ominix/models"] {
        let dir = home.join(subdir);
        if let Ok(entries) = std::fs::read_dir(&dir) {
            let mut candidates = Vec::new();
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().to_lowercase();
                if path.is_dir() && name.contains("tts") && name.contains(variant) {
                    candidates.push(path);
                }
            }

            candidates.sort_by_key(|path| {
                let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
                (
                    !has_qwen3_tts_weights(path),
                    !name.to_lowercase().contains("8bit"),
                    name.len(),
                )
            });

            if let Some(path) = candidates.into_iter().next() {
                return path.file_name().map(|s| s.to_string_lossy().to_string());
            }
        }
    }
    None
}

/// GET /v1/models/report - Model availability report
#[handler]
pub async fn models_report(res: &mut Response) {
    let report = crate::model_config::get_model_report();
    res.render(Json(report));
}

/// GET /v1/models/status - Get current model status
#[handler]
pub async fn model_status(depot: &mut Depot, res: &mut Response) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    let (response_tx, response_rx) = oneshot::channel();
    state
        .inference_tx
        .send(InferenceRequest::GetModelStatus { response_tx })
        .await
        .map_err(|_| StatusError::internal_server_error())?;

    let mut status: ModelStatus = response_rx
        .await
        .map_err(|_| StatusError::internal_server_error())?;

    // Augment with Ascend backend status
    if let Some(ref cfg) = state.ascend_config {
        status.ascend = Some(crate::inference::AscendStatus {
            llm: cfg.has_llm(),
            vlm: cfg.has_vlm(),
            asr: cfg.has_asr(),
            tts: cfg.has_tts(),
            outetts: cfg.has_outetts(),
            image: cfg.has_diffusion(),
        });
    }

    res.render(Json(serde_json::json!({
        "status": "success",
        "models": status
    })));
    Ok(())
}
