use std::time::Duration;

use salvo::prelude::*;

use crate::error::render_error;
use crate::inference::InferenceRequest;
use crate::types::ImageGenerationRequest;

use super::helpers::{get_state, send_and_wait};

/// Timeout for image generation (can be slow depending on model)
const IMAGE_TIMEOUT: Duration = Duration::from_secs(600); // 10 minutes

/// POST /v1/images/generations - Image generation
#[handler]
pub async fn images_generations(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    // Use larger body size limit for image uploads (10MB for img2img)
    let request: ImageGenerationRequest = req
        .parse_json_with_max_size(10 * 1024 * 1024)
        .await
        .map_err(|e| {
            tracing::error!("Failed to parse request: {}", e);
            StatusError::bad_request()
        })?;

    let response = send_and_wait(
        &state.inference_tx,
        |tx| InferenceRequest::Image { request, response_tx: tx },
        IMAGE_TIMEOUT,
    )
    .await?;

    res.render(Json(response));
    Ok(())
}

/// POST /v1/models/quantize - Quantize a FLUX model's transformer weights to 8-bit
///
/// Request body: { "model_id": "flux-klein-4b" } or { "model_id": "black-forest-labs/FLUX.2-klein-4B" }
///
/// This is a one-time operation. Loads the bf16 transformer, quantizes to INT8,
/// and saves alongside the original weights as `transformer/quantized_8bit.safetensors`.
/// Requires enough RAM to hold the full bf16 model (~30GB peak).
#[handler]
pub async fn quantize_model(req: &mut Request, res: &mut Response) {
    #[derive(serde::Deserialize)]
    struct QuantizeRequest {
        model_id: String,
    }

    let request: QuantizeRequest = match req.parse_json::<QuantizeRequest>().await {
        Ok(r) => r,
        Err(e) => {
            render_error(
                res,
                salvo::http::StatusCode::BAD_REQUEST,
                &format!("Invalid request: {}", e),
                "invalid_request_error",
            );
            return;
        }
    };

    let model_id = request.model_id.clone();
    tracing::info!("Quantizing FLUX model: {}", model_id);

    // Run quantization in a blocking thread (it's CPU/memory intensive)
    let result = tokio::task::spawn_blocking(move || -> eyre::Result<String> {
        use flux_klein_mlx::quantize_and_save_flux_klein;

        // Find the model directory
        let config_ids = vec!["flux-klein-4b", model_id.as_str()];
        let model_dir = 'lookup: {
            for config_id in &config_ids {
                match crate::model_config::check_model(
                    config_id,
                    crate::model_config::ModelCategory::Image,
                ) {
                    crate::model_config::ModelAvailability::Ready { local_path, .. } => {
                        let path =
                            local_path.ok_or_else(|| eyre::eyre!("Model path not available"))?;
                        break 'lookup crate::utils::resolve_hf_snapshot(&path)?;
                    }
                    crate::model_config::ModelAvailability::NotInConfig => continue,
                    _ => continue,
                }
            }
            // Try hub cache
            if let Some(hub_path) = crate::utils::resolve_from_hub_cache(&model_id) {
                hub_path
            } else {
                return Err(eyre::eyre!("Model '{}' not found", model_id));
            }
        };

        let transformer_path = model_dir.join("transformer/diffusion_pytorch_model.safetensors");
        let transformer_path = if transformer_path.exists() {
            transformer_path
        } else {
            model_dir.join("flux.safetensors")
        };
        if !transformer_path.exists() {
            return Err(eyre::eyre!(
                "Transformer weights not found at {:?}",
                transformer_path
            ));
        }

        let output_path = model_dir.join("transformer/quantized_8bit.safetensors");
        quantize_and_save_flux_klein(&transformer_path, &output_path, 64, 8)
            .map_err(|e| eyre::eyre!("Quantization failed: {}", e))?;

        Ok(format!("Quantized weights saved to {:?}", output_path))
    })
    .await;

    match result {
        Ok(Ok(msg)) => {
            res.render(Json(serde_json::json!({ "status": "success", "message": msg })));
        }
        Ok(Err(e)) => {
            render_error(
                res,
                salvo::http::StatusCode::INTERNAL_SERVER_ERROR,
                &e.to_string(),
                "quantization_error",
            );
        }
        Err(e) => {
            render_error(
                res,
                salvo::http::StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Task failed: {}", e),
                "internal_error",
            );
        }
    }
}
