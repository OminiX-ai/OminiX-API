//! VLM (Vision-Language Model) inference engine
//!
//! Wraps moxin-vlm-mlx for Moxin-7B VLM inference.

use std::path::PathBuf;

use eyre::Result;
use image::imageops::FilterType;
use mlx_rs::Array;
use tokenizers::Tokenizer;

use crate::model_config::{self, ModelAvailability, ModelCategory};
use crate::types::{VlmCompletionRequest, VlmCompletionResponse, VlmUsage};

use moxin_vlm_mlx::{
    load_model, normalize_dino, normalize_siglip, Generate, KVCache, MoxinVLM,
};

/// VLM inference engine
pub struct VlmEngine {
    model: MoxinVLM,
    tokenizer: Tokenizer,
}

impl VlmEngine {
    /// Load a VLM model from a path or model ID.
    pub fn new(model_path: &str) -> Result<Self> {
        tracing::info!("Loading VLM model: {}", model_path);

        let model_dir = resolve_model_dir(model_path)?;
        tracing::info!("Using VLM model directory: {:?}", model_dir);

        let model = load_model(&model_dir)
            .map_err(|e| eyre::eyre!("Failed to load VLM model: {}", e))?;

        // Check for quantize_config.json to auto-quantize
        let quantize_config_path = model_dir.join("quantize_config.json");
        let model = if quantize_config_path.exists() {
            let config_content = std::fs::read_to_string(&quantize_config_path)?;
            let config: serde_json::Value = serde_json::from_str(&config_content)?;
            let bits = config["quantization"]["bits"].as_i64().unwrap_or(8) as i32;
            let group_size = config["quantization"]["group_size"].as_i64().unwrap_or(64) as i32;
            tracing::info!("Quantizing VLM to {} bits (group_size={})", bits, group_size);
            model
                .quantize(group_size, bits)
                .map_err(|e| eyre::eyre!("VLM quantization failed: {}", e))?
        } else {
            model
        };

        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(eyre::eyre!("tokenizer.json not found at {:?}", tokenizer_path));
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| eyre::eyre!("Failed to load tokenizer: {}", e))?;

        tracing::info!("VLM model loaded successfully");

        Ok(Self { model, tokenizer })
    }

    /// Run VLM inference: image + prompt -> text description
    pub fn describe(&mut self, request: &VlmCompletionRequest) -> Result<VlmCompletionResponse> {
        let temperature = request.temperature.unwrap_or(0.0);
        let max_tokens = request.max_tokens.unwrap_or(256);

        // Decode base64 image
        let image_bytes = base64::Engine::decode(
            &base64::engine::general_purpose::STANDARD,
            &request.image,
        )
        .map_err(|e| eyre::eyre!("Invalid base64 image: {}", e))?;

        let img = image::load_from_memory(&image_bytes)
            .map_err(|e| eyre::eyre!("Failed to decode image: {}", e))?;

        // Resize to 224x224 and convert to float tensor
        let img = img.resize_exact(224, 224, FilterType::CatmullRom);
        let rgb = img.to_rgb8();

        let pixels: Vec<f32> = rgb
            .pixels()
            .flat_map(|p| p.0.iter().map(|&v| v as f32 / 255.0))
            .collect();
        let tensor = Array::from_slice(&pixels, &[1, 224, 224, 3]);

        // Normalize for both vision encoders
        let dino_img = normalize_dino(&tensor)
            .map_err(|e| eyre::eyre!("DINOv2 normalization failed: {}", e))?;
        let siglip_img = normalize_siglip(&tensor)
            .map_err(|e| eyre::eyre!("SigLIP normalization failed: {}", e))?;

        // Format prompt (Prismatic "Pure" format)
        let prompt_text = format!("In: {}\nOut:", request.prompt);

        // Tokenize (with BOS)
        let encoding = self
            .tokenizer
            .encode(prompt_text.as_str(), true)
            .map_err(|e| eyre::eyre!("Tokenization failed: {}", e))?;
        let prompt_tokens = encoding.get_ids().len() as u32;
        let input_ids = Array::from_iter(
            encoding.get_ids().iter().map(|&id| id as i32),
            &[1, encoding.get_ids().len() as i32],
        );

        // Generate
        let mut cache: Vec<KVCache> = Vec::new();
        let generator = Generate::new(
            &mut self.model,
            &mut cache,
            temperature,
            dino_img,
            siglip_img,
            input_ids,
        );

        let eos_token_id = 2u32; // </s>
        let mut generated_ids = Vec::new();

        for token_result in generator.take(max_tokens) {
            let token = token_result.map_err(|e| eyre::eyre!("Generation error: {}", e))?;
            let token_id = token.item::<u32>();
            if token_id == eos_token_id {
                break;
            }
            generated_ids.push(token_id);
        }

        let completion_tokens = generated_ids.len() as u32;

        let content = self
            .tokenizer
            .decode(&generated_ids, true)
            .map_err(|e| eyre::eyre!("Decoding failed: {}", e))?;

        Ok(VlmCompletionResponse {
            id: format!("vlm-{}", uuid::Uuid::new_v4()),
            object: "vlm.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: request.model.clone(),
            content,
            usage: VlmUsage {
                prompt_tokens: prompt_tokens + 256, // 256 visual tokens
                completion_tokens,
                total_tokens: prompt_tokens + 256 + completion_tokens,
            },
        })
    }
}

/// Resolve a VLM model path/ID to a local directory
fn resolve_model_dir(model_path: &str) -> Result<PathBuf> {
    // Direct filesystem path
    let direct_path = PathBuf::from(model_path);
    if direct_path.exists() && direct_path.join("config.json").exists() {
        return Ok(direct_path);
    }

    // Expand ~ prefix
    if model_path.starts_with("~/") {
        let expanded = PathBuf::from(crate::utils::expand_tilde(model_path));
        if expanded.exists() && expanded.join("config.json").exists() {
            return Ok(expanded);
        }
    }

    // Check model config
    match model_config::check_model(model_path, ModelCategory::Vlm) {
        ModelAvailability::Ready {
            local_path,
            model_name,
        } => {
            tracing::info!("Found VLM model: {} at {:?}", model_name, local_path);
            let path = local_path.ok_or_else(|| eyre::eyre!("Model path not available"))?;
            crate::utils::resolve_hf_snapshot(&path)
        }
        ModelAvailability::NotDownloaded {
            model_name,
            model_id,
        } => Err(eyre::eyre!(
            "VLM model '{}' ({}) is not downloaded.",
            model_name,
            model_id
        )),
        ModelAvailability::WrongCategory { expected, found } => Err(eyre::eyre!(
            "Model '{}' is a {:?} model, not a {:?} model",
            model_path,
            found,
            expected
        )),
        ModelAvailability::NotInConfig => {
            // Try hub caches
            if let Some(hub_path) = crate::utils::resolve_from_hub_cache(model_path) {
                if hub_path.join("config.json").exists() {
                    let _ =
                        model_config::register_model(model_path, ModelCategory::Vlm, &hub_path);
                    return Ok(hub_path);
                }
            }
            Err(eyre::eyre!(
                "VLM model '{}' not found in local configuration or hub caches.",
                model_path
            ))
        }
    }
}
