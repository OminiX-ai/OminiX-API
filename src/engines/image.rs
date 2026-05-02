//! Image generation engine supporting FLUX.2-klein, Z-Image-Turbo, and Qwen-Image
//!
//! Supported models:
//! - FLUX.2-klein-4B: 4-step denoising, ~13GB VRAM (or ~3GB with INT8)
//! - Z-Image-Turbo: 9-step denoising, ~12GB VRAM (or ~3GB with 4-bit)
//! - Qwen-Image-2512: flow-matching generation with Qwen3 text encoder, 4-bit or 8-bit

use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

use eyre::{Context, Result};
use mlx_rs::module::ModuleParameters;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;
use tokenizers::Tokenizer;

use crate::model_config::{self, ModelAvailability, ModelCategory};

use flux_klein_mlx::{
    AutoEncoderConfig, Decoder, Encoder, FluxKlein, FluxKleinParams,
    QuantizedFluxKlein, load_quantized_flux_klein,
    Qwen3Config, Qwen3TextEncoder,
    load_safetensors,
    sanitize_qwen3_weights, sanitize_vae_weights, sanitize_vae_encoder_weights,
    sanitize_klein_model_weights,
};

use zimage_mlx::{
    ZImageConfig, ZImageTransformerQuantized, create_coordinate_grid,
    load_quantized_zimage_transformer, load_safetensors as load_zimage_safetensors,
    QuantizedQwen3TextEncoder, sanitize_quantized_qwen3_weights,
};

use qwen_image_mlx::{
    QwenQuantizedTransformer, QwenConfig, QwenVAE, QwenTextEncoder,
    load_text_encoder, load_vae_from_dir, load_transformer_weights,
    FlowMatchEulerScheduler,
    pack_latents, unpack_latents, build_edit_rope, ref_shape_from_latent,
};

use crate::types::{ImageGenerationRequest, ImageGenerationResponse, ImageData};

/// Image generation model type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImageModelType {
    FluxKlein,
    FluxKleinGguf,
    ZImageTurbo,
    QwenImage,
    QwenImageEdit,
    CosmosT2I,
}

impl ImageModelType {
    /// Known config aliases for each model type (short names used by Studio)
    pub fn config_aliases(&self) -> &'static [&'static str] {
        match self {
            ImageModelType::FluxKlein => &["flux-klein-4b", "flux-klein-4b-8bit"],
            ImageModelType::FluxKleinGguf => &["flux-klein-4b-q4-gguf", "flux-klein-4b-q4"],
            ImageModelType::ZImageTurbo => &["zimage-turbo"],
            ImageModelType::QwenImage => &["qwen-image-2512-4bit", "qwen-image-2512-8bit"],
            ImageModelType::QwenImageEdit => &["qwen-image-edit", "qwen-image-edit-2511", "qwen-image-edit-2511-q4km"],
            ImageModelType::CosmosT2I => &["cosmos-predict2-14b", "cosmos-t2i", "cosmos-predict2-14b-t2i-q4km"],
        }
    }

    /// Detect model type from a user-provided model ID string
    pub fn from_model_id(model_id: &str) -> Self {
        let lower = model_id.to_lowercase();
        if lower.contains("edit") && (lower.contains("qwen") || lower.contains("image")) {
            ImageModelType::QwenImageEdit
        } else if lower.contains("cosmos") {
            ImageModelType::CosmosT2I
        } else if lower.contains("zimage") || lower.contains("z-image") {
            ImageModelType::ZImageTurbo
        } else if lower.contains("qwen-image") || lower.contains("qwen_image") {
            ImageModelType::QwenImage
        } else if lower.contains("flux") && (lower.contains("gguf") || lower.contains("q4")) {
            ImageModelType::FluxKleinGguf
        } else {
            ImageModelType::FluxKlein
        }
    }

    /// Normalized short name for tracking the current model
    pub fn normalized_name(&self) -> &'static str {
        match self {
            ImageModelType::FluxKlein => "flux",
            ImageModelType::FluxKleinGguf => "flux-gguf",
            ImageModelType::ZImageTurbo => "zimage",
            ImageModelType::QwenImage => "qwen-image",
            ImageModelType::QwenImageEdit => "qwen-image-edit",
            ImageModelType::CosmosT2I => "cosmos-t2i",
        }
    }

    /// Whether this model type uses a Python subprocess engine
    pub fn uses_pymlx(&self) -> bool {
        matches!(self, ImageModelType::QwenImageEdit | ImageModelType::CosmosT2I | ImageModelType::FluxKleinGguf)
    }
}

/// Transformer variant (FLUX, Z-Image, or Qwen-Image)
enum TransformerVariant {
    Flux(FluxKlein, FluxKleinParams),
    FluxQuantized(QuantizedFluxKlein, FluxKleinParams),
    ZImage(ZImageTransformerQuantized, ZImageConfig),
    QwenImage(QwenQuantizedTransformer, QwenConfig),
}

/// Text encoder variant
enum TextEncoderVariant {
    Standard(Qwen3TextEncoder),
    Quantized(QuantizedQwen3TextEncoder),
    /// Qwen2.5-VL based text encoder for Qwen-Image
    QwenImage(QwenTextEncoder),
}

/// Image generation engine
pub struct ImageEngine {
    model_type: ImageModelType,
    text_encoder: Option<TextEncoderVariant>,
    transformer: TransformerVariant,
    /// FLUX / Z-Image VAE decoder (None for Qwen-Image)
    vae_decoder: Option<Decoder>,
    /// FLUX / Z-Image VAE encoder for img2img (None for Qwen-Image)
    vae_encoder: Option<Encoder>,
    tokenizer: Tokenizer,
    vae_config: AutoEncoderConfig,
    /// Qwen-Image VAE (None for FLUX / Z-Image)
    qwen_vae: Option<QwenVAE>,
    /// Model directory for lazy-reloading text encoder (quantized FLUX only)
    model_dir: Option<std::path::PathBuf>,
    is_quantized_flux: bool,
}

impl ImageEngine {
    /// Estimated VRAM requirements per model type (bytes).
    fn estimated_vram(model_type: ImageModelType, is_quantized: bool) -> usize {
        match (model_type, is_quantized) {
            // FLUX quantized: 8-bit transformer ~4.1GB + VAE ~0.2GB (text encoder loaded/dropped)
            (ImageModelType::FluxKlein, true) => 4_500_000_000,
            // FLUX f32: text encoder ~15GB + transformer ~8GB + VAE ~0.2GB
            (ImageModelType::FluxKlein, false) => 23_000_000_000,
            // Z-Image 4-bit: transformer ~3GB + text encoder ~1.5GB + VAE ~0.2GB
            (ImageModelType::ZImageTurbo, _) => 5_000_000_000,
            // Qwen-Image: transformer ~3GB + text encoder ~2GB + VAE ~0.5GB
            (ImageModelType::QwenImage, _) => 6_000_000_000,
            // These use Python subprocess engines, not the Rust ImageEngine
            (ImageModelType::QwenImageEdit, _) | (ImageModelType::CosmosT2I, _) | (ImageModelType::FluxKleinGguf, _) => 0,
        }
    }

    /// Create a new image generation engine
    ///
    /// First checks ~/.OminiX/local_models_config.json for model availability.
    /// If the model is ready locally, uses that path. Otherwise, attempts
    /// to download from HuggingFace Hub.
    pub fn new(model_id: &str) -> Result<Self> {
        tracing::info!("Initializing image generation engine: {}", model_id);
        tracing::info!("Loading image model...");

        // Determine model type from the user-provided model ID
        let model_type = ImageModelType::from_model_id(model_id);
        tracing::info!("Image model type: {:?}", model_type);

        // Build config lookup IDs: known aliases, then original model_id,
        // then the registry repo_id (e.g. "moxin-org/FLUX.2-klein-4B-8bit-mlx").
        // Search all aliases + original model_id to find the registry repo_id,
        // since the registry may use a different ID than what the caller passed.
        let registry_repo_id = model_type.config_aliases()
            .iter()
            .copied()
            .chain(std::iter::once(model_id))
            .find_map(|id| {
                crate::model_registry::get_download_spec(id)
                    .and_then(|s| s.source.repo_id)
            });
        let config_model_ids: Vec<&str> = model_type.config_aliases()
            .iter()
            .copied()
            .chain(std::iter::once(model_id))
            .chain(registry_repo_id.as_deref())
            .collect();

        // Check model configuration for local availability
        let model_dir: PathBuf = 'lookup: {
            for config_model_id in &config_model_ids {
                match model_config::check_model(config_model_id, ModelCategory::Image) {
                    ModelAvailability::Ready { local_path, model_name } => {
                        tracing::info!("Found locally available model: {} at {:?}", model_name, local_path);
                        let path = local_path.ok_or_else(|| eyre::eyre!("Model path not available"))?;
                        break 'lookup crate::utils::resolve_hf_snapshot(&path)?;
                    }
                    ModelAvailability::NotDownloaded { model_name, model_id } => {
                        return Err(eyre::eyre!(
                            "Image model '{}' ({}) is not downloaded.\n\
                             Please download it using OminiX-Studio before starting the API server.",
                            model_name, model_id
                        ));
                    }
                    ModelAvailability::WrongCategory { expected, found } => {
                        return Err(eyre::eyre!(
                            "Model '{}' is a {:?} model, not a {:?} model",
                            config_model_id, found, expected
                        ));
                    }
                    ModelAvailability::NotInConfig => {
                        continue;
                    }
                }
            }

            // Not found in config — try hub caches with original model_id and repo_id
            let hub_ids: Vec<&str> = std::iter::once(model_id)
                .chain(registry_repo_id.as_deref())
                .collect();
            let mut found_hub = None;
            for hub_id in &hub_ids {
                if let Some(hub_path) = crate::utils::resolve_from_hub_cache(hub_id) {
                    tracing::info!("Found image model in hub cache: {:?}", hub_path);
                    let _ = model_config::register_model(hub_id, ModelCategory::Image, &hub_path);
                    found_hub = Some(hub_path);
                    break;
                }
            }
            match found_hub {
                Some(path) => path,
                None => return Err(eyre::eyre!(
                    "Image model '{}' not found in local configuration or hub caches.\n\
                     Please download it via OminiX-Studio or huggingface-cli.\n\
                     Searched: ~/.OminiX/local_models_config.json, ~/.cache/huggingface/hub/, ~/.cache/modelscope/hub/",
                    model_id
                )),
            }
        };

        // ── Qwen-Image: dedicated loading path ───────────────────────────────
        if model_type == ImageModelType::QwenImage {
            return Self::new_qwen_image(model_id, model_type, model_dir);
        }

        // Check if this is a quantized FLUX model (determines text encoder precision)
        let is_quantized_flux = model_type == ImageModelType::FluxKlein
            && model_dir.join("transformer/quantized_8bit.safetensors").exists();

        let estimated = Self::estimated_vram(model_type, is_quantized_flux);
        tracing::info!("Estimated VRAM: {:.1}GB", estimated as f64 / 1e9);

        // Get paths and configs based on model type
        let (transformer_path, text_encoder_paths, vae_path, tokenizer_path, vae_config) = match model_type {
            ImageModelType::FluxKlein => {
                tracing::info!("Loading FLUX.2-klein from local path: {:?}", model_dir);

                let trans = model_dir.join("transformer/diffusion_pytorch_model.safetensors");
                let trans = if trans.exists() { trans } else { model_dir.join("transformer/quantized_8bit.safetensors") };
                let trans = if trans.exists() { trans } else { model_dir.join("flux.safetensors") };
                if !trans.exists() {
                    return Err(eyre::eyre!("FLUX transformer not found at {:?}", trans));
                }

                let te1 = model_dir.join("text_encoder/model-00001-of-00002.safetensors");
                let te2 = model_dir.join("text_encoder/model-00002-of-00002.safetensors");
                if !te1.exists() || !te2.exists() {
                    return Err(eyre::eyre!("Text encoder files not found"));
                }

                let vae = model_dir.join("vae/diffusion_pytorch_model.safetensors");
                let vae = if vae.exists() { vae } else { model_dir.join("ae.safetensors") };
                if !vae.exists() {
                    return Err(eyre::eyre!("VAE not found at {:?}", vae));
                }

                let tok = model_dir.join("tokenizer/tokenizer.json");
                if !tok.exists() {
                    return Err(eyre::eyre!("Tokenizer not found at {:?}", tok));
                }

                let config = AutoEncoderConfig::flux2();
                (trans, vec![te1, te2], vae, tok, config)
            }
            ImageModelType::ZImageTurbo => {
                tracing::info!("Loading Z-Image-Turbo from local path: {:?}", model_dir);

                let trans = model_dir.join("transformer/model.safetensors");
                if !trans.exists() {
                    return Err(eyre::eyre!("Z-Image transformer not found at {:?}", trans));
                }

                let te = model_dir.join("text_encoder/model.safetensors");
                if !te.exists() {
                    return Err(eyre::eyre!("Text encoder not found at {:?}", te));
                }

                let vae = model_dir.join("vae/diffusion_pytorch_model.safetensors");
                if !vae.exists() {
                    return Err(eyre::eyre!("VAE not found at {:?}", vae));
                }

                let tok = model_dir.join("tokenizer/tokenizer.json");
                if !tok.exists() {
                    return Err(eyre::eyre!("Tokenizer not found at {:?}", tok));
                }

                let config = AutoEncoderConfig {
                    resolution: 1024,
                    in_channels: 3,
                    ch: 128,
                    out_ch: 3,
                    ch_mult: vec![1, 2, 4, 4],
                    num_res_blocks: 2,
                    z_channels: 16,
                    scale_factor: 0.3611,
                    shift_factor: 0.1159,
                };

                (trans, vec![te], vae, tok, config)
            }
            ImageModelType::QwenImage | ImageModelType::QwenImageEdit | ImageModelType::CosmosT2I | ImageModelType::FluxKleinGguf => {
                unreachable!("handled by subprocess engines")
            }
        };

        // Load Qwen3 text encoder (same config for both models)
        tracing::info!("Loading Qwen3 text encoder...");
        let qwen3_config = Qwen3Config {
            hidden_size: 2560,
            num_hidden_layers: 36,
            intermediate_size: 9728,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            rms_norm_eps: 1e-6,
            vocab_size: 151936,
            max_position_embeddings: 40960,
            rope_theta: 1000000.0,
            head_dim: 128,
        };

        // Load text encoder weights based on model type
        let text_encoder = match model_type {
            ImageModelType::FluxKlein => {
                // FLUX uses standard (non-quantized) text encoder
                let mut encoder = Qwen3TextEncoder::new(qwen3_config)?;
                let mut all_weights = HashMap::new();
                for path in &text_encoder_paths {
                    let weights = load_safetensors(path)?;
                    all_weights.extend(weights);
                }
                let weights = sanitize_qwen3_weights(all_weights);
                // Cast weights to f32 — bf16 causes Metal crash, f16 causes NaN
                // For quantized FLUX, text encoder is dropped after encoding to free ~15 GB
                let weights: HashMap<String, Array> = weights
                    .into_iter()
                    .map(|(k, v)| {
                        let v32 = v.as_type::<f32>().unwrap_or(v);
                        (k, v32)
                    })
                    .collect();
                let weights_rc: HashMap<Rc<str>, Array> = weights
                    .into_iter()
                    .map(|(k, v)| (Rc::from(k.as_str()), v))
                    .collect();
                encoder.update_flattened(weights_rc);
                TextEncoderVariant::Standard(encoder)
            }
            ImageModelType::ZImageTurbo => {
                // Z-Image uses 4-bit quantized text encoder
                let mut encoder = QuantizedQwen3TextEncoder::new(qwen3_config)?;
                let te_weights = load_safetensors(&text_encoder_paths[0])?;
                let weights = sanitize_quantized_qwen3_weights(te_weights);
                let weights_rc: HashMap<Rc<str>, Array> = weights
                    .into_iter()
                    .map(|(k, v)| (Rc::from(k.as_str()), v))
                    .collect();
                encoder.update_flattened(weights_rc);
                TextEncoderVariant::Quantized(encoder)
            }
            ImageModelType::QwenImage | ImageModelType::QwenImageEdit | ImageModelType::CosmosT2I | ImageModelType::FluxKleinGguf => {
                unreachable!("handled by subprocess engines")
            }
        };
        tracing::info!("Text encoder loaded");

        // Load transformer based on model type
        let transformer = match model_type {
            ImageModelType::FluxKlein => {
                // Check for pre-quantized 8-bit weights
                let quantized_path = model_dir.join("transformer/quantized_8bit.safetensors");
                if quantized_path.exists() {
                    tracing::info!("Loading FLUX.2-klein transformer (pre-quantized 8-bit)...");
                    let params = FluxKleinParams::default();
                    let raw_weights = load_safetensors(&quantized_path)?;
                    // Pre-quantized weights are already in our internal naming
                    let trans = load_quantized_flux_klein(raw_weights, 64, 8)
                        .map_err(|e| eyre::eyre!("Failed to load quantized FLUX transformer: {}", e))?;
                    tracing::info!("FLUX transformer loaded (8-bit quantized, ~4x memory savings)");
                    TransformerVariant::FluxQuantized(trans, params)
                } else {
                    tracing::info!("Loading FLUX.2-klein transformer (bf16)...");
                    let params = FluxKleinParams::default();
                    let mut trans = FluxKlein::new(params.clone())?;

                    let raw_weights = load_safetensors(&transformer_path)?;
                    let weights = sanitize_klein_model_weights(raw_weights);
                    let weights: HashMap<String, Array> = weights
                        .into_iter()
                        .map(|(k, v)| {
                            let v32 = v.as_type::<f32>().unwrap_or(v);
                            (k, v32)
                        })
                        .collect();

                    let weights_rc: HashMap<Rc<str>, Array> = weights
                        .into_iter()
                        .map(|(k, v)| (Rc::from(k.as_str()), v))
                        .collect();
                    trans.update_flattened(weights_rc);
                    tracing::info!("FLUX transformer loaded (bf16)");
                    TransformerVariant::Flux(trans, params)
                }
            }
            ImageModelType::ZImageTurbo => {
                tracing::info!("Loading Z-Image-Turbo transformer (4-bit quantized)...");
                let config = ZImageConfig::default();

                // Load quantized transformer weights
                let raw_weights = load_zimage_safetensors(&transformer_path)?;
                let trans = load_quantized_zimage_transformer(raw_weights, config.clone())
                    .map_err(|e| eyre::eyre!("Failed to load quantized Z-Image transformer: {}", e))?;

                tracing::info!("Z-Image transformer loaded (quantized)");
                TransformerVariant::ZImage(trans, config)
            }
            ImageModelType::QwenImage | ImageModelType::QwenImageEdit | ImageModelType::CosmosT2I | ImageModelType::FluxKleinGguf => {
                unreachable!("handled by subprocess engines")
            }
        };

        // Load VAE (decoder and encoder)
        tracing::info!("Loading VAE (z_channels={})...", vae_config.z_channels);

        // Load all VAE weights
        let all_vae_weights = load_safetensors(&vae_path)?;

        // Load decoder
        let mut vae_decoder = Decoder::new(vae_config.clone())?;
        let decoder_weights = sanitize_vae_weights(all_vae_weights.clone());
        let decoder_weights_rc: HashMap<Rc<str>, Array> = decoder_weights
            .into_iter()
            .map(|(k, v)| (Rc::from(k.as_str()), v))
            .collect();
        vae_decoder.update_flattened(decoder_weights_rc);
        tracing::info!("VAE decoder loaded");

        // Load encoder (for img2img support)
        let mut vae_encoder = Encoder::new(vae_config.clone())?;
        let encoder_weights = sanitize_vae_encoder_weights(all_vae_weights);
        let encoder_weight_count = encoder_weights.len();
        if encoder_weight_count > 0 {
            let encoder_weights_rc: HashMap<Rc<str>, Array> = encoder_weights
                .into_iter()
                .map(|(k, v)| (Rc::from(k.as_str()), v))
                .collect();
            vae_encoder.update_flattened(encoder_weights_rc);
            tracing::info!("VAE encoder loaded ({} weights) - img2img ready", encoder_weight_count);
        } else {
            tracing::warn!("VAE encoder weights not found - img2img will not work correctly");
        }

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| eyre::eyre!("Failed to load tokenizer: {}", e))?;

        tracing::info!("Image engine ready");

        Ok(Self {
            model_type,
            text_encoder: Some(text_encoder),
            transformer,
            vae_decoder: Some(vae_decoder),
            vae_encoder: Some(vae_encoder),
            tokenizer,
            vae_config,
            qwen_vae: None,
            model_dir: Some(model_dir.to_path_buf()),
            is_quantized_flux,
        })
    }

    /// Reload text encoder (called after it was dropped to free memory)
    fn reload_text_encoder(&mut self) -> Result<()> {
        let model_dir = self.model_dir.as_ref()
            .ok_or_else(|| eyre::eyre!("No model_dir stored for text encoder reload"))?;

        tracing::info!("Reloading text encoder from {:?}...", model_dir);
        let qwen3_config = Qwen3Config {
            hidden_size: 2560,
            num_hidden_layers: 36,
            intermediate_size: 9728,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            rms_norm_eps: 1e-6,
            vocab_size: 151936,
            max_position_embeddings: 40960,
            rope_theta: 1000000.0,
            head_dim: 128,
        };

        let mut encoder = Qwen3TextEncoder::new(qwen3_config)?;
        let te1 = model_dir.join("text_encoder/model-00001-of-00002.safetensors");
        let te2 = model_dir.join("text_encoder/model-00002-of-00002.safetensors");
        let mut all_weights = HashMap::new();
        for path in &[te1, te2] {
            let weights = load_safetensors(path)?;
            all_weights.extend(weights);
        }
        let weights = sanitize_qwen3_weights(all_weights);
        // Cast to f32 (bf16 crashes Metal, f16 causes NaN in forward pass)
        let weights: HashMap<String, Array> = weights
            .into_iter()
            .map(|(k, v)| {
                let v32 = v.as_type::<f32>().unwrap_or(v);
                (k, v32)
            })
            .collect();
        let weights_rc: HashMap<Rc<str>, Array> = weights
            .into_iter()
            .map(|(k, v)| (Rc::from(k.as_str()), v))
            .collect();
        encoder.update_flattened(weights_rc);
        self.text_encoder = Some(TextEncoderVariant::Standard(encoder));
        tracing::info!("FLUX text encoder reloaded");
        Ok(())
    }

    /// Generate images from a text prompt (with optional reference image for img2img)
    pub fn generate(&mut self, request: &ImageGenerationRequest) -> Result<ImageGenerationResponse> {
        tracing::debug!("Generate start");

        let (width, height) = parse_size(&request.size)?;

        // Check if we have a reference image for img2img
        let ref_latents = if let Some(ref image_b64) = request.image {
            tracing::info!("Processing reference image for img2img (strength={})", request.strength);
            Some(self.encode_reference_image(image_b64, width, height)?)
        } else {
            None
        };

        tracing::info!(
            "Image generation request: prompt='{}', size={}x{}, n={}, img2img={}",
            request.prompt,
            width,
            height,
            request.n,
            ref_latents.is_some()
        );

        let mut data = Vec::new();
        for i in 0..request.n {
            tracing::info!("Generating image {}/{}", i + 1, request.n);

            let image_bytes = if let Some(ref latents) = ref_latents {
                self.generate_img2img(&request.prompt, width, height, latents, request.strength)?
            } else {
                self.generate_single(&request.prompt, width, height)?
            };

            let b64 = base64::Engine::encode(
                &base64::engine::general_purpose::STANDARD,
                &image_bytes,
            );

            data.push(ImageData {
                url: None,
                b64_json: Some(b64),
                revised_prompt: Some(request.prompt.clone()),
            });
        }

        tracing::info!("Generate done ({} images)", data.len());

        Ok(ImageGenerationResponse {
            created: chrono::Utc::now().timestamp(),
            data,
        })
    }

    /// Encode a reference image to latents for img2img
    fn encode_reference_image(&mut self, image_b64: &str, width: u32, height: u32) -> Result<Array> {
        use base64::Engine;

        // Decode base64 image
        let image_bytes = base64::engine::general_purpose::STANDARD
            .decode(image_b64)
            .context("Failed to decode base64 image")?;

        // Parse image (PNG/JPEG) using simple decoder
        let img = image::load_from_memory(&image_bytes)
            .context("Failed to parse image")?
            .resize_exact(width, height, image::imageops::FilterType::Lanczos3);

        match self.model_type {
            ImageModelType::QwenImage => {
                // Qwen VAE expects [B, 4, H, W] (RGBA, channels-first, [-1,1])
                let rgba = img.to_rgba8();
                let pixels: Vec<f32> = rgba.pixels()
                    .flat_map(|p| p.0.iter().map(|&v| (v as f32 / 127.5) - 1.0))
                    .collect();
                // [1, H, W, 4] -> transpose to [1, 4, H, W]
                let input = Array::from_slice(&pixels, &[1, height as i32, width as i32, 4]);
                let input = input.transpose_axes(&[0, 3, 1, 2])
                    .map_err(|e| eyre::eyre!("Transpose failed: {e}"))?;

                let qwen_vae = self.qwen_vae.as_mut()
                    .ok_or_else(|| eyre::eyre!("Qwen VAE not loaded"))?;
                let latents = qwen_vae.encode(&input)
                    .map_err(|e| eyre::eyre!("Qwen VAE encode failed: {e}"))?;
                latents.eval().map_err(|e| eyre::eyre!("Eval failed: {e}"))?;
                tracing::debug!("Qwen VAE encoded ref image to latents: {:?}", latents.shape());
                Ok(latents)
            }
            _ => {
                let img = img.to_rgb8();
                // Convert to MLX array [1, H, W, 3] in range [-1, 1]
                let pixels: Vec<f32> = img.pixels()
                    .flat_map(|p| p.0.iter().map(|&v| (v as f32 / 127.5) - 1.0))
                    .collect();
                let input = Array::from_slice(&pixels, &[1, height as i32, width as i32, 3]);
                let vae_encoder = self.vae_encoder.as_mut()
                    .ok_or_else(|| eyre::eyre!("VAE encoder not available for this model type"))?;
                let latents = vae_encoder.encode_deterministic(&input)?;
                latents.eval()?;
                tracing::debug!("Encoded reference image to latents: {:?}", latents.shape());
                Ok(latents)
            }
        }
    }

    /// Generate image with img2img (starting from reference latents)
    fn generate_img2img(&mut self, prompt: &str, width: u32, height: u32, ref_latents: &Array, strength: f32) -> Result<Vec<u8>> {
        match self.model_type {
            ImageModelType::FluxKlein => self.generate_flux(prompt, width, height, Some(ref_latents), strength),
            ImageModelType::ZImageTurbo => self.generate_zimage(prompt, width, height, Some(ref_latents), strength),
            ImageModelType::QwenImage => self.generate_qwen_image_edit(prompt, width, height, ref_latents),
            ImageModelType::QwenImageEdit | ImageModelType::CosmosT2I | ImageModelType::FluxKleinGguf => {
                unreachable!("handled by subprocess engines")
            }
        }
    }

    /// Generate a single image
    fn generate_single(&mut self, prompt: &str, width: u32, height: u32) -> Result<Vec<u8>> {
        match self.model_type {
            ImageModelType::FluxKlein => self.generate_flux(prompt, width, height, None, 1.0),
            ImageModelType::ZImageTurbo => self.generate_zimage(prompt, width, height, None, 1.0),
            ImageModelType::QwenImage => self.generate_qwen_image(prompt, width, height),
            ImageModelType::QwenImageEdit | ImageModelType::CosmosT2I | ImageModelType::FluxKleinGguf => {
                unreachable!("handled by subprocess engines")
            }
        }
    }

    /// Tokenize a prompt using Qwen3 chat template
    fn tokenize_prompt(&self, prompt: &str) -> Result<(Array, Array)> {
        let batch_size = 1i32;
        let max_seq_len = 512i32;

        let chat_prompt = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            prompt
        );

        let encoding = self.tokenizer.encode(chat_prompt.as_str(), true)
            .map_err(|e| eyre::eyre!("Tokenization failed: {}", e))?;
        let ids: Vec<i32> = encoding.get_ids().iter().map(|&x| x as i32).collect();
        let num_tokens = ids.len().min(max_seq_len as usize);

        let mut padded = vec![151643i32; max_seq_len as usize];
        padded[..num_tokens].copy_from_slice(&ids[..num_tokens]);

        let mut mask = vec![0i32; max_seq_len as usize];
        for i in 0..num_tokens {
            mask[i] = 1;
        }

        let input_ids = Array::from_slice(&padded, &[batch_size, max_seq_len]);
        let attention_mask = Array::from_slice(&mask, &[batch_size, max_seq_len]);

        Ok((input_ids, attention_mask))
    }

    /// Convert VAE output to PNG bytes with FLUX-style normalization ([-1,1] → [0,255])
    fn vae_to_png_flux(&self, image: &Array) -> Result<Vec<u8>> {
        use mlx_rs::ops;

        let image = ops::add(image, &Array::from_slice(&[1.0f32], &[1]))?;
        let image = ops::multiply(&image, &Array::from_slice(&[127.5f32], &[1]))?;
        let image = ops::maximum(&image, &Array::from_slice(&[0.0f32], &[1]))?;
        let image = ops::minimum(&image, &Array::from_slice(&[255.0f32], &[1]))?;
        image.eval()?;

        let shape = image.shape();
        let out_height = shape[1] as usize;
        let out_width = shape[2] as usize;

        let image_flat = image.reshape(&[-1])?;
        let image_data: Vec<f32> = image_flat.as_slice().to_vec();
        let rgb_bytes: Vec<u8> = image_data.iter().map(|&v| v.round() as u8).collect();

        rgb_to_png(&rgb_bytes, out_width as u32, out_height as u32)
    }

    /// Convert VAE output to PNG bytes with Z-Image-style normalization (/2+0.5 → [0,255])
    fn vae_to_png_zimage(&self, image: &Array) -> Result<Vec<u8>> {
        use mlx_rs::ops;

        let image = ops::divide(image, &mlx_rs::array!(2.0f32))?;
        let image = ops::add(&image, &mlx_rs::array!(0.5f32))?;
        let image = ops::clip(&image, (0.0f32, 1.0f32))?;
        let image = ops::multiply(&image, &mlx_rs::array!(255.0f32))?;
        image.eval()?;

        let shape = image.shape();
        let out_height = shape[1] as usize;
        let out_width = shape[2] as usize;

        let image_flat = image.reshape(&[-1])?;
        let image_data: Vec<f32> = image_flat.as_slice().to_vec();
        let rgb_bytes: Vec<u8> = image_data.iter().map(|&v| v.round() as u8).collect();

        rgb_to_png(&rgb_bytes, out_width as u32, out_height as u32)
    }

    /// Unified FLUX.2-klein generation (txt2img and img2img)
    fn generate_flux(&mut self, prompt: &str, width: u32, height: u32, ref_latents: Option<&Array>, strength: f32) -> Result<Vec<u8>> {
        use mlx_rs::ops;

        let batch_size = 1i32;
        let num_steps = 4;

        // Tokenize before borrowing transformer (avoids overlapping borrow)
        let (input_ids, attention_mask) = self.tokenize_prompt(prompt)?;

        // Extract params (same for both quantized and non-quantized)
        let params = match &self.transformer {
            TransformerVariant::Flux(_, p) => p.clone(),
            TransformerVariant::FluxQuantized(_, p) => p.clone(),
            _ => return Err(eyre::eyre!("Expected FLUX transformer")),
        };

        let start_step = if ref_latents.is_some() {
            ((1.0 - strength) * num_steps as f32).round() as usize
        } else {
            0
        };

        // Encode text (reload text encoder if it was dropped for memory savings)
        if self.text_encoder.is_none() {
            self.reload_text_encoder()?;
        }
        tracing::info!("Encoding text prompt...");
        let txt_embed = match self.text_encoder.as_mut().unwrap() {
            TextEncoderVariant::Standard(enc) => enc.encode(&input_ids, Some(&attention_mask))?,
            TextEncoderVariant::Quantized(enc) => enc.encode_flux(&input_ids, Some(&attention_mask))?,
            TextEncoderVariant::QwenImage(_) => return Err(eyre::eyre!("Expected FLUX text encoder, got Qwen-Image encoder")),
        };
        let txt_embed = txt_embed.as_dtype(mlx_rs::Dtype::Float32)?;
        txt_embed.eval()?;
        tracing::info!("Text encoding complete, shape: {:?}", txt_embed.shape());

        // For quantized FLUX: drop text encoder to free ~7.5 GB before denoising
        if self.is_quantized_flux {
            tracing::info!("Dropping text encoder to free memory for denoising...");
            self.text_encoder = None;
            unsafe { mlx_sys::mlx_clear_cache(); }
        }

        // Setup latent dimensions
        let latent_height = height as i32 / 8;
        let latent_width = width as i32 / 8;
        let patch_size = 2i32;
        let patch_h = latent_height / patch_size;
        let patch_w = latent_width / patch_size;
        let img_seq_len = patch_h * patch_w;
        let in_channels = params.in_channels;
        let z_channels = self.vae_config.z_channels;
        let max_seq_len = 512i32;

        // Position IDs and RoPE
        let txt_ids = create_txt_ids(batch_size, max_seq_len)?;
        let img_ids = create_img_ids(batch_size, patch_h, patch_w)?;
        let (rope_cos, rope_sin) = FluxKlein::compute_rope(&txt_ids, &img_ids)?;

        let timesteps = flux_official_schedule(img_seq_len, num_steps);

        // Initialize latent (from noise or noised reference)
        let mut latent = if let Some(ref_lat) = ref_latents {
            let ref_lat = ref_lat.transpose_axes(&[0, 3, 1, 2])?;
            let ref_lat = ref_lat.reshape(&[batch_size, z_channels, patch_h, patch_size, patch_w, patch_size])?;
            let ref_lat = ref_lat.transpose_axes(&[0, 2, 4, 1, 3, 5])?;
            let ref_lat = ref_lat.reshape(&[batch_size, img_seq_len, in_channels])?;

            let t_start = timesteps[start_step];
            let noise = mlx_rs::random::normal::<f32>(&[batch_size, img_seq_len, in_channels], None, None, None)?;
            ops::add(
                &ops::multiply(&ref_lat, &Array::from_slice(&[1.0 - t_start], &[1]))?,
                &ops::multiply(&noise, &Array::from_slice(&[t_start], &[1]))?,
            )?
        } else {
            mlx_rs::random::normal::<f32>(&[batch_size, img_seq_len, in_channels], None, None, None)?
        };

        // Denoising loop
        tracing::info!("Running FLUX denoising ({} steps from step {})...", num_steps as usize - start_step, start_step);
        for step in start_step..num_steps as usize {
            let t_curr = timesteps[step];
            let t_next = timesteps[step + 1];
            let t_arr = Array::from_slice(&[t_curr * 1000.0], &[batch_size]);

            // Clear MLX cache between steps to free GPU memory
            unsafe { mlx_sys::mlx_clear_cache(); }

            let v_pred = match &mut self.transformer {
                TransformerVariant::Flux(t, _) => t.forward_with_rope(&latent, &txt_embed, &t_arr, &rope_cos, &rope_sin)?,
                TransformerVariant::FluxQuantized(t, _) => t.forward_with_rope(&latent, &txt_embed, &t_arr, &rope_cos, &rope_sin)?,
                _ => return Err(eyre::eyre!("Expected FLUX transformer")),
            };

            let dt = t_next - t_curr;
            let scaled_v = ops::multiply(&v_pred, &Array::from_slice(&[dt], &[1]))?;
            latent = ops::add(&latent, &scaled_v)?;
            latent.eval()
                .map_err(|e| eyre::eyre!("Denoising step eval failed: {e}"))?;

            tracing::info!("  Step {}/{}: t={:.3}->{:.3}", step + 1, num_steps, t_curr, t_next);
        }

        // Decode latents to image
        tracing::info!("Decoding latents...");
        let latent = latent.reshape(&[batch_size, patch_h, patch_w, z_channels, patch_size, patch_size])?;
        let latent = latent.transpose_axes(&[0, 1, 4, 2, 5, 3])?;
        let vae_height = patch_h * patch_size;
        let vae_width = patch_w * patch_size;
        let latent_for_vae = latent.reshape(&[batch_size, vae_height, vae_width, z_channels])?;

        let vae_decoder = self.vae_decoder.as_mut().expect("FLUX VAE decoder must be present");
        let image = vae_decoder.forward(&latent_for_vae)?;
        image.eval()
            .map_err(|e| eyre::eyre!("VAE decode eval failed: {e}"))?;
        tracing::info!("VAE decode complete, shape: {:?}", image.shape());

        self.vae_to_png_flux(&image)
    }

    // ── Qwen-Image ──────────────────────────────────────────────────────────

    /// Load Qwen-Image model from the resolved model directory.
    fn new_qwen_image(model_id: &str, model_type: ImageModelType, model_dir: std::path::PathBuf) -> Result<Self> {
        tracing::info!("Loading Qwen-Image from: {:?}", model_dir);

        // Detect quantization bits from model ID
        let lower = model_id.to_lowercase();
        let qwen_config = if lower.contains("8bit") {
            tracing::info!("Qwen-Image: using 8-bit quantized config");
            QwenConfig::with_8bit()
        } else {
            tracing::info!("Qwen-Image: using 4-bit quantized config");
            QwenConfig::default()
        };

        // Load text encoder (Qwen2.5-VL 7B, 28 layers, hidden=3584)
        tracing::info!("Loading Qwen-Image text encoder...");
        let text_encoder = load_text_encoder(&model_dir)
            .map_err(|e| eyre::eyre!("Failed to load Qwen-Image text encoder: {e}"))?;
        tracing::info!("Qwen-Image text encoder loaded");

        // Load transformer weights — auto-detect GGUF (single file) vs sharded safetensors
        tracing::info!("Loading Qwen-Image transformer ({}-bit)...", qwen_config.quantization_bits);
        let all_weights: HashMap<String, Array>;

        // Try GGUF first (edit model uses single .gguf file)
        let gguf_candidates: Vec<_> = std::fs::read_dir(&model_dir)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .filter(|e| {
                let name = e.file_name().to_string_lossy().to_lowercase();
                name.ends_with(".gguf") && (name.contains("edit") || name.contains("diffusion"))
            })
            .collect();

        if let Some(gguf_entry) = gguf_candidates.first() {
            let gguf_path = gguf_entry.path();
            tracing::info!("  Loading transformer from GGUF: {:?}", gguf_path);
            all_weights = Array::load_gguf(&gguf_path)
                .map_err(|e| eyre::eyre!("Failed to load GGUF: {e}"))?;
            tracing::info!("  GGUF loaded ({} tensors)", all_weights.len());
        } else {
            // Fall back to sharded safetensors
            let transformer_dir = model_dir.join("transformer");
            let index_path = transformer_dir.join("model.safetensors.index.json");
            if !index_path.exists() {
                return Err(eyre::eyre!(
                    "No GGUF or safetensors index found in {:?}. \
                     Expected *edit*.gguf or transformer/model.safetensors.index.json",
                    model_dir
                ));
            }
            let index_content = std::fs::read_to_string(&index_path)
                .context("Failed to read transformer index")?;
            let index: serde_json::Value = serde_json::from_str(&index_content)
                .context("Failed to parse transformer index")?;
            let weight_map = index["weight_map"].as_object()
                .ok_or_else(|| eyre::eyre!("Invalid transformer index format"))?;
            let mut shard_files: std::collections::HashSet<String> = std::collections::HashSet::new();
            for file in weight_map.values() {
                if let Some(f) = file.as_str() {
                    shard_files.insert(f.to_string());
                }
            }
            let mut weights = HashMap::new();
            for shard in &shard_files {
                let shard_path = transformer_dir.join(shard);
                tracing::info!("  Loading transformer shard: {}", shard);
                let w = load_safetensors(&shard_path)
                    .map_err(|e| eyre::eyre!("Failed to load shard {}: {}", shard, e))?;
                weights.extend(w);
            }
            all_weights = weights;
        }

        let mut transformer = QwenQuantizedTransformer::new(qwen_config.clone())
            .map_err(|e| eyre::eyre!("Failed to create Qwen-Image transformer: {e}"))?;
        load_transformer_weights(&mut transformer, all_weights)
            .map_err(|e| eyre::eyre!("Failed to load Qwen-Image transformer weights: {e}"))?;
        tracing::info!("Qwen-Image transformer loaded ({} blocks)", qwen_config.num_layers);

        // Load VAE
        tracing::info!("Loading Qwen-Image VAE...");
        let qwen_vae = load_vae_from_dir(&model_dir)
            .map_err(|e| eyre::eyre!("Failed to load Qwen-Image VAE: {e}"))?;
        tracing::info!("Qwen-Image VAE loaded");

        // Load tokenizer from tokenizer/tokenizer.json
        let tok_path = model_dir.join("tokenizer/tokenizer.json");
        if !tok_path.exists() {
            return Err(eyre::eyre!("Qwen-Image tokenizer not found at {:?}", tok_path));
        }
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| eyre::eyre!("Failed to load Qwen-Image tokenizer: {e}"))?;
        tracing::info!("Qwen-Image tokenizer loaded");

        tracing::info!("Qwen-Image engine ready");

        Ok(Self {
            model_type,
            text_encoder: Some(TextEncoderVariant::QwenImage(text_encoder)),
            transformer: TransformerVariant::QwenImage(transformer, qwen_config),
            vae_decoder: None,
            vae_encoder: None,
            tokenizer,
            vae_config: AutoEncoderConfig::flux2(), // placeholder, unused for QwenImage
            qwen_vae: Some(qwen_vae),
            model_dir: Some(model_dir),
            is_quantized_flux: false,
        })
    }

    /// Generate an image using the Qwen-Image flow-matching pipeline.
    fn generate_qwen_image(&mut self, prompt: &str, width: u32, height: u32) -> Result<Vec<u8>> {
        use mlx_rs::ops;

        let batch = 1i32;
        let num_steps = 20i32;
        let latent_channels = 16i32;
        let frames = 1i32;
        let latent_h = height as i32 / 8;
        let latent_w = width as i32 / 8;
        let patch_size = 2i32;

        // Tokenize prompt with Qwen3 chat template
        let chat_prompt = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            prompt
        );
        let encoding = self.tokenizer.encode(chat_prompt.as_str(), true)
            .map_err(|e| eyre::eyre!("Qwen-Image tokenization failed: {e}"))?;
        let ids: Vec<i32> = encoding.get_ids().iter().map(|&x| x as i32).collect();
        let seq_len = ids.len() as i32;
        let input_ids = Array::from_slice(&ids, &[batch, seq_len]);

        // Encode text
        tracing::info!("Encoding text prompt ({} tokens)...", seq_len);
        let encoder_hidden_states = match self.text_encoder.as_mut() {
            Some(TextEncoderVariant::QwenImage(enc)) => {
                enc.forward(&input_ids)
                    .map_err(|e| eyre::eyre!("Qwen-Image text encoding failed: {e}"))?
            }
            _ => return Err(eyre::eyre!("Expected QwenImage text encoder")),
        };
        let encoder_hidden_states = encoder_hidden_states.as_dtype(mlx_rs::Dtype::Float32)
            .map_err(|e| eyre::eyre!("Failed to cast text embeddings: {e}"))?;
        encoder_hidden_states.eval()
            .map_err(|e| eyre::eyre!("Failed to eval text embeddings: {e}"))?;
        tracing::info!("Text encoding complete, shape: {:?}", encoder_hidden_states.shape());

        // Flow-matching scheduler
        let scheduler = FlowMatchEulerScheduler::new(num_steps, 3.0);

        // Initialize noise latents: [B, C, F, H, W]
        let noise = mlx_rs::random::normal::<f32>(
            &[batch, latent_channels, frames, latent_h, latent_w], None, None, None,
        ).map_err(|e| eyre::eyre!("Failed to init noise: {e}"))?;
        let mut latents = scheduler.scale_noise(&noise)
            .map_err(|e| eyre::eyre!("Failed to scale noise: {e}"))?;

        // Denoising loop
        tracing::info!("Running Qwen-Image denoising ({} steps)...", num_steps);
        for (idx, &t) in scheduler.timesteps().iter().enumerate() {
            let timestep = Array::from_slice(&[t * 1000.0], &[batch]);

            unsafe { mlx_sys::mlx_clear_cache(); }

            // Patchify: [B, C, F, H, W] -> [B, F*pH*pW, C*p*p]
            let hidden_states = qwen_patchify(&latents, patch_size)
                .map_err(|e| eyre::eyre!("Patchify failed: {e}"))?;

            // Forward through quantized transformer
            let v_pred_patches = match &mut self.transformer {
                TransformerVariant::QwenImage(t, _) => {
                    t.forward(&hidden_states, &encoder_hidden_states, &timestep, None, None, None)
                        .map_err(|e| eyre::eyre!("Qwen-Image transformer forward failed: {e}"))?
                }
                _ => return Err(eyre::eyre!("Expected QwenImage transformer")),
            };

            // Unpatchify: [B, seq, C*p*p] -> [B, C, F, H, W]
            let v_pred = qwen_unpatchify(&v_pred_patches, frames, latent_h, latent_w, patch_size)
                .map_err(|e| eyre::eyre!("Unpatchify failed: {e}"))?;

            // Euler step
            latents = scheduler.step(&v_pred, idx, &latents)
                .map_err(|e| eyre::eyre!("Scheduler step failed: {e}"))?;
            latents.eval()
                .map_err(|e| eyre::eyre!("Eval after step failed: {e}"))?;

            tracing::info!("  Step {}/{}: t={:.3}", idx + 1, num_steps, t);
        }

        // Collapse frame dim: [B, C, F, H, W] -> [B, C, H, W]
        let latents_2d = {
            use mlx_rs::ops::indexing::IndexOp;
            let indexed = latents.index((.., .., 0i32, .., ..));
            let numel = indexed.shape().iter().map(|&d| d as usize).product::<usize>();
            let flat = indexed.reshape(&[numel as i32])
                .map_err(|e| eyre::eyre!("Reshape failed: {e}"))?;
            flat.reshape(&[batch, latent_channels, latent_h, latent_w])
                .map_err(|e| eyre::eyre!("Reshape to 2d failed: {e}"))?
        };

        // Decode with QwenVAE: output [B, 3, H, W]
        tracing::info!("Decoding Qwen-Image latents...");
        let image = self.qwen_vae.as_mut()
            .ok_or_else(|| eyre::eyre!("Qwen VAE not loaded"))?
            .decode(&latents_2d)
            .map_err(|e| eyre::eyre!("Qwen-Image VAE decode failed: {e}"))?;
        image.eval().map_err(|e| eyre::eyre!("Eval after decode failed: {e}"))?;
        tracing::info!("Qwen-Image decode complete, shape: {:?}", image.shape());

        // Convert [B, 3, H, W] in [-1, 1] range to PNG
        let image = ops::add(&image, &Array::from_slice(&[1.0f32], &[1]))
            .map_err(|e| eyre::eyre!("Image post-proc failed: {e}"))?;
        let image = ops::multiply(&image, &Array::from_slice(&[127.5f32], &[1]))
            .map_err(|e| eyre::eyre!("Image scale failed: {e}"))?;
        let image = ops::maximum(&image, &Array::from_slice(&[0.0f32], &[1]))
            .map_err(|e| eyre::eyre!("Image clip min failed: {e}"))?;
        let image = ops::minimum(&image, &Array::from_slice(&[255.0f32], &[1]))
            .map_err(|e| eyre::eyre!("Image clip max failed: {e}"))?;
        // Transpose [B, 3, H, W] -> [B, H, W, 3] for row-major pixel output
        let image = image.transpose_axes(&[0, 2, 3, 1])
            .map_err(|e| eyre::eyre!("Image transpose failed: {e}"))?;
        image.eval().map_err(|e| eyre::eyre!("Image eval failed: {e}"))?;

        let shape = image.shape();
        let out_height = shape[1] as u32;
        let out_width = shape[2] as u32;
        let image_flat = image.reshape(&[-1])
            .map_err(|e| eyre::eyre!("Flatten failed: {e}"))?;
        let image_data: Vec<f32> = image_flat.as_slice().to_vec();
        let rgb_bytes: Vec<u8> = image_data.iter().map(|&v| v.round() as u8).collect();

        rgb_to_png(&rgb_bytes, out_width, out_height)
    }

    /// Generate an edited image using Qwen-Image-Edit with reference latents.
    /// ref_latents: VAE-encoded reference image [1, 16, H/8, W/8] (already normalized)
    /// P0: CFG batched — cond+uncond run as batch=2 in a single forward, halving
    /// weight-bandwidth pressure on the Q4_K_M projections.
    fn generate_qwen_image_edit(
        &mut self,
        prompt: &str,
        width: u32,
        height: u32,
        ref_latents: &Array,
    ) -> Result<Vec<u8>> {
        use mlx_rs::ops;

        let batch = 1i32;
        let num_steps = 20i32;
        let cfg_scale = 2.5f32;
        let latent_channels = 16i32;
        let latent_h = height as i32 / 8;
        let latent_w = width as i32 / 8;

        tracing::info!("Encoding text prompts for CFG...");
        let (pos_embeds, neg_embeds) = match self.text_encoder.as_mut() {
            Some(TextEncoderVariant::QwenImage(enc)) => {
                // Positive prompt
                let chat_pos = format!(
                    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                    prompt
                );
                let enc_pos = self.tokenizer.encode(chat_pos.as_str(), true)
                    .map_err(|e| eyre::eyre!("Tokenization failed: {e}"))?;
                let ids_pos: Vec<i32> = enc_pos.get_ids().iter().map(|&x| x as i32).collect();
                let input_pos = Array::from_slice(&ids_pos, &[batch, ids_pos.len() as i32]);
                let pos = enc.forward(&input_pos)
                    .map_err(|e| eyre::eyre!("Text encoding failed: {e}"))?
                    .as_dtype(mlx_rs::Dtype::Float32)
                    .map_err(|e| eyre::eyre!("Cast failed: {e}"))?;
                pos.eval().map_err(|e| eyre::eyre!("Eval failed: {e}"))?;

                // Negative prompt (empty string)
                let chat_neg = "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
                let enc_neg = self.tokenizer.encode(chat_neg, true)
                    .map_err(|e| eyre::eyre!("Tokenization failed: {e}"))?;
                let ids_neg: Vec<i32> = enc_neg.get_ids().iter().map(|&x| x as i32).collect();
                let input_neg = Array::from_slice(&ids_neg, &[batch, ids_neg.len() as i32]);
                let neg = enc.forward(&input_neg)
                    .map_err(|e| eyre::eyre!("Text encoding failed: {e}"))?
                    .as_dtype(mlx_rs::Dtype::Float32)
                    .map_err(|e| eyre::eyre!("Cast failed: {e}"))?;
                neg.eval().map_err(|e| eyre::eyre!("Eval failed: {e}"))?;

                (pos, neg)
            }
            _ => return Err(eyre::eyre!("Expected QwenImage text encoder")),
        };

        // P0: Pad cond/uncond to same length, stack as batch=2
        let t_pos = pos_embeds.dim(1);
        let t_neg = neg_embeds.dim(1);
        let t_max = t_pos.max(t_neg);
        let d = pos_embeds.dim(2);
        let pos_padded = if t_pos < t_max {
            let pad = Array::zeros::<f32>(&[1, t_max - t_pos, d])
                .map_err(|e| eyre::eyre!("Pad alloc failed: {e}"))?;
            ops::concatenate_axis(&[&pos_embeds, &pad], 1)
                .map_err(|e| eyre::eyre!("Concat failed: {e}"))?
        } else {
            pos_embeds.clone()
        };
        let neg_padded = if t_neg < t_max {
            let pad = Array::zeros::<f32>(&[1, t_max - t_neg, d])
                .map_err(|e| eyre::eyre!("Pad alloc failed: {e}"))?;
            ops::concatenate_axis(&[&neg_embeds, &pad], 1)
                .map_err(|e| eyre::eyre!("Concat failed: {e}"))?
        } else {
            neg_embeds.clone()
        };
        let batched_prompt = ops::concatenate_axis(&[&pos_padded, &neg_padded], 0)
            .map_err(|e| eyre::eyre!("Batch concat failed: {e}"))?; // [2, T, D]

        // Build txt padding mask [2, T]: 1=real, 0=pad
        let pos_mask_vals: Vec<f32> = (0..t_max).map(|i| if i < t_pos { 1.0 } else { 0.0 }).collect();
        let neg_mask_vals: Vec<f32> = (0..t_max).map(|i| if i < t_neg { 1.0 } else { 0.0 }).collect();
        let pos_mask = Array::from_slice(&pos_mask_vals, &[1, t_max]);
        let neg_mask = Array::from_slice(&neg_mask_vals, &[1, t_max]);
        let txt_pad_mask = ops::concatenate_axis(&[&pos_mask, &neg_mask], 0)
            .map_err(|e| eyre::eyre!("Mask concat failed: {e}"))?; // [2, T]

        tracing::info!("CFG batched: pos_len={}, neg_len={}, padded={}, cfg_scale={}", t_pos, t_neg, t_max, cfg_scale);

        // Pack reference latents: [1, 16, H/8, W/8] -> [1, ref_seq, 64]
        let ref_latent_h = ref_latents.dim(2);
        let ref_latent_w = ref_latents.dim(3);
        let ref_5d = ref_latents.reshape(&[batch, latent_channels, 1, ref_latent_h, ref_latent_w])
            .map_err(|e| eyre::eyre!("Ref reshape failed: {e}"))?;
        let ref_packed = pack_latents(&ref_5d)
            .map_err(|e| eyre::eyre!("Ref pack failed: {e}"))?;
        ref_packed.eval().map_err(|e| eyre::eyre!("Eval failed: {e}"))?;
        tracing::info!("Ref latent packed: {:?}", ref_packed.shape());

        let ref_shape = ref_shape_from_latent(ref_latent_h, ref_latent_w);

        // Build edit RoPE (uses max of pos/neg seq lengths for text)
        let img_shape = (1i32, latent_h / 2, latent_w / 2);
        let ((img_cos, img_sin), (txt_cos, txt_sin)) = build_edit_rope(
            img_shape,
            &[ref_shape],
            t_max,
            10000.0,
            [16, 56, 56],
        ).map_err(|e| eyre::eyre!("Edit RoPE build failed: {e}"))?;

        // Flow-matching scheduler
        let scheduler = FlowMatchEulerScheduler::new(num_steps, 3.0);

        // Initialize noise and pack: [1, C, 1, H, W] -> [1, main_seq, 64]
        let noise = mlx_rs::random::normal::<f32>(
            &[batch, latent_channels, 1, latent_h, latent_w], None, None, None,
        ).map_err(|e| eyre::eyre!("Noise init failed: {e}"))?;
        let noise_packed = pack_latents(&noise)
            .map_err(|e| eyre::eyre!("Pack failed: {e}"))?;
        let mut latents = scheduler.scale_noise(&noise_packed)
            .map_err(|e| eyre::eyre!("Scale noise failed: {e}"))?;

        let main_seq = latents.dim(1);
        let packed_dim = latents.dim(2);

        // Denoising loop with batched CFG
        tracing::info!("Running Qwen-Image edit denoising ({} steps, CFG batched)...", num_steps);
        for (idx, &t) in scheduler.timesteps().iter().enumerate() {
            let timestep = Array::from_slice(&[t * 1000.0], &[batch]);

            unsafe { mlx_sys::mlx_clear_cache(); }

            // Broadcast latents to batch=2 for CFG
            let latents_b = ops::broadcast_to(&latents, &[2, main_seq, packed_dim])
                .map_err(|e| eyre::eyre!("Broadcast failed: {e}"))?;

            // Single forward for cond+uncond
            let both_pred = match &mut self.transformer {
                TransformerVariant::QwenImage(transformer, _) => {
                    transformer.forward_edit(
                        &latents_b,
                        &batched_prompt,
                        &timestep,
                        &[&ref_packed],
                        (&img_cos, &img_sin),
                        (&txt_cos, &txt_sin),
                        Some(&txt_pad_mask),
                    ).map_err(|e| eyre::eyre!("Edit forward failed: {e}"))?
                }
                _ => return Err(eyre::eyre!("Expected QwenImage transformer")),
            };

            // Split batch: cond=[0:1], uncond=[1:2]
            let cond_pred = both_pred.index((0..1, .., ..));
            let uncond_pred = both_pred.index((1..2, .., ..));

            // CFG: combined = uncond + cfg_scale * (cond - uncond)
            let diff = ops::subtract(&cond_pred, &uncond_pred)
                .map_err(|e| eyre::eyre!("CFG diff failed: {e}"))?;
            let scaled = ops::multiply(&diff, &Array::from_f32(cfg_scale))
                .map_err(|e| eyre::eyre!("CFG scale failed: {e}"))?;
            let v_pred = ops::add(&uncond_pred, &scaled)
                .map_err(|e| eyre::eyre!("CFG add failed: {e}"))?;

            // Euler step (already in packed space)
            latents = scheduler.step(&v_pred, idx, &latents)
                .map_err(|e| eyre::eyre!("Scheduler step failed: {e}"))?;
            latents.eval().map_err(|e| eyre::eyre!("Eval failed: {e}"))?;

            tracing::info!("  Step {}/{}: t={:.3}", idx + 1, num_steps, t);
        }

        // Unpack: [B, seq, 64] -> [B, 16, 1, H, W] -> [B, 16, H, W]
        let latents_5d = unpack_latents(&latents, width as i32, height as i32)
            .map_err(|e| eyre::eyre!("Unpack failed: {e}"))?;
        let latents_2d = {
            use mlx_rs::ops::indexing::IndexOp;
            let indexed = latents_5d.index((.., .., 0i32, .., ..));
            let numel = indexed.shape().iter().map(|&d| d as usize).product::<usize>();
            let flat = indexed.reshape(&[numel as i32])
                .map_err(|e| eyre::eyre!("Reshape failed: {e}"))?;
            flat.reshape(&[batch, latent_channels, latent_h, latent_w])
                .map_err(|e| eyre::eyre!("Reshape to 2d failed: {e}"))?
        };

        // Denormalize and decode
        let latents_2d = QwenVAE::denormalize_latent(&latents_2d)
            .map_err(|e| eyre::eyre!("Denormalize failed: {e}"))?;

        tracing::info!("Decoding Qwen-Image edit result...");
        let image = self.qwen_vae.as_mut()
            .ok_or_else(|| eyre::eyre!("Qwen VAE not loaded"))?
            .decode(&latents_2d)
            .map_err(|e| eyre::eyre!("VAE decode failed: {e}"))?;
        image.eval().map_err(|e| eyre::eyre!("Eval failed: {e}"))?;

        // Convert [B, 3, H, W] in [-1, 1] to PNG
        let image = ops::add(&image, &Array::from_slice(&[1.0f32], &[1]))
            .map_err(|e| eyre::eyre!("Post-proc failed: {e}"))?;
        let image = ops::multiply(&image, &Array::from_slice(&[127.5f32], &[1]))
            .map_err(|e| eyre::eyre!("Scale failed: {e}"))?;
        let image = ops::maximum(&image, &Array::from_slice(&[0.0f32], &[1]))
            .map_err(|e| eyre::eyre!("Clip min failed: {e}"))?;
        let image = ops::minimum(&image, &Array::from_slice(&[255.0f32], &[1]))
            .map_err(|e| eyre::eyre!("Clip max failed: {e}"))?;
        let image = image.transpose_axes(&[0, 2, 3, 1])
            .map_err(|e| eyre::eyre!("Transpose failed: {e}"))?;
        image.eval().map_err(|e| eyre::eyre!("Eval failed: {e}"))?;

        let shape = image.shape();
        let out_height = shape[1] as u32;
        let out_width = shape[2] as u32;
        let image_flat = image.reshape(&[-1])
            .map_err(|e| eyre::eyre!("Flatten failed: {e}"))?;
        let image_data: Vec<f32> = image_flat.as_slice().to_vec();
        let rgb_bytes: Vec<u8> = image_data.iter().map(|&v| v.round() as u8).collect();

        rgb_to_png(&rgb_bytes, out_width, out_height)
    }

    /// Unified Z-Image-Turbo generation (txt2img and img2img)
    fn generate_zimage(&mut self, prompt: &str, width: u32, height: u32, ref_latents: Option<&Array>, strength: f32) -> Result<Vec<u8>> {
        use mlx_rs::ops;

        let batch_size = 1i32;
        let num_steps = 9;

        let zimage_config = match &self.transformer {
            TransformerVariant::ZImage(_, c) => c.clone(),
            _ => return Err(eyre::eyre!("Expected Z-Image transformer")),
        };

        let start_step = if ref_latents.is_some() {
            ((1.0 - strength) * num_steps as f32).round() as usize
        } else {
            0
        };

        // Tokenize and encode text
        let (input_ids, attention_mask) = self.tokenize_prompt(prompt)?;
        if self.text_encoder.is_none() {
            self.reload_text_encoder()?;
        }
        tracing::debug!("Encoding text prompt (Z-Image style)...");
        let txt_embed = match self.text_encoder.as_mut().unwrap() {
            TextEncoderVariant::Standard(enc) => enc.encode_zimage(&input_ids, Some(&attention_mask))?,
            TextEncoderVariant::Quantized(enc) => enc.encode_zimage(&input_ids, Some(&attention_mask))?,
            TextEncoderVariant::QwenImage(_) => return Err(eyre::eyre!("Expected ZImage text encoder, got Qwen-Image encoder")),
        };
        let txt_embed = txt_embed.as_dtype(mlx_rs::Dtype::Float32)?;
        txt_embed.eval()?;

        // Pad caption to multiple of 32
        let cap_len = txt_embed.dim(1) as i32;
        let pad_to = ((cap_len + 31) / 32) * 32;
        let txt_embed = if pad_to > cap_len {
            let last_token = txt_embed.index((.., (cap_len - 1)..));
            let padding = Array::repeat_axis::<f32>(last_token, pad_to - cap_len, 1)?;
            ops::concatenate_axis(&[&txt_embed, &padding], 1)?
        } else {
            txt_embed
        };
        let cap_len = txt_embed.dim(1) as i32;

        // Setup latent dimensions
        let latent_height = height as i32 / 8;
        let latent_width = width as i32 / 8;
        let patch_size = zimage_config.patch_size;
        let h_tok = latent_height / patch_size;
        let w_tok = latent_width / patch_size;
        let img_seq_len = h_tok * w_tok;
        let in_channels = zimage_config.in_channels;

        // Position encodings
        let img_pos = create_coordinate_grid((1, h_tok, w_tok), (cap_len + 1, 0, 0))?;
        let img_pos = img_pos.reshape(&[1, img_seq_len, 3])?;
        let cap_pos = create_coordinate_grid((cap_len, 1, 1), (1, 0, 0))?;
        let cap_pos = cap_pos.reshape(&[1, cap_len, 3])?;

        let (cos, sin) = match &mut self.transformer {
            TransformerVariant::ZImage(trans, _) => trans.compute_rope(&img_pos, &cap_pos)?,
            _ => return Err(eyre::eyre!("Expected Z-Image transformer")),
        };

        let mu = calculate_shift(img_seq_len, 256, 4096, 0.5, 1.15);
        let sigmas = generate_sigmas(num_steps as i32, mu);

        // Initialize latent (from noise or noised reference)
        let mut latents = if let Some(ref_lat) = ref_latents {
            let mut lat = ref_lat.transpose_axes(&[0, 3, 1, 2])?;
            let sigma_start = sigmas[start_step];
            let noise = mlx_rs::random::normal::<f32>(&[batch_size, in_channels, latent_height, latent_width], None, None, None)?;
            lat = ops::add(
                &ops::multiply(&lat, &Array::from_slice(&[1.0 - sigma_start], &[1]))?,
                &ops::multiply(&noise, &Array::from_slice(&[sigma_start], &[1]))?,
            )?;
            lat
        } else {
            mlx_rs::random::normal::<f32>(&[batch_size, in_channels, latent_height, latent_width], None, None, None)?
        };

        // Denoising loop
        tracing::debug!("Running Z-Image denoising ({} steps from step {})...", num_steps as usize - start_step, start_step);
        for step in start_step..num_steps as usize {
            let sigma_curr = sigmas[step];
            let sigma_next = sigmas[step + 1];
            let t = Array::from_slice(&[1.0 - sigma_curr], &[1]);

            let latents_patched = patchify(&latents, h_tok, w_tok, in_channels)?;

            let model_out = match &mut self.transformer {
                TransformerVariant::ZImage(trans, _) => trans.forward_with_rope(
                    &latents_patched, &t, &txt_embed, &img_pos, &cap_pos, &cos, &sin, None, None
                )?,
                _ => return Err(eyre::eyre!("Expected Z-Image transformer")),
            };

            let noise_pred = unpatchify(&model_out, h_tok, w_tok, in_channels)?;
            let noise_pred = ops::negative(&noise_pred)?;

            let dt = sigma_next - sigma_curr;
            let scaled_noise = ops::multiply(&noise_pred, &Array::from_slice(&[dt], &[1]))?;
            latents = ops::add(&latents, &scaled_noise)?;
            latents.eval()?;

            tracing::debug!("  Step {}/{}: sigma={:.3}->{:.3}", step + 1, num_steps, sigma_curr, sigma_next);
        }

        // Decode: [B, C, H, W] → [B, H, W, C]
        tracing::debug!("Decoding latents...");
        let latents = latents.transpose_axes(&[0, 2, 3, 1])?;

        let vae_decoder = self.vae_decoder.as_mut().expect("Z-Image VAE decoder must be present");
        let image = vae_decoder.forward(&latents)?;
        image.eval()?;

        self.vae_to_png_zimage(&image)
    }
}

// ── Qwen-Image helpers ──────────────────────────────────────────────────────

/// Patchify latents for Qwen-Image: [B, C, F, H, W] -> [B, F*pH*pW, C*p*p]
fn qwen_patchify(x: &Array, patch_size: i32) -> Result<Array, mlx_rs::error::Exception> {
    let batch = x.dim(0);
    let channels = x.dim(1);
    let frames = x.dim(2);
    let height = x.dim(3);
    let width = x.dim(4);
    let p = patch_size;
    let patch_h = height / p;
    let patch_w = width / p;

    // [B, C, F, H, W] -> [B, C, F, pH, p, pW, p]
    let x = x.reshape(&[batch, channels, frames, patch_h, p, patch_w, p])?;
    // -> [B, F, pH, pW, C, p, p]
    let x = x.transpose_axes(&[0, 2, 3, 5, 1, 4, 6])?;
    // -> [B, F*pH*pW, C*p*p]
    let num_patches = frames * patch_h * patch_w;
    let patch_dim = channels * p * p;
    x.reshape(&[batch, num_patches, patch_dim])
}

/// Unpatchify for Qwen-Image: [B, seq, out_ch*p*p] -> [B, out_ch, F, H, W]
fn qwen_unpatchify(
    x: &Array,
    frames: i32,
    height: i32,
    width: i32,
    patch_size: i32,
) -> Result<Array, mlx_rs::error::Exception> {
    let batch = x.dim(0);
    let patch_dim = x.dim(2);
    let p = patch_size;
    let out_channels = patch_dim / (p * p);
    let patch_h = height / p;
    let patch_w = width / p;

    // [B, F*pH*pW, C*p*p] -> [B, F, pH, pW, C, p, p]
    let x = x.reshape(&[batch, frames, patch_h, patch_w, out_channels, p, p])?;
    // -> [B, C, F, pH, p, pW, p]
    let x = x.transpose_axes(&[0, 4, 1, 2, 5, 3, 6])?;
    // -> [B, C, F, H, W]
    x.reshape(&[batch, out_channels, frames, height, width])
}

/// Maximum allowed image dimension
const MAX_IMAGE_DIM: u32 = 2048;
/// Minimum allowed image dimension
const MIN_IMAGE_DIM: u32 = 64;

/// Parse size string like "512x512" into (width, height) with validation
fn parse_size(size: &str) -> Result<(u32, u32)> {
    let parts: Vec<&str> = size.split('x').collect();
    if parts.len() != 2 {
        return Err(eyre::eyre!("Invalid size format: {}. Expected WIDTHxHEIGHT (e.g., 512x512)", size));
    }

    let width: u32 = parts[0].parse().context("Invalid width")?;
    let height: u32 = parts[1].parse().context("Invalid height")?;

    // Validate dimensions to prevent OOM
    if width < MIN_IMAGE_DIM || width > MAX_IMAGE_DIM {
        return Err(eyre::eyre!("Width must be between {} and {}, got {}", MIN_IMAGE_DIM, MAX_IMAGE_DIM, width));
    }
    if height < MIN_IMAGE_DIM || height > MAX_IMAGE_DIM {
        return Err(eyre::eyre!("Height must be between {} and {}, got {}", MIN_IMAGE_DIM, MAX_IMAGE_DIM, height));
    }

    // Ensure dimensions are multiples of 8 (required for VAE)
    if width % 8 != 0 || height % 8 != 0 {
        return Err(eyre::eyre!("Width and height must be multiples of 8, got {}x{}", width, height));
    }

    Ok((width, height))
}

/// Convert RGB bytes to PNG format
fn rgb_to_png(rgb_bytes: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    use image::{ImageBuffer, Rgb, ImageEncoder};

    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(width, height, rgb_bytes.to_vec())
        .ok_or_else(|| eyre::eyre!("Failed to create image buffer"))?;

    let mut png_data = Vec::new();
    let encoder = image::codecs::png::PngEncoder::new(&mut png_data);
    encoder.write_image(
        img.as_raw(),
        width,
        height,
        image::ExtendedColorType::Rgb8,
    ).context("Failed to encode PNG")?;

    Ok(png_data)
}

/// Create image position IDs for 4D RoPE
fn create_img_ids(batch: i32, h: i32, w: i32) -> Result<Array, mlx_rs::error::Exception> {
    let mut ids = Vec::with_capacity((batch * h * w * 4) as usize);

    for _ in 0..batch {
        for y in 0..h {
            for x in 0..w {
                ids.push(0.0f32);      // T position = 0 for images
                ids.push(y as f32);    // H position
                ids.push(x as f32);    // W position
                ids.push(0.0f32);      // Extra dim = 0
            }
        }
    }

    Ok(Array::from_slice(&ids, &[batch, h * w, 4]))
}

/// Create text position IDs for 4D RoPE
fn create_txt_ids(batch: i32, seq_len: i32) -> Result<Array, mlx_rs::error::Exception> {
    let mut ids = Vec::with_capacity((batch * seq_len * 4) as usize);

    for _ in 0..batch {
        for s in 0..seq_len {
            ids.push(0.0f32);      // T position = 0 for text
            ids.push(0.0f32);      // H position = 0 for text
            ids.push(0.0f32);      // W position = 0 for text
            ids.push(s as f32);    // L position = sequence index
        }
    }

    Ok(Array::from_slice(&ids, &[batch, seq_len, 4]))
}

/// Compute empirical mu for SNR shift
fn compute_empirical_mu(image_seq_len: i32, num_steps: i32) -> f32 {
    const A1: f32 = 8.73809524e-05;
    const B1: f32 = 1.89833333;
    const A2: f32 = 0.00016927;
    const B2: f32 = 0.45666666;

    if image_seq_len > 4300 {
        return A2 * (image_seq_len as f32) + B2;
    }

    let m_200 = A2 * (image_seq_len as f32) + B2;
    let m_10 = A1 * (image_seq_len as f32) + B1;
    let a = (m_200 - m_10) / 190.0;
    let b = m_200 - 200.0 * a;
    a * (num_steps as f32) + b
}

/// Apply generalized time SNR shift
fn generalized_time_snr_shift(t: f32, mu: f32, sigma: f32) -> f32 {
    if t <= 0.0 {
        return 0.0;
    }
    if t >= 1.0 {
        return 1.0;
    }
    mu.exp() / (mu.exp() + (1.0 / t - 1.0).powf(sigma))
}

/// Generate the official FLUX timestep schedule with SNR shift
fn flux_official_schedule(image_seq_len: i32, num_steps: i32) -> Vec<f32> {
    let mu = compute_empirical_mu(image_seq_len, num_steps);
    let sigma = 1.0f32;

    let mut timesteps = Vec::with_capacity((num_steps + 1) as usize);
    for i in 0..=num_steps {
        let t_linear = 1.0 - (i as f32) / (num_steps as f32);
        let t_shifted = generalized_time_snr_shift(t_linear, mu, sigma);
        timesteps.push(t_shifted);
    }
    timesteps
}

// ============================================================================
// Z-Image Helper Functions
// ============================================================================

/// Calculate dynamic time shift for Z-Image
fn calculate_shift(
    image_seq_len: i32,
    base_seq_len: i32,
    max_seq_len: i32,
    base_shift: f32,
    max_shift: f32,
) -> f32 {
    let m = (max_shift - base_shift) / ((max_seq_len - base_seq_len) as f32);
    let b = base_shift - m * (base_seq_len as f32);
    (image_seq_len as f32) * m + b
}

/// Generate sigma schedule for Z-Image
fn generate_sigmas(num_steps: i32, mu: f32) -> Vec<f32> {
    let mut sigmas = Vec::with_capacity((num_steps + 1) as usize);
    for i in 0..=num_steps {
        // Linear schedule from 1 (noise) to 0 (clean)
        let sigma_linear = 1.0 - (i as f32) / (num_steps as f32);
        // Apply time shift: exp(mu) / (exp(mu) + (1/sigma - 1))
        let sigma_shifted = if sigma_linear <= 0.0 {
            0.0
        } else if sigma_linear >= 1.0 {
            1.0
        } else {
            mu.exp() / (mu.exp() + (1.0 / sigma_linear - 1.0))
        };
        sigmas.push(sigma_shifted);
    }
    sigmas
}

/// Patchify latents: [1, C, H, W] -> [1, seq, C*4]
fn patchify(x: &Array, h_tok: i32, w_tok: i32, in_channels: i32) -> Result<Array, mlx_rs::error::Exception> {
    // [1, C, H, W] -> [C, 1, 1, H_tok, 2, W_tok, 2]
    let x = x.reshape(&[in_channels, 1, 1, h_tok, 2, w_tok, 2])?;
    // transpose to [1, 1, H_tok, W_tok, 2, 2, C]
    let x = x.transpose_axes(&[1, 2, 3, 5, 4, 6, 0])?;
    // reshape to [1, seq, C*4]
    x.reshape(&[1, h_tok * w_tok, in_channels * 4])
}

/// Unpatchify: [1, seq, C*4] -> [1, C, H, W]
fn unpatchify(x: &Array, h_tok: i32, w_tok: i32, in_channels: i32) -> Result<Array, mlx_rs::error::Exception> {
    // [1, seq, C*4] -> [1, 1, H_tok, W_tok, 2, 2, C]
    let x = x.reshape(&[1, 1, h_tok, w_tok, 2, 2, in_channels])?;
    // transpose to [C, 1, 1, H_tok, 2, W_tok, 2]
    let x = x.transpose_axes(&[6, 0, 1, 2, 4, 3, 5])?;
    // reshape to [1, C, H, W]
    x.reshape(&[1, in_channels, h_tok * 2, w_tok * 2])
}
