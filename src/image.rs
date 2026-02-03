//! Image generation engine supporting FLUX.2-klein and Z-Image-Turbo
//!
//! Supported models:
//! - FLUX.2-klein-4B: 4-step denoising, ~13GB VRAM (or ~3GB with INT8)
//! - Z-Image-Turbo: 9-step denoising, ~12GB VRAM (or ~3GB with 4-bit)
//!
//! Both models support:
//! - Text-to-image generation
//! - Image-to-image generation (img2img) with reference image

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
    Qwen3Config, Qwen3TextEncoder,
    load_safetensors, sanitize_qwen3_weights, sanitize_vae_weights, sanitize_vae_encoder_weights,
    sanitize_klein_model_weights,
};

use zimage_mlx::{
    ZImageConfig, ZImageTransformerQuantized, create_coordinate_grid,
    load_quantized_zimage_transformer, load_safetensors as load_zimage_safetensors,
    QuantizedQwen3TextEncoder, sanitize_quantized_qwen3_weights,
};

use crate::types::{ImageGenerationRequest, ImageGenerationResponse, ImageData};

/// Image generation model type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImageModelType {
    FluxKlein,
    ZImageTurbo,
}

/// Transformer variant (FLUX or Z-Image)
enum TransformerVariant {
    Flux(FluxKlein, FluxKleinParams),
    ZImage(ZImageTransformerQuantized, ZImageConfig),
}

/// Text encoder variant (quantized or non-quantized)
enum TextEncoderVariant {
    Standard(Qwen3TextEncoder),
    Quantized(QuantizedQwen3TextEncoder),
}

/// Image generation engine
pub struct ImageEngine {
    model_type: ImageModelType,
    text_encoder: TextEncoderVariant,
    transformer: TransformerVariant,
    vae_decoder: Decoder,
    vae_encoder: Encoder,
    tokenizer: Tokenizer,
    vae_config: AutoEncoderConfig,
}

impl ImageEngine {
    /// Create a new image generation engine
    ///
    /// First checks ~/.moly/local_models_config.json for model availability.
    /// If the model is ready locally, uses that path. Otherwise, attempts
    /// to download from HuggingFace Hub.
    pub fn new(model_id: &str) -> Result<Self> {
        tracing::info!("Initializing image generation engine: {}", model_id);

        // Determine model type
        let model_type = if model_id.contains("zimage") || model_id.contains("z-image") || model_id.contains("Z-Image") {
            ImageModelType::ZImageTurbo
        } else {
            ImageModelType::FluxKlein
        };

        tracing::info!("Image model type: {:?}", model_type);

        // Determine the config ID to check
        let config_model_id = match model_type {
            ImageModelType::FluxKlein => "flux-klein-4b",
            ImageModelType::ZImageTurbo => "zimage-turbo",
        };

        // Check model configuration for local availability
        let model_dir: PathBuf = match model_config::check_model(config_model_id, ModelCategory::Image) {
            ModelAvailability::Ready { local_path, model_name } => {
                tracing::info!("Found locally available model: {} at {:?}", model_name, local_path);
                let path = local_path.ok_or_else(|| eyre::eyre!("Model path not available"))?;

                // For HuggingFace cache structure, we need to find the snapshots directory
                let snapshots_dir = path.join("snapshots");
                if snapshots_dir.exists() {
                    let snapshot = std::fs::read_dir(&snapshots_dir)?
                        .filter_map(|e| e.ok())
                        .filter(|e| e.path().is_dir())
                        .next()
                        .ok_or_else(|| eyre::eyre!("No snapshot found in {:?}", snapshots_dir))?;
                    snapshot.path()
                } else {
                    path
                }
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
                    model_id, found, expected
                ));
            }
            ModelAvailability::NotInConfig => {
                return Err(eyre::eyre!(
                    "Image model '{}' not found in local configuration.\n\
                     Please add this model to OminiX-Studio and download it there first.\n\
                     Available models can be viewed at: ~/.moly/local_models_config.json",
                    model_id
                ));
            }
        };

        // Get paths and configs based on model type
        let (transformer_path, text_encoder_paths, vae_path, tokenizer_path, vae_config) = match model_type {
            ImageModelType::FluxKlein => {
                tracing::info!("Loading FLUX.2-klein from local path: {:?}", model_dir);

                let trans = model_dir.join("transformer/diffusion_pytorch_model.safetensors");
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
        };
        tracing::info!("Text encoder loaded");

        // Load transformer based on model type
        let transformer = match model_type {
            ImageModelType::FluxKlein => {
                tracing::info!("Loading FLUX.2-klein transformer...");
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
                tracing::info!("FLUX transformer loaded");
                TransformerVariant::Flux(trans, params)
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

        tracing::info!("Image generation engine ready");

        Ok(Self {
            model_type,
            text_encoder,
            transformer,
            vae_decoder,
            vae_encoder,
            tokenizer,
            vae_config,
        })
    }

    /// Generate images from a text prompt (with optional reference image for img2img)
    pub fn generate(&mut self, request: &ImageGenerationRequest) -> Result<ImageGenerationResponse> {
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
            .resize_exact(width, height, image::imageops::FilterType::Lanczos3)
            .to_rgb8();

        // Convert to MLX array [1, H, W, 3] in range [-1, 1]
        let pixels: Vec<f32> = img.pixels()
            .flat_map(|p| p.0.iter().map(|&v| (v as f32 / 127.5) - 1.0))
            .collect();

        let input = Array::from_slice(&pixels, &[1, height as i32, width as i32, 3]);

        // Encode through VAE
        let latents = self.vae_encoder.encode_deterministic(&input)?;
        latents.eval()?;

        tracing::debug!("Encoded reference image to latents: {:?}", latents.shape());

        Ok(latents)
    }

    /// Generate image with img2img (starting from reference latents)
    fn generate_img2img(&mut self, prompt: &str, width: u32, height: u32, ref_latents: &Array, strength: f32) -> Result<Vec<u8>> {
        match self.model_type {
            ImageModelType::FluxKlein => self.generate_img2img_flux(prompt, width, height, ref_latents, strength),
            ImageModelType::ZImageTurbo => self.generate_img2img_zimage(prompt, width, height, ref_latents, strength),
        }
    }

    /// FLUX.2-klein img2img generation
    fn generate_img2img_flux(&mut self, prompt: &str, width: u32, height: u32, ref_latents: &Array, strength: f32) -> Result<Vec<u8>> {
        use mlx_rs::ops;

        let batch_size = 1i32;
        let max_seq_len = 512i32;
        let num_steps = 4; // FLUX.2-klein uses 4 steps

        // Get params from transformer variant
        let (flux_trans, params) = match &mut self.transformer {
            TransformerVariant::Flux(t, p) => (t, p.clone()),
            _ => return Err(eyre::eyre!("Expected FLUX transformer")),
        };

        // Calculate how many steps to skip based on strength
        let start_step = ((1.0 - strength) * num_steps as f32).round() as usize;
        let actual_steps = num_steps as usize - start_step;
        tracing::debug!("img2img: starting from step {} (running {} steps)", start_step, actual_steps);

        // Apply Qwen3 chat template
        let chat_prompt = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            prompt
        );

        // Tokenize
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

        // Encode text
        let txt_embed = match &mut self.text_encoder {
            TextEncoderVariant::Standard(enc) => enc.encode(&input_ids, Some(&attention_mask))?,
            TextEncoderVariant::Quantized(enc) => enc.encode_flux(&input_ids, Some(&attention_mask))?,
        };
        let txt_embed = txt_embed.as_dtype(mlx_rs::Dtype::Float32)?;
        txt_embed.eval()?;

        // Setup latent dimensions
        let img_height = height as i32;
        let img_width = width as i32;
        let latent_height = img_height / 8;
        let latent_width = img_width / 8;
        let patch_size = 2i32;
        let patch_h = latent_height / patch_size;
        let patch_w = latent_width / patch_size;
        let img_seq_len = patch_h * patch_w;
        let in_channels = params.in_channels;
        let z_channels = self.vae_config.z_channels;

        // Create position IDs
        let txt_ids = create_txt_ids(batch_size, max_seq_len)?;
        let img_ids = create_img_ids(batch_size, patch_h, patch_w)?;
        let (rope_cos, rope_sin) = FluxKlein::compute_rope(&txt_ids, &img_ids)?;

        // Generate timestep schedule
        let timesteps = flux_official_schedule(img_seq_len, num_steps);

        // Convert reference latents to sequence format
        // ref_latents shape: [1, latent_h, latent_w, z_channels]
        // Need to patchify to: [1, seq_len, in_channels]
        let ref_latents = ref_latents.transpose_axes(&[0, 3, 1, 2])?; // [1, C, H, W]
        let ref_latents = ref_latents.reshape(&[batch_size, z_channels, patch_h, patch_size, patch_w, patch_size])?;
        let ref_latents = ref_latents.transpose_axes(&[0, 2, 4, 1, 3, 5])?; // [1, ph, pw, C, 2, 2]
        let ref_latents = ref_latents.reshape(&[batch_size, img_seq_len, in_channels])?;

        // Add noise to reference latents based on starting timestep
        let t_start = timesteps[start_step];
        let noise = mlx_rs::random::normal::<f32>(
            &[batch_size, img_seq_len, in_channels],
            None, None, None,
        )?;

        // Mix reference with noise: latent = ref * (1-t) + noise * t
        let ref_weight = Array::from_slice(&[1.0 - t_start], &[1]);
        let noise_weight = Array::from_slice(&[t_start], &[1]);
        let mut latent = ops::add(
            &ops::multiply(&ref_latents, &ref_weight)?,
            &ops::multiply(&noise, &noise_weight)?,
        )?;

        // Denoising loop (starting from start_step)
        tracing::debug!("Running FLUX denoising ({} steps from step {})...", actual_steps, start_step);
        for step in start_step..num_steps as usize {
            let t_curr = timesteps[step];
            let t_next = timesteps[step + 1];
            let t_arr = Array::from_slice(&[t_curr * 1000.0], &[batch_size]);

            let v_pred = flux_trans.forward_with_rope(
                &latent, &txt_embed, &t_arr, &rope_cos, &rope_sin
            )?;

            let dt = t_next - t_curr;
            let scaled_v = ops::multiply(&v_pred, &Array::from_slice(&[dt], &[1]))?;
            latent = ops::add(&latent, &scaled_v)?;
            latent.eval()?;

            tracing::debug!("  Step {}/{}: t={:.3}->{:.3}", step + 1, num_steps, t_curr, t_next);
        }

        // Decode latents to image
        let latent = latent.reshape(&[batch_size, patch_h, patch_w, z_channels, patch_size, patch_size])?;
        let latent = latent.transpose_axes(&[0, 1, 4, 2, 5, 3])?;
        let vae_height = patch_h * patch_size;
        let vae_width = patch_w * patch_size;
        let latent_for_vae = latent.reshape(&[batch_size, vae_height, vae_width, z_channels])?;

        let image = self.vae_decoder.forward(&latent_for_vae)?;
        image.eval()?;

        // Convert to RGB bytes
        let image = ops::add(&image, &Array::from_slice(&[1.0f32], &[1]))?;
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

        // Convert to PNG
        rgb_to_png(&rgb_bytes, out_width as u32, out_height as u32)
    }

    /// Z-Image-Turbo img2img generation
    fn generate_img2img_zimage(&mut self, prompt: &str, width: u32, height: u32, ref_latents: &Array, strength: f32) -> Result<Vec<u8>> {
        use mlx_rs::ops;

        let batch_size = 1i32;
        let max_seq_len = 512i32;
        let num_steps = 9; // Z-Image uses 9 steps

        // Get config from transformer variant
        let zimage_config = match &self.transformer {
            TransformerVariant::ZImage(_, c) => c.clone(),
            _ => return Err(eyre::eyre!("Expected Z-Image transformer")),
        };

        // Calculate how many steps to skip based on strength
        let start_step = ((1.0 - strength) * num_steps as f32).round() as usize;
        let actual_steps = num_steps as usize - start_step;
        tracing::debug!("Z-Image img2img: starting from step {} (running {} steps)", start_step, actual_steps);

        // Apply Qwen3 chat template
        let chat_prompt = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            prompt
        );

        // Tokenize
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

        // Encode text (Z-Image uses layer 34 extraction)
        let txt_embed = match &mut self.text_encoder {
            TextEncoderVariant::Standard(enc) => enc.encode_zimage(&input_ids, Some(&attention_mask))?,
            TextEncoderVariant::Quantized(enc) => enc.encode_zimage(&input_ids, Some(&attention_mask))?,
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
        let img_height = height as i32;
        let img_width = width as i32;
        let latent_height = img_height / 8;
        let latent_width = img_width / 8;
        let patch_size = zimage_config.patch_size;
        let h_tok = latent_height / patch_size;
        let w_tok = latent_width / patch_size;
        let img_seq_len = h_tok * w_tok;
        let in_channels = zimage_config.in_channels;
        let _z_channels = self.vae_config.z_channels; // Z-Image: 16, used by VAE decoder

        // Create Z-Image position encodings
        let img_pos = create_coordinate_grid((1, h_tok, w_tok), (cap_len + 1, 0, 0))?;
        let img_pos = img_pos.reshape(&[1, img_seq_len, 3])?;
        let cap_pos = create_coordinate_grid((cap_len, 1, 1), (1, 0, 0))?;
        let cap_pos = cap_pos.reshape(&[1, cap_len, 3])?;

        // Pre-compute RoPE
        let (cos, sin) = match &mut self.transformer {
            TransformerVariant::ZImage(trans, _) => trans.compute_rope(&img_pos, &cap_pos)?,
            _ => return Err(eyre::eyre!("Expected Z-Image transformer")),
        };

        // Z-Image's dynamic shift calculation
        let mu = calculate_shift(img_seq_len, 256, 4096, 0.5, 1.15);
        let sigmas = generate_sigmas(num_steps as i32, mu);

        // ref_latents is [1, H, W, C] from VAE encoder
        // Convert to Z-Image format [1, C, H, W]
        let mut latents = ref_latents.transpose_axes(&[0, 3, 1, 2])?;

        // Add noise based on starting sigma
        let sigma_start = sigmas[start_step];
        let noise = mlx_rs::random::normal::<f32>(
            &[batch_size, in_channels, latent_height, latent_width],
            None, None, None,
        )?;
        // latent = ref * (1-sigma) + noise * sigma
        let ref_weight = Array::from_slice(&[1.0 - sigma_start], &[1]);
        let noise_weight = Array::from_slice(&[sigma_start], &[1]);
        latents = ops::add(
            &ops::multiply(&latents, &ref_weight)?,
            &ops::multiply(&noise, &noise_weight)?,
        )?;

        // Denoising loop
        tracing::debug!("Running Z-Image denoising ({} steps from step {})...", actual_steps, start_step);
        for step in start_step..num_steps as usize {
            let sigma_curr = sigmas[step];
            let sigma_next = sigmas[step + 1];
            let t_model = 1.0 - sigma_curr;
            let t = Array::from_slice(&[t_model], &[1]);

            // Patchify latents
            let latents_patched = patchify(&latents, h_tok, w_tok, in_channels)?;

            let model_out = match &mut self.transformer {
                TransformerVariant::ZImage(trans, _) => trans.forward_with_rope(
                    &latents_patched, &t, &txt_embed, &img_pos, &cap_pos, &cos, &sin, None, None
                )?,
                _ => return Err(eyre::eyre!("Expected Z-Image transformer")),
            };

            // Unpatchify and negate
            let noise_pred = unpatchify(&model_out, h_tok, w_tok, in_channels)?;
            let noise_pred = ops::negative(&noise_pred)?;

            // Euler step
            let dt = sigma_next - sigma_curr;
            let scaled_noise = ops::multiply(&noise_pred, &Array::from_slice(&[dt], &[1]))?;
            latents = ops::add(&latents, &scaled_noise)?;
            latents.eval()?;

            tracing::debug!("  Step {}/{}: sigma={:.3}->{:.3}", step + 1, num_steps, sigma_curr, sigma_next);
        }

        // Decode latents to image
        // Latents are [B, C, H, W], VAE expects [B, H, W, C]
        let latents = latents.transpose_axes(&[0, 2, 3, 1])?;

        let image = self.vae_decoder.forward(&latents)?;
        image.eval()?;

        // Convert to RGB bytes (Z-Image style normalization)
        let image = ops::divide(&image, &mlx_rs::array!(2.0f32))?;
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

        // Convert to PNG
        rgb_to_png(&rgb_bytes, out_width as u32, out_height as u32)
    }

    /// Generate a single image
    fn generate_single(&mut self, prompt: &str, width: u32, height: u32) -> Result<Vec<u8>> {
        match self.model_type {
            ImageModelType::FluxKlein => self.generate_single_flux(prompt, width, height),
            ImageModelType::ZImageTurbo => self.generate_single_zimage(prompt, width, height),
        }
    }

    /// FLUX.2-klein text-to-image generation
    fn generate_single_flux(&mut self, prompt: &str, width: u32, height: u32) -> Result<Vec<u8>> {
        use mlx_rs::ops;

        let batch_size = 1i32;
        let max_seq_len = 512i32;
        let num_steps = 4; // FLUX.2-klein uses 4 steps

        // Get params from transformer variant
        let (flux_trans, params) = match &mut self.transformer {
            TransformerVariant::Flux(t, p) => (t, p.clone()),
            _ => return Err(eyre::eyre!("Expected FLUX transformer")),
        };

        // Apply Qwen3 chat template
        let chat_prompt = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            prompt
        );

        // Tokenize
        let encoding = self.tokenizer.encode(chat_prompt.as_str(), true)
            .map_err(|e| eyre::eyre!("Tokenization failed: {}", e))?;
        let ids: Vec<i32> = encoding.get_ids().iter().map(|&x| x as i32).collect();
        let num_tokens = ids.len().min(max_seq_len as usize);

        // Pad tokens
        let mut padded = vec![151643i32; max_seq_len as usize]; // Qwen3 pad token
        padded[..num_tokens].copy_from_slice(&ids[..num_tokens]);

        // Create attention mask
        let mut mask = vec![0i32; max_seq_len as usize];
        for i in 0..num_tokens {
            mask[i] = 1;
        }

        let input_ids = Array::from_slice(&padded, &[batch_size, max_seq_len]);
        let attention_mask = Array::from_slice(&mask, &[batch_size, max_seq_len]);

        // Encode text
        tracing::debug!("Encoding text prompt...");
        let txt_embed = match &mut self.text_encoder {
            TextEncoderVariant::Standard(enc) => enc.encode(&input_ids, Some(&attention_mask))?,
            TextEncoderVariant::Quantized(enc) => enc.encode_flux(&input_ids, Some(&attention_mask))?,
        };
        let txt_embed = txt_embed.as_dtype(mlx_rs::Dtype::Float32)?;
        txt_embed.eval()?;

        // Setup latent dimensions
        let img_height = height as i32;
        let img_width = width as i32;
        let latent_height = img_height / 8;
        let latent_width = img_width / 8;
        let patch_size = 2i32;
        let patch_h = latent_height / patch_size;
        let patch_w = latent_width / patch_size;
        let img_seq_len = patch_h * patch_w;
        let in_channels = params.in_channels;
        let z_channels = self.vae_config.z_channels;

        // Create position IDs
        let txt_ids = create_txt_ids(batch_size, max_seq_len)?;
        let img_ids = create_img_ids(batch_size, patch_h, patch_w)?;
        let (rope_cos, rope_sin) = FluxKlein::compute_rope(&txt_ids, &img_ids)?;

        // Generate timestep schedule
        let timesteps = flux_official_schedule(img_seq_len, num_steps);

        // Start with random noise
        let mut latent = mlx_rs::random::normal::<f32>(
            &[batch_size, img_seq_len, in_channels],
            None,
            None,
            None,
        )?;

        // Denoising loop
        tracing::debug!("Running FLUX denoising ({} steps)...", num_steps);
        for step in 0..num_steps as usize {
            let t_curr = timesteps[step];
            let t_next = timesteps[step + 1];
            let t_arr = Array::from_slice(&[t_curr * 1000.0], &[batch_size]);

            let v_pred = flux_trans.forward_with_rope(
                &latent, &txt_embed, &t_arr, &rope_cos, &rope_sin
            )?;

            // Euler step
            let dt = t_next - t_curr;
            let scaled_v = ops::multiply(&v_pred, &Array::from_slice(&[dt], &[1]))?;
            latent = ops::add(&latent, &scaled_v)?;
            latent.eval()?;

            tracing::debug!("  Step {}/{}: t={:.3}->{:.3}", step + 1, num_steps, t_curr, t_next);
        }

        // Decode latents to image
        tracing::debug!("Decoding latents...");
        let latent = latent.reshape(&[batch_size, patch_h, patch_w, z_channels, patch_size, patch_size])?;
        let latent = latent.transpose_axes(&[0, 1, 4, 2, 5, 3])?;
        let vae_height = patch_h * patch_size;
        let vae_width = patch_w * patch_size;
        let latent_for_vae = latent.reshape(&[batch_size, vae_height, vae_width, z_channels])?;

        let image = self.vae_decoder.forward(&latent_for_vae)?;
        image.eval()?;

        // Convert to RGB bytes
        let image = ops::add(&image, &Array::from_slice(&[1.0f32], &[1]))?;
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

        // Convert to PNG
        rgb_to_png(&rgb_bytes, out_width as u32, out_height as u32)
    }

    /// Z-Image-Turbo text-to-image generation
    fn generate_single_zimage(&mut self, prompt: &str, width: u32, height: u32) -> Result<Vec<u8>> {
        use mlx_rs::ops;

        let batch_size = 1i32;
        let max_seq_len = 512i32;
        let num_steps = 9; // Z-Image uses 9 steps

        // Get config from transformer variant
        let zimage_config = match &self.transformer {
            TransformerVariant::ZImage(_, c) => c.clone(),
            _ => return Err(eyre::eyre!("Expected Z-Image transformer")),
        };

        // Apply Qwen3 chat template
        let chat_prompt = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            prompt
        );

        // Tokenize
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

        // Encode text (Z-Image uses layer 34 extraction)
        tracing::debug!("Encoding text prompt (Z-Image style)...");
        let txt_embed = match &mut self.text_encoder {
            TextEncoderVariant::Standard(enc) => enc.encode_zimage(&input_ids, Some(&attention_mask))?,
            TextEncoderVariant::Quantized(enc) => enc.encode_zimage(&input_ids, Some(&attention_mask))?,
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
        let img_height = height as i32;
        let img_width = width as i32;
        let latent_height = img_height / 8;
        let latent_width = img_width / 8;
        let patch_size = zimage_config.patch_size;
        let h_tok = latent_height / patch_size;
        let w_tok = latent_width / patch_size;
        let img_seq_len = h_tok * w_tok;
        let in_channels = zimage_config.in_channels;

        // Create Z-Image position encodings
        let img_pos = create_coordinate_grid((1, h_tok, w_tok), (cap_len + 1, 0, 0))?;
        let img_pos = img_pos.reshape(&[1, img_seq_len, 3])?;
        let cap_pos = create_coordinate_grid((cap_len, 1, 1), (1, 0, 0))?;
        let cap_pos = cap_pos.reshape(&[1, cap_len, 3])?;

        // Pre-compute RoPE
        let (cos, sin) = match &mut self.transformer {
            TransformerVariant::ZImage(trans, _) => trans.compute_rope(&img_pos, &cap_pos)?,
            _ => return Err(eyre::eyre!("Expected Z-Image transformer")),
        };

        // Z-Image's dynamic shift calculation
        let mu = calculate_shift(img_seq_len, 256, 4096, 0.5, 1.15);
        let sigmas = generate_sigmas(num_steps as i32, mu);

        // Start with random noise in [B, C, H, W] format
        let mut latents = mlx_rs::random::normal::<f32>(
            &[batch_size, in_channels, latent_height, latent_width],
            None, None, None,
        )?;

        // Denoising loop
        tracing::debug!("Running Z-Image denoising ({} steps)...", num_steps);
        for step in 0..num_steps as usize {
            let sigma_curr = sigmas[step];
            let sigma_next = sigmas[step + 1];
            let t_model = 1.0 - sigma_curr;
            let t = Array::from_slice(&[t_model], &[1]);

            // Patchify latents
            let latents_patched = patchify(&latents, h_tok, w_tok, in_channels)?;

            let model_out = match &mut self.transformer {
                TransformerVariant::ZImage(trans, _) => trans.forward_with_rope(
                    &latents_patched, &t, &txt_embed, &img_pos, &cap_pos, &cos, &sin, None, None
                )?,
                _ => return Err(eyre::eyre!("Expected Z-Image transformer")),
            };

            // Unpatchify and negate
            let noise_pred = unpatchify(&model_out, h_tok, w_tok, in_channels)?;
            let noise_pred = ops::negative(&noise_pred)?;

            // Euler step
            let dt = sigma_next - sigma_curr;
            let scaled_noise = ops::multiply(&noise_pred, &Array::from_slice(&[dt], &[1]))?;
            latents = ops::add(&latents, &scaled_noise)?;
            latents.eval()?;

            tracing::debug!("  Step {}/{}: sigma={:.3}->{:.3}", step + 1, num_steps, sigma_curr, sigma_next);
        }

        // Decode latents to image
        // Latents are [B, C, H, W], VAE expects [B, H, W, C]
        tracing::debug!("Decoding latents...");
        let latents = latents.transpose_axes(&[0, 2, 3, 1])?;

        let image = self.vae_decoder.forward(&latents)?;
        image.eval()?;

        // Convert to RGB bytes (Z-Image style normalization)
        let image = ops::divide(&image, &mlx_rs::array!(2.0f32))?;
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

        // Convert to PNG
        rgb_to_png(&rgb_bytes, out_width as u32, out_height as u32)
    }
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
