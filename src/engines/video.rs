//! Video generation engine for Wan2.2 text-to-video
//!
//! Reuses the qwen-image-mlx DiT transformer and Wan VAE for multi-frame generation.

use eyre::{Context, Result};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;
use image::{ImageEncoder, Rgb, ImageBuffer};
use tokenizers::Tokenizer;

use crate::model_config::{self, ModelAvailability, ModelCategory};

use qwen_image_mlx::{
    QwenQuantizedTransformer, QwenConfig, QwenVAE, QwenTextEncoder,
    load_text_encoder, load_vae_from_dir, load_transformer_weights,
    FlowMatchEulerScheduler,
    pack_latents, unpack_latents,
    weights::load_safetensors_shards,
};

use crate::types::{VideoGenerationRequest, VideoGenerationResponse, VideoFrameData};

pub struct VideoEngine {
    transformer: QwenQuantizedTransformer,
    qwen_config: QwenConfig,
    text_encoder: QwenTextEncoder,
    qwen_vae: QwenVAE,
    tokenizer: Tokenizer,
}

impl VideoEngine {
    pub fn new(model_id: &str) -> Result<Self> {
        let model_dir = Self::resolve_model_dir(model_id)?;

        tracing::info!("Loading Wan2.2 video model from {:?}", model_dir);

        // Detect quantization from model files
        let qwen_config = if model_dir.join("transformer").read_dir().ok()
            .and_then(|mut d| d.find(|e| e.as_ref().ok()
                .map_or(false, |e| e.file_name().to_string_lossy().contains("8bit"))))
            .is_some()
        {
            QwenConfig::with_8bit()
        } else {
            QwenConfig::default()
        };

        // Load text encoder
        tracing::info!("Loading text encoder...");
        let text_encoder = load_text_encoder(&model_dir)
            .map_err(|e| eyre::eyre!("Failed to load text encoder: {e}"))?;

        // Load tokenizer
        let tok_path = model_dir.join("tokenizer/tokenizer.json");
        if !tok_path.exists() {
            return Err(eyre::eyre!("Tokenizer not found at {:?}", tok_path));
        }
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| eyre::eyre!("Failed to load tokenizer: {e}"))?;

        // Load transformer
        tracing::info!("Loading DiT transformer...");
        let safetensors_files: Vec<_> = std::fs::read_dir(model_dir.join("transformer"))
            .context("Missing transformer directory")?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "safetensors"))
            .map(|e| e.path())
            .collect();

        let all_weights = load_safetensors_shards(&safetensors_files)
            .map_err(|e| eyre::eyre!("Failed to load transformer weights: {e}"))?;

        let mut transformer = QwenQuantizedTransformer::new(qwen_config.clone())
            .map_err(|e| eyre::eyre!("Failed to create transformer: {e}"))?;
        load_transformer_weights(&mut transformer, all_weights)
            .map_err(|e| eyre::eyre!("Failed to load transformer weights: {e}"))?;
        tracing::info!("Transformer loaded ({} blocks)", qwen_config.num_layers);

        // Load VAE
        tracing::info!("Loading Wan VAE...");
        let qwen_vae = load_vae_from_dir(&model_dir)
            .map_err(|e| eyre::eyre!("Failed to load Wan VAE: {e}"))?;

        tracing::info!("Wan2.2 video engine ready");

        Ok(Self {
            transformer,
            qwen_config,
            text_encoder,
            qwen_vae,
            tokenizer,
        })
    }

    fn resolve_model_dir(model_id: &str) -> Result<std::path::PathBuf> {
        let config_aliases = ["wan2.2-5b-q4km", "wan2.2-5b-q8"];
        let try_ids: Vec<&str> = config_aliases.iter().copied()
            .chain(std::iter::once(model_id))
            .collect();

        for config_id in &try_ids {
            match model_config::check_model(config_id, ModelCategory::Image) {
                ModelAvailability::Ready { local_path, .. } => {
                    if let Some(path) = local_path {
                        return crate::utils::resolve_hf_snapshot(&path);
                    }
                }
                _ => continue,
            }
        }

        // Try hub cache
        if let Some(hub_path) = crate::utils::resolve_from_hub_cache(model_id) {
            return Ok(hub_path);
        }

        Err(eyre::eyre!(
            "Video model '{}' not found. Download it first via /v1/models/download.",
            model_id
        ))
    }

    pub fn generate(&mut self, request: &VideoGenerationRequest) -> Result<VideoGenerationResponse> {
        use mlx_rs::ops;

        let batch = 1i32;
        let latent_channels = 16i32;

        // Parse dimensions
        let (width, height) = parse_size(&request.size)?;
        let width = (width / 16) * 16;
        let height = (height / 16) * 16;
        let num_frames = request.num_frames.max(1);
        let num_steps = request.steps.max(1);
        let latent_h = height / 8;
        let latent_w = width / 8;
        // Wan2.2 operates on temporal patches — latent_t depends on num_frames
        let latent_t = ((num_frames + 3) / 4).max(1);

        tracing::info!(
            "Generating video: {}x{} x {} frames, {} steps",
            width, height, num_frames, num_steps
        );

        // Tokenize prompt
        let chat_prompt = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            request.prompt
        );
        let encoding = self.tokenizer.encode(chat_prompt.as_str(), true)
            .map_err(|e| eyre::eyre!("Tokenization failed: {e}"))?;
        let ids: Vec<i32> = encoding.get_ids().iter().map(|&x| x as i32).collect();
        let seq_len = ids.len() as i32;
        let input_ids = Array::from_slice(&ids, &[batch, seq_len]);

        // Encode text
        tracing::info!("Encoding text ({} tokens)...", seq_len);
        let encoder_hidden_states = self.text_encoder.forward(&input_ids)
            .map_err(|e| eyre::eyre!("Text encoding failed: {e}"))?;
        let encoder_hidden_states = encoder_hidden_states
            .as_dtype(mlx_rs::Dtype::Float32)
            .map_err(|e| eyre::eyre!("Cast failed: {e}"))?;
        encoder_hidden_states.eval()
            .map_err(|e| eyre::eyre!("Eval failed: {e}"))?;

        // Flow-matching scheduler (shift=3.0 for Wan2.2)
        let scheduler = FlowMatchEulerScheduler::new(num_steps, 3.0);

        // Initialize noise: [B, C, T, H, W]
        let noise = mlx_rs::random::normal::<f32>(
            &[batch, latent_channels, latent_t, latent_h, latent_w],
            None, None, None,
        ).map_err(|e| eyre::eyre!("Noise init failed: {e}"))?;

        // Pack latents for the DiT
        let noise_packed = pack_latents(&noise)
            .map_err(|e| eyre::eyre!("Pack failed: {e}"))?;
        let mut latents = scheduler.scale_noise(&noise_packed)
            .map_err(|e| eyre::eyre!("Scale noise failed: {e}"))?;

        // Denoising loop
        tracing::info!("Denoising ({} steps)...", num_steps);
        for (idx, &t) in scheduler.timesteps().iter().enumerate() {
            let timestep = Array::from_slice(&[t * 1000.0], &[batch]);

            unsafe { mlx_sys::mlx_clear_cache(); }

            let v_pred = self.transformer.forward(
                &latents,
                &encoder_hidden_states,
                &timestep,
                None, None, None,
            ).map_err(|e| eyre::eyre!("Transformer forward failed: {e}"))?;

            latents = scheduler.step(&v_pred, idx, &latents)
                .map_err(|e| eyre::eyre!("Step failed: {e}"))?;
            latents.eval().map_err(|e| eyre::eyre!("Eval failed: {e}"))?;

            tracing::info!("  Step {}/{}: t={:.3}", idx + 1, num_steps, t);
        }

        // Unpack latents: [B, seq, C*4] -> [B, C, T, H, W]
        // For video, we need to unpack considering the temporal dimension
        let latents_5d = unpack_latents(&latents, width, height)
            .map_err(|e| eyre::eyre!("Unpack failed: {e}"))?;

        // Denormalize
        let latents_denorm = QwenVAE::denormalize_latent(&latents_5d)
            .map_err(|e| eyre::eyre!("Denormalize failed: {e}"))?;

        // Decode each frame through VAE
        tracing::info!("Decoding {} frames...", latent_t);
        let actual_frames = latent_t.min(num_frames);
        let mut frame_data = Vec::with_capacity(actual_frames as usize);

        for frame_idx in 0..actual_frames {
            // Extract single frame: [B, C, 1, H, W]
            let frame_latent = latents_denorm.index((.., .., frame_idx..frame_idx + 1, .., ..));
            let frame_2d = frame_latent.reshape(&[batch, latent_channels, latent_h, latent_w])
                .map_err(|e| eyre::eyre!("Frame reshape failed: {e}"))?;

            let decoded = self.qwen_vae.decode(&frame_2d)
                .map_err(|e| eyre::eyre!("VAE decode failed: {e}"))?;
            decoded.eval().map_err(|e| eyre::eyre!("Eval failed: {e}"))?;

            // Convert [B, 3, H, W] in [-1, 1] to PNG bytes
            let image = ops::add(&decoded, &Array::from_slice(&[1.0f32], &[1]))
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
            let out_h = shape[1] as u32;
            let out_w = shape[2] as u32;
            let flat = image.reshape(&[-1])
                .map_err(|e| eyre::eyre!("Flatten failed: {e}"))?;
            let pixels: Vec<f32> = flat.as_slice().to_vec();
            let rgb: Vec<u8> = pixels.iter().map(|&v| v.round() as u8).collect();
            let png = rgb_to_png(&rgb, out_w, out_h)?;
            let b64 = base64::Engine::encode(
                &base64::engine::general_purpose::STANDARD,
                &png,
            );

            frame_data.push(VideoFrameData {
                frame_index: frame_idx as usize,
                b64_json: Some(b64),
            });

            if frame_idx % 10 == 0 || frame_idx == actual_frames - 1 {
                tracing::info!("  Decoded frame {}/{}", frame_idx + 1, actual_frames);
            }
        }

        tracing::info!("Video generation complete ({} frames)", frame_data.len());

        Ok(VideoGenerationResponse {
            created: chrono::Utc::now().timestamp(),
            data: frame_data,
        })
    }
}

fn parse_size(size: &str) -> Result<(i32, i32)> {
    let parts: Vec<&str> = size.split('x').collect();
    if parts.len() != 2 {
        return Err(eyre::eyre!("Invalid size format '{}', expected WxH (e.g., 480x320)", size));
    }
    let w: i32 = parts[0].parse().context("Invalid width")?;
    let h: i32 = parts[1].parse().context("Invalid height")?;
    Ok((w, h))
}

fn rgb_to_png(rgb_bytes: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
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
