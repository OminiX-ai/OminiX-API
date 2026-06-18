//! Cosmos Predict2 subprocess engine for text-to-image and video-to-world inference.
//!
//! Wraps `infer_cosmos_gguf.py` as a subprocess, supporting both T2I (text-to-image)
//! and V2W (video-to-world) modes via GGUF-quantized Cosmos Predict2 models on MLX.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use eyre::{Context, Result};

use crate::engines::pymlx;
use crate::types::{
    ImageData, ImageGenerationRequest, ImageGenerationResponse, VideoFrameData,
    VideoGenerationRequest, VideoGenerationResponse,
};

const SCRIPT_NAME: &str = "infer_cosmos_gguf.py";
const DEFAULT_MODEL_SUBDIR: &str = ".OminiX/models/cosmos";
const TIMEOUT: Duration = Duration::from_secs(900);

pub struct PymlxCosmosEngine {
    python: PathBuf,
    script: PathBuf,
    model_dir: PathBuf,
}

impl PymlxCosmosEngine {
    pub fn new(model_id: &str) -> Result<Self> {
        let python = pymlx::find_python().ok_or_else(|| {
            eyre::eyre!("No Python 3 binary found. Set OMINIX_PYTHON or install python3.")
        })?;

        let script = pymlx::find_script(SCRIPT_NAME).ok_or_else(|| {
            eyre::eyre!(
                "Inference script '{}' not found. Set OMINIX_INFERENCE_DIR or place it in ~/.OminiX/inference/scripts/",
                SCRIPT_NAME
            )
        })?;

        let model_dir = resolve_model_dir()?;

        tracing::info!(
            "PymlxCosmosEngine ready: model_id={} python={} script={} model_dir={}",
            model_id,
            python.display(),
            script.display(),
            model_dir.display()
        );

        Ok(Self {
            python,
            script,
            model_dir,
        })
    }

    /// Text-to-image generation using Cosmos Predict2 T2I model.
    pub fn generate_image(
        &self,
        request: &ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse> {
        let t0 = Instant::now();

        let (width, height) = pymlx::parse_size(&request.size)?;

        let dit_model = find_model_file(&self.model_dir, "t2i")
            .context("No Cosmos T2I DiT model (GGUF containing 't2i') found in model dir")?;
        let t5_model = find_model_file(&self.model_dir, "t5xxl")
            .context("No T5 text encoder GGUF (containing 't5xxl') found in model dir")?;
        let vae_model = find_model_file(&self.model_dir, "vae")
            .context("No VAE GGUF (containing 'vae') found in model dir")?;

        let output_path = pymlx::temp_output_path("cosmos-t2i", "png");

        let width_str = width.to_string();
        let height_str = height.to_string();
        let dit_str = dit_model.to_string_lossy().to_string();
        let t5_str = t5_model.to_string_lossy().to_string();
        let vae_str = vae_model.to_string_lossy().to_string();
        let output_str = output_path.to_string_lossy().to_string();

        let mut args: Vec<&str> = vec![
            "--mode",
            "t2i",
            "--model",
            &dit_str,
            "--t5",
            &t5_str,
            "--vae",
            &vae_str,
            "--prompt",
            &request.prompt,
            "--width",
            &width_str,
            "--height",
            &height_str,
            "--output",
            &output_str,
        ];

        // If a reference image is provided, write it to disk and pass --input-image
        let temp_image_path;
        let temp_image_str;
        if let Some(ref image_b64) = request.image {
            let image_bytes = base64::Engine::decode(
                &base64::engine::general_purpose::STANDARD,
                image_b64,
            )
            .context("Failed to decode base64 reference image")?;
            temp_image_path = pymlx::write_temp_file("cosmos-ref", "png", &image_bytes)?;
            temp_image_str = temp_image_path.to_string_lossy().to_string();
            // With a reference image, switch to v2w mode
            args[1] = "v2w";
            args.push("--input-image");
            args.push(&temp_image_str);
        }

        tracing::info!(
            "cosmos t2i: generating {}x{} image, prompt='{}' ...",
            width,
            height,
            truncate_prompt(&request.prompt, 60)
        );

        let image_bytes = pymlx::run_and_read_output(
            &self.python,
            &self.script,
            &args,
            &output_path,
            TIMEOUT,
        )
        .context("Cosmos T2I inference failed")?;

        let elapsed = t0.elapsed();
        tracing::info!(
            "cosmos t2i: done in {:.1}s ({} bytes)",
            elapsed.as_secs_f32(),
            image_bytes.len()
        );

        let b64 = base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            &image_bytes,
        );

        let data = vec![ImageData {
            url: None,
            b64_json: Some(b64),
            revised_prompt: Some(request.prompt.clone()),
        }];

        Ok(ImageGenerationResponse {
            created: chrono::Utc::now().timestamp(),
            data,
        })
    }

    /// Video-to-world generation using Cosmos Predict2 V2W model.
    pub fn generate_video(
        &self,
        request: &VideoGenerationRequest,
    ) -> Result<VideoGenerationResponse> {
        let t0 = Instant::now();

        let (width, height) = pymlx::parse_size(&request.size)?;

        let dit_model = find_model_file(&self.model_dir, "v2w")
            .context("No Cosmos V2W DiT model (GGUF containing 'v2w') found in model dir")?;
        let t5_model = find_model_file(&self.model_dir, "t5xxl")
            .context("No T5 text encoder GGUF (containing 't5xxl') found in model dir")?;
        let vae_model = find_model_file(&self.model_dir, "vae")
            .context("No VAE GGUF (containing 'vae') found in model dir")?;

        let output_path = pymlx::temp_output_path("cosmos-v2w", "mp4");

        let width_str = width.to_string();
        let height_str = height.to_string();
        let num_frames_str = request.num_frames.to_string();
        let steps_str = request.steps.to_string();
        let dit_str = dit_model.to_string_lossy().to_string();
        let t5_str = t5_model.to_string_lossy().to_string();
        let vae_str = vae_model.to_string_lossy().to_string();
        let output_str = output_path.to_string_lossy().to_string();

        let args: Vec<&str> = vec![
            "--mode",
            "v2w",
            "--model",
            &dit_str,
            "--t5",
            &t5_str,
            "--vae",
            &vae_str,
            "--prompt",
            &request.prompt,
            "--width",
            &width_str,
            "--height",
            &height_str,
            "--num-frames",
            &num_frames_str,
            "--steps",
            &steps_str,
            "--output",
            &output_str,
        ];

        tracing::info!(
            "cosmos v2w: generating {}x{} video ({} frames, {} steps), prompt='{}' ...",
            width,
            height,
            request.num_frames,
            request.steps,
            truncate_prompt(&request.prompt, 60)
        );

        let video_bytes = pymlx::run_and_read_output(
            &self.python,
            &self.script,
            &args,
            &output_path,
            TIMEOUT,
        )
        .context("Cosmos V2W inference failed")?;

        let elapsed = t0.elapsed();
        tracing::info!(
            "cosmos v2w: done in {:.1}s ({} bytes)",
            elapsed.as_secs_f32(),
            video_bytes.len()
        );

        // Store entire MP4 as base64 in a single frame entry (frame_index=0).
        // Extracting individual frames from MP4 in Rust is complex; this pragmatic
        // approach lets the client decode the full video.
        let b64 = base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            &video_bytes,
        );

        let data = vec![VideoFrameData {
            frame_index: 0,
            b64_json: Some(b64),
        }];

        Ok(VideoGenerationResponse {
            created: chrono::Utc::now().timestamp(),
            data,
        })
    }
}

/// Resolve the Cosmos model directory.
///
/// Uses `OMINIX_COSMOS_MODEL_DIR` env var if set, otherwise defaults to
/// `~/.OminiX/models/cosmos/`.
fn resolve_model_dir() -> Result<PathBuf> {
    if let Ok(dir) = std::env::var("OMINIX_COSMOS_MODEL_DIR") {
        let p = PathBuf::from(dir);
        if p.is_dir() {
            return Ok(p);
        }
        return Err(eyre::eyre!(
            "OMINIX_COSMOS_MODEL_DIR='{}' is not a valid directory",
            p.display()
        ));
    }

    let home = std::env::var("HOME")
        .context("HOME environment variable not set")?;
    let default_dir = PathBuf::from(home).join(DEFAULT_MODEL_SUBDIR);

    if !default_dir.is_dir() {
        return Err(eyre::eyre!(
            "Cosmos model directory not found at '{}'. Download models or set OMINIX_COSMOS_MODEL_DIR.",
            default_dir.display()
        ));
    }

    Ok(default_dir)
}

/// Find the first GGUF file in model_dir whose name contains the given pattern.
fn find_model_file(model_dir: &PathBuf, pattern: &str) -> Result<PathBuf> {
    let entries = std::fs::read_dir(model_dir)
        .with_context(|| format!("Cannot read model directory: {}", model_dir.display()))?;

    for entry in entries.flatten() {
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            let lower = name.to_lowercase();
            if lower.contains(pattern) && lower.ends_with(".gguf") {
                return Ok(path);
            }
        }
    }

    Err(eyre::eyre!(
        "No GGUF file matching '{}' found in {}",
        pattern,
        model_dir.display()
    ))
}

/// Truncate a prompt string for log display.
fn truncate_prompt(prompt: &str, max_len: usize) -> String {
    if prompt.len() <= max_len {
        prompt.to_string()
    } else {
        format!("{}...", &prompt[..max_len])
    }
}
