//! Python MLX subprocess engine for Qwen-Image-Edit-2511 image editing.
//!
//! Wraps `infer_qwen_image_edit_op_p1.py` which performs diffusion-based image
//! editing using a Qwen2.5-VL text encoder, a GGUF diffusion model, and a VAE.
//! Supports both reference-image editing (img2img) and text-only generation.

use std::path::{Path, PathBuf};
use std::time::Duration;

use eyre::{Context, Result};

use crate::engines::pymlx;
use crate::types::{ImageData, ImageGenerationRequest, ImageGenerationResponse};

const SCRIPT_NAME: &str = "infer_qwen_image_edit_op_p1.py";
const DEFAULT_MODEL_SUBDIR: &str = ".OminiX/models/qwen-image-edit-2511";
const TIMEOUT: Duration = Duration::from_secs(600);

pub struct PymlxImageEditEngine {
    python: PathBuf,
    script: PathBuf,
    model_dir: PathBuf,
}

impl PymlxImageEditEngine {
    pub fn new(model_id: &str) -> Result<Self> {
        let python = pymlx::find_python()
            .ok_or_else(|| eyre::eyre!(
                "Python 3 not found. Set OMINIX_PYTHON or install python3."
            ))?;

        let script = pymlx::find_script(SCRIPT_NAME)
            .ok_or_else(|| eyre::eyre!(
                "Inference script '{}' not found. Set OMINIX_INFERENCE_DIR or place it in ~/.OminiX/inference/scripts/",
                SCRIPT_NAME
            ))?;

        let model_dir = resolve_model_dir()?;

        tracing::info!(
            "PymlxImageEditEngine ready: model_id={} python={} script={} model_dir={}",
            model_id,
            python.display(),
            script.display(),
            model_dir.display(),
        );

        Ok(Self {
            python,
            script,
            model_dir,
        })
    }

    pub fn generate(&self, request: &ImageGenerationRequest) -> Result<ImageGenerationResponse> {
        let (width, height) = pymlx::parse_size(&request.size)?;

        // Round width/height to multiples of 16 as required by the script.
        let width = (width / 16) * 16;
        let height = (height / 16) * 16;

        let mut data = Vec::new();

        for i in 0..request.n {
            tracing::info!(
                "pymlx_image_edit: generating image {}/{} ({}x{})",
                i + 1,
                request.n,
                width,
                height
            );
            let t0 = std::time::Instant::now();

            let image_bytes = self.run_edit(request, width, height, i as u32)?;

            let elapsed = t0.elapsed();
            tracing::info!(
                "pymlx_image_edit: image {}/{} done in {:.1}s ({} bytes)",
                i + 1,
                request.n,
                elapsed.as_secs_f32(),
                image_bytes.len()
            );

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

    fn run_edit(
        &self,
        request: &ImageGenerationRequest,
        width: u32,
        height: u32,
        index: u32,
    ) -> Result<Vec<u8>> {
        let output_path = pymlx::temp_output_path("image-edit", "png");

        // Resolve model files from model_dir.
        let diffusion_model = find_model_file(&self.model_dir, &["edit"], "gguf")
            .ok_or_else(|| eyre::eyre!(
                "Diffusion GGUF model not found in {}",
                self.model_dir.display()
            ))?;

        let vae_model = find_model_file(&self.model_dir, &["vae"], "safetensors")
            .ok_or_else(|| eyre::eyre!(
                "VAE safetensors file not found in {}",
                self.model_dir.display()
            ))?;

        let llm_model = find_model_file(&self.model_dir, &["qwen2.5", "vl", "text_encoder"], "gguf")
            .ok_or_else(|| eyre::eyre!(
                "LLM text encoder GGUF not found in {}",
                self.model_dir.display()
            ))?;

        // Build argument list.
        let width_str = width.to_string();
        let height_str = height.to_string();
        let seed_str = index.to_string();
        let output_str = output_path.to_string_lossy().to_string();
        let diffusion_str = diffusion_model.to_string_lossy().to_string();
        let vae_str = vae_model.to_string_lossy().to_string();
        let llm_str = llm_model.to_string_lossy().to_string();

        let mut args: Vec<&str> = Vec::new();

        args.push("--diffusion-model");
        args.push(&diffusion_str);
        args.push("--vae");
        args.push(&vae_str);
        args.push("--llm");
        args.push(&llm_str);
        args.push("--prompt");
        args.push(&request.prompt);
        args.push("--width");
        args.push(&width_str);
        args.push("--height");
        args.push(&height_str);
        args.push("--seed");
        args.push(&seed_str);
        args.push("--output");
        args.push(&output_str);
        args.push("--qwen-image-zero-cond-t");
        args.push("--verbose");

        // Write reference image to temp file if provided (img2img mode).
        let ref_image_path: Option<PathBuf>;
        let ref_image_str: String;

        if let Some(ref b64_image) = request.image {
            let image_data = base64::Engine::decode(
                &base64::engine::general_purpose::STANDARD,
                b64_image,
            ).context("Failed to decode base64 reference image")?;

            let temp_path = pymlx::write_temp_file("image-edit-ref", "png", &image_data)?;
            ref_image_str = temp_path.to_string_lossy().to_string();
            ref_image_path = Some(temp_path);

            args.push("--reference-image");
            args.push(&ref_image_str);
        } else {
            ref_image_path = None;
            ref_image_str = String::new();
            let _ = &ref_image_str; // suppress unused warning
        }

        // Run the Python script and read output.
        let image_bytes = pymlx::run_and_read_output(
            &self.python,
            &self.script,
            &args,
            &output_path,
            TIMEOUT,
        ).context("Qwen-Image-Edit inference failed")?;

        // Clean up reference image temp file.
        if let Some(ref path) = ref_image_path {
            let _ = std::fs::remove_file(path);
        }

        Ok(image_bytes)
    }
}

/// Resolve the model directory from env var or default location.
fn resolve_model_dir() -> Result<PathBuf> {
    if let Ok(dir) = std::env::var("OMINIX_IMAGE_EDIT_MODEL_DIR") {
        let p = PathBuf::from(&dir);
        if p.is_dir() {
            return Ok(p);
        }
        tracing::warn!(
            "OMINIX_IMAGE_EDIT_MODEL_DIR={} does not exist, falling back to default",
            dir
        );
    }

    let home = std::env::var("HOME")
        .context("HOME env var not set")?;
    let default_dir = PathBuf::from(home).join(DEFAULT_MODEL_SUBDIR);

    if !default_dir.is_dir() {
        return Err(eyre::eyre!(
            "Model directory not found: {}. Set OMINIX_IMAGE_EDIT_MODEL_DIR or place models there.",
            default_dir.display()
        ));
    }

    Ok(default_dir)
}

/// Find a model file in a directory matching any of the given keywords and extension.
///
/// Returns the first file whose name (lowercased) contains at least one keyword
/// and ends with the given extension.
fn find_model_file(dir: &Path, keywords: &[&str], extension: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let file_name = path.file_name()?.to_string_lossy().to_lowercase();

        // Check extension.
        let expected_ext = format!(".{}", extension);
        if !file_name.ends_with(&expected_ext) {
            continue;
        }

        // Check if any keyword matches.
        for keyword in keywords {
            if file_name.contains(&keyword.to_lowercase()) {
                return Some(path);
            }
        }
    }

    None
}
