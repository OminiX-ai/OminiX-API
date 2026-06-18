//! Python MLX subprocess engine for FLUX.2-klein GGUF text-to-image.
//!
//! Wraps `infer_flux2.py` which performs 4-step diffusion using a GGUF-quantized
//! FLUX.2-klein model with a Qwen3-4B text encoder.

use std::path::{Path, PathBuf};
use std::time::Duration;

use eyre::{Context, Result};

use crate::engines::pymlx;
use crate::types::{ImageData, ImageGenerationRequest, ImageGenerationResponse};

const SCRIPT_NAME: &str = "infer_flux2.py";
const DEFAULT_MODEL_SUBDIR: &str = ".OminiX/models/flux-klein-4b-q4-gguf";
const TIMEOUT: Duration = Duration::from_secs(300);

pub struct PymlxFluxEngine {
    python: PathBuf,
    script: PathBuf,
    model_dir: PathBuf,
}

impl PymlxFluxEngine {
    pub fn new(model_id: &str) -> Result<Self> {
        let python = pymlx::find_python()
            .ok_or_else(|| eyre::eyre!("Python 3 not found. Set OMINIX_PYTHON or install python3."))?;

        let script = pymlx::find_script(SCRIPT_NAME)
            .ok_or_else(|| eyre::eyre!(
                "Inference script '{}' not found. Set OMINIX_INFERENCE_DIR or place it in ~/.OminiX/inference/scripts/",
                SCRIPT_NAME
            ))?;

        let model_dir = resolve_model_dir()?;

        tracing::info!(
            "PymlxFluxEngine ready: model_id={} python={} script={} model_dir={}",
            model_id, python.display(), script.display(), model_dir.display(),
        );

        Ok(Self { python, script, model_dir })
    }

    pub fn generate(&self, request: &ImageGenerationRequest) -> Result<ImageGenerationResponse> {
        let (width, height) = pymlx::parse_size(&request.size)?;

        let mut data = Vec::new();
        for i in 0..request.n {
            tracing::info!("pymlx_flux: generating image {}/{} ({}x{})", i + 1, request.n, width, height);
            let t0 = std::time::Instant::now();

            let output_path = pymlx::temp_output_path("flux", "png");

            let diffusion = find_model_file(&self.model_dir, &["flux", "klein"], "gguf")
                .unwrap_or_else(|| self.model_dir.join("flux-2-klein-4b-Q4_0.gguf"));
            let vae = find_model_file(&self.model_dir, &["vae", "ae"], "safetensors")
                .unwrap_or_else(|| self.model_dir.join("flux2-vae.safetensors"));
            let llm = find_model_file(&self.model_dir, &["qwen3", "qwen"], "gguf")
                .unwrap_or_else(|| self.model_dir.join("Qwen3-4B-Q4_0.gguf"));

            let width_s = width.to_string();
            let height_s = height.to_string();
            let seed_s = (i as u32).to_string();
            let output_s = output_path.display().to_string();
            let diffusion_s = diffusion.display().to_string();
            let vae_s = vae.display().to_string();
            let llm_s = llm.display().to_string();

            let args: Vec<&str> = vec![
                "--diffusion-model", &diffusion_s,
                "--vae", &vae_s,
                "--llm", &llm_s,
                "-p", &request.prompt,
                "--width", &width_s,
                "--height", &height_s,
                "--steps", "4",
                "--seed", &seed_s,
                "--output", &output_s,
                "-v",
            ];

            let image_bytes = pymlx::run_and_read_output(
                &self.python, &self.script, &args, &output_path, TIMEOUT,
            )?;

            let elapsed = t0.elapsed();
            tracing::info!("pymlx_flux: image {}/{} done in {:.1}s ({} bytes)",
                i + 1, request.n, elapsed.as_secs_f32(), image_bytes.len());

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
}

fn resolve_model_dir() -> Result<PathBuf> {
    if let Ok(dir) = std::env::var("OMINIX_FLUX_GGUF_MODEL_DIR") {
        let p = PathBuf::from(&dir);
        if p.is_dir() { return Ok(p); }
    }
    if let Some(home) = std::env::var("HOME").ok().map(PathBuf::from) {
        let p = home.join(DEFAULT_MODEL_SUBDIR);
        if p.is_dir() { return Ok(p); }
        // Fall back to ~/Downloads where models may have been downloaded
        let dl = home.join("Downloads");
        if dl.is_dir() { return Ok(dl); }
    }
    Err(eyre::eyre!("FLUX GGUF model directory not found. Set OMINIX_FLUX_GGUF_MODEL_DIR or place models in ~/.OminiX/models/flux-klein-4b-q4-gguf/"))
}

fn find_model_file(dir: &Path, keywords: &[&str], extension: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some(extension) {
            continue;
        }
        let name = path.file_name()?.to_string_lossy().to_lowercase();
        if keywords.iter().any(|kw| name.contains(kw)) {
            return Some(path);
        }
    }
    None
}
