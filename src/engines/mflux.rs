//! mflux subprocess engine for Qwen-Image-2512 generation.
//!
//! Wraps the `mflux-generate-qwen` CLI tool, which produces high-quality
//! images via Apple MLX. This replaces the broken Rust Qwen-Image pipeline
//! that outputs noise.

use std::path::{Path, PathBuf};
use std::process::Command;

use eyre::{Context, Result};

use crate::types::{ImageData, ImageGenerationRequest, ImageGenerationResponse};

const DEFAULT_MODEL: &str = "mlx-community/Qwen-Image-2512-4bit";
const DEFAULT_STEPS: u32 = 20;

pub struct MfluxEngine {
    model: String,
    mflux_bin: PathBuf,
}

impl MfluxEngine {
    pub fn new(model_id: &str) -> Result<Self> {
        let mflux_bin = find_mflux_binary()
            .ok_or_else(|| eyre::eyre!(
                "mflux-generate-qwen not found. Install with: pip install mflux"
            ))?;

        let lower = model_id.to_lowercase();
        let model = if lower.contains("8bit") || lower.contains("8-bit") {
            "mlx-community/Qwen-Image-2512-8bit".to_string()
        } else if lower.starts_with("mlx-community/") {
            model_id.to_string()
        } else {
            DEFAULT_MODEL.to_string()
        };

        tracing::info!("MfluxEngine ready: model={model} bin={}", mflux_bin.display());
        Ok(Self { model, mflux_bin })
    }

    pub fn generate(&self, request: &ImageGenerationRequest) -> Result<ImageGenerationResponse> {
        let (width, height) = parse_size(&request.size)?;

        let mut data = Vec::new();
        for i in 0..request.n {
            tracing::info!("mflux: generating image {}/{} ({}x{})", i + 1, request.n, width, height);
            let t0 = std::time::Instant::now();

            let image_bytes = self.run_mflux(&request.prompt, width, height, i as u32)?;

            let elapsed = t0.elapsed();
            tracing::info!("mflux: image {}/{} done in {:.1}s ({} bytes)",
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

    fn run_mflux(&self, prompt: &str, width: u32, height: u32, index: u32) -> Result<Vec<u8>> {
        let output_path = format!("/tmp/ominix-mflux-{}-{}.png", std::process::id(), index);

        let mut cmd = Command::new(&self.mflux_bin);
        cmd.arg("--model").arg(&self.model)
            .arg("--prompt").arg(prompt)
            .arg("--width").arg(width.to_string())
            .arg("--height").arg(height.to_string())
            .arg("--steps").arg(DEFAULT_STEPS.to_string())
            .arg("--seed").arg(index.to_string())
            .arg("--output").arg(&output_path);

        tracing::debug!("mflux cmd: {:?}", cmd);

        let output = cmd.output()
            .context("Failed to spawn mflux-generate-qwen")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(eyre::eyre!(
                "mflux-generate-qwen failed (exit {}): {}\n{}",
                output.status.code().unwrap_or(-1),
                stderr.chars().take(500).collect::<String>(),
                stdout.chars().take(200).collect::<String>(),
            ));
        }

        let image_bytes = std::fs::read(&output_path)
            .context("Failed to read mflux output image")?;

        let _ = std::fs::remove_file(&output_path);

        Ok(image_bytes)
    }
}

fn find_mflux_binary() -> Option<PathBuf> {
    if let Ok(val) = std::env::var("MFLUX_BIN") {
        let p = PathBuf::from(val);
        if p.exists() {
            return Some(p);
        }
    }

    for dir in std::env::var("PATH").unwrap_or_default().split(':') {
        let candidate = Path::new(dir).join("mflux-generate-qwen");
        if candidate.exists() {
            return Some(candidate);
        }
    }

    let common_paths = [
        "/opt/anaconda3/bin/mflux-generate-qwen",
        "/opt/homebrew/bin/mflux-generate-qwen",
        "/usr/local/bin/mflux-generate-qwen",
    ];
    for path in common_paths {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    None
}

fn parse_size(size: &str) -> Result<(u32, u32)> {
    let parts: Vec<&str> = size.split('x').collect();
    if parts.len() != 2 {
        return Err(eyre::eyre!("Invalid size format '{}', expected WIDTHxHEIGHT", size));
    }
    let width: u32 = parts[0].parse().context("Invalid width")?;
    let height: u32 = parts[1].parse().context("Invalid height")?;
    Ok((width, height))
}
