//! Python MLX subprocess engine for Wan 2.2 video generation.
//!
//! Supports two modes:
//! - MLX safetensors (via `infer_wan22_mlx.py`) — flat model dir with config.json + safetensors
//! - GGUF (via `infer_wan22_gguf.py`) — separate GGUF files for diffusion, T5, VAE

use std::path::{Path, PathBuf};
use std::time::Duration;

use eyre::{Context, Result};

use crate::engines::pymlx;
use crate::types::{VideoGenerationRequest, VideoGenerationResponse, VideoFrameData};

const SCRIPT_GGUF: &str = "infer_wan22_gguf.py";
const SCRIPT_MLX: &str = "infer_wan22_mlx.py";
const TIMEOUT: Duration = Duration::from_secs(900);

const DEFAULT_GUIDANCE_SCALE: f32 = 5.0;
const DEFAULT_FPS: i32 = 16;

#[derive(Debug)]
enum ModelFormat {
    Mlx { model_dir: PathBuf },
    Gguf { model_dir: PathBuf },
}

pub struct PymlxWan22Engine {
    python: PathBuf,
    script: PathBuf,
    format: ModelFormat,
}

impl PymlxWan22Engine {
    pub fn new(model_id: &str) -> Result<Self> {
        let python = pymlx::find_python()
            .ok_or_else(|| eyre::eyre!(
                "Python 3 not found. Set OMINIX_PYTHON or install python3."
            ))?;

        let (format, script_name) = detect_model_format()?;
        let script = pymlx::find_script(script_name)
            .ok_or_else(|| eyre::eyre!(
                "Inference script '{}' not found. Set OMINIX_INFERENCE_DIR or place it in ~/.OminiX/inference/scripts/",
                script_name
            ))?;

        let model_dir = match &format {
            ModelFormat::Mlx { model_dir } => model_dir,
            ModelFormat::Gguf { model_dir } => model_dir,
        };

        tracing::info!(
            "PymlxWan22Engine ready: model_id={} format={:?} python={} script={} model_dir={}",
            model_id,
            format,
            python.display(),
            script.display(),
            model_dir.display(),
        );

        Ok(Self { python, script, format })
    }

    pub fn generate(&self, request: &VideoGenerationRequest) -> Result<VideoGenerationResponse> {
        let (width, height) = pymlx::parse_size(&request.size)?;
        let width = round_to_multiple(width, 32);
        let height = round_to_multiple(height, 32);
        let num_frames = round_to_4n_plus_1(request.num_frames);

        tracing::info!(
            "pymlx_wan22: generating video {}x{} x {} frames, {} steps",
            width, height, num_frames, request.steps
        );
        let t0 = std::time::Instant::now();

        let video_bytes = match &self.format {
            ModelFormat::Mlx { model_dir } =>
                self.run_mlx(request, model_dir, width, height, num_frames)?,
            ModelFormat::Gguf { model_dir } =>
                self.run_gguf(request, model_dir, width, height, num_frames)?,
        };

        let elapsed = t0.elapsed();
        tracing::info!(
            "pymlx_wan22: video generation done in {:.1}s ({} bytes)",
            elapsed.as_secs_f32(), video_bytes.len()
        );

        let b64 = base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            &video_bytes,
        );

        Ok(VideoGenerationResponse {
            created: chrono::Utc::now().timestamp(),
            data: vec![VideoFrameData {
                frame_index: 0,
                b64_json: Some(b64),
            }],
        })
    }

    fn run_mlx(
        &self,
        request: &VideoGenerationRequest,
        model_dir: &Path,
        width: u32,
        height: u32,
        num_frames: i32,
    ) -> Result<Vec<u8>> {
        let output_path = pymlx::temp_output_path("wan22", "mp4");

        let width_str = width.to_string();
        let height_str = height.to_string();
        let num_frames_str = num_frames.to_string();
        let steps_str = request.steps.to_string();
        let guidance_str = format!("{:.1}", DEFAULT_GUIDANCE_SCALE);
        let output_str = output_path.to_string_lossy().to_string();
        let model_dir_str = model_dir.to_string_lossy().to_string();

        let args: Vec<&str> = vec![
            "generate",
            "--model-dir", &model_dir_str,
            "-p", &request.prompt,
            "--width", &width_str,
            "--height", &height_str,
            "--num-frames", &num_frames_str,
            "--steps", &steps_str,
            "--guide-scale", &guidance_str,
            "--seed", "-1",
            "--output", &output_str,
        ];

        pymlx::run_and_read_output(
            &self.python, &self.script, &args, &output_path, TIMEOUT,
        ).context("Wan2.2 MLX video inference failed")
    }

    fn run_gguf(
        &self,
        request: &VideoGenerationRequest,
        model_dir: &Path,
        width: u32,
        height: u32,
        num_frames: i32,
    ) -> Result<Vec<u8>> {
        let output_path = pymlx::temp_output_path("wan22", "mp4");

        let diffusion_model = find_model_file(model_dir, &["wan2", "wan22"], "gguf")
            .ok_or_else(|| eyre::eyre!("Wan2.2 transformer GGUF not found in {}", model_dir.display()))?;
        let t5_model = find_model_file(model_dir, &["t5xxl"], "gguf")
            .ok_or_else(|| eyre::eyre!("T5-XXL encoder GGUF not found in {}", model_dir.display()))?;
        let vae_model = find_vae_file(model_dir)
            .ok_or_else(|| eyre::eyre!("VAE model not found in {}", model_dir.display()))?;

        let width_str = width.to_string();
        let height_str = height.to_string();
        let num_frames_str = num_frames.to_string();
        let steps_str = request.steps.to_string();
        let fps_str = DEFAULT_FPS.to_string();
        let guidance_str = format!("{:.1}", DEFAULT_GUIDANCE_SCALE);
        let output_str = output_path.to_string_lossy().to_string();
        let diffusion_str = diffusion_model.to_string_lossy().to_string();
        let t5_str = t5_model.to_string_lossy().to_string();
        let vae_str = vae_model.to_string_lossy().to_string();

        let args: Vec<&str> = vec![
            "--diffusion-model", &diffusion_str,
            "--t5xxl", &t5_str,
            "--vae", &vae_str,
            "--prompt", &request.prompt,
            "--width", &width_str,
            "--height", &height_str,
            "--video-frames", &num_frames_str,
            "--fps", &fps_str,
            "--sampling-steps", &steps_str,
            "--sampling-method", "euler",
            "--cfg-scale", &guidance_str,
            "--seed", "-1",
            "--mode", "vid_gen",
            "--verbose",
            "--output", &output_str,
        ];

        pymlx::run_and_read_output(
            &self.python, &self.script, &args, &output_path, TIMEOUT,
        ).context("Wan2.2 GGUF video inference failed")
    }
}

fn detect_model_format() -> Result<(ModelFormat, &'static str)> {
    let home = std::env::var("HOME").context("HOME env var not set")?;

    // Check env override first
    if let Ok(dir) = std::env::var("OMINIX_WAN22_MODEL_DIR") {
        let p = PathBuf::from(&dir);
        if p.is_dir() {
            if p.join("config.json").exists() {
                return Ok((ModelFormat::Mlx { model_dir: p }, SCRIPT_MLX));
            }
            return Ok((ModelFormat::Gguf { model_dir: p }, SCRIPT_GGUF));
        }
    }

    // Prefer MLX model (flat dir with config.json + safetensors)
    let mlx_dir = PathBuf::from(&home).join(".OminiX/models/wan2.2-5b/mlx_model_4bit");
    if mlx_dir.join("config.json").exists() {
        return Ok((ModelFormat::Mlx { model_dir: mlx_dir }, SCRIPT_MLX));
    }

    // Fall back to GGUF
    let gguf_dir = PathBuf::from(&home).join(".OminiX/models/wan2.2");
    if gguf_dir.is_dir() {
        return Ok((ModelFormat::Gguf { model_dir: gguf_dir }, SCRIPT_GGUF));
    }

    Err(eyre::eyre!(
        "Wan2.2 model not found. Place MLX model in ~/.OminiX/models/wan2.2-5b/mlx_model_4bit/ or GGUF model in ~/.OminiX/models/wan2.2/"
    ))
}

fn find_model_file(dir: &Path, keywords: &[&str], extension: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() { continue; }
        let file_name = path.file_name()?.to_string_lossy().to_lowercase();
        let expected_ext = format!(".{}", extension);
        if !file_name.ends_with(&expected_ext) { continue; }
        for keyword in keywords {
            if file_name.contains(&keyword.to_lowercase()) {
                return Some(path);
            }
        }
    }
    None
}

fn find_vae_file(dir: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() { continue; }
        let file_name = path.file_name()?.to_string_lossy().to_lowercase();
        if !file_name.contains("vae") { continue; }
        if file_name.ends_with(".gguf") || file_name.ends_with(".safetensors") {
            return Some(path);
        }
    }
    None
}

fn round_to_multiple(value: u32, m: u32) -> u32 {
    ((value + m / 2) / m) * m
}

fn round_to_4n_plus_1(frames: i32) -> i32 {
    if frames <= 1 { return 1; }
    let n = ((frames - 1) as f32 / 4.0).round() as i32;
    (n * 4) + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_to_multiple() {
        assert_eq!(round_to_multiple(480, 32), 480);
        assert_eq!(round_to_multiple(481, 32), 480);
        assert_eq!(round_to_multiple(496, 32), 512);
        assert_eq!(round_to_multiple(832, 32), 832);
    }

    #[test]
    fn test_round_to_4n_plus_1() {
        assert_eq!(round_to_4n_plus_1(49), 49);
        assert_eq!(round_to_4n_plus_1(50), 49);
        assert_eq!(round_to_4n_plus_1(51), 53);
        assert_eq!(round_to_4n_plus_1(33), 33);
        assert_eq!(round_to_4n_plus_1(1), 1);
        assert_eq!(round_to_4n_plus_1(0), 1);
    }
}
