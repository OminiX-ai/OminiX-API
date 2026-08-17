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
const TIMEOUT: Duration = Duration::from_secs(3600);

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
        let guidance_str = format!("{:.2}", request.guide_scale.unwrap_or(DEFAULT_GUIDANCE_SCALE));
        let seed_str = request.seed.unwrap_or(-1).to_string();
        let output_str = output_path.to_string_lossy().to_string();
        let model_dir_str = model_dir.to_string_lossy().to_string();
        let prepared_image = prepare_input_image(request.image.as_deref())?;
        let image_str = prepared_image.as_ref().map(|(path, _)| path.clone());

        let mut args: Vec<String> = vec![
            "generate".to_string(),
            "--model-dir".to_string(), model_dir_str,
            "-p".to_string(), request.prompt.clone(),
            "--width".to_string(), width_str,
            "--height".to_string(), height_str,
            "--num-frames".to_string(), num_frames_str,
            "--steps".to_string(), steps_str,
            "--guide-scale".to_string(), guidance_str,
            "--seed".to_string(), seed_str,
            "--output".to_string(), output_str,
        ];

        if let Some(negative_prompt) = non_empty(request.negative_prompt.as_deref()) {
            args.push("--negative-prompt".to_string());
            args.push(negative_prompt.to_string());
        }

        if let Some(image) = image_str {
            args.push("--image".to_string());
            args.push(image);
        }

        let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
        let result = pymlx::run_and_read_output(
            &self.python, &self.script, &arg_refs, &output_path, TIMEOUT,
        ).context("Wan2.2 MLX video inference failed");

        if let Some((_, Some(temp_path))) = prepared_image {
            let _ = std::fs::remove_file(temp_path);
        }

        result
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
        let guidance_str = format!("{:.2}", request.guide_scale.unwrap_or(DEFAULT_GUIDANCE_SCALE));
        let seed_str = request.seed.unwrap_or(-1).to_string();
        let output_str = output_path.to_string_lossy().to_string();
        let diffusion_str = diffusion_model.to_string_lossy().to_string();
        let t5_str = t5_model.to_string_lossy().to_string();
        let vae_str = vae_model.to_string_lossy().to_string();

        if non_empty(request.image.as_deref()).is_some() {
            return Err(eyre::eyre!(
                "Wan2.2 image-to-video requires the MLX safetensors backend; the GGUF wrapper only supports text-to-video."
            ));
        }

        let mut args: Vec<String> = vec![
            "--diffusion-model".to_string(), diffusion_str,
            "--t5xxl".to_string(), t5_str,
            "--vae".to_string(), vae_str,
            "--prompt".to_string(), request.prompt.clone(),
            "--width".to_string(), width_str,
            "--height".to_string(), height_str,
            "--video-frames".to_string(), num_frames_str,
            "--fps".to_string(), fps_str,
            "--sampling-steps".to_string(), steps_str,
            "--sampling-method".to_string(), "euler".to_string(),
            "--cfg-scale".to_string(), guidance_str,
            "--seed".to_string(), seed_str,
            "--mode".to_string(), "vid_gen".to_string(),
            "--verbose".to_string(),
            "--output".to_string(), output_str,
        ];

        if let Some(negative_prompt) = non_empty(request.negative_prompt.as_deref()) {
            args.push("--negative-prompt".to_string());
            args.push(negative_prompt.to_string());
        }

        let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
        pymlx::run_and_read_output(
            &self.python, &self.script, &arg_refs, &output_path, TIMEOUT,
        ).context("Wan2.2 GGUF video inference failed")
    }
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|s| !s.is_empty())
}

fn prepare_input_image(image: Option<&str>) -> Result<Option<(String, Option<PathBuf>)>> {
    let Some(image) = non_empty(image) else {
        return Ok(None);
    };

    let expanded = crate::utils::expand_tilde(image);
    let path = PathBuf::from(&expanded);
    if path.exists() {
        return Ok(Some((expanded, None)));
    }

    let data = image
        .split_once(',')
        .filter(|(prefix, _)| prefix.starts_with("data:"))
        .map(|(_, payload)| payload)
        .unwrap_or(image);

    let bytes = base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD,
        data,
    ).context("Failed to decode video input image as base64")?;

    let temp_path = pymlx::write_temp_file("wan22-ref", "png", &bytes)?;
    Ok(Some((temp_path.to_string_lossy().to_string(), Some(temp_path))))
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

    // Fall back to GGUF installs.
    for relative_dir in [
        ".OminiX/models/wan2.2-5b-q4km",
        ".OminiX/models/wan2.2-5b-q8",
        ".OminiX/models/wan2.2",
    ] {
        let gguf_dir = PathBuf::from(&home).join(relative_dir);
        if gguf_dir.is_dir() {
            return Ok((ModelFormat::Gguf { model_dir: gguf_dir }, SCRIPT_GGUF));
        }
    }

    Err(eyre::eyre!(
        "Wan2.2 model not found. Place MLX model in ~/.OminiX/models/wan2.2-5b/mlx_model_4bit/ or GGUF model in ~/.OminiX/models/wan2.2-5b-q4km/, ~/.OminiX/models/wan2.2-5b-q8/, or ~/.OminiX/models/wan2.2/"
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
