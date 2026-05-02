//! Python MLX subprocess engine for Wan 2.2 GGUF video generation.
//!
//! Wraps `infer_wan22_gguf.py` which performs diffusion-based text-to-video
//! generation using a Wan2.2 transformer GGUF, T5-XXL text encoder, and VAE.
//! Produces MP4 output returned as base64.

use std::path::{Path, PathBuf};
use std::time::Duration;

use eyre::{Context, Result};

use crate::engines::pymlx;
use crate::types::{VideoGenerationRequest, VideoGenerationResponse, VideoFrameData};

const SCRIPT_NAME: &str = "infer_wan22_gguf.py";
const DEFAULT_MODEL_SUBDIR: &str = ".OminiX/models/wan2.2";
const TIMEOUT: Duration = Duration::from_secs(900);

const DEFAULT_SAMPLING_METHOD: &str = "euler";
const DEFAULT_GUIDANCE_SCALE: f32 = 5.0;
const DEFAULT_FPS: i32 = 16;

pub struct PymlxWan22Engine {
    python: PathBuf,
    script: PathBuf,
    model_dir: PathBuf,
}

impl PymlxWan22Engine {
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
            "PymlxWan22Engine ready: model_id={} python={} script={} model_dir={}",
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

    pub fn generate(&self, request: &VideoGenerationRequest) -> Result<VideoGenerationResponse> {
        let (width, height) = pymlx::parse_size(&request.size)?;

        // Width/height must be divisible by 32 — round to nearest valid value.
        let width = round_to_multiple(width, 32);
        let height = round_to_multiple(height, 32);

        // num_frames must be 4n+1 — adjust to nearest valid value.
        let num_frames = round_to_4n_plus_1(request.num_frames);

        tracing::info!(
            "pymlx_wan22: generating video {}x{} x {} frames, {} steps",
            width,
            height,
            num_frames,
            request.steps
        );
        let t0 = std::time::Instant::now();

        let video_bytes = self.run_generation(request, width, height, num_frames)?;

        let elapsed = t0.elapsed();
        tracing::info!(
            "pymlx_wan22: video generation done in {:.1}s ({} bytes)",
            elapsed.as_secs_f32(),
            video_bytes.len()
        );

        // Return entire MP4 as base64 in a single VideoFrameData with frame_index=0.
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

    fn run_generation(
        &self,
        request: &VideoGenerationRequest,
        width: u32,
        height: u32,
        num_frames: i32,
    ) -> Result<Vec<u8>> {
        let output_path = pymlx::temp_output_path("wan22", "mp4");

        // Resolve model files from model_dir.
        let diffusion_model = find_model_file(&self.model_dir, &["wan2", "wan22"], "gguf")
            .ok_or_else(|| eyre::eyre!(
                "Wan2.2 transformer GGUF not found in {}. Expected a .gguf file containing 'wan2' or 'wan22' in its name.",
                self.model_dir.display()
            ))?;

        let t5_model = find_model_file(&self.model_dir, &["t5xxl"], "gguf")
            .ok_or_else(|| eyre::eyre!(
                "T5-XXL encoder GGUF not found in {}. Expected a .gguf file containing 't5xxl' in its name.",
                self.model_dir.display()
            ))?;

        let vae_model = find_vae_file(&self.model_dir)
            .ok_or_else(|| eyre::eyre!(
                "VAE model not found in {}. Expected a .gguf or .safetensors file containing 'vae' in its name.",
                self.model_dir.display()
            ))?;

        // Build argument list.
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

        let mut args: Vec<&str> = Vec::new();

        args.push("--diffusion-model");
        args.push(&diffusion_str);
        args.push("--t5xxl");
        args.push(&t5_str);
        args.push("--vae");
        args.push(&vae_str);
        args.push("--prompt");
        args.push(&request.prompt);
        args.push("--width");
        args.push(&width_str);
        args.push("--height");
        args.push(&height_str);
        args.push("--video-frames");
        args.push(&num_frames_str);
        args.push("--fps");
        args.push(&fps_str);
        args.push("--sampling-steps");
        args.push(&steps_str);
        args.push("--sampling-method");
        args.push(DEFAULT_SAMPLING_METHOD);
        args.push("--guidance-scale");
        args.push(&guidance_str);
        args.push("--seed");
        args.push("-1");
        args.push("--mode");
        args.push("vid_gen");
        args.push("--verbose");
        args.push("--output");
        args.push(&output_str);

        // Run the Python script and read output MP4.
        let video_bytes = pymlx::run_and_read_output(
            &self.python,
            &self.script,
            &args,
            &output_path,
            TIMEOUT,
        ).context("Wan2.2 GGUF video inference failed")?;

        Ok(video_bytes)
    }
}

/// Resolve the model directory from env var or default location.
fn resolve_model_dir() -> Result<PathBuf> {
    if let Ok(dir) = std::env::var("OMINIX_WAN22_MODEL_DIR") {
        let p = PathBuf::from(&dir);
        if p.is_dir() {
            return Ok(p);
        }
        tracing::warn!(
            "OMINIX_WAN22_MODEL_DIR={} does not exist, falling back to default",
            dir
        );
    }

    let home = std::env::var("HOME")
        .context("HOME env var not set")?;
    let default_dir = PathBuf::from(home).join(DEFAULT_MODEL_SUBDIR);

    if !default_dir.is_dir() {
        return Err(eyre::eyre!(
            "Model directory not found: {}. Set OMINIX_WAN22_MODEL_DIR or place models there.",
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

        let expected_ext = format!(".{}", extension);
        if !file_name.ends_with(&expected_ext) {
            continue;
        }

        for keyword in keywords {
            if file_name.contains(&keyword.to_lowercase()) {
                return Some(path);
            }
        }
    }

    None
}

/// Find a VAE model file — accepts either .gguf or .safetensors extension.
fn find_vae_file(dir: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let file_name = path.file_name()?.to_string_lossy().to_lowercase();

        if !file_name.contains("vae") {
            continue;
        }

        if file_name.ends_with(".gguf") || file_name.ends_with(".safetensors") {
            return Some(path);
        }
    }

    None
}

/// Round a value to the nearest multiple of `m`.
fn round_to_multiple(value: u32, m: u32) -> u32 {
    ((value + m / 2) / m) * m
}

/// Round num_frames to the nearest valid value of the form 4n+1.
/// Valid values: 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, ...
fn round_to_4n_plus_1(frames: i32) -> i32 {
    if frames <= 1 {
        return 1;
    }
    // Find nearest 4n+1: subtract 1, round to nearest multiple of 4, add 1.
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
        assert_eq!(round_to_multiple(500, 32), 512);
        assert_eq!(round_to_multiple(832, 32), 832);
    }

    #[test]
    fn test_round_to_4n_plus_1() {
        assert_eq!(round_to_4n_plus_1(49), 49);
        assert_eq!(round_to_4n_plus_1(50), 49);
        assert_eq!(round_to_4n_plus_1(51), 53);
        assert_eq!(round_to_4n_plus_1(33), 33);
        assert_eq!(round_to_4n_plus_1(32), 33);
        assert_eq!(round_to_4n_plus_1(65), 65);
        assert_eq!(round_to_4n_plus_1(1), 1);
        assert_eq!(round_to_4n_plus_1(0), 1);
    }
}
