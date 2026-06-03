use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use base64::Engine;
use eyre::{Context, Result};

use crate::types::{VideoData, VideoGenerationRequest, VideoGenerationResponse};

const DEFAULT_WAN_MODEL_DIR: &str = "~/.OminiX/models/wan2.2-5b/mlx_model_4bit";

#[derive(Debug, Clone)]
pub struct PythonMlxVideoConfig {
    pub python_bin: String,
    pub module: String,
    pub model_dir: String,
    pub timeout_secs: u64,
}

impl PythonMlxVideoConfig {
    pub fn from_env(model_hint: Option<&str>) -> Self {
        let python_bin =
            std::env::var("WAN_VIDEO_PYTHON").unwrap_or_else(|_| "python3".to_string());
        let module = std::env::var("WAN_VIDEO_MODULE")
            .unwrap_or_else(|_| "mlx_video.models.wan_2.generate".to_string());
        let timeout_secs = std::env::var("WAN_VIDEO_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1800);
        let model_dir = resolve_model_dir(model_hint);

        Self {
            python_bin,
            module,
            model_dir,
            timeout_secs,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoInvocation {
    pub program: String,
    pub args: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PythonMlxVideoEngine {
    config: PythonMlxVideoConfig,
}

impl PythonMlxVideoEngine {
    pub fn new(config: PythonMlxVideoConfig) -> Self {
        Self { config }
    }

    pub fn generate(&self, request: &VideoGenerationRequest) -> Result<VideoGenerationResponse> {
        if request.n != 1 {
            return Err(eyre::eyre!(
                "Python MLX video backend supports n=1, got {}",
                request.n
            ));
        }

        let output = tempfile::Builder::new()
            .prefix("ominix-video-")
            .suffix(".mp4")
            .tempfile()
            .context("create temporary video output")?;

        let image_tmp = if let Some(image_b64) = request.image.as_deref() {
            Some(write_base64_image_tempfile(image_b64)?)
        } else {
            None
        };

        let image_path = image_tmp.as_ref().map(|tmp| tmp.path());
        let invocation = self.build_invocation(request, output.path(), image_path)?;
        let command_output = run_with_timeout(&invocation, self.config.timeout_secs)?;

        if !command_output.status.success() {
            let stderr = String::from_utf8_lossy(&command_output.stderr);
            let stdout = String::from_utf8_lossy(&command_output.stdout);
            return Err(eyre::eyre!(
                "mlx-video generation failed with status {:?}: stderr={} stdout={}",
                command_output.status.code(),
                stderr.trim(),
                stdout.trim()
            ));
        }

        let video_bytes = std::fs::read(output.path())
            .with_context(|| format!("read generated video {}", output.path().display()))?;
        if video_bytes.is_empty() {
            return Err(eyre::eyre!(
                "mlx-video completed but produced an empty output at {}",
                output.path().display()
            ));
        }

        let b64 = base64::engine::general_purpose::STANDARD.encode(video_bytes);
        Ok(VideoGenerationResponse {
            created: chrono::Utc::now().timestamp(),
            data: vec![VideoData {
                url: None,
                b64_json: Some(b64),
                revised_prompt: None,
            }],
        })
    }

    pub fn build_invocation(
        &self,
        request: &VideoGenerationRequest,
        output_path: &Path,
        image_path: Option<&Path>,
    ) -> Result<VideoInvocation> {
        let (width, height) = parse_size(&request.size)?;
        if request.num_frames < 1 || (request.num_frames - 1) % 4 != 0 {
            return Err(eyre::eyre!(
                "num_frames must satisfy 4n+1, got {}",
                request.num_frames
            ));
        }

        let model_dir = expand_tilde(&self.config.model_dir);
        let mut args = vec![
            "-m".to_string(),
            self.config.module.clone(),
            "--model-dir".to_string(),
            model_dir,
            "--prompt".to_string(),
            request.prompt.clone(),
        ];

        if let Some(image_path) = image_path {
            args.push("--image".to_string());
            args.push(image_path.display().to_string());
        }

        if let Some(negative_prompt) = request.negative_prompt.as_ref() {
            if negative_prompt.is_empty() {
                args.push("--no-negative-prompt".to_string());
            } else {
                args.push("--negative-prompt".to_string());
                args.push(negative_prompt.clone());
            }
        }

        args.extend([
            "--width".to_string(),
            width.to_string(),
            "--height".to_string(),
            height.to_string(),
            "--num-frames".to_string(),
            request.num_frames.to_string(),
        ]);

        if let Some(steps) = request.steps {
            args.push("--steps".to_string());
            args.push(steps.to_string());
        }
        if let Some(guide_scale) = request.guide_scale.as_ref() {
            args.push("--guide-scale".to_string());
            args.push(guide_scale.clone());
        }
        if let Some(seed) = request.seed {
            args.push("--seed".to_string());
            args.push(seed.to_string());
        }

        args.push("--output-path".to_string());
        args.push(output_path.display().to_string());

        if let Some(scheduler) = request.scheduler.as_ref() {
            args.push("--scheduler".to_string());
            args.push(scheduler.clone());
        }
        if let Some(tiling) = request.tiling.as_ref() {
            args.push("--tiling".to_string());
            args.push(tiling.clone());
        }

        Ok(VideoInvocation {
            program: self.config.python_bin.clone(),
            args,
        })
    }
}

fn resolve_model_dir(model_hint: Option<&str>) -> String {
    if let Some(hint) = model_hint.map(str::trim).filter(|s| !s.is_empty()) {
        let expanded = expand_tilde(hint);
        if Path::new(&expanded).exists() || hint.starts_with('/') || hint.starts_with('~') {
            return expanded;
        }
    }

    std::env::var("WAN_VIDEO_MODEL_DIR").unwrap_or_else(|_| DEFAULT_WAN_MODEL_DIR.to_string())
}

fn expand_tilde(path: &str) -> String {
    if path == "~" {
        return dirs::home_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| path.to_string());
    }
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest).display().to_string();
        }
    }
    path.to_string()
}

fn parse_size(size: &str) -> Result<(usize, usize)> {
    let (w, h) = size
        .split_once('x')
        .ok_or_else(|| eyre::eyre!("size must be formatted as WIDTHxHEIGHT, got '{size}'"))?;
    let width = w
        .parse::<usize>()
        .with_context(|| format!("invalid video width '{w}'"))?;
    let height = h
        .parse::<usize>()
        .with_context(|| format!("invalid video height '{h}'"))?;
    if width == 0 || height == 0 {
        return Err(eyre::eyre!("video width and height must be positive"));
    }
    Ok((width, height))
}

fn write_base64_image_tempfile(image_b64: &str) -> Result<tempfile::NamedTempFile> {
    let payload = image_b64
        .split_once(',')
        .map(|(_, payload)| payload)
        .unwrap_or(image_b64);
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload)
        .context("Invalid base64 in image field")?;
    let mut tmp = tempfile::Builder::new()
        .prefix("ominix-video-input-")
        .suffix(".png")
        .tempfile()
        .context("create temporary video input image")?;
    tmp.write_all(&bytes)
        .context("write temporary video input image")?;
    tmp.flush().context("flush temporary video input image")?;
    Ok(tmp)
}

fn run_with_timeout(
    invocation: &VideoInvocation,
    timeout_secs: u64,
) -> Result<std::process::Output> {
    let mut child = Command::new(&invocation.program)
        .args(&invocation.args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| {
            format!(
                "spawn video backend: {} {}",
                invocation.program,
                invocation.args.join(" ")
            )
        })?;

    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    loop {
        if child.try_wait()?.is_some() {
            return child
                .wait_with_output()
                .context("collect video backend output");
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err(eyre::eyre!(
                "mlx-video generation timed out after {} seconds",
                timeout_secs
            ));
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::types::VideoGenerationRequest;

    use super::{PythonMlxVideoConfig, PythonMlxVideoEngine};

    #[test]
    fn builds_mlx_video_wan_command_from_request() {
        let engine = PythonMlxVideoEngine::new(PythonMlxVideoConfig {
            python_bin: "python3".to_string(),
            module: "mlx_video.models.wan_2.generate".to_string(),
            model_dir: "/models/wan22-ti2v-5b-q4".to_string(),
            timeout_secs: 1800,
        });
        let request = VideoGenerationRequest {
            model: Some("wan2.2-5b-4bit-mlx".to_string()),
            prompt: "a quiet lake at sunrise".to_string(),
            negative_prompt: Some("blur, distortion".to_string()),
            image: None,
            n: 1,
            size: "704x1280".to_string(),
            response_format: "b64_json".to_string(),
            num_frames: 41,
            steps: Some(20),
            guide_scale: Some("5.0".to_string()),
            seed: Some(42),
            scheduler: Some("unipc".to_string()),
            tiling: Some("auto".to_string()),
        };

        let invocation = engine
            .build_invocation(&request, Path::new("/tmp/out.mp4"), None)
            .expect("request should build a valid invocation");

        assert_eq!(invocation.program, "python3");
        assert_eq!(
            invocation.args,
            vec![
                "-m",
                "mlx_video.models.wan_2.generate",
                "--model-dir",
                "/models/wan22-ti2v-5b-q4",
                "--prompt",
                "a quiet lake at sunrise",
                "--negative-prompt",
                "blur, distortion",
                "--width",
                "704",
                "--height",
                "1280",
                "--num-frames",
                "41",
                "--steps",
                "20",
                "--guide-scale",
                "5.0",
                "--seed",
                "42",
                "--output-path",
                "/tmp/out.mp4",
                "--scheduler",
                "unipc",
                "--tiling",
                "auto",
            ]
        );
    }
}
