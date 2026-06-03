//! stable-diffusion.cpp subprocess image-edit backend.
//!
//! Spawns the `sd-cli` binary from leejet/stable-diffusion.cpp. Used for
//! Qwen-Image-Edit-2511 on 16GB Apple Silicon, where MLX-native paths run
//! into the Q4_0 naive-quantization precision ceiling (see
//! `Moxin-Studio/docs/2026-05-31-qwen-image-edit-2511-implementation-report.md`
//! §3.8). sd.cpp uses ggml k-quant which is outlier-aware, so the same model
//! at the same memory budget actually converges.
//!
//! Subprocess pattern mirrors `engines::ascend::AscendImageEngine` (different
//! binary, no CANN env vars).

use std::path::PathBuf;
use std::process::{Command, Stdio};

use eyre::{Context, Result};

use crate::types::{
    ImageData, ImageGenerationRequest, ImageGenerationResponse,
};

// ============================================================================
// Configuration
// ============================================================================

#[derive(Debug, Clone)]
pub struct SdCppConfig {
    pub sd_bin: PathBuf,
    pub diffusion_model: PathBuf,
    pub vae: PathBuf,
    pub llm: PathBuf,
    pub steps: u32,
    pub cfg_scale: f32,
    pub flow_shift: f32,
    pub offload_to_cpu: bool,
    /// Required for Qwen-Image-Edit-2511 — without it, edit quality
    /// degrades significantly per sd.cpp docs/qwen_image_edit.md.
    pub qwen_image_zero_cond_t: bool,
    /// Keep VAE on CPU. Required on Apple Silicon — current sd.cpp Metal
    /// backend lacks `PAD` op, so the WAN VAE encoder aborts when run on
    /// Metal. Verified 2026-05-31 on M4.
    pub vae_on_cpu: bool,
    /// Keep CLIP/text-encoder on CPU. Same Metal-op-coverage reason as VAE.
    pub clip_on_cpu: bool,
}

impl SdCppConfig {
    /// Build from environment variables. Returns `None` if any required path
    /// is missing.
    pub fn from_env() -> Option<Self> {
        let sd_bin = std::env::var("SDCPP_BIN").ok().map(PathBuf::from)?;
        let diffusion_model = std::env::var("SDCPP_DIFFUSION_MODEL").ok().map(PathBuf::from)?;
        let vae = std::env::var("SDCPP_VAE").ok().map(PathBuf::from)?;
        let llm = std::env::var("SDCPP_LLM").ok().map(PathBuf::from)?;

        let steps = std::env::var("SDCPP_STEPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(20);
        let cfg_scale = std::env::var("SDCPP_CFG_SCALE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2.5);
        let flow_shift = std::env::var("SDCPP_FLOW_SHIFT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3.0);
        let offload_to_cpu = std::env::var("SDCPP_OFFLOAD_TO_CPU")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(true);
        let qwen_image_zero_cond_t = std::env::var("SDCPP_QWEN_IMAGE_ZERO_COND_T")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(true);
        let vae_on_cpu = std::env::var("SDCPP_VAE_ON_CPU")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(true);
        let clip_on_cpu = std::env::var("SDCPP_CLIP_ON_CPU")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(true);

        Some(Self {
            sd_bin,
            diffusion_model,
            vae,
            llm,
            steps,
            cfg_scale,
            flow_shift,
            offload_to_cpu,
            qwen_image_zero_cond_t,
            vae_on_cpu,
            clip_on_cpu,
        })
    }

    /// True iff all required binaries and weight files exist on disk.
    pub fn is_available(&self) -> bool {
        self.sd_bin.exists()
            && self.diffusion_model.exists()
            && self.vae.exists()
            && self.llm.exists()
    }
}

// ============================================================================
// Argument builder (pure, for TDD)
// ============================================================================

/// Build the argv passed to `sd-cli` (excluding the binary itself).
///
/// `ref_img` is the path to the reference image for edit; passed as `-r`
/// (`--ref-image`), NOT `-i` (`--init-img`). Qwen-Image-Edit uses the
/// Flux-Kontext-style reference channel.
pub fn build_sd_args(
    cfg: &SdCppConfig,
    prompt: &str,
    w: u32,
    h: u32,
    out: &str,
    ref_img: Option<&str>,
) -> Vec<String> {
    let mut a: Vec<String> = vec![
        "--diffusion-model".into(),
        cfg.diffusion_model.to_string_lossy().into_owned(),
        "--vae".into(),
        cfg.vae.to_string_lossy().into_owned(),
        "--llm".into(),
        cfg.llm.to_string_lossy().into_owned(),
        "--cfg-scale".into(),
        cfg.cfg_scale.to_string(),
        "--sampling-method".into(),
        "euler".into(),
        "--diffusion-fa".into(),
        "--flow-shift".into(),
        cfg.flow_shift.to_string(),
        "--steps".into(),
        cfg.steps.to_string(),
        "-W".into(),
        w.to_string(),
        "-H".into(),
        h.to_string(),
        "-p".into(),
        prompt.into(),
        "-o".into(),
        out.into(),
    ];
    if cfg.offload_to_cpu {
        a.push("--offload-to-cpu".into());
    }
    if cfg.qwen_image_zero_cond_t {
        a.push("--qwen-image-zero-cond-t".into());
    }
    if cfg.vae_on_cpu {
        a.push("--vae-on-cpu".into());
    }
    if cfg.clip_on_cpu {
        a.push("--clip-on-cpu".into());
    }
    if let Some(r) = ref_img {
        a.push("-r".into());
        a.push(r.into());
    }
    a
}

// ============================================================================
// Engine
// ============================================================================

pub struct SdCppImageEngine {
    config: SdCppConfig,
}

impl SdCppImageEngine {
    pub fn new(config: SdCppConfig) -> Result<Self> {
        if !config.sd_bin.exists() {
            eyre::bail!("sd-cli binary not found: {}", config.sd_bin.display());
        }
        if !config.diffusion_model.exists() {
            eyre::bail!(
                "Diffusion model not found: {}",
                config.diffusion_model.display()
            );
        }
        if !config.vae.exists() {
            eyre::bail!("VAE file not found: {}", config.vae.display());
        }
        if !config.llm.exists() {
            eyre::bail!("LLM file not found: {}", config.llm.display());
        }
        tracing::info!(
            "sd.cpp image-edit ready: bin={}, diff={}, vae={}, llm={}",
            config.sd_bin.display(),
            config.diffusion_model.display(),
            config.vae.display(),
            config.llm.display(),
        );
        Ok(Self { config })
    }

    pub fn generate(&self, request: &ImageGenerationRequest) -> Result<ImageGenerationResponse> {
        let (width, height) = parse_size(&request.size);

        let tmp_output = tempfile::Builder::new()
            .suffix(".png")
            .tempfile()
            .context("Failed to create temp output file")?;

        // Reference image is required for edit. Write b64 to a temp file so
        // sd-cli can read it as a path via `-r`.
        let _tmp_ref_guard;
        let ref_path_str: Option<String> = match request.image.as_deref() {
            Some(b64) => {
                use base64::Engine;
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(b64)
                    .context("Invalid base64 in image field")?;
                let mut tmp = tempfile::Builder::new()
                    .suffix(".png")
                    .tempfile()
                    .context("Failed to create temp ref-image file")?;
                std::io::Write::write_all(&mut tmp, &bytes)?;
                let p = tmp.path().to_string_lossy().into_owned();
                _tmp_ref_guard = tmp; // keep file alive for subprocess
                Some(p)
            }
            None => None,
        };

        let args = build_sd_args(
            &self.config,
            &request.prompt,
            width,
            height,
            tmp_output.path().to_str().unwrap(),
            ref_path_str.as_deref(),
        );

        tracing::info!(
            "Running sd-cli: prompt='{}', {}x{}, ref={}",
            request.prompt,
            width,
            height,
            ref_path_str.as_deref().unwrap_or("<none>")
        );

        let output = Command::new(&self.config.sd_bin)
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .with_context(|| format!("Failed to spawn {}", self.config.sd_bin.display()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let last = stderr.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("unknown");
            eyre::bail!("sd-cli failed: {}", last);
        }

        let png_data =
            std::fs::read(tmp_output.path()).context("Failed to read sd-cli output PNG")?;

        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&png_data);

        Ok(ImageGenerationResponse {
            created: chrono::Utc::now().timestamp(),
            data: vec![ImageData {
                url: None,
                b64_json: Some(b64),
                revised_prompt: None,
            }],
        })
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn parse_size(size: &str) -> (u32, u32) {
    let parts: Vec<&str> = size.split('x').collect();
    if parts.len() == 2 {
        let w = parts[0].parse().unwrap_or(512);
        let h = parts[1].parse().unwrap_or(512);
        (w, h)
    } else {
        (512, 512)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_for_test() -> SdCppConfig {
        SdCppConfig {
            sd_bin: PathBuf::from("/opt/sd/bin/sd-cli"),
            diffusion_model: PathBuf::from("/m/qwen-image-edit-2511-Q2_K.gguf"),
            vae: PathBuf::from("/m/qwen_image_vae.safetensors"),
            llm: PathBuf::from("/m/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"),
            steps: 20,
            cfg_scale: 2.5,
            flow_shift: 3.0,
            offload_to_cpu: true,
            qwen_image_zero_cond_t: true,
            vae_on_cpu: true,
            clip_on_cpu: true,
        }
    }

    #[test]
    fn build_sd_args_uses_ref_image_dash_r_for_edit() {
        let cfg = cfg_for_test();
        let args = build_sd_args(&cfg, "make sky sunset", 768, 768, "/tmp/out.png", Some("/tmp/ref.png"));
        // Reference image must be passed as -r (not -i)
        let i = args.iter().position(|a| a == "-r").expect("missing -r");
        assert_eq!(args[i + 1], "/tmp/ref.png");
        assert!(!args.iter().any(|a| a == "-i"), "must not use --init-img for edit");
    }

    #[test]
    fn build_sd_args_includes_qwen_image_zero_cond_t_for_2511() {
        let cfg = cfg_for_test();
        let args = build_sd_args(&cfg, "p", 512, 512, "/tmp/o.png", None);
        assert!(
            args.iter().any(|a| a == "--qwen-image-zero-cond-t"),
            "2511 requires --qwen-image-zero-cond-t per sd.cpp docs"
        );
    }

    #[test]
    fn build_sd_args_includes_offload_to_cpu_for_low_memory() {
        let cfg = cfg_for_test();
        let args = build_sd_args(&cfg, "p", 512, 512, "/tmp/o.png", None);
        assert!(args.iter().any(|a| a == "--offload-to-cpu"));
    }

    #[test]
    fn build_sd_args_keeps_vae_and_clip_on_cpu_for_apple_silicon() {
        // Required: sd.cpp Metal backend (as of 2026-05) lacks PAD op, so
        // VAE encoder aborts if it runs on Metal. CLIP/TE shares similar
        // op-coverage gaps. Verified on M4 against Qwen-Image-Edit-2511.
        let cfg = cfg_for_test();
        let args = build_sd_args(&cfg, "p", 512, 512, "/tmp/o.png", None);
        assert!(args.iter().any(|a| a == "--vae-on-cpu"));
        assert!(args.iter().any(|a| a == "--clip-on-cpu"));
    }

    #[test]
    fn build_sd_args_omits_offload_when_disabled() {
        let mut cfg = cfg_for_test();
        cfg.offload_to_cpu = false;
        let args = build_sd_args(&cfg, "p", 512, 512, "/tmp/o.png", None);
        assert!(!args.iter().any(|a| a == "--offload-to-cpu"));
    }

    #[test]
    fn build_sd_args_threads_paths_correctly() {
        let cfg = cfg_for_test();
        let args = build_sd_args(&cfg, "p", 1024, 768, "/tmp/o.png", None);
        let dm = args.iter().position(|a| a == "--diffusion-model").unwrap();
        assert_eq!(args[dm + 1], "/m/qwen-image-edit-2511-Q2_K.gguf");
        let vae = args.iter().position(|a| a == "--vae").unwrap();
        assert_eq!(args[vae + 1], "/m/qwen_image_vae.safetensors");
        let llm = args.iter().position(|a| a == "--llm").unwrap();
        assert_eq!(args[llm + 1], "/m/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf");
        let w = args.iter().position(|a| a == "-W").unwrap();
        assert_eq!(args[w + 1], "1024");
        let h = args.iter().position(|a| a == "-H").unwrap();
        assert_eq!(args[h + 1], "768");
    }

    #[test]
    fn build_sd_args_uses_euler_sampling_and_flow_shift() {
        let cfg = cfg_for_test();
        let args = build_sd_args(&cfg, "p", 512, 512, "/tmp/o.png", None);
        let sm = args.iter().position(|a| a == "--sampling-method").unwrap();
        assert_eq!(args[sm + 1], "euler");
        assert!(args.iter().any(|a| a == "--diffusion-fa"));
        let fs = args.iter().position(|a| a == "--flow-shift").unwrap();
        assert_eq!(args[fs + 1], "3");
    }
}
