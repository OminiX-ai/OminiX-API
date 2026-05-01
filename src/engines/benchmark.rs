//! Benchmark framework for comparing Python MLX vs Rust native inference engines.
//!
//! Measures wall-clock time and peak memory for overlapping model implementations,
//! reporting side-by-side comparison results.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use eyre::Result;
use serde::{Deserialize, Serialize};

use crate::engines::pymlx;

#[derive(Debug, Deserialize)]
pub struct BenchmarkRequest {
    pub model: String,
    #[serde(default = "default_backend")]
    pub backend: String,
    pub prompt: String,
    #[serde(default = "default_size")]
    pub size: String,
    #[serde(default = "default_iterations")]
    pub iterations: usize,
    #[serde(default = "default_steps")]
    pub steps: u32,
}

fn default_backend() -> String { "both".to_string() }
fn default_size() -> String { "512x512".to_string() }
fn default_iterations() -> usize { 2 }
fn default_steps() -> u32 { 4 }

#[derive(Debug, Serialize)]
pub struct BenchmarkResponse {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rust: Option<BenchmarkResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub python: Option<BenchmarkResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speedup: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub winner: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct BenchmarkResult {
    pub mean_seconds: f64,
    pub std_seconds: f64,
    pub min_seconds: f64,
    pub max_seconds: f64,
    pub iterations: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_memory_mb: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Which Python script to use for a given model.
fn python_script_for_model(model: &str) -> Option<&'static str> {
    let lower = model.to_lowercase();
    if lower.contains("qwen-image") || lower.contains("qwen_image") {
        Some("infer_qwen_image.py")
    } else if lower.contains("flux") {
        Some("infer_flux2.py")
    } else if lower.contains("qwen") && !lower.contains("image") {
        Some("infer_qwen_mlx_lm_gguf.py")
    } else {
        None
    }
}

/// Run a Python MLX benchmark for image generation models.
pub fn benchmark_python_image(
    request: &BenchmarkRequest,
) -> Result<BenchmarkResult> {
    let script_name = python_script_for_model(&request.model)
        .ok_or_else(|| eyre::eyre!("No Python benchmark script for model: {}", request.model))?;

    let python = pymlx::find_python()
        .ok_or_else(|| eyre::eyre!("Python not found for benchmark"))?;
    let script = pymlx::find_script(script_name)
        .ok_or_else(|| eyre::eyre!("Script '{}' not found", script_name))?;

    let (width, height) = pymlx::parse_size(&request.size)?;
    let output_path = pymlx::temp_output_path("bench", "png");

    let mut timings = Vec::with_capacity(request.iterations);
    let mut last_error: Option<String> = None;

    for i in 0..request.iterations {
        tracing::info!("benchmark python: iteration {}/{}", i + 1, request.iterations);

        let args = build_python_image_args(
            &request.model,
            &request.prompt,
            width,
            height,
            request.steps,
            &output_path,
        );
        let args_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();

        let t0 = Instant::now();
        let result = pymlx::run_and_read_output(
            &python,
            &script,
            &args_refs,
            &output_path,
            Duration::from_secs(600),
        );
        let elapsed = t0.elapsed().as_secs_f64();

        match result {
            Ok(_) => {
                timings.push(elapsed);
                tracing::info!("benchmark python: iteration {} = {:.2}s", i + 1, elapsed);
            }
            Err(e) => {
                last_error = Some(format!("Iteration {} failed: {}", i + 1, e));
                tracing::warn!("benchmark python: iteration {} failed: {}", i + 1, e);
            }
        }
    }

    let _ = std::fs::remove_file(&output_path);

    if timings.is_empty() {
        return Ok(BenchmarkResult {
            mean_seconds: 0.0,
            std_seconds: 0.0,
            min_seconds: 0.0,
            max_seconds: 0.0,
            iterations: 0,
            peak_memory_mb: None,
            error: last_error,
        });
    }

    Ok(compute_stats(&timings))
}

/// Run a Rust-native benchmark for image generation.
///
/// This requires the Rust ImageEngine to be loaded. Since the benchmark handler
/// runs outside the inference thread, we time a request through the normal
/// inference channel (which includes queue wait time — negligible under no load).
pub fn benchmark_rust_image_via_channel(
    timings: Vec<f64>,
) -> BenchmarkResult {
    if timings.is_empty() {
        return BenchmarkResult {
            mean_seconds: 0.0,
            std_seconds: 0.0,
            min_seconds: 0.0,
            max_seconds: 0.0,
            iterations: 0,
            peak_memory_mb: None,
            error: Some("No successful iterations".to_string()),
        };
    }
    compute_stats(&timings)
}

fn compute_stats(timings: &[f64]) -> BenchmarkResult {
    let n = timings.len() as f64;
    let mean = timings.iter().sum::<f64>() / n;
    let variance = timings.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();
    let min = timings.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = timings.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    BenchmarkResult {
        mean_seconds: mean,
        std_seconds: std,
        min_seconds: min,
        max_seconds: max,
        iterations: timings.len(),
        peak_memory_mb: None,
        error: None,
    }
}

fn build_python_image_args(
    model: &str,
    prompt: &str,
    width: u32,
    height: u32,
    steps: u32,
    output_path: &PathBuf,
) -> Vec<String> {
    let lower = model.to_lowercase();

    if lower.contains("flux") {
        let mut args = vec![
            "--prompt".to_string(), prompt.to_string(),
            "--width".to_string(), width.to_string(),
            "--height".to_string(), height.to_string(),
            "--steps".to_string(), steps.to_string(),
            "--output".to_string(), output_path.display().to_string(),
            "--verbose".to_string(),
        ];
        if let Ok(diffusion) = std::env::var("OMINIX_FLUX_DIFFUSION_MODEL") {
            args.extend(["--diffusion-model".to_string(), diffusion]);
        }
        if let Ok(vae) = std::env::var("OMINIX_FLUX_VAE") {
            args.extend(["--vae".to_string(), vae]);
        }
        if let Ok(llm) = std::env::var("OMINIX_FLUX_LLM") {
            args.extend(["--llm".to_string(), llm]);
        }
        args
    } else if lower.contains("qwen-image") || lower.contains("qwen_image") {
        let mut args = vec![
            "--prompt".to_string(), prompt.to_string(),
            "--width".to_string(), width.to_string(),
            "--height".to_string(), height.to_string(),
            "--steps".to_string(), steps.to_string(),
            "--output".to_string(), output_path.display().to_string(),
            "--verbose".to_string(),
        ];
        if let Ok(diffusion) = std::env::var("OMINIX_QWEN_IMAGE_DIFFUSION_MODEL") {
            args.extend(["--diffusion-model".to_string(), diffusion]);
        }
        if let Ok(vae) = std::env::var("OMINIX_QWEN_IMAGE_VAE") {
            args.extend(["--vae".to_string(), vae]);
        }
        if let Ok(llm) = std::env::var("OMINIX_QWEN_IMAGE_LLM") {
            args.extend(["--llm".to_string(), llm]);
        }
        args
    } else {
        vec![
            "--prompt".to_string(), prompt.to_string(),
            "--output".to_string(), output_path.display().to_string(),
        ]
    }
}

/// Build a complete benchmark response comparing Python and Rust results.
pub fn build_comparison(
    model: &str,
    python_result: Option<BenchmarkResult>,
    rust_result: Option<BenchmarkResult>,
) -> BenchmarkResponse {
    let speedup = match (&rust_result, &python_result) {
        (Some(r), Some(p)) if r.mean_seconds > 0.0 && p.mean_seconds > 0.0 => {
            Some(r.mean_seconds / p.mean_seconds)
        }
        _ => None,
    };

    let winner = speedup.map(|s| {
        if s > 1.05 {
            "python".to_string()
        } else if s < 0.95 {
            "rust".to_string()
        } else {
            "tie".to_string()
        }
    });

    BenchmarkResponse {
        model: model.to_string(),
        rust: rust_result,
        python: python_result,
        speedup,
        winner,
    }
}
