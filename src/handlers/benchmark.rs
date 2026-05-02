//! Benchmark API handler for comparing Python vs Rust inference performance.

use std::time::{Duration, Instant};

use salvo::prelude::*;

use crate::engines::benchmark::{self, BenchmarkRequest};
use crate::inference::InferenceRequest;
use crate::types::ImageGenerationRequest;

use super::helpers::{get_state, send_and_wait};

const BENCHMARK_TIMEOUT: Duration = Duration::from_secs(900);

/// POST /v1/benchmark — Run inference benchmarks comparing backends
#[handler]
pub async fn run_benchmark(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    let bench_req: BenchmarkRequest = req
        .parse_json_with_max_size(1024 * 1024)
        .await
        .map_err(|e| {
            tracing::error!("Failed to parse benchmark request: {}", e);
            StatusError::bad_request()
        })?;

    tracing::info!(
        "Benchmark: model={} backend={} iterations={} size={}",
        bench_req.model, bench_req.backend, bench_req.iterations, bench_req.size
    );

    let run_python = bench_req.backend == "python" || bench_req.backend == "both";
    let run_rust = bench_req.backend == "rust" || bench_req.backend == "both";

    // Run Python benchmark (blocking, spawned on blocking thread pool)
    let python_result = if run_python {
        let bench_req_clone = BenchmarkRequest {
            model: bench_req.model.clone(),
            backend: bench_req.backend.clone(),
            prompt: bench_req.prompt.clone(),
            size: bench_req.size.clone(),
            iterations: bench_req.iterations,
            steps: bench_req.steps,
        };

        let result = tokio::task::spawn_blocking(move || {
            benchmark::benchmark_python_image(&bench_req_clone)
        })
        .await
        .map_err(|e| {
            tracing::error!("Python benchmark task failed: {}", e);
            StatusError::internal_server_error()
        })?;

        match result {
            Ok(r) => Some(r),
            Err(e) => {
                tracing::warn!("Python benchmark error: {}", e);
                Some(benchmark::BenchmarkResult {
                    mean_seconds: 0.0,
                    std_seconds: 0.0,
                    min_seconds: 0.0,
                    max_seconds: 0.0,
                    iterations: 0,
                    peak_memory_mb: None,
                    error: Some(e.to_string()),
                })
            }
        }
    } else {
        None
    };

    // Run Rust benchmark via the inference channel
    let rust_result = if run_rust {
        let mut timings = Vec::new();

        for i in 0..bench_req.iterations {
            tracing::info!("benchmark rust: iteration {}/{}", i + 1, bench_req.iterations);

            let image_request = ImageGenerationRequest {
                prompt: bench_req.prompt.clone(),
                model: Some(bench_req.model.clone()),
                n: 1,
                size: bench_req.size.clone(),
                response_format: "b64_json".to_string(),
                quality: None,
                image: None,
                strength: 0.75,
            };

            let t0 = Instant::now();
            let result = send_and_wait(
                &state.inference_tx,
                |tx| InferenceRequest::Image { request: image_request, response_tx: tx },
                BENCHMARK_TIMEOUT,
            )
            .await;
            let elapsed = t0.elapsed().as_secs_f64();

            match result {
                Ok(_) => {
                    timings.push(elapsed);
                    tracing::info!("benchmark rust: iteration {} = {:.2}s", i + 1, elapsed);
                }
                Err(e) => {
                    tracing::warn!("benchmark rust: iteration {} failed: {:?}", i + 1, e);
                }
            }
        }

        Some(benchmark::benchmark_rust_image_via_channel(timings))
    } else {
        None
    };

    let response = benchmark::build_comparison(&bench_req.model, python_result, rust_result);

    res.render(Json(response));
    Ok(())
}
