//! Model download thread
//!
//! Handles downloading models from HuggingFace and ModelScope, including
//! special cases like Paraformer PyTorch→MLX conversion and multi-component
//! funasr-qwen4b downloads.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::{broadcast, mpsc, oneshot};

use crate::model_registry::{CatalogModel, SourceType};
use crate::types::*;

/// Cancel flags: map of model_id → AtomicBool
pub type DownloadCancelFlags = Arc<std::sync::Mutex<HashMap<String, Arc<AtomicBool>>>>;

/// Messages sent to the download thread
pub enum DownloadRequest {
    StartDownload {
        task_id: String,
        spec: CatalogModel,
        response_tx: oneshot::Sender<Result<(), String>>,
    },
}

// ============================================================================
// HuggingFace / ModelScope API types
// ============================================================================

#[derive(serde::Deserialize)]
struct HuggingFaceItem {
    #[serde(rename = "type")]
    item_type: String,
    path: String,
    size: Option<u64>,
}

#[derive(serde::Deserialize)]
struct ModelScopeResponse {
    #[serde(rename = "Code")]
    code: i32,
    #[serde(rename = "Data")]
    data: Option<ModelScopeData>,
}

#[derive(serde::Deserialize)]
struct ModelScopeData {
    #[serde(rename = "Files")]
    files: Vec<ModelScopeFile>,
}

#[derive(serde::Deserialize)]
struct ModelScopeFile {
    #[serde(rename = "Path")]
    path: String,
    #[serde(rename = "Size")]
    size: u64,
    #[serde(rename = "Type")]
    file_type: String,
}

// ============================================================================
// Progress tracker
// ============================================================================

struct DownloadTracker {
    model_id: String,
    task_id: String,
    state: DownloadState,
    current_file: Option<String>,
    current_file_index: usize,
    total_files: usize,
    downloaded_bytes: u64,
    total_bytes: u64,
    progress_tx: broadcast::Sender<DownloadProgressEvent>,
    last_event_time: Instant,
    speed_start_time: Instant,
    speed_start_bytes: u64,
    cancel_flag: Arc<AtomicBool>,
}

impl DownloadTracker {
    fn new(
        model_id: String,
        task_id: String,
        progress_tx: broadcast::Sender<DownloadProgressEvent>,
        cancel_flag: Arc<AtomicBool>,
    ) -> Self {
        let now = Instant::now();
        Self {
            model_id,
            task_id,
            state: DownloadState::Queued,
            current_file: None,
            current_file_index: 0,
            total_files: 0,
            downloaded_bytes: 0,
            total_bytes: 0,
            progress_tx,
            last_event_time: now,
            speed_start_time: now,
            speed_start_bytes: 0,
            cancel_flag,
        }
    }

    fn is_cancelled(&self) -> bool {
        self.cancel_flag.load(Ordering::Relaxed)
    }

    fn speed_bytes_per_sec(&self) -> u64 {
        let elapsed = self.speed_start_time.elapsed().as_secs_f64();
        if elapsed > 0.5 {
            ((self.downloaded_bytes - self.speed_start_bytes) as f64 / elapsed) as u64
        } else {
            0
        }
    }

    fn send_progress(&mut self) {
        let progress = if self.total_bytes > 0 {
            self.downloaded_bytes as f32 / self.total_bytes as f32
        } else {
            0.0
        };

        let speed = self.speed_bytes_per_sec();
        let eta_seconds = if speed > 0 && self.total_bytes > self.downloaded_bytes {
            Some((self.total_bytes - self.downloaded_bytes) / speed)
        } else {
            None
        };

        let message = match self.state {
            DownloadState::Listing => "Listing files...".to_string(),
            DownloadState::Converting => "Converting model...".to_string(),
            DownloadState::Downloading => {
                if let Some(ref file) = self.current_file {
                    format!(
                        "Downloading [{}/{}]: {}",
                        self.current_file_index + 1,
                        self.total_files,
                        file
                    )
                } else {
                    "Downloading...".to_string()
                }
            }
            _ => format!("{:?}", self.state),
        };

        let event = DownloadProgressEvent {
            model_id: self.model_id.clone(),
            task_id: self.task_id.clone(),
            state: self.state.clone(),
            current_file: self.current_file.clone(),
            current_file_index: self.current_file_index,
            total_files: self.total_files,
            downloaded_bytes: self.downloaded_bytes,
            total_bytes: self.total_bytes,
            progress,
            speed_bytes_per_sec: speed,
            eta_seconds,
            message,
            is_complete: false,
            error: None,
        };
        let _ = self.progress_tx.send(event);
        self.last_event_time = Instant::now();
    }

    fn send_complete(&self, message: &str) {
        let event = DownloadProgressEvent {
            model_id: self.model_id.clone(),
            task_id: self.task_id.clone(),
            state: DownloadState::Ready,
            current_file: None,
            current_file_index: self.total_files,
            total_files: self.total_files,
            downloaded_bytes: self.total_bytes,
            total_bytes: self.total_bytes,
            progress: 1.0,
            speed_bytes_per_sec: 0,
            eta_seconds: None,
            message: message.to_string(),
            is_complete: true,
            error: None,
        };
        let _ = self.progress_tx.send(event);
    }

    fn send_error(&self, error: &str) {
        let event = DownloadProgressEvent {
            model_id: self.model_id.clone(),
            task_id: self.task_id.clone(),
            state: DownloadState::Error,
            current_file: self.current_file.clone(),
            current_file_index: self.current_file_index,
            total_files: self.total_files,
            downloaded_bytes: self.downloaded_bytes,
            total_bytes: self.total_bytes,
            progress: self.downloaded_bytes as f32 / self.total_bytes.max(1) as f32,
            speed_bytes_per_sec: 0,
            eta_seconds: None,
            message: format!("Download failed: {}", error),
            is_complete: true,
            error: Some(error.to_string()),
        };
        let _ = self.progress_tx.send(event);
    }

    fn send_cancelled(&self) {
        let event = DownloadProgressEvent {
            model_id: self.model_id.clone(),
            task_id: self.task_id.clone(),
            state: DownloadState::Cancelled,
            current_file: None,
            current_file_index: self.current_file_index,
            total_files: self.total_files,
            downloaded_bytes: self.downloaded_bytes,
            total_bytes: self.total_bytes,
            progress: self.downloaded_bytes as f32 / self.total_bytes.max(1) as f32,
            speed_bytes_per_sec: 0,
            eta_seconds: None,
            message: "Download cancelled".to_string(),
            is_complete: true,
            error: None,
        };
        let _ = self.progress_tx.send(event);
    }

    fn add_bytes(&mut self, bytes: u64) {
        self.downloaded_bytes += bytes;
        // Send progress event at most every 500ms
        if self.last_event_time.elapsed() > std::time::Duration::from_millis(500) {
            self.send_progress();
        }
    }
}

// ============================================================================
// Download thread
// ============================================================================

/// Download thread entry point — processes download requests sequentially
pub fn download_thread(
    mut rx: mpsc::Receiver<DownloadRequest>,
    progress_tx: broadcast::Sender<DownloadProgressEvent>,
    cancel_flags: DownloadCancelFlags,
) {
    tracing::info!("Download thread started");

    while let Some(request) = rx.blocking_recv() {
        match request {
            DownloadRequest::StartDownload {
                task_id,
                spec,
                response_tx,
            } => {
                let model_id = spec.id.clone();

                // Create cancel flag for this download
                let cancel_flag = Arc::new(AtomicBool::new(false));
                {
                    let mut flags = cancel_flags.lock().unwrap();
                    flags.insert(model_id.clone(), cancel_flag.clone());
                }

                // Acknowledge receipt
                let _ = response_tx.send(Ok(()));

                // Run download
                let mut tracker = DownloadTracker::new(
                    model_id.clone(),
                    task_id.clone(),
                    progress_tx.clone(),
                    cancel_flag.clone(),
                );

                let result = run_download(&spec, &mut tracker);

                // Clean up cancel flag
                {
                    let mut flags = cancel_flags.lock().unwrap();
                    flags.remove(&model_id);
                }

                match result {
                    Ok(()) => {
                        tracing::info!("Download complete: {}", model_id);
                        tracker.send_complete(&format!("Model {} downloaded successfully", model_id));

                        // Register model in config
                        let dest_path = std::path::PathBuf::from(
                            crate::utils::expand_tilde(&spec.storage.local_path),
                        );
                        if let Err(e) = crate::model_config::register_model(
                            spec.source.repo_id.as_deref().unwrap_or(&model_id),
                            spec.category.clone(),
                            &dest_path,
                        ) {
                            tracing::warn!("Failed to register model in config: {}", e);
                        }
                    }
                    Err(e) => {
                        if cancel_flag.load(Ordering::Relaxed) {
                            tracing::info!("Download cancelled: {}", model_id);
                            tracker.send_cancelled();
                        } else {
                            tracing::error!("Download failed for {}: {}", model_id, e);
                            tracker.send_error(&e);
                        }
                    }
                }
            }
        }
    }

    tracing::info!("Download thread shutting down");
}

// ============================================================================
// Download logic
// ============================================================================

fn run_download(spec: &CatalogModel, tracker: &mut DownloadTracker) -> Result<(), String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(3600))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

    let dest_path = crate::utils::expand_tilde(&spec.storage.local_path);

    // Create destination directory
    std::fs::create_dir_all(&dest_path)
        .map_err(|e| format!("Failed to create model directory: {}", e))?;

    // Special handling for multi-component models
    if spec.id == "funasr-qwen4b" {
        return download_funasr_qwen4b(&client, tracker, &dest_path);
    }

    // Try primary URL first, then backups
    let all_urls: Vec<&str> = std::iter::once(spec.source.primary_url.as_str())
        .chain(spec.source.backup_urls.iter().map(|s| s.as_str()))
        .collect();

    let mut last_error = String::new();

    for (url_index, url) in all_urls.iter().enumerate() {
        if url_index > 0 {
            tracing::info!("Trying backup URL {}: {}", url_index, url);
        }

        let result = match spec.source.source_type {
            SourceType::Modelscope => download_from_modelscope(&client, tracker, url, &dest_path),
            SourceType::Huggingface => download_from_huggingface(&client, tracker, url, &dest_path),
        };

        match result {
            Ok(()) => return Ok(()),
            Err(e) => {
                if tracker.is_cancelled() {
                    return Err("Download cancelled".to_string());
                }
                last_error = e;
                tracing::warn!("Download failed from {}: {}", url, last_error);
            }
        }
    }

    Err(last_error)
}

/// Read HuggingFace token from ~/.cache/huggingface/token if available.
/// Cached per download session to avoid repeated disk reads.
fn get_hf_token() -> Option<String> {
    use std::sync::OnceLock;
    static TOKEN: OnceLock<Option<String>> = OnceLock::new();
    TOKEN
        .get_or_init(|| {
            let token_path = dirs::home_dir()?.join(".cache/huggingface/token");
            std::fs::read_to_string(token_path)
                .ok()
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
        })
        .clone()
}

/// Extract the base URL (host) from a HuggingFace-style URL
fn hf_base_url(url: &str) -> &str {
    if url.contains("hf-mirror.com") {
        "https://hf-mirror.com"
    } else {
        "https://huggingface.co"
    }
}

/// Download from HuggingFace
fn download_from_huggingface(
    client: &reqwest::blocking::Client,
    tracker: &mut DownloadTracker,
    url: &str,
    dest_path: &str,
) -> Result<(), String> {
    let repo_id = parse_huggingface_repo_id(url)?;
    let base = hf_base_url(url);
    tracing::info!("Downloading HuggingFace repo: {} to {} (via {})", repo_id, dest_path, base);

    let hf_token = get_hf_token();

    // List files
    tracker.state = DownloadState::Listing;
    tracker.send_progress();
    let files = list_huggingface_files(client, &repo_id, "", base, hf_token.as_deref())?;

    // Accumulate total size (don't overwrite — supports multi-phase downloads)
    let phase_size: u64 = files.iter().map(|(_, size)| *size).sum();
    tracker.total_bytes += phase_size;
    tracker.total_files += files.len();
    tracker.state = DownloadState::Downloading;
    tracker.speed_start_time = Instant::now();
    tracker.speed_start_bytes = tracker.downloaded_bytes;
    tracing::info!(
        "Phase download size: {} bytes ({} files)",
        phase_size,
        files.len()
    );

    // Download each file
    for (file_index, (file_path, _file_size)) in files.iter().enumerate() {
        if tracker.is_cancelled() {
            let _ = std::fs::remove_dir_all(dest_path);
            return Err("Download cancelled".to_string());
        }

        tracker.current_file_index = file_index;
        tracker.current_file = Some(file_path.clone());
        tracker.send_progress();

        let local_path = Path::new(dest_path).join(file_path);
        if let Some(parent) = local_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory: {}", e))?;
        }

        let download_url = format!(
            "{}/{}/resolve/main/{}",
            base, repo_id, file_path
        );
        tracing::info!(
            "Downloading [{}/{}]: {}",
            file_index + 1,
            files.len(),
            file_path
        );
        download_file_streaming(client, &download_url, &local_path, tracker, hf_token.as_deref())?;
    }

    tracing::info!("Download complete: {}", dest_path);
    Ok(())
}

/// Download from ModelScope (with automatic PyTorch→MLX conversion for Paraformer)
fn download_from_modelscope(
    client: &reqwest::blocking::Client,
    tracker: &mut DownloadTracker,
    url: &str,
    dest_path: &str,
) -> Result<(), String> {
    let model_id = parse_modelscope_model_id(url)?;
    let is_paraformer = url.contains("paraformer");

    // For Paraformer, download to temp dir first, then convert
    let download_dir = if is_paraformer {
        let temp_dir = std::env::temp_dir().join("ominix-paraformer-download");
        temp_dir.to_string_lossy().to_string()
    } else {
        dest_path.to_string()
    };

    tracing::info!(
        "Downloading ModelScope model: {} to {}",
        model_id,
        download_dir
    );

    std::fs::create_dir_all(&download_dir)
        .map_err(|e| format!("Failed to create directory: {}", e))?;

    // List files
    tracker.state = DownloadState::Listing;
    tracker.send_progress();
    let files = list_modelscope_files(client, &model_id, "")?;

    // Calculate total size (add 10% for conversion overhead)
    let download_size: u64 = files.iter().map(|(_, size)| *size).sum();
    let phase_size = if is_paraformer {
        download_size + download_size / 10
    } else {
        download_size
    };
    tracker.total_bytes += phase_size;
    tracker.total_files += files.len();
    tracker.state = DownloadState::Downloading;
    tracker.speed_start_time = Instant::now();
    tracker.speed_start_bytes = tracker.downloaded_bytes;
    tracing::info!(
        "Total download size: {} bytes ({} files)",
        download_size,
        files.len()
    );

    // Download each file
    for (file_index, (file_path, _file_size)) in files.iter().enumerate() {
        if tracker.is_cancelled() {
            let _ = std::fs::remove_dir_all(&download_dir);
            return Err("Download cancelled".to_string());
        }

        tracker.current_file_index = file_index;
        tracker.current_file = Some(file_path.clone());
        tracker.send_progress();

        let local_path = Path::new(&download_dir).join(file_path);
        if let Some(parent) = local_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory: {}", e))?;
        }

        let download_url = format!(
            "https://modelscope.cn/models/{}/resolve/master/{}",
            model_id, file_path
        );
        tracing::info!(
            "Downloading [{}/{}]: {}",
            file_index + 1,
            files.len(),
            file_path
        );
        download_file_streaming(client, &download_url, &local_path, tracker, None)?;
    }

    // Convert Paraformer model from PyTorch to MLX format
    if is_paraformer {
        tracing::info!("Converting Paraformer model to MLX format...");
        tracker.state = DownloadState::Converting;
        tracker.current_file = Some("PyTorch → MLX conversion".to_string());
        tracker.send_progress();

        let input_dir = Path::new(&download_dir);
        let output_dir = Path::new(dest_path);

        std::fs::create_dir_all(output_dir)
            .map_err(|e| format!("Failed to create output directory: {}", e))?;

        let (converted, unmapped) =
            mlx_rs_core::convert::convert_paraformer(input_dir, output_dir)
                .map_err(|e| format!("Conversion failed: {}", e))?;

        tracing::info!("Converted {} tensors ({} unmapped)", converted, unmapped);

        // Update progress to 100%
        tracker.downloaded_bytes = tracker.total_bytes;

        // Clean up temp directory
        let _ = std::fs::remove_dir_all(&download_dir);

        tracing::info!("Paraformer conversion complete: {}", dest_path);
    }

    Ok(())
}

/// Download FunASR-Qwen4B multi-component ASR model
fn download_funasr_qwen4b(
    client: &reqwest::blocking::Client,
    tracker: &mut DownloadTracker,
    dest_path: &str,
) -> Result<(), String> {
    tracing::info!(
        "Downloading FunASR-Qwen4B multi-component ASR model to {}",
        dest_path
    );

    // Phase 1: Qwen3-4B-8bit from HuggingFace
    let qwen_dest = Path::new(dest_path).join("models").join("Qwen3-4B-8bit");
    std::fs::create_dir_all(&qwen_dest)
        .map_err(|e| format!("Failed to create Qwen3 directory: {}", e))?;

    tracing::info!("Phase 1/2: Downloading Qwen3-4B-8bit...");
    tracker.current_file = Some("Qwen3-4B-8bit".to_string());
    download_from_huggingface(
        client,
        tracker,
        "https://huggingface.co/mlx-community/Qwen3-4B-8bit",
        qwen_dest.to_str().unwrap(),
    )?;

    if tracker.is_cancelled() {
        return Err("Download cancelled".to_string());
    }

    // Phase 2: SenseVoice encoder + Adaptor from custom repo
    tracing::info!("Phase 2/2: Downloading SenseVoice encoder and adaptor...");
    tracker.current_file = Some("sensevoice + adaptor".to_string());
    download_from_huggingface(
        client,
        tracker,
        "https://huggingface.co/yuechen/funasr-qwen4b-mlx",
        dest_path,
    )?;

    tracing::info!("FunASR-Qwen4B download complete: {}", dest_path);
    Ok(())
}

// ============================================================================
// File listing
// ============================================================================

/// List files in a HuggingFace repository recursively
fn list_huggingface_files(
    client: &reqwest::blocking::Client,
    repo_id: &str,
    path_prefix: &str,
    base_url: &str,
    token: Option<&str>,
) -> Result<Vec<(String, u64)>, String> {
    let api_url = if path_prefix.is_empty() {
        format!("{}/api/models/{}/tree/main", base_url, repo_id)
    } else {
        format!(
            "{}/api/models/{}/tree/main/{}",
            base_url, repo_id, path_prefix
        )
    };

    let mut req = client
        .get(&api_url)
        .header("User-Agent", "ominix-api/1.0");

    if let Some(token) = token {
        req = req.header("Authorization", format!("Bearer {}", token));
    }

    let response = req
        .send()
        .map_err(|e| format!("Failed to list files: {}", e))?;

    if !response.status().is_success() {
        return Err(format!(
            "Failed to list files: HTTP {}",
            response.status()
        ));
    }

    let items: Vec<HuggingFaceItem> = response
        .json()
        .map_err(|e| format!("Failed to parse file list: {}", e))?;

    let mut files = Vec::new();
    for item in items {
        if item.item_type == "file" {
            files.push((item.path, item.size.unwrap_or(0)));
        } else if item.item_type == "directory" {
            let sub_files = list_huggingface_files(client, repo_id, &item.path, base_url, token)?;
            files.extend(sub_files);
        }
    }

    Ok(files)
}

/// List files in a ModelScope repository recursively
fn list_modelscope_files(
    client: &reqwest::blocking::Client,
    model_id: &str,
    path_prefix: &str,
) -> Result<Vec<(String, u64)>, String> {
    let api_url = if path_prefix.is_empty() {
        format!(
            "https://modelscope.cn/api/v1/models/{}/repo/files",
            model_id
        )
    } else {
        format!(
            "https://modelscope.cn/api/v1/models/{}/repo/files?Root={}",
            model_id, path_prefix
        )
    };

    let response = client
        .get(&api_url)
        .header("User-Agent", "ominix-api/1.0")
        .send()
        .map_err(|e| format!("Failed to list files: {}", e))?;

    if !response.status().is_success() {
        return Err(format!(
            "Failed to list files: HTTP {}",
            response.status()
        ));
    }

    let api_response: ModelScopeResponse = response
        .json()
        .map_err(|e| format!("Failed to parse file list: {}", e))?;

    if api_response.code != 200 {
        return Err(format!(
            "ModelScope API error: code {}",
            api_response.code
        ));
    }

    let data = api_response
        .data
        .ok_or("No data in ModelScope response")?;

    let mut files = Vec::new();
    for item in data.files {
        if item.file_type == "blob" {
            files.push((item.path, item.size));
        } else if item.file_type == "tree" {
            let sub_files = list_modelscope_files(client, model_id, &item.path)?;
            files.extend(sub_files);
        }
    }

    Ok(files)
}

// ============================================================================
// Streaming download
// ============================================================================

/// Download a file with streaming and progress tracking
fn download_file_streaming(
    client: &reqwest::blocking::Client,
    url: &str,
    local_path: &Path,
    tracker: &mut DownloadTracker,
    auth_token: Option<&str>,
) -> Result<(), String> {
    let mut req = client
        .get(url)
        .header("User-Agent", "ominix-api/1.0");

    if let Some(token) = auth_token {
        req = req.header("Authorization", format!("Bearer {}", token));
    }

    let response = req
        .send()
        .map_err(|e| format!("Failed to download: {}", e))?;

    if !response.status().is_success() {
        return Err(format!(
            "Failed to download: HTTP {}",
            response.status()
        ));
    }

    let mut file = std::fs::File::create(local_path)
        .map_err(|e| format!("Failed to create file: {}", e))?;

    let mut reader = response;
    let mut buffer = [0u8; 8192];

    loop {
        if tracker.is_cancelled() {
            return Err("Download cancelled".to_string());
        }

        let bytes_read = reader
            .read(&mut buffer)
            .map_err(|e| format!("Failed to read data: {}", e))?;

        if bytes_read == 0 {
            break;
        }

        file.write_all(&buffer[..bytes_read])
            .map_err(|e| format!("Failed to write data: {}", e))?;

        tracker.add_bytes(bytes_read as u64);
    }

    Ok(())
}

// ============================================================================
// URL parsing
// ============================================================================

/// Parse HuggingFace URL to extract repo ID (org/repo)
fn parse_huggingface_repo_id(url: &str) -> Result<String, String> {
    let url = url.trim_end_matches('/');

    // Try huggingface.co
    if let Some(stripped) = url.strip_prefix("https://huggingface.co/") {
        let parts: Vec<&str> = stripped.split('/').collect();
        if parts.len() >= 2 {
            return Ok(format!("{}/{}", parts[0], parts[1]));
        }
    }

    // Try hf-mirror.com
    if let Some(stripped) = url.strip_prefix("https://hf-mirror.com/") {
        let parts: Vec<&str> = stripped.split('/').collect();
        if parts.len() >= 2 {
            return Ok(format!("{}/{}", parts[0], parts[1]));
        }
    }

    Err(format!("Invalid HuggingFace URL: {}", url))
}

/// Parse ModelScope URL to extract model ID (org/model)
fn parse_modelscope_model_id(url: &str) -> Result<String, String> {
    let url = url.trim_end_matches('/');

    if let Some(stripped) = url.strip_prefix("https://modelscope.cn/models/") {
        let parts: Vec<&str> = stripped.split('/').collect();
        if parts.len() >= 2 {
            return Ok(format!("{}/{}", parts[0], parts[1]));
        }
    }

    Err(format!("Invalid ModelScope URL: {}", url))
}

// ============================================================================
// Cancel helper
// ============================================================================

/// Cancel a download by model_id. Returns Ok if the flag was set.
pub fn cancel_download(cancel_flags: &DownloadCancelFlags, model_id: &str) -> Result<(), String> {
    let flags = cancel_flags.lock().unwrap();
    if let Some(flag) = flags.get(model_id) {
        flag.store(true, Ordering::Relaxed);
        Ok(())
    } else {
        Err(format!("No active download for model: {}", model_id))
    }
}

/// Remove a downloaded model from disk and update config
pub fn remove_model(model_id: &str) -> Result<String, String> {
    // Look up the model in registry or config
    let dest_path = if let Some(spec) = crate::model_registry::get_download_spec(model_id) {
        crate::utils::expand_tilde(&spec.storage.local_path)
    } else {
        // Try to find in config
        let config = crate::model_config::LocalModelsConfig::load()
            .ok_or_else(|| format!("Model '{}' not found in catalog or config", model_id))?;
        let model = config
            .find_by_id(model_id)
            .ok_or_else(|| format!("Model '{}' not found", model_id))?;
        crate::utils::expand_tilde(&model.storage.local_path)
    };

    let path = std::path::PathBuf::from(&dest_path);
    if !path.exists() {
        return Err(format!("Model directory does not exist: {}", dest_path));
    }

    // Remove the directory
    std::fs::remove_dir_all(&path)
        .map_err(|e| format!("Failed to remove model directory: {}", e))?;

    // Update config: set status to not_downloaded
    if let Some(mut config) = crate::model_config::LocalModelsConfig::load() {
        if let Some(model) = config.models.iter_mut().find(|m| m.id == model_id) {
            model.status.state = "not_downloaded".to_string();
            model.status.downloaded_bytes = None;
            model.status.downloaded_files = None;
        }
        config.last_updated = Some(chrono::Utc::now().to_rfc3339());
        let _ = config.save();
    }

    Ok(format!("Removed model '{}' from {}", model_id, dest_path))
}
