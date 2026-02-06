//! Model configuration for ~/.OminiX/local_models_config.json
//!
//! Reads model configuration from OminiX-Studio, scans hub caches
//! (HuggingFace, ModelScope) on startup, and provides a report API.

use std::collections::HashMap;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};

/// Path to the local models config file
const CONFIG_PATH: &str = "~/.OminiX/local_models_config.json";

/// Model category
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum ModelCategory {
    Image,
    Llm,
    Asr,
    Tts,
}

/// Model source information
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelSource {
    #[allow(dead_code)]
    pub primary_url: Option<String>,
    #[allow(dead_code)]
    pub source_type: Option<String>,
    pub repo_id: Option<String>,
}

/// Model storage information
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelStorage {
    pub local_path: String,
    #[allow(dead_code)]
    pub total_size_bytes: Option<u64>,
    #[allow(dead_code)]
    pub total_size_display: Option<String>,
}

/// Model status information
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelStatusInfo {
    pub state: String,
    #[allow(dead_code)]
    pub downloaded_bytes: Option<u64>,
    #[allow(dead_code)]
    pub downloaded_files: Option<u32>,
}

/// Individual model configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LocalModel {
    pub id: String,
    pub name: String,
    #[allow(dead_code)]
    pub description: Option<String>,
    pub category: ModelCategory,
    pub source: ModelSource,
    pub storage: ModelStorage,
    pub status: ModelStatusInfo,
}

impl LocalModel {
    /// Check if the model is ready (downloaded and available)
    pub fn is_ready(&self) -> bool {
        self.status.state == "ready"
    }

    /// Get the expanded local path (with ~ expanded to home directory)
    pub fn local_path(&self) -> Option<PathBuf> {
        let path = &self.storage.local_path;
        if path.starts_with("~/") {
            dirs::home_dir().map(|home| home.join(&path[2..]))
        } else {
            Some(PathBuf::from(path))
        }
    }

    /// Check if the model files actually exist on disk
    pub fn files_exist(&self) -> bool {
        if let Some(path) = self.local_path() {
            path.exists()
        } else {
            false
        }
    }
}

/// Local models configuration (V2 format)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LocalModelsConfig {
    pub version: String,
    pub last_updated: Option<String>,
    pub models: Vec<LocalModel>,
}

impl LocalModelsConfig {
    /// Create an empty config (for when no file exists yet)
    pub fn default_empty() -> Self {
        Self {
            version: "2".to_string(),
            last_updated: None,
            models: Vec::new(),
        }
    }

    /// Load the configuration from ~/.OminiX/local_models_config.json
    pub fn load() -> Option<Self> {
        let config_path = expand_path(CONFIG_PATH)?;

        if !config_path.exists() {
            tracing::debug!("Model config not found at {:?}", config_path);
            return None;
        }

        let content = std::fs::read_to_string(&config_path).ok()?;
        match serde_json::from_str(&content) {
            Ok(config) => Some(config),
            Err(e) => {
                tracing::warn!("Failed to parse model config: {}", e);
                None
            }
        }
    }

    /// Save the configuration to ~/.OminiX/local_models_config.json
    pub fn save(&self) -> Result<(), String> {
        let config_path = match expand_path(CONFIG_PATH) {
            Some(p) => p,
            None => return Err("Could not determine config path".to_string()),
        };

        if let Some(parent) = config_path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return Err(format!("Failed to create config directory: {}", e));
            }
        }

        let content = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize config: {}", e))?;

        std::fs::write(&config_path, content)
            .map_err(|e| format!("Failed to write config: {}", e))
    }

    /// Merge a scanned model into the config.
    /// Only adds if no existing entry has the same repo_id or id.
    /// Returns true if the model was added.
    pub fn merge_scanned_model(&mut self, model: LocalModel) -> bool {
        if let Some(ref repo_id) = model.source.repo_id {
            if self.models.iter().any(|m| m.source.repo_id.as_deref() == Some(repo_id)) {
                return false;
            }
        }
        if self.models.iter().any(|m| m.id == model.id) {
            return false;
        }
        self.models.push(model);
        true
    }

    /// Find a model by its ID
    pub fn find_by_id(&self, id: &str) -> Option<&LocalModel> {
        self.models.iter().find(|m| m.id == id)
    }

    /// Find a model by its HuggingFace repo ID
    pub fn find_by_repo_id(&self, repo_id: &str) -> Option<&LocalModel> {
        self.models.iter().find(|m| {
            m.source.repo_id.as_deref() == Some(repo_id)
        })
    }

    /// Find models by category
    pub fn find_by_category(&self, category: ModelCategory) -> Vec<&LocalModel> {
        self.models.iter().filter(|m| m.category == category).collect()
    }

    /// Get all models that are ready
    #[allow(dead_code)]
    pub fn get_ready_models(&self) -> Vec<&LocalModel> {
        self.models.iter().filter(|m| m.is_ready()).collect()
    }

    /// Get all models that are not ready (need download)
    pub fn get_missing_models(&self) -> Vec<&LocalModel> {
        self.models.iter().filter(|m| !m.is_ready()).collect()
    }

    /// Check model availability and return status report
    pub fn check_model_availability(&self, model_id: &str, category: ModelCategory) -> ModelAvailability {
        // First try to find by repo_id (HuggingFace style)
        if let Some(model) = self.find_by_repo_id(model_id) {
            if model.category != category {
                return ModelAvailability::WrongCategory {
                    expected: category,
                    found: model.category.clone(),
                };
            }
            if model.is_ready() && model.files_exist() {
                return ModelAvailability::Ready {
                    local_path: model.local_path(),
                    model_name: model.name.clone(),
                };
            } else {
                return ModelAvailability::NotDownloaded {
                    model_name: model.name.clone(),
                    model_id: model.id.clone(),
                };
            }
        }

        // Try to find by model ID
        if let Some(model) = self.find_by_id(model_id) {
            if model.category != category {
                return ModelAvailability::WrongCategory {
                    expected: category,
                    found: model.category.clone(),
                };
            }
            if model.is_ready() && model.files_exist() {
                return ModelAvailability::Ready {
                    local_path: model.local_path(),
                    model_name: model.name.clone(),
                };
            } else {
                return ModelAvailability::NotDownloaded {
                    model_name: model.name.clone(),
                    model_id: model.id.clone(),
                };
            }
        }

        // Check for partial matches (e.g., "zimage" matches "zimage-turbo")
        let model_id_lower = model_id.to_lowercase();
        for model in &self.models {
            if model.category != category {
                continue;
            }
            let id_lower = model.id.to_lowercase();
            let name_lower = model.name.to_lowercase();

            if id_lower.contains(&model_id_lower) || model_id_lower.contains(&id_lower) ||
               name_lower.contains(&model_id_lower) || model_id_lower.contains(&name_lower) {
                if model.is_ready() && model.files_exist() {
                    return ModelAvailability::Ready {
                        local_path: model.local_path(),
                        model_name: model.name.clone(),
                    };
                } else {
                    return ModelAvailability::NotDownloaded {
                        model_name: model.name.clone(),
                        model_id: model.id.clone(),
                    };
                }
            }
        }

        ModelAvailability::NotInConfig
    }

    /// Print a summary of model availability
    pub fn print_status_report(&self) {
        tracing::info!("=== Model Configuration Status ===");

        for category in [ModelCategory::Llm, ModelCategory::Image, ModelCategory::Asr, ModelCategory::Tts] {
            let models = self.find_by_category(category.clone());
            if models.is_empty() {
                continue;
            }

            let category_name = match category {
                ModelCategory::Llm => "LLM",
                ModelCategory::Image => "Image",
                ModelCategory::Asr => "ASR",
                ModelCategory::Tts => "TTS",
            };

            tracing::info!("--- {} Models ---", category_name);
            for model in models {
                let status = if model.is_ready() && model.files_exist() {
                    "Ready"
                } else if model.is_ready() {
                    "Config says ready but files missing"
                } else {
                    "Not downloaded"
                };
                tracing::info!("  {} ({}): {}", model.name, model.id, status);
            }
        }

        let missing = self.get_missing_models();
        if !missing.is_empty() {
            tracing::warn!("=== Missing Models ===");
            tracing::warn!("The following models need to be downloaded via OminiX-Studio:");
            for model in missing {
                tracing::warn!("  - {} ({})", model.name, model.id);
            }
        }
    }
}

/// Result of checking model availability
#[derive(Debug)]
pub enum ModelAvailability {
    /// Model is ready and available locally
    Ready {
        local_path: Option<PathBuf>,
        model_name: String,
    },
    /// Model is in config but not downloaded
    NotDownloaded {
        model_name: String,
        model_id: String,
    },
    /// Model found but wrong category
    WrongCategory {
        expected: ModelCategory,
        found: ModelCategory,
    },
    /// Model not found in config (will try to download)
    NotInConfig,
}

// ============================================================================
// Report API types
// ============================================================================

/// Model report returned by GET /v1/models/report
#[derive(Debug, Serialize)]
pub struct ModelReport {
    pub total: usize,
    pub ready: usize,
    pub by_category: HashMap<String, CategoryCount>,
    pub models: Vec<ModelSummary>,
}

/// Per-category count
#[derive(Debug, Serialize)]
pub struct CategoryCount {
    pub total: usize,
    pub ready: usize,
}

/// Summary of a single model for the report
#[derive(Debug, Serialize)]
pub struct ModelSummary {
    pub id: String,
    pub name: String,
    pub category: String,
    pub status: String,
    pub local_path: String,
    pub source: String,
}

// ============================================================================
// Hub cache scanning
// ============================================================================

/// Scan HuggingFace and ModelScope hub caches, plus ~/.OminiX/models/,
/// detect their category, and merge into the config.
/// Returns the number of newly registered models.
pub fn scan_hub_caches() -> usize {
    let mut config = LocalModelsConfig::load()
        .unwrap_or_else(LocalModelsConfig::default_empty);

    let mut added = 0;

    // Scan ~/.OminiX/models/ (locally placed models)
    if let Some(home) = dirs::home_dir() {
        let local_models_dir = home.join(".OminiX/models");
        if local_models_dir.exists() {
            added += scan_local_models(&local_models_dir, &mut config);
        }
    }

    for root in get_hf_cache_roots() {
        if root.exists() {
            added += scan_hf_cache(&root, &mut config);
        }
    }

    for root in get_ms_cache_roots() {
        if root.exists() {
            added += scan_ms_cache(&root, &mut config);
        }
    }

    if added > 0 {
        config.last_updated = Some(chrono::Utc::now().to_rfc3339());
        if let Err(e) = config.save() {
            tracing::warn!("Failed to save updated config after scan: {}", e);
        } else {
            tracing::info!("Model scan: registered {} new model(s)", added);
        }
    } else {
        tracing::debug!("Model scan: no new models found");
    }

    added
}

/// Register a model in the config after discovery or download.
/// Returns Ok(true) if newly added, Ok(false) if already existed.
pub fn register_model(repo_id: &str, category: ModelCategory, local_path: &PathBuf) -> Result<bool, String> {
    let mut config = LocalModelsConfig::load()
        .unwrap_or_else(LocalModelsConfig::default_empty);

    let display_name = repo_id.split('/').last().unwrap_or(repo_id).to_string();

    let model = LocalModel {
        id: repo_id.to_string(),
        name: display_name,
        description: Some("Auto-registered".to_string()),
        category,
        source: ModelSource {
            primary_url: None,
            source_type: Some("hub_cache".to_string()),
            repo_id: Some(repo_id.to_string()),
        },
        storage: ModelStorage {
            local_path: local_path.to_string_lossy().to_string(),
            total_size_bytes: None,
            total_size_display: None,
        },
        status: ModelStatusInfo {
            state: "ready".to_string(),
            downloaded_bytes: None,
            downloaded_files: None,
        },
    };

    let was_added = config.merge_scanned_model(model);
    if was_added {
        config.last_updated = Some(chrono::Utc::now().to_rfc3339());
        config.save()?;
        tracing::info!("Registered model in config: {}", repo_id);
    }

    Ok(was_added)
}

/// Build a model report for the API
pub fn get_model_report() -> ModelReport {
    let config = LocalModelsConfig::load()
        .unwrap_or_else(LocalModelsConfig::default_empty);

    let mut by_category: HashMap<String, CategoryCount> = HashMap::new();
    let mut models = Vec::new();
    let mut total_ready = 0;

    for model in &config.models {
        let cat_name = category_name(&model.category);
        let is_ready = model.is_ready() && model.files_exist();

        let entry = by_category.entry(cat_name.to_string()).or_insert(CategoryCount {
            total: 0,
            ready: 0,
        });
        entry.total += 1;
        if is_ready {
            entry.ready += 1;
            total_ready += 1;
        }

        models.push(ModelSummary {
            id: model.id.clone(),
            name: model.name.clone(),
            category: cat_name.to_string(),
            status: if is_ready { "ready".to_string() } else { model.status.state.clone() },
            local_path: model.storage.local_path.clone(),
            source: model.source.source_type.clone().unwrap_or_default(),
        });
    }

    ModelReport {
        total: config.models.len(),
        ready: total_ready,
        by_category,
        models,
    }
}

// ============================================================================
// Internal scanning helpers
// ============================================================================

fn get_hf_cache_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(val) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        roots.push(PathBuf::from(val));
    }
    if let Ok(val) = std::env::var("HF_HOME") {
        roots.push(PathBuf::from(val).join("hub"));
    }
    if let Some(home) = dirs::home_dir() {
        roots.push(home.join(".cache/huggingface/hub"));
    }
    roots
}

fn get_ms_cache_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(val) = std::env::var("MODELSCOPE_CACHE") {
        roots.push(PathBuf::from(val).join("hub"));
    }
    if let Some(home) = dirs::home_dir() {
        roots.push(home.join(".cache/modelscope/hub"));
    }
    roots
}

/// Scan a HuggingFace hub cache root for models
fn scan_hf_cache(cache_root: &PathBuf, config: &mut LocalModelsConfig) -> usize {
    let entries = match std::fs::read_dir(cache_root) {
        Ok(e) => e,
        Err(_) => return 0,
    };

    let mut added = 0;

    for entry in entries.filter_map(|e| e.ok()) {
        let dir_name = entry.file_name().to_string_lossy().to_string();

        // Only process model directories (skip datasets--, spaces--, etc.)
        if !dir_name.starts_with("models--") {
            continue;
        }

        // Reconstruct repo_id: "models--org--name" -> "org/name"
        let repo_id = match dir_name.strip_prefix("models--") {
            Some(rest) => rest.replacen("--", "/", 1),
            None => continue,
        };

        // Resolve to snapshot directory
        let model_dir = entry.path();
        let snapshot_dir = match crate::utils::resolve_hf_snapshot(&model_dir) {
            Ok(p) => p,
            Err(_) => continue,
        };

        let category = match detect_model_category(&snapshot_dir) {
            Some(c) => c,
            None => continue,
        };

        let display_name = repo_id.split('/').last().unwrap_or(&repo_id).to_string();

        let model = LocalModel {
            id: repo_id.clone(),
            name: display_name,
            description: Some("Scanned from HuggingFace cache".to_string()),
            category: category.clone(),
            source: ModelSource {
                primary_url: Some(format!("https://huggingface.co/{}", repo_id)),
                source_type: Some("huggingface".to_string()),
                repo_id: Some(repo_id.clone()),
            },
            storage: ModelStorage {
                local_path: model_dir.to_string_lossy().to_string(),
                total_size_bytes: dir_size_bytes(&snapshot_dir),
                total_size_display: None,
            },
            status: ModelStatusInfo {
                state: "ready".to_string(),
                downloaded_bytes: None,
                downloaded_files: None,
            },
        };

        if config.merge_scanned_model(model) {
            added += 1;
            tracing::info!("  Registered from HF cache: {} ({:?})", repo_id, category);
        }
    }

    added
}

/// Scan a ModelScope hub cache root for models
fn scan_ms_cache(cache_root: &PathBuf, config: &mut LocalModelsConfig) -> usize {
    let mut added = 0;

    // Scan top-level org directories (e.g., damo/, iic/)
    let entries = match std::fs::read_dir(cache_root) {
        Ok(e) => e,
        Err(_) => return 0,
    };

    for org_entry in entries.filter_map(|e| e.ok()) {
        if !org_entry.path().is_dir() {
            continue;
        }
        let org_name = org_entry.file_name().to_string_lossy().to_string();

        let models = match std::fs::read_dir(org_entry.path()) {
            Ok(e) => e,
            Err(_) => continue,
        };

        for model_entry in models.filter_map(|e| e.ok()) {
            if !model_entry.path().is_dir() {
                continue;
            }
            let model_name = model_entry.file_name().to_string_lossy().to_string();
            let repo_id = format!("{}/{}", org_name, model_name);
            let model_dir = model_entry.path();

            let category = match detect_model_category(&model_dir) {
                Some(c) => c,
                None => continue,
            };

            let display_name = model_name.clone();

            let model = LocalModel {
                id: repo_id.clone(),
                name: display_name,
                description: Some("Scanned from ModelScope cache".to_string()),
                category: category.clone(),
                source: ModelSource {
                    primary_url: Some(format!("https://modelscope.cn/models/{}", repo_id)),
                    source_type: Some("modelscope".to_string()),
                    repo_id: Some(repo_id.clone()),
                },
                storage: ModelStorage {
                    local_path: model_dir.to_string_lossy().to_string(),
                    total_size_bytes: dir_size_bytes(&model_dir),
                    total_size_display: None,
                },
                status: ModelStatusInfo {
                    state: "ready".to_string(),
                    downloaded_bytes: None,
                    downloaded_files: None,
                },
            };

            if config.merge_scanned_model(model) {
                added += 1;
                tracing::info!("  Registered from ModelScope cache: {} ({:?})", repo_id, category);
            }
        }
    }

    added
}

/// Scan ~/.OminiX/models/ for locally placed models
fn scan_local_models(models_dir: &PathBuf, config: &mut LocalModelsConfig) -> usize {
    let entries = match std::fs::read_dir(models_dir) {
        Ok(e) => e,
        Err(_) => return 0,
    };

    let mut added = 0;

    for entry in entries.filter_map(|e| e.ok()) {
        if !entry.path().is_dir() {
            continue;
        }

        let model_name = entry.file_name().to_string_lossy().to_string();
        let model_dir = entry.path();

        let category = match detect_model_category(&model_dir) {
            Some(c) => c,
            None => continue,
        };

        let model_id = format!("local/{}", model_name);

        let model = LocalModel {
            id: model_id.clone(),
            name: model_name,
            description: Some("Scanned from ~/.OminiX/models/".to_string()),
            category: category.clone(),
            source: ModelSource {
                primary_url: None,
                source_type: Some("local".to_string()),
                repo_id: None,
            },
            storage: ModelStorage {
                local_path: model_dir.to_string_lossy().to_string(),
                total_size_bytes: dir_size_bytes(&model_dir),
                total_size_display: None,
            },
            status: ModelStatusInfo {
                state: "ready".to_string(),
                downloaded_bytes: None,
                downloaded_files: None,
            },
        };

        if config.merge_scanned_model(model) {
            added += 1;
            tracing::info!("  Registered from local models: {} ({:?})", model_id, category);
        }
    }

    added
}

/// Detect the category of a model by examining its files.
/// Detection order matters: Image -> ASR -> LLM (most specific first).
fn detect_model_category(model_dir: &PathBuf) -> Option<ModelCategory> {
    // Image: transformer/ + vae/ directories with weight files
    if model_dir.join("transformer").is_dir() && model_dir.join("vae").is_dir() {
        let has_weights = has_safetensors_in(&model_dir.join("transformer"));
        if has_weights {
            return Some(ModelCategory::Image);
        }
    }

    // ASR variant 1: classic paraformer (paraformer.safetensors + am.mvn)
    if model_dir.join("paraformer.safetensors").exists()
        && model_dir.join("am.mvn").exists()
    {
        return Some(ModelCategory::Asr);
    }

    // ASR variant 2: funasr-style (config.json with model_type="funasr" + weight files)
    let config_path = model_dir.join("config.json");
    if config_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(model_type) = config.get("model_type").and_then(|v| v.as_str()) {
                    if model_type.starts_with("funasr") || model_type == "paraformer" {
                        if has_safetensors_in(model_dir) {
                            return Some(ModelCategory::Asr);
                        }
                    }
                }
            }
        }
    }

    // LLM: config.json with model_type + tokenizer.json + weight files
    let tokenizer_path = model_dir.join("tokenizer.json");
    if config_path.exists() && tokenizer_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
                if config.get("model_type").and_then(|v| v.as_str()).is_some() {
                    if has_safetensors_in(model_dir) {
                        return Some(ModelCategory::Llm);
                    }
                }
            }
        }
    }

    None
}

/// Check if a directory contains any .safetensors files
fn has_safetensors_in(dir: &PathBuf) -> bool {
    std::fs::read_dir(dir)
        .ok()
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .any(|e| {
                    e.path()
                        .extension()
                        .map_or(false, |ext| ext == "safetensors")
                })
        })
        .unwrap_or(false)
}

/// Calculate total size of files in a directory (1 level of subdirs)
fn dir_size_bytes(dir: &PathBuf) -> Option<u64> {
    let mut total: u64 = 0;
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.filter_map(|e| e.ok()) {
        if let Ok(meta) = entry.metadata() {
            if meta.is_file() {
                total += meta.len();
            } else if meta.is_dir() {
                if let Ok(sub_entries) = std::fs::read_dir(entry.path()) {
                    for sub in sub_entries.filter_map(|e| e.ok()) {
                        if let Ok(sub_meta) = sub.metadata() {
                            if sub_meta.is_file() {
                                total += sub_meta.len();
                            }
                        }
                    }
                }
            }
        }
    }
    if total > 0 { Some(total) } else { None }
}

fn category_name(cat: &ModelCategory) -> &'static str {
    match cat {
        ModelCategory::Llm => "llm",
        ModelCategory::Image => "image",
        ModelCategory::Asr => "asr",
        ModelCategory::Tts => "tts",
    }
}

// ============================================================================
// Public API (unchanged from before)
// ============================================================================

/// Expand ~ to home directory and return as PathBuf
fn expand_path(path: &str) -> Option<PathBuf> {
    Some(PathBuf::from(crate::utils::expand_tilde(path)))
}

/// Global config instance (lazy loaded)
pub fn get_config() -> Option<LocalModelsConfig> {
    LocalModelsConfig::load()
}

/// Check if a specific model is available
pub fn check_model(model_id: &str, category: ModelCategory) -> ModelAvailability {
    match get_config() {
        Some(config) => config.check_model_availability(model_id, category),
        None => ModelAvailability::NotInConfig,
    }
}

/// Print startup status report
pub fn print_startup_report() {
    match get_config() {
        Some(config) => config.print_status_report(),
        None => {
            tracing::info!("No model configuration found at {}", CONFIG_PATH);
            tracing::info!("Models will be resolved from hub caches as needed.");
        }
    }
}
