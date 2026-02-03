//! Model configuration reader for ~/.moly/local_models_config.json
//!
//! This module reads the model configuration from OminiX-Studio to check
//! which models are available locally without downloading.

use std::path::PathBuf;
use serde::Deserialize;

/// Path to the local models config file
const CONFIG_PATH: &str = "~/.moly/local_models_config.json";

/// Model category
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub enum ModelCategory {
    Image,
    Llm,
    Asr,
    Tts,
}

/// Model source information
#[derive(Debug, Clone, Deserialize)]
pub struct ModelSource {
    pub primary_url: Option<String>,
    pub source_type: Option<String>,
    pub repo_id: Option<String>,
}

/// Model storage information
#[derive(Debug, Clone, Deserialize)]
pub struct ModelStorage {
    pub local_path: String,
    pub total_size_bytes: Option<u64>,
    pub total_size_display: Option<String>,
}

/// Model status information
#[derive(Debug, Clone, Deserialize)]
pub struct ModelStatusInfo {
    pub state: String,
    pub downloaded_bytes: Option<u64>,
    pub downloaded_files: Option<u32>,
}

/// Individual model configuration
#[derive(Debug, Clone, Deserialize)]
pub struct LocalModel {
    pub id: String,
    pub name: String,
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
#[derive(Debug, Clone, Deserialize)]
pub struct LocalModelsConfig {
    pub version: String,
    pub last_updated: Option<String>,
    pub models: Vec<LocalModel>,
}

impl LocalModelsConfig {
    /// Load the configuration from ~/.moly/local_models_config.json
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
                    "✓ Ready"
                } else if model.is_ready() {
                    "⚠ Config says ready but files missing"
                } else {
                    "✗ Not downloaded"
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

/// Expand ~ to home directory
fn expand_path(path: &str) -> Option<PathBuf> {
    if path.starts_with("~/") {
        dirs::home_dir().map(|home| home.join(&path[2..]))
    } else {
        Some(PathBuf::from(path))
    }
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
            tracing::info!("Models will be downloaded from HuggingFace Hub as needed.");
        }
    }
}
