//! Server configuration — gatekeeper for model loading.
//!
//! Loaded from `~/.OminiX/server_config.json` (override with
//! `OMINIX_SERVER_CONFIG` env var).  When the file is absent,
//! all models are allowed (backward-compatible default).

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Server configuration controlling which models may be loaded.
///
/// # Example `server_config.json`
/// ```json
/// {
///   "allowed_models": {
///     "llm": ["*"],
///     "image": ["flux-8bit", "zimage"],
///     "asr": ["qwen3-asr"],
///     "tts": ["qwen3-tts", "qwen3-tts-base"],
///     "vlm": []
///   }
/// }
/// ```
///
/// - `["*"]`         → any model allowed for this category
/// - `["id1","id2"]` → only listed IDs (case-insensitive substring match)
/// - `[]`            → category disabled (no models can be loaded)
/// - key absent      → any model allowed (backward-compatible)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default)]
    pub allowed_models: HashMap<String, Vec<String>>,
}

impl ServerConfig {
    /// Load from disk, falling back to a permissive default.
    pub fn load() -> Self {
        let path = config_path();
        match std::fs::read_to_string(&path) {
            Ok(content) => match serde_json::from_str(&content) {
                Ok(cfg) => {
                    tracing::info!("Loaded server config from {}", path.display());
                    cfg
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to parse server config {}: {} — using defaults",
                        path.display(),
                        e
                    );
                    Self::default()
                }
            },
            Err(_) => {
                tracing::info!(
                    "No server config at {} — all models allowed",
                    path.display()
                );
                Self::default()
            }
        }
    }

    /// Check whether loading `model_id` is allowed for the given `category`.
    pub fn is_model_allowed(&self, category: &str, model_id: &str) -> bool {
        let list = match self.allowed_models.get(category) {
            Some(l) => l,
            None => return true, // category not listed → allow all
        };
        if list.iter().any(|s| s == "*") {
            return true;
        }
        let id_lower = model_id.to_lowercase();
        list.iter().any(|pattern| {
            let p = pattern.to_lowercase();
            id_lower.contains(&p) || p.contains(&id_lower)
        })
    }

    /// Check if a category is fully disabled (empty allow-list).
    pub fn is_category_disabled(&self, category: &str) -> bool {
        self.allowed_models
            .get(category)
            .map(|list| list.is_empty())
            .unwrap_or(false)
    }
}

fn config_path() -> PathBuf {
    if let Ok(p) = std::env::var("OMINIX_SERVER_CONFIG") {
        return PathBuf::from(p);
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".OminiX/server_config.json")
}
