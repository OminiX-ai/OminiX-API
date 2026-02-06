//! Shared utility functions

use std::path::PathBuf;

use eyre::Result;

/// Expand `~` prefix to the user's home directory.
///
/// Returns the original string unchanged if it doesn't start with `~/`
/// or if the home directory cannot be determined.
pub fn expand_tilde(path: &str) -> String {
    if path.starts_with("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return format!("{}/{}", home.to_string_lossy(), &path[2..]);
        }
    }
    path.to_string()
}

/// Resolve a HuggingFace Hub cache directory to its snapshot path.
///
/// HuggingFace stores models in `<model_dir>/snapshots/<hash>/`. This function
/// checks for that structure and returns the snapshot path, or the original
/// path if no snapshots directory exists.
pub fn resolve_hf_snapshot(model_dir: &PathBuf) -> Result<PathBuf> {
    let snapshots_dir = model_dir.join("snapshots");
    if snapshots_dir.exists() {
        let snapshot = std::fs::read_dir(&snapshots_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .next()
            .ok_or_else(|| eyre::eyre!("No snapshot found in {:?}", snapshots_dir))?;
        Ok(snapshot.path())
    } else {
        Ok(model_dir.clone())
    }
}

/// Resolve a model ID (e.g. "mlx-community/Qwen3-4B-bf16") by searching
/// standard model hub cache directories.
///
/// Checks the following locations in order:
/// 1. HuggingFace Hub: `~/.cache/huggingface/hub/models--{org}--{model}/`
/// 2. ModelScope: `~/.cache/modelscope/hub/{org}/{model}/`
/// 3. Custom env overrides: `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `MODELSCOPE_CACHE`
///
/// Returns the resolved path with snapshots navigated, or None if not found.
pub fn resolve_from_hub_cache(model_id: &str) -> Option<PathBuf> {
    let home = dirs::home_dir()?;

    // HuggingFace cache: models--{org}--{model} (slashes become --)
    let hf_dir_name = format!("models--{}", model_id.replace('/', "--"));
    let hf_cache_roots = [
        std::env::var("HUGGINGFACE_HUB_CACHE")
            .map(PathBuf::from)
            .ok(),
        std::env::var("HF_HOME")
            .map(|h| PathBuf::from(h).join("hub"))
            .ok(),
        Some(home.join(".cache/huggingface/hub")),
    ];

    for root in hf_cache_roots.iter().flatten() {
        let model_dir = root.join(&hf_dir_name);
        if model_dir.exists() {
            if let Ok(resolved) = resolve_hf_snapshot(&model_dir) {
                tracing::info!("Found model in HuggingFace cache: {:?}", resolved);
                return Some(resolved);
            }
        }
    }

    // ModelScope cache: {org}/{model}
    let ms_cache_roots = [
        std::env::var("MODELSCOPE_CACHE")
            .map(|c| PathBuf::from(c).join("hub"))
            .ok(),
        Some(home.join(".cache/modelscope/hub")),
    ];

    for root in ms_cache_roots.iter().flatten() {
        let model_dir = root.join(model_id);
        if model_dir.exists() {
            tracing::info!("Found model in ModelScope cache: {:?}", model_dir);
            return Some(model_dir);
        }
    }

    None
}

/// Sanitize a voice name to prevent path traversal attacks.
///
/// Only allows alphanumeric characters, hyphens, and underscores.
/// Length must be between 1 and 64 characters.
pub fn sanitize_voice_name(name: &str) -> Result<String> {
    let name = name.trim();
    if name.is_empty() || name.len() > 64 {
        return Err(eyre::eyre!("voice_name must be 1-64 characters"));
    }
    if !name
        .chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
    {
        return Err(eyre::eyre!(
            "voice_name may only contain alphanumeric characters, hyphens, and underscores"
        ));
    }
    Ok(name.to_string())
}

/// Validate that a path does not contain traversal components (`..`).
pub fn is_safe_path(path: &str) -> bool {
    !path.contains("..")
}
