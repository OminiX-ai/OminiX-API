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
