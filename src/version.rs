//! Capability versioning and app manifest validation.
//!
//! Each model crate declares its version and capabilities via `ominix.toml`.
//! This module provides:
//! - A compiled-in registry of available capabilities and their versions
//! - App manifest parsing (`ominix.toml`) for declaring requirements
//! - Startup validation to check that all requirements are satisfied

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// API version (from Cargo.toml).
pub const API_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Git commit hash (short, 7 chars). Empty if built outside a git repo.
pub const GIT_HASH: &str = env!("OMINIX_GIT_HASH");

/// Full version string: "0.1.0+a3f9b2c" or "0.1.0" if no git hash.
pub fn full_version() -> String {
    if GIT_HASH.is_empty() {
        API_VERSION.to_string()
    } else {
        format!("{API_VERSION}+{GIT_HASH}")
    }
}

/// A capability provided by the server.
#[derive(Debug, Clone, Serialize)]
pub struct CapabilityInfo {
    pub name: String,
    pub version: String,
    pub category: String,
    pub description: String,
    pub capabilities: Vec<String>,
}

/// Full version response for `/v1/version`.
#[derive(Debug, Clone, Serialize)]
pub struct VersionResponse {
    pub ominix_api: String,
    /// All available capabilities, keyed by crate name.
    pub capabilities: HashMap<String, CapabilityInfo>,
    pub models_loaded: Vec<String>,
}

// Include the build-script-generated registry from OminiX-MLX ominix.toml files.
include!(concat!(env!("OUT_DIR"), "/capability_registry.rs"));

/// Get the compiled-in capability registry, keyed by crate name.
///
/// Auto-generated at build time from each OminiX-MLX crate's `ominix.toml`.
/// When you update a crate's version or capabilities, just rebuild.
pub fn capability_registry() -> HashMap<String, CapabilityInfo> {
    let mut caps = HashMap::new();
    for (category, name, version, description, cap_list) in generated_capability_registry() {
        caps.insert(name.to_string(), CapabilityInfo {
            name: name.to_string(),
            version: version.to_string(),
            category: category.to_string(),
            description: description.to_string(),
            capabilities: cap_list.iter().map(|s| s.to_string()).collect(),
        });
    }
    caps
}

/// Get capabilities grouped by category (for requirement checking).
/// When multiple crates share a category, the highest version wins.
fn registry_by_category() -> HashMap<String, CapabilityInfo> {
    let mut by_cat: HashMap<String, CapabilityInfo> = HashMap::new();
    for (_, cap) in capability_registry() {
        let entry = by_cat.entry(cap.category.clone()).or_insert_with(|| cap.clone());
        // Merge capabilities from multiple crates in same category
        if entry.name != cap.name {
            for c in &cap.capabilities {
                if !entry.capabilities.contains(c) {
                    entry.capabilities.push(c.clone());
                }
            }
            // Keep the higher version
            if let (Ok(existing), Ok(new)) = (
                semver::Version::parse(&entry.version),
                semver::Version::parse(&cap.version),
            ) {
                if new > existing {
                    entry.version = cap.version.clone();
                    entry.name = cap.name.clone();
                }
            }
        }
    }
    by_cat
}

// --- App Manifest ---

/// Parsed app manifest from `ominix.toml`.
#[derive(Debug, Deserialize)]
pub struct AppManifest {
    pub app: AppInfo,
    #[serde(default)]
    pub requires: HashMap<String, toml::Value>,
}

#[derive(Debug, Deserialize)]
pub struct AppInfo {
    pub name: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub ominix_api: String,
}

/// A parsed requirement from the manifest.
#[derive(Debug)]
pub struct Requirement {
    pub category: String,
    pub version_req: String,
    pub capabilities: Vec<String>,
}

/// Parse an app manifest file.
pub fn parse_manifest(path: &str) -> eyre::Result<AppManifest> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| eyre::eyre!("Failed to read manifest {}: {}", path, e))?;
    let manifest: AppManifest = toml::from_str(&content)
        .map_err(|e| eyre::eyre!("Failed to parse manifest {}: {}", path, e))?;
    Ok(manifest)
}

/// Extract requirements from a parsed manifest.
pub fn extract_requirements(manifest: &AppManifest) -> Vec<Requirement> {
    let mut reqs = Vec::new();
    for (category, value) in &manifest.requires {
        match value {
            toml::Value::String(version_req) => {
                reqs.push(Requirement {
                    category: category.clone(),
                    version_req: version_req.clone(),
                    capabilities: vec![],
                });
            }
            toml::Value::Table(table) => {
                let version_req = table
                    .get("version")
                    .and_then(|v| v.as_str())
                    .unwrap_or(">=0.0.0")
                    .to_string();
                let capabilities = table
                    .get("capabilities")
                    .and_then(|v| v.as_table())
                    .map(|t| {
                        t.iter()
                            .filter(|(_, v)| v.as_bool().unwrap_or(false))
                            .map(|(k, _)| k.clone())
                            .collect()
                    })
                    .unwrap_or_default();
                reqs.push(Requirement {
                    category: category.clone(),
                    version_req,
                    capabilities,
                });
            }
            _ => {}
        }
    }
    reqs
}

/// Check requirements against available capabilities.
/// Returns a list of error messages (empty = all satisfied).
pub fn check_requirements(requirements: &[Requirement]) -> Vec<String> {
    let registry = registry_by_category();
    let mut errors = Vec::new();

    for req in requirements {
        match registry.get(&req.category) {
            None => {
                errors.push(format!(
                    "Required capability '{}' is not available",
                    req.category
                ));
            }
            Some(cap) => {
                // Check version
                if !req.version_req.is_empty() {
                    if let Ok(version_req) = semver::VersionReq::parse(&req.version_req) {
                        if let Ok(version) = semver::Version::parse(&cap.version) {
                            if !version_req.matches(&version) {
                                errors.push(format!(
                                    "Capability '{}' version {} does not satisfy requirement {}",
                                    req.category, cap.version, req.version_req
                                ));
                            }
                        }
                    }
                }
                // Check capabilities
                for required_cap in &req.capabilities {
                    if !cap.capabilities.contains(required_cap) {
                        errors.push(format!(
                            "Capability '{}' does not support '{}' (available: {:?})",
                            req.category, required_cap, cap.capabilities
                        ));
                    }
                }
            }
        }
    }

    errors
}

/// Validate an app manifest and log results.
/// Returns Ok(()) if all requirements are satisfied, Err with details otherwise.
pub fn validate_manifest(path: &str) -> eyre::Result<()> {
    let manifest = parse_manifest(path)?;
    tracing::info!(
        "Validating app manifest: {} v{}",
        manifest.app.name,
        manifest.app.version
    );

    // Check API version requirement
    if !manifest.app.ominix_api.is_empty() {
        if let Ok(req) = semver::VersionReq::parse(&manifest.app.ominix_api) {
            if let Ok(current) = semver::Version::parse(API_VERSION) {
                if !req.matches(&current) {
                    return Err(eyre::eyre!(
                        "App requires ominix_api {}, but server is {}",
                        manifest.app.ominix_api,
                        API_VERSION
                    ));
                }
            }
        }
    }

    let requirements = extract_requirements(&manifest);
    let errors = check_requirements(&requirements);

    if errors.is_empty() {
        tracing::info!(
            "All {} requirements satisfied for '{}'",
            requirements.len(),
            manifest.app.name
        );
        Ok(())
    } else {
        for err in &errors {
            tracing::error!("  {}", err);
        }
        Err(eyre::eyre!(
            "App '{}' has {} unsatisfied requirements",
            manifest.app.name,
            errors.len()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_have_all_crates_in_registry() {
        let reg = capability_registry();
        assert!(reg.contains_key("qwen3-tts"), "missing qwen3-tts");
        assert!(reg.contains_key("qwen3-asr"), "missing qwen3-asr");
        assert!(reg.contains_key("qwen3-llm"), "missing qwen3-llm");
        assert!(reg.contains_key("flux-klein"), "missing flux-klein");
        assert!(reg.contains_key("moxin-vlm"), "missing moxin-vlm");
        assert!(reg.contains_key("deepseek-ocr2"), "missing deepseek-ocr2");
    }

    #[test]
    fn should_have_all_categories() {
        let reg = registry_by_category();
        assert!(reg.contains_key("tts"));
        assert!(reg.contains_key("asr"));
        assert!(reg.contains_key("llm"));
        assert!(reg.contains_key("image"));
        assert!(reg.contains_key("vlm"));
        assert!(reg.contains_key("ocr"));
    }

    #[test]
    fn should_merge_capabilities_for_same_category() {
        let reg = registry_by_category();
        let image = reg.get("image").unwrap();
        assert!(image.capabilities.contains(&"text_to_image".to_string()));
        assert!(image.capabilities.contains(&"quantization".to_string()));
    }

    #[test]
    fn should_satisfy_matching_version() {
        let reqs = vec![Requirement {
            category: "tts".into(),
            version_req: ">=1.0.0".into(),
            capabilities: vec![],
        }];
        let errors = check_requirements(&reqs);
        assert!(errors.is_empty(), "errors: {:?}", errors);
    }

    #[test]
    fn should_reject_too_high_version() {
        let reqs = vec![Requirement {
            category: "asr".into(),
            version_req: ">=2.0.0".into(),
            capabilities: vec![],
        }];
        let errors = check_requirements(&reqs);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("does not satisfy"));
    }

    #[test]
    fn should_reject_missing_capability() {
        let reqs = vec![Requirement {
            category: "asr".into(),
            version_req: ">=0.1.0".into(),
            capabilities: vec!["voice_cloning".into()],
        }];
        let errors = check_requirements(&reqs);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("does not support"));
    }

    #[test]
    fn should_reject_unknown_category() {
        let reqs = vec![Requirement {
            category: "video".into(),
            version_req: ">=0.1.0".into(),
            capabilities: vec![],
        }];
        let errors = check_requirements(&reqs);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("not available"));
    }

    #[test]
    fn should_parse_manifest_toml() {
        let toml_str = r#"
[app]
name = "test-app"
version = "0.1.0"
ominix_api = ">=0.1.0"

[requires]
tts = ">=1.0.0"
asr = ">=0.1.0"
"#;
        let manifest: AppManifest = toml::from_str(toml_str).unwrap();
        assert_eq!(manifest.app.name, "test-app");
        let reqs = extract_requirements(&manifest);
        assert_eq!(reqs.len(), 2);
    }

    #[test]
    fn should_parse_manifest_with_capabilities() {
        let toml_str = r#"
[app]
name = "voice-bot"
version = "0.1.0"

[requires.tts]
version = ">=1.0.0"
[requires.tts.capabilities]
voice_cloning = true
streaming = true
"#;
        let manifest: AppManifest = toml::from_str(toml_str).unwrap();
        let reqs = extract_requirements(&manifest);
        assert_eq!(reqs.len(), 1);
        assert_eq!(reqs[0].category, "tts");
        assert!(reqs[0].capabilities.contains(&"voice_cloning".to_string()));
        assert!(reqs[0].capabilities.contains(&"streaming".to_string()));
    }
}
