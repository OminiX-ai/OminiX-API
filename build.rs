//! Build script: reads `ominix.toml` manifests from OminiX-MLX crates
//! and generates a compiled-in capability registry.
//!
//! Also embeds a git-based micro version for accurate build identification.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn main() {
    // Embed git commit hash for micro-versioning.
    // Format: "<cargo_version>+<short_hash>" (e.g. "0.1.0+a3f9b2c")
    // Falls back to cargo version alone if git is unavailable.
    println!("cargo:rerun-if-changed=.git/HEAD");
    let git_hash = std::process::Command::new("git")
        .args(["rev-parse", "--short=7", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    println!("cargo:rustc-env=OMINIX_GIT_HASH={git_hash}");

    let mlx_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../OminiX-MLX");

    // Version parity check: OminiX-API and OminiX-MLX must share the same version.
    let api_version = env!("CARGO_PKG_VERSION");
    println!("cargo:rerun-if-changed=../OminiX-MLX/Cargo.toml");
    let mlx_cargo = std::fs::read_to_string(mlx_dir.join("Cargo.toml"))
        .expect("Cannot read OminiX-MLX/Cargo.toml — is it a sibling directory?");
    let mlx_version = mlx_cargo
        .lines()
        .find(|l| l.starts_with("version") && !l.contains('#'))
        .and_then(|l| l.split_once('='))
        .map(|(_, v)| v.trim().trim_matches('"').to_string())
        .expect("Cannot parse version from OminiX-MLX/Cargo.toml");
    if api_version != mlx_version {
        panic!(
            "Version mismatch: OminiX-API={} but OminiX-MLX={}. Both must be the same.",
            api_version, mlx_version
        );
    }

    // Find all ominix.toml files
    let mut manifests: BTreeMap<String, ManifestData> = BTreeMap::new();

    if let Ok(entries) = std::fs::read_dir(&mlx_dir) {
        for entry in entries.flatten() {
            let manifest_path = entry.path().join("ominix.toml");
            if manifest_path.exists() {
                // Re-run build if any manifest changes
                println!("cargo:rerun-if-changed={}", manifest_path.display());
                if let Some(data) = parse_manifest(&manifest_path) {
                    manifests.insert(data.name.clone(), data);
                }
            }
        }
    }

    // Generate Rust code
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let dest = Path::new(&out_dir).join("capability_registry.rs");

    let mut code = String::new();
    code.push_str("/// Auto-generated capability registry from OminiX-MLX ominix.toml files.\n");
    code.push_str("pub fn generated_capability_registry() -> Vec<(&'static str, &'static str, &'static str, &'static str, &'static [&'static str])> {\n");
    code.push_str("    vec![\n");

    for (_key, data) in &manifests {
        let caps_str: Vec<String> = data.capabilities.iter().map(|c| format!("\"{}\"", c)).collect();
        code.push_str(&format!(
            "        (\"{}\", \"{}\", \"{}\", \"{}\", &[{}]),\n",
            data.category,
            data.name.replace('"', "\\\""),
            data.version,
            data.description.replace('"', "\\\""),
            caps_str.join(", "),
        ));
    }

    code.push_str("    ]\n");
    code.push_str("}\n");

    std::fs::write(&dest, code).expect("Failed to write generated registry");
}

struct ManifestData {
    name: String,
    version: String,
    category: String,
    description: String,
    capabilities: Vec<String>,
}

fn parse_manifest(path: &Path) -> Option<ManifestData> {
    let content = std::fs::read_to_string(path).ok()?;

    // Simple TOML parsing without pulling in the toml crate for build.rs
    let mut name = String::new();
    let mut version = String::new();
    let mut category = String::new();
    let mut description = String::new();
    let mut capabilities = Vec::new();

    let mut in_package = false;
    let mut in_capabilities = false;

    for line in content.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with('[') {
            in_package = trimmed == "[package]";
            in_capabilities = trimmed == "[capabilities]";
            continue;
        }

        if let Some((key, val)) = trimmed.split_once('=') {
            let key = key.trim();
            let val = val.trim().trim_matches('"');

            if in_package {
                match key {
                    "name" => name = val.to_string(),
                    "version" => version = val.to_string(),
                    "category" => category = val.to_string(),
                    "description" => description = val.to_string(),
                    _ => {}
                }
            } else if in_capabilities {
                // Boolean capabilities: key = true
                if val == "true" {
                    capabilities.push(key.to_string());
                }
            }
        }
    }

    if name.is_empty() || category.is_empty() {
        eprintln!("cargo:warning=Skipping malformed ominix.toml: {}", path.display());
        return None;
    }

    Some(ManifestData { name, version, category, description, capabilities })
}
