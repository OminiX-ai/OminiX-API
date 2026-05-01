//! Shared utilities for Python MLX subprocess inference engines.
//!
//! Provides Python binary discovery, inference script resolution, and subprocess
//! execution. All Python-based engines (image edit, Cosmos, Wan2.2 GGUF) share
//! this infrastructure.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use eyre::{Context, Result};

const INFERENCE_SCRIPTS_DIR: &str = ".OminiX/inference/scripts";
const INFERENCE_VENV_DIR: &str = ".OminiX/inference/venv";

/// Find a working Python 3 binary.
///
/// Search order:
/// 1. `OMINIX_PYTHON` env var (explicit path)
/// 2. `~/.OminiX/inference/venv/bin/python3` (managed venv)
/// 3. PATH lookup for `python3`
/// 4. Hardcoded common locations
pub fn find_python() -> Option<PathBuf> {
    if let Ok(val) = std::env::var("OMINIX_PYTHON") {
        let p = PathBuf::from(&val);
        if p.exists() {
            return Some(p);
        }
    }

    if let Some(home) = home_dir() {
        let venv_python = home.join(INFERENCE_VENV_DIR).join("bin/python3");
        if venv_python.exists() {
            return Some(venv_python);
        }
    }

    for dir in std::env::var("PATH").unwrap_or_default().split(':') {
        let candidate = Path::new(dir).join("python3");
        if candidate.exists() {
            return Some(candidate);
        }
    }

    let common_paths = [
        "/opt/homebrew/bin/python3",
        "/usr/local/bin/python3",
        "/opt/anaconda3/bin/python3",
        "/usr/bin/python3",
    ];
    for path in common_paths {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    None
}

/// Find an inference script by filename.
///
/// Search order:
/// 1. `OMINIX_INFERENCE_DIR` env var
/// 2. `~/.OminiX/inference/scripts/`
/// 3. Relative to the API binary: `../scripts/inference-reference/`
pub fn find_script(name: &str) -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("OMINIX_INFERENCE_DIR") {
        let p = PathBuf::from(&dir).join(name);
        if p.exists() {
            return Some(p);
        }
    }

    if let Some(home) = home_dir() {
        let p = home.join(INFERENCE_SCRIPTS_DIR).join(name);
        if p.exists() {
            return Some(p);
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let p = parent.join("../scripts/inference-reference").join(name);
            if p.exists() {
                return Some(p);
            }
        }
    }

    None
}

/// Get the directory containing inference scripts (for PYTHONPATH).
pub fn scripts_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("OMINIX_INFERENCE_DIR") {
        let p = PathBuf::from(&dir);
        if p.is_dir() {
            return Some(p);
        }
    }

    if let Some(home) = home_dir() {
        let p = home.join(INFERENCE_SCRIPTS_DIR);
        if p.is_dir() {
            return Some(p);
        }
    }

    None
}

/// Run a Python inference script as a subprocess.
///
/// Returns the raw stdout bytes on success. The script is expected to write its
/// output to a file (path passed as `--output`) and print the path to stdout.
pub fn run_python_script(
    python: &Path,
    script: &Path,
    args: &[&str],
    timeout: Duration,
) -> Result<String> {
    let scripts_path = script.parent().unwrap_or(Path::new("."));

    let mut cmd = Command::new(python);
    cmd.arg(script)
        .args(args)
        .env("PYTHONPATH", scripts_path);

    if let Ok(endpoint) = std::env::var("HF_ENDPOINT") {
        cmd.env("HF_ENDPOINT", endpoint);
    }

    tracing::debug!("pymlx cmd: {:?}", cmd);

    let output = run_with_timeout(&mut cmd, timeout)
        .context("Failed to run Python inference script")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(eyre::eyre!(
            "Python inference failed (exit {}): {}\nstdout: {}",
            output.status.code().unwrap_or(-1),
            stderr.chars().take(1000).collect::<String>(),
            stdout.chars().take(500).collect::<String>(),
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    Ok(stdout)
}

/// Run a Python inference script and read the output file it produces.
///
/// The script must write its output to the path given via `--output` arg.
/// Returns the raw bytes of the output file.
pub fn run_and_read_output(
    python: &Path,
    script: &Path,
    args: &[&str],
    output_path: &Path,
    timeout: Duration,
) -> Result<Vec<u8>> {
    run_python_script(python, script, args, timeout)?;

    let bytes = std::fs::read(output_path)
        .with_context(|| format!("Failed to read output file: {}", output_path.display()))?;

    let _ = std::fs::remove_file(output_path);

    Ok(bytes)
}

/// Write bytes to a temp file, returning the path.
pub fn write_temp_file(prefix: &str, extension: &str, data: &[u8]) -> Result<PathBuf> {
    let path = PathBuf::from(format!(
        "/tmp/ominix-{}-{}.{}",
        prefix,
        std::process::id(),
        extension
    ));
    std::fs::write(&path, data)
        .with_context(|| format!("Failed to write temp file: {}", path.display()))?;
    Ok(path)
}

/// Generate a unique temp output path.
pub fn temp_output_path(engine: &str, extension: &str) -> PathBuf {
    PathBuf::from(format!(
        "/tmp/ominix-{}-{}-{}.{}",
        engine,
        std::process::id(),
        rand_suffix(),
        extension
    ))
}

/// Parse "WIDTHxHEIGHT" size string.
pub fn parse_size(size: &str) -> Result<(u32, u32)> {
    let parts: Vec<&str> = size.split('x').collect();
    if parts.len() != 2 {
        return Err(eyre::eyre!("Invalid size format '{}', expected WIDTHxHEIGHT", size));
    }
    let width: u32 = parts[0].parse().context("Invalid width")?;
    let height: u32 = parts[1].parse().context("Invalid height")?;
    Ok((width, height))
}

fn home_dir() -> Option<PathBuf> {
    std::env::var("HOME").ok().map(PathBuf::from)
}

fn rand_suffix() -> u32 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0)
}

fn run_with_timeout(cmd: &mut Command, timeout: Duration) -> std::io::Result<std::process::Output> {
    use std::process::Stdio;

    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let deadline = std::time::Instant::now() + timeout;

    loop {
        match child.try_wait()? {
            Some(_) => return child.wait_with_output(),
            None => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    return child.wait_with_output();
                }
                std::thread::sleep(Duration::from_millis(500));
            }
        }
    }
}
