//! Build script for qwen-tts-ascend-sys.
//!
//! Responsibilities:
//!   1. Run bindgen over the vendored `wrapper/qwen_tts_api.h` and emit
//!      `bindings.rs` to `$OUT_DIR`.
//!   2. When the `ascend-available` feature is enabled, tell cargo to link
//!      `libqwen_tts_api.so` using:
//!        (a) `ASCEND_TTS_LIB_DIR` env var if set, otherwise
//!        (b) `pkg-config --libs qwen_tts_api` if available.
//!      Without the feature, no link directives are emitted and the crate
//!      compiles (but does not expose linkable FFI) — useful on Mac dev
//!      hosts that do not have the Ascend runtime.
//!
//! Host-OS gating: even with `ascend-available`, we only attempt to link
//! when the target is Linux (Ascend is Linux-only). Other targets get a
//! warning and skip linking, so `cargo check --features ascend-available`
//! on macOS still succeeds at the build-script level (though the final
//! binary will fail to link — expected).

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let header = manifest_dir.join("wrapper").join("qwen_tts_api.h");
    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rerun-if-env-changed=ASCEND_TTS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");

    // --- bindgen -------------------------------------------------------
    let bindings = bindgen::Builder::default()
        .header(header.to_str().expect("header path must be UTF-8"))
        // Only bind the qwen_tts_* surface — avoids pulling stdint/stddef noise.
        .allowlist_function("qwen_tts_.*")
        .allowlist_type("qwen_tts_.*")
        // The opaque handle stays opaque on the Rust side.
        .opaque_type("qwen_tts_ctx")
        // Required ABI hygiene.
        .derive_default(false)
        .layout_tests(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen failed to generate qwen_tts_api bindings");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("failed to write bindings.rs");

    // --- linking -------------------------------------------------------
    // Only emit link directives when the ascend-available feature is on
    // AND we're targeting Linux. This lets `cargo check` on Mac compile
    // both without the feature (no linking) and with the feature (still
    // no linking — we defensively skip non-Linux).
    let feature_on = env::var_os("CARGO_FEATURE_ASCEND_AVAILABLE").is_some();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if !feature_on {
        return;
    }
    if target_os != "linux" {
        println!(
            "cargo:warning=qwen-tts-ascend-sys: feature `ascend-available` is on but target_os={target_os}; skipping link directives (Ascend is Linux-only)"
        );
        return;
    }

    if let Some(lib_dir) = env::var_os("ASCEND_TTS_LIB_DIR") {
        let lib_dir = PathBuf::from(lib_dir);
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=dylib=qwen_tts_api");
        // Let runtime loaders find the .so next to the binary during dev.
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
        return;
    }

    // pkg-config fallback. We shell out manually to avoid adding a
    // pkg-config build-dep; this is a best-effort path for CI hosts.
    if let Ok(output) = std::process::Command::new("pkg-config")
        .args(["--libs", "--cflags", "qwen_tts_api"])
        .output()
    {
        if output.status.success() {
            let flags = String::from_utf8_lossy(&output.stdout);
            for token in flags.split_whitespace() {
                if let Some(path) = token.strip_prefix("-L") {
                    println!("cargo:rustc-link-search=native={path}");
                } else if let Some(name) = token.strip_prefix("-l") {
                    println!("cargo:rustc-link-lib=dylib={name}");
                }
            }
            return;
        }
    }

    // Neither env nor pkg-config available: warn loudly. The link step
    // will still fail downstream, which is the intended signal on a real
    // Ascend deploy that forgot to install the .so.
    println!(
        "cargo:warning=qwen-tts-ascend-sys: ASCEND_TTS_LIB_DIR not set and pkg-config qwen_tts_api not found; the final link will fail"
    );
    println!("cargo:rustc-link-lib=dylib=qwen_tts_api");
}
