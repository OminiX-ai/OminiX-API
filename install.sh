#!/bin/sh
# OminiX-API installer — one-line install for macOS Apple Silicon
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/OminiX-ai/OminiX-API/main/install.sh | sh
#
# Options (via environment variables):
#   VERSION=0.1.0    Pin a specific release version
#   INSTALL_DIR=/usr/local/bin   Override install directory

set -eu

REPO="OminiX-ai/OminiX-API"
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"

# ---------------------------------------------------------------------------
# Platform check
# ---------------------------------------------------------------------------
OS=$(uname -s)
ARCH=$(uname -m)

if [ "$OS" != "Darwin" ]; then
  printf "Error: OminiX-API only supports macOS (detected: %s)\n" "$OS" >&2
  exit 1
fi

if [ "$ARCH" != "arm64" ]; then
  printf "Error: OminiX-API requires Apple Silicon (detected: %s)\n" "$ARCH" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Check for Xcode Command Line Tools (provides Metal framework)
# ---------------------------------------------------------------------------
if ! xcode-select -p >/dev/null 2>&1; then
  printf "Xcode Command Line Tools not found. Installing...\n"
  xcode-select --install 2>/dev/null
  printf "Please complete the Xcode CLT installation prompt, then re-run this script.\n"
  exit 1
fi

# ---------------------------------------------------------------------------
# Resolve version
# ---------------------------------------------------------------------------
if [ -z "${VERSION:-}" ]; then
  printf "Fetching latest release...\n"
  VERSION=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
    | grep '"tag_name"' \
    | sed 's/.*"v\([^"]*\)".*/\1/')
fi

if [ -z "$VERSION" ]; then
  printf "Error: could not determine latest release version\n" >&2
  exit 1
fi

TARBALL="ominix-api-${VERSION}-darwin-aarch64.tar.gz"
BASE_URL="https://github.com/${REPO}/releases/download/v${VERSION}"

printf "Installing ominix-api v%s...\n" "$VERSION"

# ---------------------------------------------------------------------------
# Download and verify
# ---------------------------------------------------------------------------
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

curl -fSL "${BASE_URL}/${TARBALL}" -o "${TMP}/${TARBALL}"
curl -fSL "${BASE_URL}/checksums.txt" -o "${TMP}/checksums.txt"

printf "Verifying checksum...\n"
cd "$TMP"
if ! shasum -a 256 -c checksums.txt --status 2>/dev/null; then
  # --status may not be available on older macOS; fall back to manual check
  EXPECTED=$(grep "$TARBALL" checksums.txt | awk '{print $1}')
  ACTUAL=$(shasum -a 256 "$TARBALL" | awk '{print $1}')
  if [ "$EXPECTED" != "$ACTUAL" ]; then
    printf "Error: checksum mismatch\n  expected: %s\n  actual:   %s\n" "$EXPECTED" "$ACTUAL" >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Extract and install binary
# ---------------------------------------------------------------------------
tar xzf "$TARBALL"

if [ -w "$INSTALL_DIR" ]; then
  mv ominix-api "${INSTALL_DIR}/ominix-api"
else
  printf "Installing to %s (requires sudo)...\n" "$INSTALL_DIR"
  sudo mv ominix-api "${INSTALL_DIR}/ominix-api"
fi
chmod +x "${INSTALL_DIR}/ominix-api"

# ---------------------------------------------------------------------------
# Bootstrap config directory
# ---------------------------------------------------------------------------
OMINIX_DIR="${HOME}/.OminiX"
mkdir -p "${OMINIX_DIR}/models"

if [ ! -f "${OMINIX_DIR}/server_config.json" ]; then
  cat > "${OMINIX_DIR}/server_config.json" << 'CONF'
{
  "allowed_models": {
    "llm": ["*"],
    "image": ["flux-8bit", "zimage"],
    "asr": ["qwen3-asr", "paraformer"],
    "tts": ["qwen3-tts", "qwen3-tts-base"],
    "vlm": ["*"]
  }
}
CONF
  printf "Created %s/server_config.json\n" "$OMINIX_DIR"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
printf "\n"
printf "  ominix-api v%s installed to %s/ominix-api\n" "$VERSION" "$INSTALL_DIR"
printf "\n"
printf "  Quick start:\n"
printf "    ominix-api --help\n"
printf "    LLM_MODEL=mlx-community/Qwen3-4B-bf16 ominix-api\n"
printf "\n"
