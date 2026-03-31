#!/usr/bin/env bash
# install-uno-q.sh — Install BirdNET-Go natively on the Arduino Uno Q
#
# This script installs BirdNET-Go as a native binary (no Docker) on the
# Arduino Uno Q (Qualcomm QRB2210 / QCM2290). It handles:
#
#   1. Hardware detection (Arduino Uno Q with Kryo-V2 cores)
#   2. Binary download and installation
#   3. ONNX Runtime shared library installation
#   4. Model file placement (CNN sub-model + preprocessing constants)
#   5. Systemd service creation (real files, not symlinks — required for Uno Q)
#   6. Auto-configuration (ONNX backend, 4 threads, etc.)
#
# Usage:
#   sudo bash install-uno-q.sh
#
# Prerequisites:
#   - Arduino Uno Q running Debian (aarch64)
#   - Internet access for downloading binaries
#   - Root privileges
#
# The BirdNET-Q model directory must contain:
#   - birdnet_cnn.onnx          (CNN sub-model for ONNX backend)
#   - birdnet_preproc.npz       (preprocessing constants)
# These can be copied from a BirdNET-Q installation's model/ directory.

set -euo pipefail

# ── Configuration ───────────────────────────────────────────────────────────

# Installation paths
INSTALL_DIR="/opt/birdnet-go"
CONFIG_DIR="/etc/birdnet-go"
DATA_DIR="/var/lib/birdnet-go"
MODEL_DIR="${INSTALL_DIR}/model"
LOG_DIR="/var/log/birdnet-go"

# Binary name
BINARY_NAME="birdnet-go"

# ONNX Runtime version
ORT_VERSION="1.17.1"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-aarch64-${ORT_VERSION}.tgz"
ORT_LIB_DIR="/usr/local/lib"

# Service user (matches Arduino Uno Q default user)
SERVICE_USER="${BIRDNET_USER:-arduino}"
SERVICE_GROUP="${BIRDNET_GROUP:-arduino}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ── Helper functions ────────────────────────────────────────────────────────

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

die() { log_error "$@"; exit 1; }

# ── Hardware detection ──────────────────────────────────────────────────────

detect_hardware() {
    log_info "Detecting hardware..."

    # Check architecture
    local arch
    arch=$(uname -m)
    if [[ "$arch" != "aarch64" ]]; then
        die "This script requires aarch64 (ARM64). Detected: $arch"
    fi

    # Check for Arduino Uno Q via device tree
    local is_uno_q=false
    if [[ -f /proc/device-tree/model ]]; then
        local model
        model=$(tr -d '\0' < /proc/device-tree/model)
        if [[ "$model" == *"Arduino"* ]] || [[ "$model" == *"Imola"* ]] || [[ "$model" == *"QRB2210"* ]]; then
            is_uno_q=true
            log_info "Detected Arduino Uno Q: $model"
        fi
    fi

    # Fallback: check /proc/cpuinfo for Qualcomm Kryo-V2
    if ! $is_uno_q; then
        if grep -q "CPU implementer.*0x51" /proc/cpuinfo 2>/dev/null && \
           grep -q "CPU part.*0x804" /proc/cpuinfo 2>/dev/null; then
            is_uno_q=true
            log_info "Detected Qualcomm QRB2210 (Kryo-V2) via cpuinfo"
        fi
    fi

    if ! $is_uno_q; then
        log_warn "Arduino Uno Q not detected. This script is designed for the Uno Q."
        log_warn "Proceeding anyway — the ONNX backend works on any ARM64 Linux system."
        read -rp "Continue? [y/N] " confirm
        [[ "$confirm" =~ ^[Yy]$ ]] || exit 0
    fi

    # Check CPU features
    local features
    features=$(grep -m1 "^Features" /proc/cpuinfo 2>/dev/null | cut -d: -f2 || echo "")
    if [[ "$features" != *"asimddp"* ]]; then
        log_info "CPU lacks dot-product extension (asimddp) — ONNX backend recommended"
    fi

    local cpu_count
    cpu_count=$(nproc)
    log_info "CPU cores: $cpu_count"
}

# ── Prerequisite checks ────────────────────────────────────────────────────

check_prerequisites() {
    log_info "Checking prerequisites..."

    if [[ $EUID -ne 0 ]]; then
        die "This script must be run as root (sudo)."
    fi

    # Check for required tools
    for cmd in curl tar systemctl id; do
        if ! command -v "$cmd" &>/dev/null; then
            die "Required command not found: $cmd"
        fi
    done

    # Check that the service user exists
    if ! id "$SERVICE_USER" &>/dev/null; then
        die "Service user '$SERVICE_USER' does not exist. Create it first or set BIRDNET_USER."
    fi

    # Check disk space (need at least 200MB)
    local free_mb
    free_mb=$(df -m /opt 2>/dev/null | awk 'NR==2{print $4}' || echo "0")
    if [[ "$free_mb" -lt 200 ]]; then
        die "Insufficient disk space on /opt: ${free_mb}MB free, need at least 200MB"
    fi

    log_info "Prerequisites OK"
}

# ── Directory setup ─────────────────────────────────────────────────────────

setup_directories() {
    log_info "Creating directories..."

    mkdir -p "$INSTALL_DIR"
    mkdir -p "$CONFIG_DIR"
    mkdir -p "$DATA_DIR"/{clips,database,logs,backups}
    mkdir -p "$MODEL_DIR"
    mkdir -p "$LOG_DIR"

    # Set ownership
    chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "$DATA_DIR"
    chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "$LOG_DIR"
    chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "$CONFIG_DIR"

    log_info "Directories created"
}

# ── ONNX Runtime installation ──────────────────────────────────────────────

install_onnx_runtime() {
    # Check if already installed
    if [[ -f "${ORT_LIB_DIR}/libonnxruntime.so.${ORT_VERSION}" ]]; then
        log_info "ONNX Runtime ${ORT_VERSION} already installed"
        return
    fi

    log_info "Installing ONNX Runtime ${ORT_VERSION} for aarch64..."

    local tmp_dir
    tmp_dir=$(mktemp -d)
    trap 'rm -rf "$tmp_dir"' EXIT

    curl -sSL -o "${tmp_dir}/ort.tgz" "$ORT_URL" || die "Failed to download ONNX Runtime"
    tar -xzf "${tmp_dir}/ort.tgz" -C "$tmp_dir" || die "Failed to extract ONNX Runtime"

    local ort_dir
    ort_dir=$(find "$tmp_dir" -maxdepth 1 -name "onnxruntime-*" -type d | head -1)
    if [[ -z "$ort_dir" ]]; then
        die "ONNX Runtime directory not found after extraction"
    fi

    # Install shared library
    cp "${ort_dir}/lib/libonnxruntime.so.${ORT_VERSION}" "$ORT_LIB_DIR/"
    ln -sf "libonnxruntime.so.${ORT_VERSION}" "${ORT_LIB_DIR}/libonnxruntime.so"
    ldconfig

    trap - EXIT
    rm -rf "$tmp_dir"

    log_info "ONNX Runtime installed to ${ORT_LIB_DIR}"
}

# ── Model file placement ───────────────────────────────────────────────────

install_model_files() {
    log_info "Setting up model files..."

    local source_model_dir=""

    # Check common locations for the ONNX model files
    local search_paths=(
        "/home/${SERVICE_USER}/BirdNET-Pi/model"
        "/home/${SERVICE_USER}/BirdNET-Q/model"
        "./model"
        "../BirdNET-Q/model"
    )

    for path in "${search_paths[@]}"; do
        if [[ -f "${path}/birdnet_cnn.onnx" ]] && [[ -f "${path}/birdnet_preproc.npz" ]]; then
            source_model_dir="$path"
            break
        fi
    done

    if [[ -n "$source_model_dir" ]]; then
        log_info "Found ONNX model files in: $source_model_dir"
        cp -v "${source_model_dir}/birdnet_cnn.onnx" "$MODEL_DIR/"
        cp -v "${source_model_dir}/birdnet_preproc.npz" "$MODEL_DIR/"
    else
        log_warn "ONNX model files not found automatically."
        log_warn "Please copy these files to ${MODEL_DIR}/:"
        log_warn "  - birdnet_cnn.onnx      (CNN sub-model)"
        log_warn "  - birdnet_preproc.npz    (preprocessing constants)"
        log_warn ""
        log_warn "These can be obtained from a BirdNET-Q installation's model/ directory."
    fi

    # Also copy the TFLite models if available (for range filter)
    for tflite_file in BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite \
                       BirdNET_GLOBAL_6K_V2.4_MData_Model_V2_FP16.tflite \
                       BirdNET_GLOBAL_6K_V2.4_MData_Model_FP16.tflite; do
        for path in "${search_paths[@]}"; do
            if [[ -f "${path}/${tflite_file}" ]]; then
                cp -v "${path}/${tflite_file}" "$MODEL_DIR/"
                break
            fi
        done
    done

    chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "$MODEL_DIR"
    log_info "Model files setup complete"
}

# ── Configuration ───────────────────────────────────────────────────────────

create_config() {
    local config_file="${CONFIG_DIR}/config.yaml"

    if [[ -f "$config_file" ]]; then
        log_info "Configuration file already exists: $config_file"
        return
    fi

    log_info "Creating default configuration for Arduino Uno Q..."

    cat > "$config_file" << 'YAML'
# BirdNET-Go configuration for Arduino Uno Q
# See https://github.com/tphakala/birdnet-go for full documentation

birdnet:
  # Use ONNX backend for best performance on Qualcomm QRB2210 (Kryo-V2)
  # The ONNX backend uses Go-native FFT preprocessing + CNN sub-model,
  # achieving ~150ms per 3-second chunk vs ~292ms with TFLite.
  backend: onnx

  # Use all 4 Kryo-V2 cores
  threads: 4

  # XNNPACK is less effective without dot-product extension (asimddp)
  # Irrelevant when using ONNX backend, but set false as documentation
  usexnnpack: false

  sensitivity: 1.0
  threshold: 0.8
  overlap: 0.0
  locale: en

  rangefilter:
    model: latest
    threshold: 0.01

webserver:
  enabled: true
  port: 8080
  autotls: false

realtime:
  audio:
    source: sysdefault
    export:
      enabled: true
      type: wav

output:
  sqlite:
    enabled: true
    path: /var/lib/birdnet-go/data/database/birdnet.db

  log:
    enabled: true
    path: /var/log/birdnet-go/birdnet.log
YAML

    chown "${SERVICE_USER}:${SERVICE_GROUP}" "$config_file"
    log_info "Configuration written to $config_file"
}

# ── Systemd service ────────────────────────────────────────────────────────

install_service() {
    local service_file="/etc/systemd/system/birdnet-go.service"

    log_info "Installing systemd service..."

    # IMPORTANT: Write the service file directly (not as a symlink).
    # On the Uno Q, /home/arduino is on a separate partition that may not
    # be mounted when systemd loads unit files during early boot.
    cat > "$service_file" << EOF
[Unit]
Description=BirdNET-Go Bird Sound Identification
After=network-online.target sound.target
Wants=network-online.target

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
WorkingDirectory=${INSTALL_DIR}

# ONNX Runtime library path
Environment=ORT_LIBRARY_PATH=${ORT_LIB_DIR}/libonnxruntime.so
Environment=LD_LIBRARY_PATH=${ORT_LIB_DIR}

ExecStart=${INSTALL_DIR}/${BINARY_NAME} serve --config ${CONFIG_DIR}/config.yaml
Restart=always
RestartSec=5

# Audio device access
SupplementaryGroups=audio

# Resource limits appropriate for QRB2210 (4GB RAM)
MemoryMax=512M
MemoryHigh=384M

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=birdnet-go

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    log_info "Service installed: $service_file"
}

# ── Binary installation ─────────────────────────────────────────────────────

install_binary() {
    log_info "Installing BirdNET-Go binary..."

    # Check if a local binary is provided
    local local_binary=""
    for candidate in ./${BINARY_NAME} ./build/${BINARY_NAME} ../build/${BINARY_NAME}; do
        if [[ -x "$candidate" ]]; then
            local_binary="$candidate"
            break
        fi
    done

    if [[ -n "$local_binary" ]]; then
        log_info "Using local binary: $local_binary"
        cp "$local_binary" "${INSTALL_DIR}/${BINARY_NAME}"
    else
        log_warn "No local binary found."
        log_warn "Please build BirdNET-Go for linux/arm64 and copy the binary to:"
        log_warn "  ${INSTALL_DIR}/${BINARY_NAME}"
        log_warn ""
        log_warn "Build command (from BirdNET-Go source tree):"
        log_warn "  task linux_arm64"
        log_warn ""
        log_warn "Or cross-compile manually:"
        log_warn "  GOOS=linux GOARCH=arm64 CGO_ENABLED=1 \\"
        log_warn "    CC=aarch64-linux-gnu-gcc go build -o ${BINARY_NAME} ."
        return
    fi

    chmod 755 "${INSTALL_DIR}/${BINARY_NAME}"
    chown root:root "${INSTALL_DIR}/${BINARY_NAME}"
    log_info "Binary installed to ${INSTALL_DIR}/${BINARY_NAME}"
}

# ── Main ────────────────────────────────────────────────────────────────────

main() {
    echo "================================================================"
    echo "  BirdNET-Go Installer for Arduino Uno Q"
    echo "================================================================"
    echo

    detect_hardware
    check_prerequisites
    setup_directories
    install_onnx_runtime
    install_binary
    install_model_files
    create_config
    install_service

    echo
    echo "================================================================"
    log_info "Installation complete!"
    echo "================================================================"
    echo
    echo "Next steps:"
    echo "  1. Ensure the binary is installed:"
    echo "       ls -la ${INSTALL_DIR}/${BINARY_NAME}"
    echo
    echo "  2. Ensure ONNX model files are in place:"
    echo "       ls -la ${MODEL_DIR}/birdnet_cnn.onnx"
    echo "       ls -la ${MODEL_DIR}/birdnet_preproc.npz"
    echo
    echo "  3. Edit the configuration if needed:"
    echo "       nano ${CONFIG_DIR}/config.yaml"
    echo
    echo "  4. Start the service:"
    echo "       sudo systemctl enable --now birdnet-go"
    echo
    echo "  5. Check status:"
    echo "       sudo systemctl status birdnet-go"
    echo "       journalctl -fu birdnet-go"
    echo
    echo "  Web interface: http://\$(hostname -I | awk '{print \$1}'):8080"
    echo
}

main "$@"
