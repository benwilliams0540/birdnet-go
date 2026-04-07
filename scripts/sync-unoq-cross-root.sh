#!/usr/bin/env bash
set -euo pipefail

CROSS_ROOT="${CROSS_ROOT:-$HOME/.cross/birdnetq}"
UNOQ_HOST="${UNOQ_HOST:-birdnetq.local}"
UNOQ_USER="${UNOQ_USER:-arduino}"
UNOQ_PASSWORD="${UNOQ_PASSWORD:-}"

REMOTE_TFLITE_LIB_DIR="${REMOTE_TFLITE_LIB_DIR:-/usr/local/lib}"
REMOTE_TFLITE_SRC_ROOT="${REMOTE_TFLITE_SRC_ROOT:-/home/arduino/src/tensorflow}"
REMOTE_SYS_LIB_DIR="${REMOTE_SYS_LIB_DIR:-/usr/lib/aarch64-linux-gnu}"

SYSROOT_LIB_DIR="$CROSS_ROOT/sysroot/usr/local/lib"
SYSROOT_INCLUDE_DIR="$CROSS_ROOT/sysroot/usr/local/include"
SYSROOT_TARGET_LIB_DIR="$CROSS_ROOT/sysroot/usr/lib/aarch64-linux-gnu"

SSH_OPTS=(-o StrictHostKeyChecking=no)

ssh_wrap() {
  if [[ -n "$UNOQ_PASSWORD" ]]; then
    sshpass -p "$UNOQ_PASSWORD" ssh "${SSH_OPTS[@]}" "$UNOQ_USER@$UNOQ_HOST" "$@"
    return
  fi

  ssh "${SSH_OPTS[@]}" "$UNOQ_USER@$UNOQ_HOST" "$@"
}

scp_wrap() {
  if [[ -n "$UNOQ_PASSWORD" ]]; then
    sshpass -p "$UNOQ_PASSWORD" scp -O "${SSH_OPTS[@]}" "$@"
    return
  fi

  scp -O "${SSH_OPTS[@]}" "$@"
}

mkdir -p "$SYSROOT_LIB_DIR" "$SYSROOT_INCLUDE_DIR" "$SYSROOT_TARGET_LIB_DIR"
chmod -f u+w "$SYSROOT_LIB_DIR"/libtensorflowlite_c.so* "$SYSROOT_TARGET_LIB_DIR"/librt* 2>/dev/null || true
rm -f "$SYSROOT_LIB_DIR"/libtensorflowlite_c.so "$SYSROOT_LIB_DIR"/libtensorflowlite_c.so.* "$SYSROOT_TARGET_LIB_DIR"/librt.so "$SYSROOT_TARGET_LIB_DIR"/librt.so.1 "$SYSROOT_TARGET_LIB_DIR"/librt.a

echo "Syncing TFLite shared library from $UNOQ_USER@$UNOQ_HOST ..."
scp_wrap "$UNOQ_USER@$UNOQ_HOST:$REMOTE_TFLITE_LIB_DIR/libtensorflowlite_c.so*" "$SYSROOT_LIB_DIR/"

echo "Syncing target runtime linker shims ..."
scp_wrap \
  "$UNOQ_USER@$UNOQ_HOST:$REMOTE_SYS_LIB_DIR/librt.a" \
  "$UNOQ_USER@$UNOQ_HOST:$REMOTE_SYS_LIB_DIR/librt.so.1" \
  "$SYSROOT_TARGET_LIB_DIR/"
ln -sf librt.so.1 "$SYSROOT_TARGET_LIB_DIR/librt.so"

echo "Syncing TFLite headers into $SYSROOT_INCLUDE_DIR ..."
ssh_wrap "
  set -euo pipefail
  cd '$REMOTE_TFLITE_SRC_ROOT'
  tar -cf - \
    tensorflow/lite/c \
    tensorflow/lite/core/c \
    tensorflow/lite/core/async/c \
    tensorflow/lite/delegates/xnnpack \
    tensorflow/lite/builtin_ops.h \
    tensorflow/lite/context.h
" | tar -xf - -C "$SYSROOT_INCLUDE_DIR"

echo "Uno Q cross-root updated:"
echo "  lib dir:     $SYSROOT_LIB_DIR"
echo "  include dir: $SYSROOT_INCLUDE_DIR"
echo "You can now run: task unoq"
