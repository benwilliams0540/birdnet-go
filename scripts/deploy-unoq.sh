#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_BINARY="${LOCAL_BINARY:-${1:-$REPO_ROOT/bin/birdnet-go-unoq}}"
BUILD_FLAVOR="${BUILD_FLAVOR:-unoq}"

UNOQ_HOST="${UNOQ_HOST:-birdnetq.local}"
UNOQ_USER="${UNOQ_USER:-arduino}"
UNOQ_PASSWORD="${UNOQ_PASSWORD:-}"

REMOTE_INSTALL_DIR="${REMOTE_INSTALL_DIR:-/opt/birdnet-go}"
REMOTE_BINARY_PATH="$REMOTE_INSTALL_DIR/birdnet-go"
REMOTE_BACKUP_ROOT="${REMOTE_BACKUP_ROOT:-$REMOTE_INSTALL_DIR/backups}"
REMOTE_SERVICE_NAME="${REMOTE_SERVICE_NAME:-birdnet-go}"
REMOTE_TMP_DIR="${REMOTE_TMP_DIR:-/home/$UNOQ_USER/.cache/birdnet-go-deploy}"

TIMESTAMP_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_DIR="$REMOTE_BACKUP_ROOT/$TIMESTAMP_UTC-$BUILD_FLAVOR"
REMOTE_TMP_BINARY="$REMOTE_TMP_DIR/birdnet-go-$TIMESTAMP_UTC"
REMOTE_TMP_BUILD_INFO="$REMOTE_TMP_DIR/BUILD_INFO-$TIMESTAMP_UTC"

GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"

SSH_OPTS=(-o StrictHostKeyChecking=no)

if [[ -n "$UNOQ_PASSWORD" ]]; then
  REMOTE_SUDO_PREFIX="printf '%s\\n' '$UNOQ_PASSWORD' | sudo -S -p ''"
else
  REMOTE_SUDO_PREFIX="sudo"
fi

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

if [[ ! -x "$LOCAL_BINARY" ]]; then
  echo "Local binary is missing or not executable: $LOCAL_BINARY" >&2
  exit 1
fi

echo "Preparing remote deploy workspace on $UNOQ_USER@$UNOQ_HOST ..."
ssh_wrap "mkdir -p '$REMOTE_TMP_DIR'"

echo "Uploading $(basename "$LOCAL_BINARY") ..."
scp_wrap "$LOCAL_BINARY" "$UNOQ_USER@$UNOQ_HOST:$REMOTE_TMP_BINARY"

echo "Stopping $REMOTE_SERVICE_NAME and installing new binary ..."
ssh_wrap "
  set -euo pipefail

  SUDO_PREFIX=\"$REMOTE_SUDO_PREFIX\"

  eval \"\$SUDO_PREFIX mkdir -p '$BACKUP_DIR'\"
  eval \"\$SUDO_PREFIX systemctl stop '$REMOTE_SERVICE_NAME'\"

  if [[ -x '$REMOTE_BINARY_PATH' ]]; then
    eval \"\$SUDO_PREFIX cp '$REMOTE_BINARY_PATH' '$BACKUP_DIR/birdnet-go'\"
  fi

  eval \"\$SUDO_PREFIX install -m 0755 '$REMOTE_TMP_BINARY' '$REMOTE_BINARY_PATH'\"
  cat > '$REMOTE_TMP_BUILD_INFO' <<INFO
deployed_at_utc=$TIMESTAMP_UTC
build_flavor=$BUILD_FLAVOR
git_commit=$GIT_COMMIT
binary_name=$(basename "$LOCAL_BINARY")
backup_dir=$BACKUP_DIR
INFO
  eval \"\$SUDO_PREFIX install -m 0644 '$REMOTE_TMP_BUILD_INFO' '$REMOTE_INSTALL_DIR/BUILD_INFO'\"
  rm -f '$REMOTE_TMP_BINARY' '$REMOTE_TMP_BUILD_INFO'

  eval \"\$SUDO_PREFIX systemctl start '$REMOTE_SERVICE_NAME'\"
  eval \"\$SUDO_PREFIX systemctl --no-pager --full status '$REMOTE_SERVICE_NAME'\" | sed -n '1,20p'
"

echo "Deployment complete."
echo "  host:         $UNOQ_HOST"
echo "  service:      $REMOTE_SERVICE_NAME"
echo "  build flavor: $BUILD_FLAVOR"
echo "  backup dir:   $BACKUP_DIR"
