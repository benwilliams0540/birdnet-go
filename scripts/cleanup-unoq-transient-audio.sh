#!/usr/bin/env bash
set -euo pipefail

UNOQ_HOST="${UNOQ_HOST:-birdnetq.local}"
UNOQ_USER="${UNOQ_USER:-arduino}"
UNOQ_PASSWORD="${UNOQ_PASSWORD:-}"
UNOQ_DATA_ROOT="${UNOQ_DATA_ROOT:-/var/lib/birdnet-go}"

SSH_OPTS=(-o StrictHostKeyChecking=no)

if [[ -n "$UNOQ_PASSWORD" ]]; then
  REMOTE_SUDO_PREFIX="printf '%s\\n' '$UNOQ_PASSWORD' | sudo -S"
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

ssh_wrap "
  set -euo pipefail

  SUDO_PREFIX=\"$REMOTE_SUDO_PREFIX\"
  data_root='$UNOQ_DATA_ROOT'
  transient_dirs=(
    \"\$data_root/tmp\"
    \"\$data_root/spool\"
    \"\$data_root/unprocessed\"
    \"\$data_root/data/tmp\"
    \"\$data_root/data/spool\"
    \"\$data_root/data/unprocessed\"
  )

  found=0
  for dir in \"\${transient_dirs[@]}\"; do
    if [[ -d \"\$dir\" ]]; then
      found=1
      echo \"Cleaning transient audio under \$dir\"
      eval \"\$SUDO_PREFIX find '\$dir' -mindepth 1 -type f \\( \
        -name '*.wav' -o \
        -name '*.flac' -o \
        -name '*.pcm' -o \
        -name '*.raw' -o \
        -name '*.tmp' \
      \\) -print -delete\"
      eval \"\$SUDO_PREFIX find '\$dir' -mindepth 1 -type d -empty -delete\"
    fi
  done

  if [[ \"\$found\" -eq 0 ]]; then
    echo 'No transient audio spool directories found. Nothing removed.'
  fi
"
