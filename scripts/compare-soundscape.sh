#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_AUDIO="${LOCAL_AUDIO:-${1:-$REPO_ROOT/soundscape.wav}}"
LOCAL_ONNX_MODEL="${LOCAL_ONNX_MODEL:-$REPO_ROOT/source_models/BirdNET_V2.4.onnx}"
LOCAL_LABELS="${LOCAL_LABELS:-$REPO_ROOT/source_models/BirdNET_GLOBAL_6K_V2.4_Labels.txt}"

UNOQ_HOST="${UNOQ_HOST:-birdnetq.local}"
UNOQ_USER="${UNOQ_USER:-arduino}"
UNOQ_PASSWORD="${UNOQ_PASSWORD:-}"

REMOTE_BINARY="${REMOTE_BINARY:-/opt/birdnet-go/birdnet-go}"
REMOTE_CONFIG="${REMOTE_CONFIG:-/etc/birdnet-go/config.yaml}"
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-/home/$UNOQ_USER/.cache/birdnet-go-compare}"
REMOTE_AUDIO="$REMOTE_BASE_DIR/$(basename "$LOCAL_AUDIO")"
REMOTE_ONNX_MODEL="$REMOTE_BASE_DIR/$(basename "$LOCAL_ONNX_MODEL")"
REMOTE_LABELS="$REMOTE_BASE_DIR/$(basename "$LOCAL_LABELS")"
REMOTE_ONNX_RUNTIME_PATH="${REMOTE_ONNX_RUNTIME_PATH:-/opt/birdnet-go/lib/libonnxruntime.so.1.24.1}"
REMOTE_NCNN_SOURCE_DIR="${REMOTE_NCNN_SOURCE_DIR:-}"

TOP_K="${TOP_K:-5}"
SUMMARY_K="${SUMMARY_K:-10}"
MAX_CHUNKS="${MAX_CHUNKS:-0}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/compare_results/$STAMP}"

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

remote_run() {
  local quoted=()
  local arg
  for arg in "$@"; do
    quoted+=("$(printf '%q' "$arg")")
  done
  ssh_wrap "${quoted[*]}"
}

normalize_json_file() {
  local path="$1"

  python3 - "$path" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
raw = path.read_text()
start = raw.find("{")
end = raw.rfind("}")

if start == -1 or end == -1 or end < start:
    raise SystemExit(f"Failed to locate JSON payload in {path}")

path.write_text(raw[start:end + 1] + "\n")
PY
}

run_remote_compare() {
  local name="$1"
  shift
  local output_file="$OUTPUT_DIR/$name.json"

  echo "Running $name comparison..."
  if remote_run "$REMOTE_BINARY" --config "$REMOTE_CONFIG" compare-audio "$@" --json --summary-only \
      >"$output_file"; then
    normalize_json_file "$output_file"
    echo "  saved: $output_file"
  else
    echo "  failed: $name" >&2
    return 1
  fi
}

mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$LOCAL_AUDIO" ]]; then
  echo "Audio file not found: $LOCAL_AUDIO" >&2
  exit 1
fi

echo "Preparing compare workspace on $UNOQ_USER@$UNOQ_HOST..."
ssh_wrap "mkdir -p $(printf '%q' "$REMOTE_BASE_DIR")"

echo "Uploading audio fixture..."
scp_wrap "$LOCAL_AUDIO" "$UNOQ_USER@$UNOQ_HOST:$REMOTE_AUDIO"

if [[ -f "$LOCAL_ONNX_MODEL" ]]; then
  echo "Uploading ONNX model..."
  scp_wrap "$LOCAL_ONNX_MODEL" "$UNOQ_USER@$UNOQ_HOST:$REMOTE_ONNX_MODEL"
fi

if [[ -f "$LOCAL_LABELS" ]]; then
  echo "Uploading labels..."
  scp_wrap "$LOCAL_LABELS" "$UNOQ_USER@$UNOQ_HOST:$REMOTE_LABELS"
fi

run_remote_compare \
  tflite \
  --audio "$REMOTE_AUDIO" \
  --backend tflite \
  --version 2.4 \
  --top-k "$TOP_K" \
  --summary-k "$SUMMARY_K" \
  --max-chunks "$MAX_CHUNKS"

if [[ -f "$LOCAL_ONNX_MODEL" && -f "$LOCAL_LABELS" ]]; then
  run_remote_compare \
    onnx \
    --audio "$REMOTE_AUDIO" \
    --backend onnx \
    --version 2.4 \
    --model-path "$REMOTE_ONNX_MODEL" \
    --label-path "$REMOTE_LABELS" \
    --onnx-runtime-path "$REMOTE_ONNX_RUNTIME_PATH" \
    --top-k "$TOP_K" \
    --summary-k "$SUMMARY_K" \
    --max-chunks "$MAX_CHUNKS"
fi

if [[ -n "$REMOTE_NCNN_SOURCE_DIR" ]]; then
  REMOTE_NCNN_CASE_DIR="$REMOTE_BASE_DIR/ncnn-case-$STAMP"
  echo "Preparing validated NCNN wrapper directory from $REMOTE_NCNN_SOURCE_DIR ..."
  ssh_wrap "
    set -euo pipefail
    src=$(printf '%q' "$REMOTE_NCNN_SOURCE_DIR")
    dst=$(printf '%q' "$REMOTE_NCNN_CASE_DIR")
    rm -rf \"\$dst\"
    mkdir -p \"\$dst\"
    copied=0
    for pair in \
      'birdnet_cnn_only.param birdnet_cnn_only.bin' \
      'birdnet.pnnx.param birdnet.pnnx.bin' \
      'birdnet_cnn.param birdnet_cnn.bin' \
      'birdnet_v2_cnn_sim.ncnn.param birdnet_v2_cnn_sim.ncnn.bin' \
      'model.param model.bin'
    do
      set -- \$pair
      if [[ -f \"\$src/\$1\" && -f \"\$src/\$2\" ]]; then
        cp \"\$src/\$1\" \"\$dst/\$1\"
        cp \"\$src/\$2\" \"\$dst/\$2\"
        copied=1
        break
      fi
    done
    if [[ \$copied -ne 1 ]]; then
      echo 'No supported NCNN file pair found in source directory' >&2
      exit 1
    fi
    touch \"\$dst/birdnet-go.ncnn-validated\"
  "

  run_remote_compare \
    ncnn \
    --audio "$REMOTE_AUDIO" \
    --backend ncnn \
    --version 2.4 \
    --ncnn-model-dir "$REMOTE_NCNN_CASE_DIR" \
    --top-k "$TOP_K" \
    --summary-k "$SUMMARY_K" \
    --max-chunks "$MAX_CHUNKS"
fi

python - <<'PY' "$OUTPUT_DIR"
import json
import pathlib
import sys

output_dir = pathlib.Path(sys.argv[1])
files = sorted(output_dir.glob("*.json"))
print("\nSummary")
for path in files:
    data = json.loads(path.read_text())
    summary = data["summary"]
    top = summary.get("topSpeciesByMax", [])
    best = top[0] if top else {}
    name = best.get("commonName") or best.get("scientificName") or best.get("label", "n/a")
    score = best.get("maxConfidence", 0)
    print(
        f"- {path.stem}: backend={summary['backend']} model={summary['modelId']} "
        f"chunks={summary['processedChunkCount']} top={name} ({score:.3f})"
    )
PY

echo
echo "Full JSON results saved under: $OUTPUT_DIR"
