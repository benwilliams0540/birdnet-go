# Uno Q GPU Acceleration Summary

## Outcome

BirdNET-Go now has a validated NCNN path for the Arduino Uno Q that matches the
trusted TFLite and ONNX backends on the `soundscape.wav` fixture.

The effective recovery path is:

1. Use `source_models/BirdNET_V2.4.onnx` as the source of truth.
2. Split away the unsupported raw-audio frontend with
   `source_models/split_birdnet_v24.py`.
3. Convert the split model with `pnnx`, not plain `onnx2ncnn`.
4. Run parity checks before marking the NCNN model directory as validated.

## Key Findings

### QNN is not a viable route on this device

- The Uno Q exposes a Hexagon DSP, not a supported QNN/NPU target for this use case.
- The Qualcomm QNN SDK is not a practical path for BirdNET-Go on this hardware.

### The main NCNN blocker was conversion fidelity, not BirdNET preprocessing

- The Go-side split preprocessing now matches the trusted ONNX model closely.
- The split ONNX tail matches the full `BirdNET_V2.4.onnx` logits exactly when
  fed the correct intermediate tensor.
- The old `onnx2ncnn` artifacts were numerically wrong even when the input
  tensor was correct.

Symptoms from the bad `onnx2ncnn` path:

- logits in the thousands
- saturated post-sigmoid predictions at `1.000`
- bogus top labels unrelated to the fixture audio

### `pnnx` produced the first parity-correct NCNN artifact

The `pnnx`-generated NCNN model produced sensible logits and matched the first
chunk top result from ONNX and TFLite:

- NCNN: Black-capped Chickadee `0.815`
- ONNX: Black-capped Chickadee `0.814`
- TFLite: Black-capped Chickadee `0.814`

On the full `soundscape.wav` compare, the leading species stack also aligned:

- Black-capped Chickadee
- Dark-eyed Junco
- House Finch
- Engine
- American Goldfinch

## System Overview Notes

### GPU metric

The system overview GPU card already uses a percentage-like load metric as its
primary value and sparkline. The frequency shown in MHz is only the secondary
footer detail.

That percentage is an estimate derived from Linux `devfreq/trans_stat`, not a
vendor-specific direct utilization counter.

### Temperature graph

The missing temperature sparkline came from a backend mismatch:

- `/api/v2/system/temperature/cpu` already had a permissive fallback for
  untyped thermal zones
- the metrics collector did not

That meant the UI could show a current temperature while the history series
stayed empty. The collector now uses the same fallback behavior, so the
temperature card can build a sparkline on devices like the Uno Q.

## Repo Changes

### Added or updated

- `source_models/build_ncnn_model.py`
- `source_models/split_birdnet_v24.py`
- `internal/inference/ncnn/melspec.go`
- `internal/inference/ncnn/ncnn.go`
- `internal/observability/collector.go`
- `internal/observability/collector_test.go`
- `docs/ncnn-conversion.md`

### Cleaned up

Removed stale or misleading local artifacts that were no longer useful:

- `source_models/BirdNET_V2.4.pnnxsim.onnx`
- `source_models/BirdNET_V2.4.cnn_only.pnnxsim.onnx`
- generated local compare output directories
- generated local binaries
- transient NCNN progress notes

The repo now keeps the source models and scripts, while derived NCNN artifacts
are meant to be regenerated on demand.

## Uno Q State

Validated NCNN model directory on the device:

- `/opt/birdnet-go/model-v24-split`

Current safe live config remains:

- `backend: tflite`
- `version: "2.4"`
- `ncnnmodeldir: /opt/birdnet-go/model-v24-split`

That keeps the service on the conservative baseline while leaving the validated
NCNN model ready to switch to.

## Regeneration Workflow

To rebuild the NCNN artifact locally:

```bash
python3 source_models/build_ncnn_model.py \
  --pnnx /tmp/pnnx-venv/bin/pnnx \
  --output-dir build/ncnn-v24
```

Then validate it against the Uno Q fixture flow before creating the validation
marker in the deployed model directory.
