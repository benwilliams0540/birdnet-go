# Uno Q Acceleration Summary

## Outcome

BirdNET-Go now has a validated NCNN path for the Arduino Uno Q that matches the
trusted TFLite and ONNX backends on the `soundscape.wav` fixture.

The recovery path is:

1. Use `source_models/BirdNET_V2.4.onnx` as the source of truth.
2. Split away the unsupported raw-audio front end with
   `source_models/split_birdnet_v24.py`.
3. Convert the split model with `pnnx`, not plain `onnx2ncnn`.
4. Run parity checks before marking an NCNN model directory as validated.

## Key Findings

### QNN is not a viable route on this device

- The Uno Q exposes a Hexagon DSP, not a practical QNN target for this use
  case.
- The Qualcomm QNN SDK is not a workable deployment path for BirdNET-Go on
  this hardware.

### The main NCNN blocker was conversion fidelity

- The Go-side split preprocessing matches the trusted ONNX model closely.
- The split ONNX tail matches the full `BirdNET_V2.4.onnx` logits when fed the
  correct intermediate tensor.
- The old `onnx2ncnn` artifacts were numerically wrong even with correct input.

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

## System Overview Notes

### GPU metric

The system overview GPU card uses a percentage-like load metric as its primary
value and sparkline. The frequency shown in MHz is only the secondary detail.

That percentage is an estimate derived from Linux `devfreq/trans_stat`, not a
vendor-direct utilization counter.

### Temperature graph

The missing temperature sparkline came from a backend mismatch:

- `/api/v2/system/temperature/cpu` already had a permissive fallback for
  untyped thermal zones
- the metrics collector did not

The collector now uses the same fallback behavior, so the temperature card can
build a sparkline on devices like the Uno Q.

## Uno Q State

Validated NCNN model directory on the device:

- `/opt/birdnet-go/model-v24-split`

Current safe live config:

- `backend: tflite`
- `version: "2.4"`
- `ncnnmodeldir: /opt/birdnet-go/model-v24-split`

That keeps the service on the conservative baseline while leaving the validated
NCNN model ready to switch to.

## Related Notes

- [NCNN conversion notes](./ncnn-conversion.md)
