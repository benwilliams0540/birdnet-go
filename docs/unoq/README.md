# Arduino Uno Q Notes

This folder collects the current BirdNET-Go guidance for the Arduino Uno Q.

## Current Recommendation

For a portable Uno Q deployment, the safest live configuration is:

- `backend: tflite`
- `version: "2.4"`

ONNX remains useful for comparison and debugging. NCNN is available for
validated split-model experiments, but it is not the default recommendation for
production use on this device.

## Included Notes

- [Acceleration summary](./acceleration-summary.md)
- [NCNN conversion notes](./ncnn-conversion.md)

## Important Constraints

- Qualcomm QNN is not a viable path for this board in this fork.
- Existing NCNN artifacts should be treated as untrusted until they are
  regenerated from the source models and validated against fixed fixtures.
- Large model assets in `source_models/` are intentionally kept out of git.
  See [source_models/README.md](../../source_models/README.md).
