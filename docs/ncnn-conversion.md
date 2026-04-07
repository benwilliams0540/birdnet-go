# NCNN Conversion Notes

This repo treats all pre-existing NCNN artifacts as untrusted until they are
regenerated from a known ONNX source and verified against the TFLite baseline.

## Trusted Source Models

Treat the pristine assets in `source_models/` as the source of truth for NCNN
conversion attempts in this fork, especially:

- `source_models/BirdNET_V2.4.onnx`
- `source_models/BirdNET_GLOBAL_6K_V2.4_Labels.txt`

The embedded CNN-only ONNX asset remains useful for experiments, but it is not
the primary provenance source for this fork's NCNN recovery work.

## Current Conversion Attempts

The Arduino Uno Q has `onnx2ncnn` installed at:

- `/home/arduino/ncnn/tools/onnx/onnx2ncnn`

### Direct full-model attempt from `source_models/BirdNET_V2.4.onnx`

```bash
scp -O source_models/BirdNET_V2.4.onnx arduino@birdnetq.local:/home/arduino/birdnet-ncnn-full/BirdNET_V2.4.onnx

ssh arduino@birdnetq.local \
  "/home/arduino/ncnn/tools/onnx/onnx2ncnn \
    /home/arduino/birdnet-ncnn-full/BirdNET_V2.4.onnx \
    /home/arduino/birdnet-ncnn-full/birdnet.pnnx.param \
    /home/arduino/birdnet-ncnn-full/birdnet.pnnx.bin"
```

Current result on the Uno Q:

```text
Gather not supported yet!
DFT not supported yet!
Unsupported slice step !
```

This means the full BirdNET V2.4 ONNX graph is not directly convertible with the
current `onnx2ncnn` tool on the device.

### Full-model `pnnx` experiment from `source_models/BirdNET_V2.4.onnx`

We also tried direct full-graph conversion with `pnnx` from the same trusted
source model in order to preserve the raw-audio input path and give NCNN a real
chance to accelerate more than just the CNN tail.

Observed results:

- `pnnx` default settings (`optlevel=2`) crashed during later lowering passes
  after successfully parsing the ONNX graph and enumerating BirdNET's frontend
  ops such as `DFT` and `Gather`.
- `pnnx optlevel=0` emitted files, but the resulting `.param` still contained
  PNNX dialect operators such as `aten::sub` and `prim::Constant`, which the
  Uno Q's NCNN runtime could not load.
- `pnnx optlevel=1` emitted a more NCNN-native `.param`, but BirdNET-Go still
  failed to load it on the Uno Q because the runtime reported:

```text
layer Gather not exists or registered
```

The generated full-graph NCNN param also still contained frontend-specific ops
that are not part of the currently working split-model route, including:

- `Gather`
- `DFT`
- `pnnx.Expression`
- `F.linear`

Current conclusion:

- A full raw-audio `birdnet.pnnx.param/bin` pair is not deployable on the Uno Q
  with the current stock NCNN runtime integration.
- The remaining blocker is no longer ONNX conversion alone. It is runtime layer
  support for BirdNET's frontend graph.
- If we continue the full-graph NCNN path, the most realistic next step is not
  generic converter experimentation, but custom-layer work in the runtime.

### Split-model route from the same source ONNX

The viable NCNN path still starts from the trusted
`source_models/BirdNET_V2.4.onnx`, but it first splits away the unsupported
raw-audio front end and then uses `pnnx` to emit the final NCNN model.

1. If you want to inspect the split point directly, generate a CNN-only ONNX
   from the trusted source model:

```bash
python3 source_models/split_birdnet_v24.py \
  --input source_models/BirdNET_V2.4.onnx \
  --output source_models/BirdNET_V2.4.cnn_only.onnx
```

2. The preferred route is to let the build helper do the split in temporary
   workspace and convert it with `pnnx`:

```bash
python3 source_models/build_ncnn_model.py \
  --pnnx /tmp/pnnx-venv/bin/pnnx \
  --output-dir build/ncnn-v24
```

This script writes:

- `build/ncnn-v24/birdnet_cnn_only.param`
- `build/ncnn-v24/birdnet_cnn_only.bin`

It does not need to leave the intermediate split ONNX or `.pnnx` byproducts in
`source_models/`.

Current verified result:

```text
-rw-r--r-- ... birdnet_cnn_only.bin   24M
-rw-r--r-- ... birdnet_cnn_only.param 18K
```

This split point expects the exact tensor produced immediately after the source
ONNX graph's spectrogram affine normalization and transpose:

- input shape: `[1, 2, 96, 511]`
- channel 0: long-window mel path
- channel 1: short-window mel path

BirdNET-Go's NCNN backend now reproduces that preprocessing in Go before
feeding the split NCNN classifier.

The current parity check on `soundscape.wav` shows all three backends agreeing
on the same top species:

```text
- ncnn:  Black-capped Chickadee (0.815)
- onnx:  Black-capped Chickadee (0.814)
- tflite: Black-capped Chickadee (0.814)
```

### Why not `onnx2ncnn` for the split model?

The earlier split-model conversion using `onnx2ncnn` produced a syntactically
loadable NCNN model, but the logits were numerically wrong:

- direct split conversion: output length 6522, but logits in the thousands and
  post-sigmoid detections saturated to `1.000`
- simplified split via `onnxsim` plus `onnx2ncnn`: output came back as
  `8 x 6522`, which is also wrong for BirdNET's single-logit vector

This is why the repo now prefers `pnnx` for the final NCNN artifact generation.

### Earlier CNN-only experiment

The earlier CNN-only experiment from the embedded asset still fails:

```text
Unsupported transpose type !
```

## Conversion Policy

- Do not commit or deploy NCNN model files generated from a failed
  `onnx2ncnn` run.
- Keep the NCNN backend behind explicit verification against fixed fixtures.
- Only mark an NCNN model directory as selectable after parity checks by
  creating `birdnet-go.ncnn-validated` beside the `.param` and `.bin` files.
- Prefer the split-model path from `source_models/BirdNET_V2.4.onnx` plus
  `pnnx` over the older embedded CNN experiment, because it preserves
  provenance from the source-of-truth model while avoiding the unsupported
  DFT/Gather front end and the incorrect `onnx2ncnn` conversion.
- Treat any future full-graph `birdnet.pnnx.param/bin` attempt as experimental
  until the required frontend operators are implemented or registered in the
  target NCNN runtime.
