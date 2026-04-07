# Uno Q GPU Acceleration Options

This note summarizes the current BirdNET-Go findings for GPU acceleration on the
Arduino Uno Q and ranks the realistic next paths.

## Device Facts

Observed directly on the Uno Q:

- Vulkan is available through Mesa Turnip on `Adreno (TM) 702`.
- OpenCL is available through Mesa `rusticl`.
- The current validated NCNN split-model path is numerically correct, but slow
  because BirdNET preprocessing still runs on the CPU in Go before NCNN sees
  the CNN input tensor.

## Current NCNN Situation

The working NCNN model in production is a split CNN-only graph:

- input: `[1, 2, 96, 511]`
- preprocessing: implemented in Go
- CNN inference: handled by NCNN

This is why current timings are disappointing: the expensive frontend work is
still CPU-bound.

We tested a direct full-graph `pnnx` route from `source_models/BirdNET_V2.4.onnx`.
That work established two important facts:

1. The current BirdNET-Go NCNN integration can already detect and load raw-audio
   style `birdnet.pnnx.param/bin` artifacts when they are valid.
2. The generated full-graph model still depends on frontend operators that the
   Uno Q's NCNN runtime does not currently provide, notably `Gather`.

## Best NCNN Custom-Layer Strategy

If we stay with NCNN, the most realistic design is:

### `BirdNETFrontend` custom layer

Implement a single custom NCNN layer that:

- accepts raw audio `[1, 144000]`
- reproduces the proven BirdNET V2.4 frontend now implemented in
  `internal/inference/ncnn/melspec.go`
- emits the validated split tensor `[1, 2, 96, 511]`

Then keep using the already validated split CNN model for the rest of the
network.

Why this is better than adding generic missing layers one by one:

- It avoids implementing and maintaining a broad set of generic ONNX-style
  frontend ops such as `Gather`, `DFT`, `pnnx.Expression`, and `F.linear`.
- It reuses the existing, parity-correct split CNN artifact.
- It keeps the acceleration problem focused on BirdNET's actual hotspot.
- It lets us ship a CPU custom layer first, then add a Vulkan path only if the
  CPU custom layer proves worthwhile.

## Other Viable GPU Paths

### 1. MNN

Most promising non-NCNN option.

Reasons:

- Officially supports `ONNX` and `embedded devices with POSIX interface`.
- Official docs describe adding both `Vulkan` and `OpenCL` operator
  implementations.
- Recent official MNN releases continue improving both `OpenCL` and `Vulkan`
  backends.
- The Uno Q already exposes both Vulkan and OpenCL.

Practical caveat:

- We have not yet proven that BirdNET V2.4 imports cleanly end to end.
- If BirdNET's frontend ops are not supported by the converter/runtime, we may
  still need a custom op path.

Why it is still attractive:

- OpenCL on Adreno is real and available on this device today.
- MNN appears to have a more explicit GPU custom-op story than our current
  NCNN integration.

### 2. IREE

Technically plausible, but higher risk.

Reasons:

- Officially supports `ONNX` import.
- Officially supports `Vulkan` deployment.
- Supports Linux and ARM targets.

Practical caveat:

- Official IREE docs explicitly note that ONNX operator support is still an
  active investment area and missing operators can fail during legalization.
- We have not verified BirdNET's frontend ops against IREE's import/lowering
  pipeline yet.

Why it is interesting:

- If BirdNET imports and lowers successfully, IREE offers a serious full-graph
  Vulkan path rather than a partial delegate path.

### 3. TVM

Potentially viable, but integration cost is high.

Reasons:

- Official TVM docs include OpenCL deployment for Adreno targets.
- TVM can generate Adreno-specific OpenCL code paths using
  `target="opencl -device=adreno"`.

Practical caveat:

- The official Adreno deployment flow is strongly Android-oriented.
- Integrating TVM into BirdNET-Go would require a bigger compile/deploy
  pipeline than NCNN or MNN.

Why it is interesting:

- The Uno Q has a working OpenCL device, so TVM's Adreno path is not purely
  theoretical here.

## Paths That Look Weak Right Now

### LiteRT / TFLite GPU delegate on Linux

Unpromising as an off-the-shelf solution.

- Official LiteRT GPU docs are published around Android and iOS platform flows.
- LiteRT's delegate API does allow a custom delegate, but that would be a new
  accelerator integration project, not a quick drop-in fix for this Debian
  device.

### ONNX Runtime as the direct GPU runtime

Unpromising by itself on this hardware.

- Official ONNX Runtime provider docs do not list a built-in Vulkan or OpenCL
  execution provider.
- The relevant non-vendor alternatives are community-maintained or preview
  paths such as TVM.
- A custom ONNX Runtime GPU path would mean building a plugin execution
  provider, which is a larger project than an NCNN custom layer.

## Recommendation

Ranked next steps:

1. Build a `BirdNETFrontend` custom layer for NCNN and keep the validated split
   CNN artifact.
2. In parallel or immediately after, prototype `MNN` on the Uno Q because it is
   the strongest non-NCNN option given the device's working OpenCL and Vulkan
   stacks.
3. Treat `IREE` as a research branch, not the primary path.
4. Treat `TVM` as a higher-effort alternative if both NCNN custom-layer and MNN
   stall.

If we optimize for fastest route to a plausible win, NCNN custom frontend is
the best next move. If we optimize for hedge value, MNN is the best backup.
