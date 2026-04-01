# GPU Acceleration on Arduino SA Imola (Qualcomm Adreno)

## Hardware

| Property | Value |
|---|---|
| Board | Arduino SA Imola |
| GPU | Qualcomm Adreno (modalias: `qcom,adreno-07000200`) |
| Kernel | Linux 6.16.7 aarch64 |
| GPU devfreq path | `/sys/class/devfreq/5900000.gpu` |
| GPU freq range | 355 MHz idle → 845 MHz max |
| ONNX Runtime deployed | v1.24.1 (CPU-only build) |

## Why GPU Acceleration Isn't Active Now

ONNX Runtime 1.24.1 was built without the **QNN (Qualcomm Neural Network) Execution Provider**. The standard Linux ARM64 release only includes the CPU execution provider. All inference currently runs on CPU.

Additionally, the BirdNET+ V3.0 preview3 ONNX model on the device (`BirdNET+_V3.0-preview3_Global_11K_FP32.onnx`) has fully dynamic input shapes (both batch and sample-count dimensions are `-1`), which is incompatible with the fixed-shape inference pipeline in birdnet-go regardless of execution provider.

---

## What's Required

### 1. Qualcomm AI Engine Direct SDK (QNN SDK)

The QNN SDK is Qualcomm's inference framework that underpins the ONNX Runtime QNN EP. It is free but requires registration.

- **Download:** [Qualcomm Developer Portal](https://developer.qualcomm.com/software/ai-engine-direct-sdk)
- **Toolchain needed:** `aarch64-oe-linux-gcc11.2` (the Linux embedded variant — *not* Android)
- **Size:** ~2 GB

### 2. ONNX Runtime Built with QNN EP

No official pre-built Linux ARM64 binary with QNN EP exists for v1.24.1. Options:

#### Option A — Community wheel (v1.23.2, faster path)

Radxa maintains an aarch64 Linux wheel with QNN support:

```bash
pip3 install https://github.com/ZIFENG278/onnxruntime/releases/download/v1.23.2/onnxruntime_qnn-1.23.2-cp312-cp312-linux_aarch64.whl
```

This gives a Python wheel only. To use it from Go, you would need to extract the shared library (`libonnxruntime.so`) from the wheel.

#### Option B — Build from source (v1.24.1, full control)

Clone and build ONNX Runtime with QNN support:

```bash
git clone --branch v1.24.1 https://github.com/microsoft/onnxruntime
cd onnxruntime

./build.sh \
  --use_qnn \
  --qnn_home /path/to/qnn-sdk \
  --build_shared_lib \
  --config Release \
  --parallel \
  --skip_tests \
  --build_dir build/Linux
```

**Required CMake patch** — the official build assumes Android; swap it for the Linux OE toolchain by editing `cmake/CMakeLists.txt` around line 840:

```cmake
# Before:
set(QNN_ARCH_ABI aarch64-android)

# After:
set(QNN_ARCH_ABI aarch64-oe-linux-gcc11.2)
```

Output: `build/Linux/Release/libonnxruntime.so` — drop this in `/opt/birdnet-go/lib/` and update `onnxruntimepath` in the config.

### 3. A Compatible V3.0 ONNX Model

The BirdNET+ V3.0 preview3 model has dynamic input shapes and cannot be used as-is. A valid V3.0 ONNX model must have a **fixed input shape**.

The birdnet-go inference code (`internal/inference/onnx/detection.go`) auto-detects V3.0 by matching:
- Last input dimension = **160,000 samples** (32 kHz × 5 s)
- Output count = **2** (embeddings + logits)

The correct model to target is the **BirdNET+ V3.0 production release** (not the preview3 PyTorch export). Watch the [BirdNET releases page](https://github.com/kahst/BirdNET-Analyzer/releases) for a fixed-shape ONNX export.

### 4. Session Provider Selection in Go

Once a QNN-enabled `libonnxruntime.so` is in place, the execution provider must be set when creating the ONNX session. In `internal/inference/onnx/classifier.go`, the session options would need a QNN provider entry:

```go
// Adreno GPU backend
sessionOptions.AppendExecutionProviderQNN(map[string]string{
    "backend_path": "libQnnHtp.so",   // HTP/NPU (quantized models, fastest)
    // or:
    "backend_path": "libQnnGpu.so",   // Adreno GPU (float32 models)
})
```

The `libQnnHtp.so` / `libQnnGpu.so` libraries ship with the QNN SDK and need to be present on the device alongside the ONNX Runtime library.

---

## Backend Comparison

| Backend | Target | Model type | Notes |
|---|---|---|---|
| **CPU** | All cores | Any | Current state, no changes needed |
| **HTP (Hexagon DSP)** | Qualcomm Hexagon DSP tensor extensions | Quantized int8 | Fastest; requires model quantization (QDQ format) |
| **Adreno GPU** | Adreno 700 series | Float32 or float16 | Newer, less mature; good for unquantized models |

For maximum throughput with a quantized model, the HTP path uses the Hexagon DSP's tensor extensions and will outperform the Adreno GPU. The GPU backend is the better fit if staying with an unquantized float32 model.

> **Note on Hexagon terminology:** The Hexagon is a DSP, not a discrete NPU. The "Tensor Processor" (HTP) designation refers to tensor-operation extensions added to the Hexagon DSP in newer Qualcomm SoCs to accelerate AI workloads — it is part of the DSP subsystem, not a separate neural processing unit.

---

## Summary Checklist

- [ ] Register and download the Qualcomm AI Engine Direct (QNN) SDK
- [ ] Build ONNX Runtime v1.24.1 from source with `--use_qnn` and the Linux OE CMake patch
- [ ] Deploy the new `libonnxruntime.so` to `/opt/birdnet-go/lib/` on the device
- [ ] Copy QNN backend libraries (`libQnnHtp.so`, `libQnnGpu.so`, etc.) to the device
- [ ] Obtain a V3.0 ONNX model with fixed input shape (160,000 samples, 2 outputs) when the BirdNET+ V3.0 production release is available
- [ ] Add QNN execution provider selection to `internal/inference/onnx/classifier.go`
- [ ] For HTP backend: quantize the model to int8 using the QNN model quantization tools
- [ ] Test and benchmark both HTP and GPU backends
