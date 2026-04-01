# QNN Hardware Acceleration — Arduino SA Imola (QCM2290)

This document describes how to enable Qualcomm QNN hardware acceleration for
BirdNET-Go on the Arduino SA Imola board.  The current implementation uses the
**native QNN C API** (no ONNX Runtime rebuild required) and supports two
backends: Adreno GPU and Hexagon DSP (HTP).

---

## Hardware Summary

| Property | Value |
|---|---|
| Board | Arduino SA Imola |
| SoC | Qualcomm QCM2290 |
| CPU | Cortex-A53 × 4 @ 1.8 GHz |
| GPU | Adreno 702 (`a702_sqe.fw`) |
| DSP | Hexagon 686 (v68 architecture), running as ADSP |
| Kernel | Linux 6.16.7 aarch64 |
| OpenCL | 3.0 via Mesa Rusticl (device: `FD702`) |
| QAIRT SDK | 2.39.0.250926 |

---

## Backend Status

| Backend | Library | Status | Notes |
|---|---|---|---|
| **CPU** | — | ✅ Working | Default; no changes needed |
| **GPU (Adreno 702)** | `libQnnGpu.so` | ⚠️ Untested | OpenCL available via Mesa Rusticl; `libQnnGpu.so` compatibility unknown — needs testing |
| **HTP (Hexagon 686 v68)** | `libQnnHtp.so` | ❌ Blocked | `fastrpc` kernel module loaded and ADSP running, but no `/dev/fastrpc*` character device exposed to userspace |

> **Hexagon terminology:** The Hexagon is a DSP (Digital Signal Processor), not
> a discrete NPU.  The "HTP" (Hexagon Tensor Processor) designation refers to
> tensor-operation extensions added to the Hexagon DSP for AI workloads — it is
> part of the DSP subsystem.

---

## Architecture Overview

The QNN backend in birdnet-go uses the **native QNN C API** loaded at runtime
via `dlopen`.  The binary itself has no compile-time dependency on the QAIRT
SDK.

Two model loading modes are supported:

1. **Model library** (portable) — a compiled `.so` generated from the ONNX
   model on any Linux/Windows host with the QAIRT SDK.  The QNN backend
   JIT-compiles the graph for the target hardware on first inference.

2. **Context binary** (fastest) — a device/driver-specific pre-compiled binary
   generated on the target device.  Eliminates JIT compilation overhead on
   every restart.

The recommended workflow is to start with the model library (generated on
Windows below), and optionally generate a context binary on the device later.

---

## Prerequisites

| Requirement | Where to get it |
|---|---|
| QAIRT SDK 2.39+ | [Qualcomm Developer Portal](https://developer.qualcomm.com/software/ai-engine-direct-sdk) (free, registration required) |
| Python 3.10.x | [python.org](https://www.python.org/downloads/) |
| `birdnet_int8_cnn.onnx` | `internal/classifier/data/birdnet_int8_cnn.onnx` in this repo |
| `zig` toolchain | [ziglang.org/download](https://ziglang.org/download/) (for cross-compiling the Go binary) |

---

## Step 1 — Generate the QNN Model Library on Windows

These steps run on your **Windows machine** using the QAIRT SDK tools.

### 1.1 — Set up the QAIRT environment

Open PowerShell, navigate to the QAIRT SDK directory, and run the environment
setup script:

```powershell
cd C:\path\to\qairt\2.39.0.250926
.\bin\envsetup.ps1
```

This adds the SDK tools to your `PATH` and sets required environment variables.

Verify the tools are available:

```powershell
qnn-onnx-converter --version
qnn-model-lib-generator --version
```

### 1.2 — Install Python dependencies

The converter requires several Python packages:

```powershell
pip install onnx onnxruntime numpy
```

### 1.3 — Convert the ONNX model to QNN format

```powershell
qnn-onnx-converter `
    -i C:\path\to\birdnet_int8_cnn.onnx `
    --input_dim input_1 1,144000 `
    -o C:\qnn_model\birdnet_int8_cnn.cpp
```

- `--input_dim input_1 1,144000` — specifies the input tensor name and shape
  (batch size 1, 144 000 float32 samples = 48 kHz × 3 s)
- The output is a `.cpp` model description and a `.bin` weights file

> **Tip:** To find the correct input tensor name and shape, you can inspect the
> model first:
> ```powershell
> qnn-netron C:\path\to\birdnet_int8_cnn.onnx
> ```
> This opens a browser-based graph viewer.

### 1.4 — Compile the model library

```powershell
qnn-model-lib-generator `
    -m C:\qnn_model\birdnet_int8_cnn.cpp `
    -b C:\qnn_model\birdnet_int8_cnn.bin `
    -o C:\qnn_model\libs\ `
    --lib_targets aarch64-oe-linux-gcc11.2
```

Output: `C:\qnn_model\libs\aarch64-oe-linux-gcc11.2\libmodel_net.so`

---

## Step 2 — Deploy Artifacts to the Device

Copy the QNN backend libraries from the QAIRT SDK and the compiled model library
to the device.  `/opt/qnn/` is used here but any directory works.

```powershell
# From PowerShell on Windows — adjust the device IP and password as needed

$DEVICE = "arduino@192.168.86.27"
$QAIRT  = "C:\path\to\qairt\2.39.0.250926"
$MODEL  = "C:\qnn_model\libs\aarch64-oe-linux-gcc11.2"

# Create target directory on device
ssh $DEVICE "mkdir -p /opt/qnn"

# QNN runtime libraries from the QAIRT SDK
scp "$QAIRT\lib\aarch64-oe-linux-gcc11.2\libQnnGpu.so"           "${DEVICE}:/opt/qnn/"
scp "$QAIRT\lib\aarch64-oe-linux-gcc11.2\libQnnSystem.so"         "${DEVICE}:/opt/qnn/"
scp "$QAIRT\lib\aarch64-oe-linux-gcc11.2\libQnnCpu.so"            "${DEVICE}:/opt/qnn/"

# Compiled model library
scp "$MODEL\libmodel_net.so"                                      "${DEVICE}:/opt/qnn/"
```

For HTP (Hexagon DSP) you additionally need:

```powershell
scp "$QAIRT\lib\aarch64-oe-linux-gcc11.2\libQnnHtp.so"            "${DEVICE}:/opt/qnn/"
scp "$QAIRT\lib\aarch64-oe-linux-gcc11.2\libQnnHtpPrepare.so"     "${DEVICE}:/opt/qnn/"
scp "$QAIRT\lib\aarch64-oe-linux-gcc11.2\libQnnHtpV68Stub.so"     "${DEVICE}:/opt/qnn/"
scp "$QAIRT\lib\aarch64-oe-linux-gcc11.2\libQnnHtpV68CalculatorStub.so" "${DEVICE}:/opt/qnn/"
```

---

## Step 3 — Build birdnet-go with the QNN Tag

The QNN backend is gated behind the `qnn` build tag and requires the QAIRT SDK
headers at compile time.

### 3.1 — Symlink the QAIRT headers

From the root of the birdnet-go repository on your development machine:

```bash
# macOS / Linux:
ln -s /path/to/qairt/2.39.0.250926/include vendor/qairt/include

# Windows (PowerShell, run as Administrator):
New-Item -ItemType Junction `
    -Path "vendor\qairt\include" `
    -Target "C:\path\to\qairt\2.39.0.250926\include"
```

### 3.2 — Build the binary

```bash
# Ensure the TFLite ARM64 library is available (downloads automatically via task):
# task linux_arm64   ← uses the Taskfile; or manually:

TFLITE_LIB_DIR=/tmp   # path containing libtensorflowlite_c.so
CC="zig cc -target aarch64-linux-gnu" \
CXX="zig c++ -target aarch64-linux-gnu" \
CGO_ENABLED=1 \
GOOS=linux GOARCH=arm64 \
CGO_CFLAGS="-I$HOME/src/tensorflow" \
CGO_LDFLAGS="-L${TFLITE_LIB_DIR} -ltensorflowlite_c" \
go build -tags onnx,qnn -o birdnet-go-linux-arm64 .
```

> **Note:** The `qnn` tag compiles `internal/inference/qnn/qnn_backend.c` which
> `#include`s headers from `vendor/qairt/include/QNN/`.  Without the symlink
> the build will fail with "file not found" errors.

### 3.3 — Deploy the new binary

```bash
scp birdnet-go-linux-arm64 arduino@192.168.86.27:/tmp/birdnet-go-new
ssh arduino@192.168.86.27 "sudo sh -c 'systemctl stop birdnet-go && \
    mv /tmp/birdnet-go-new /opt/birdnet-go/birdnet-go && \
    chmod +x /opt/birdnet-go/birdnet-go && \
    systemctl start birdnet-go'"
```

---

## Step 4 — Configure birdnet-go

In the web UI go to **Settings → BirdNET** and set:

| Setting | Value |
|---|---|
| Model Version | BirdNET Global 6K V2.4 INT8 CNN (ONNX) |
| QNN Backend | GPU — Adreno via OpenCL |
| QNN Lib Dir | `/opt/qnn` |
| QNN Model Lib Dir | `/opt/qnn` |

Or edit `/etc/birdnet-go/config.yaml` directly:

```yaml
birdnet:
  version: "2.4-int8-cnn"
  qnnbackend: "gpu"          # "gpu" or "htp"
  qnnlibdir: "/opt/qnn"
  qnnmodellibdir: "/opt/qnn"
```

Restart the service.  If QNN initialisation fails (e.g. library incompatibility)
birdnet-go logs a warning and automatically falls back to CPU ONNX inference.
No manual intervention is needed.

---

## Step 5 (Optional) — Generate a Context Binary on the Device

A context binary is a pre-compiled, device-specific representation of the model
for a particular backend.  It eliminates the JIT compilation on every restart,
reducing start-up time from ~10 s to ~1 s.

Copy the `qnn-context-binary-generator` tool from the QAIRT SDK to the device
and run it:

```bash
# Copy the tool (Linux aarch64 binary is in the SDK)
scp /path/to/qairt/2.39.0.250926/bin/aarch64-oe-linux-gcc11.2/qnn-context-binary-generator \
    arduino@device:/opt/qnn/

# On the device:
ssh arduino@device
LD_LIBRARY_PATH=/opt/qnn /opt/qnn/qnn-context-binary-generator \
    --model       /opt/qnn/libmodel_net.so \
    --backend     /opt/qnn/libQnnGpu.so \
    --binary_file /opt/qnn/birdnet_int8_cnn_gpu_context.bin
```

birdnet-go automatically prefers the context binary over the model library when
it finds a file named `<model_name>_<backend>_context.bin` in the model lib
directory.  With the paths above it will find
`/opt/qnn/birdnet_int8_cnn_gpu_context.bin` automatically.

---

## HTP (Hexagon DSP) — Current Blocker

The Hexagon 686 DSP is running (ADSP remoteproc is active, `fastrpc` kernel
module is loaded) but there is currently **no `/dev/fastrpc*` character device**
node on the device.  The QNN HTP backend uses FastRPC to communicate with the
DSP, so without the device node it will fail to initialise.

**Possible fixes:**

1. **Kernel configuration** — the `CONFIG_QCOM_FASTRPC` driver may need
   `CONFIG_QCOM_FASTRPC_ADSP` or a device tree change to expose the char device.
   Check:
   ```bash
   zcat /proc/config.gz | grep FASTRPC
   ```

2. **Manual `mknod`** (temporary test) — if the major/minor numbers are known:
   ```bash
   sudo mknod /dev/adsprpc-smd c <major> <minor>
   ```
   Determine the numbers from `/proc/devices` or `dmesg | grep fastrpc`.

3. **udev rule** — add a udev rule to create the node automatically on boot once
   the driver is confirmed working.

---

## Troubleshooting

### "QNN initialization failed, falling back to ONNX CPU"
Check the logs for the specific error:
```bash
journalctl -u birdnet-go -n 50 | grep -i qnn
```

Common causes:
- `libQnnGpu.so` not found in `qnnlibdir` — check the path and file permissions
- `libQnnSystem.so` not found — required alongside the backend library
- OpenCL platform incompatibility — `libQnnGpu.so` may require Qualcomm's
  proprietary OpenCL rather than Mesa Rusticl; check `clinfo` on the device

### Model library not found
birdnet-go looks for:
1. `<qnnmodellibdir>/birdnet_int8_cnn_gpu_context.bin` (context binary, preferred)
2. `<qnnmodellibdir>/libbirdnet_int8_cnn_net.so` (model library)
3. `<qnnmodellibdir>/libmodel_net.so` (default `qnn-model-lib-generator` output)

### OpenCL device check
```bash
clinfo | grep -A3 "Device Name"
```
The device should appear as `FD702` (Freedreno 702).  If no devices are listed,
OpenCL is not working and the GPU backend will fail.

### Testing `libQnnGpu.so` manually
```bash
LD_LIBRARY_PATH=/opt/qnn /opt/qnn/qnn-net-run \
    --model /opt/qnn/libmodel_net.so \
    --backend /opt/qnn/libQnnGpu.so \
    --input_list /tmp/test_input.txt
```
This validates the QNN GPU pipeline independently of birdnet-go.

---

## File Reference

| File | Purpose |
|---|---|
| `internal/inference/qnn/qnn_backend.h` | C header for the QNN shim |
| `internal/inference/qnn/qnn_backend.c` | QNN C API wrapper (dlopen-based) |
| `internal/inference/qnn/classifier.go` | Go wrapper (`-tags qnn`) |
| `internal/inference/qnn.go` | Go adapter (`-tags qnn`) |
| `internal/inference/qnn_stub.go` | No-op stub for standard builds |
| `internal/classifier/model_qnn.go` | BirdNET QNN initializer (`-tags qnn`) |
| `internal/classifier/model_qnn_stub.go` | No-op stub for standard builds |
| `vendor/qairt/include` | Symlink to QAIRT SDK `include/` (not committed) |
