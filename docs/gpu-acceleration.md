# QNN Hardware Acceleration — Arduino Uno Q (QRB2210 / QCM2290)

This document describes how to enable Qualcomm QNN hardware acceleration for
BirdNET-Go on the Arduino Uno Q board. The current implementation uses the
**native QNN C API** (no ONNX Runtime rebuild required) and supports the
**CPU backend** (confirmed working). The GPU backend is blocked pending
OpenCL driver support (see [GPU Backend Status](#gpu-backend-adreno-702)).

---

## Hardware Summary

| Property | Value |
|---|---|
| Board | Arduino Uno Q |
| SoC | Qualcomm QRB2210 (QCM2290-compatible) |
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
| **CPU** | `libQnnCpu.so` | ✅ **Working** | Confirmed end-to-end; requires `fake_soc.so` (see below) |
| **GPU (Adreno 702)** | `libQnnGpu.so` | ❌ Blocked | Mesa Rusticl missing `clGetDeviceImageInfoQCOM` extension; Qualcomm proprietary OpenCL required |
| **HTP (Hexagon 686 v68)** | `libQnnHtp.so` | ❌ Blocked | `fastrpc` kernel module loaded but no `/dev/fastrpc*` char device exposed to userspace |

---

## Critical Findings (Read Before Deploying)

### 1. SoC ID Whitelist Bypass (`fake_soc.so`)

**Problem:** `libQnnCpu.so` reads `/sys/devices/soc0/soc_id` during
`QnnBackend_Create` and checks the value against an internal whitelist. The
QRB2210 board reports a soc_id that is **not in the whitelist**, causing
`QnnBackend_Create` to return `rc=1006` (initialization failure).

**Solution:** Compile a small `LD_PRELOAD` library that intercepts `open` and
`openat` syscalls, detects the soc_id path, and returns a pipe pre-filled with
a whitelisted soc_id (`356` = SM8250 / Snapdragon 865).

Source: `internal/inference/qnn/fake_soc.c`

```c
// fake_soc.c - compile with:
// aarch64-linux-gnu-gcc -shared -fPIC -o fake_soc.so fake_soc.c -ldl
// Intercepts open/openat for /sys/devices/soc0/soc_id
// Returns a pipe FD pre-filled with FAKE_SOC_ID (default: "356\n")
```

Deploy and enable:
```bash
# Compile on dev machine (requires aarch64 cross-compiler)
aarch64-linux-gnu-gcc -shared -fPIC -o fake_soc.so fake_soc.c -ldl
scp fake_soc.so arduino@192.168.86.27:/opt/qnn/

# Enable permanently via systemd override
sudo tee -a /etc/systemd/system/birdnet-go.service <<'EOF'
Environment=LD_PRELOAD=/opt/qnn/fake_soc.so
Environment=FAKE_SOC_ID=356
EOF
sudo systemctl daemon-reload && sudo systemctl restart birdnet-go
```

Verify it's working:
```bash
journalctl -u birdnet-go -n 5 | grep fake_soc
# Expect: [fake_soc] Loaded, will fake soc_id as: 356
# Expect: [fake_soc] Intercepted open(/sys/devices/soc0/soc_id) -> returning fake fd with '356
```

### 2. RTLD_LOCAL → RTLD_GLOBAL for Backend Library

**Problem:** Loading `libQnnCpu.so` with `RTLD_LOCAL` (the default) caused
a SIGSEGV inside `composeGraphs` at a memory address that spelled "birdnet"
in ASCII (`0x74656e64726962`) — a classic sign of a stale function pointer
being called as a code address. The QNN backend's internal dispatch relies on
globally-visible symbols from the backend library.

**Solution:** Load the backend library with `RTLD_GLOBAL` instead of `RTLD_LOCAL`:

```c
// qnn_backend.c — line in qnn_create_from_model_lib:
// ❌ WRONG: RTLD_LOCAL causes SIGSEGV in composeGraphs
// s->backend_lib = dlopen(backend_lib_path, RTLD_NOW | RTLD_LOCAL);

// ✅ CORRECT: RTLD_GLOBAL allows backend's symbols to be found by model .so
s->backend_lib = dlopen(backend_lib_path, RTLD_NOW | RTLD_GLOBAL);
```

This fix is already in `internal/inference/qnn/qnn_backend.c`.

### 3. QNN Interface Struct ABI (ARM64 Calling Convention)

The `QNN_INTERFACE_VER_TYPE` struct is 552 bytes — too large to fit in ARM64
registers. Per the AAPCS64 ABI, structs >16 bytes are passed by hidden pointer.
The `composeGraphs` call signature must pass this struct correctly:

```c
typedef int (*ComposeGraphsFn_t)(
    Qnn_BackendHandle_t,
    QNN_INTERFACE_VER_TYPE,   // 552 bytes — passed via hidden ptr in ARM64
    Qnn_ContextHandle_t,
    const GraphConfigInfo_t **,
    uint32_t,
    GraphInfo_t ***,
    uint32_t *,
    bool,
    QnnLog_Callback_t,
    QnnLog_Level_t);

// Call site — pass the struct field directly, not a pointer:
rc = compose(s->backend_handle,
             s->iface.QNN_INTERFACE_VER_NAME,   // compiler generates hidden-ptr call
             s->context_handle, ...);
```

---

## Model Architecture

The original BirdNET V2.4 ONNX model contains 4 `DFT` ops (for mel spectrogram
computation via STFT) that the QAIRT converter cannot translate. The model was
**split at the mel spectrogram boundary**:

- **Removed from QNN** (nodes 0–38): audio normalization → framing → Hann window
  → 2×STFT → mel filterbank → power compression
- **Kept in QNN** (nodes 39–302): pure CNN (Conv, Sigmoid, Relu, etc.)

The Go code in `internal/inference/qnn/melspec.go` reimplements the removed
preprocessing and produces the two mel spectrograms that the QNN CNN model expects.

### CNN Model I/O

| | Shape | Meaning |
|---|---|---|
| Input 1 | `[1, 511, 96]` | MEL_SPEC1 (time-freq power spectrum) |
| Input 2 | `[1, 511, 96]` | MEL_SPEC2 (independent mel computation) |
| Output | `[1, 6522]` | Per-species confidence scores |

> **NumSpecies = 6522** (not 6523 like BirdNET TFLite). The CNN subgraph
> output was verified by QNN tensor introspection.

---

## Available Model Libraries on Device

All models are in `/opt/qnn/` on the device:

| File | Precision | Size | Notes |
|---|---|---|---|
| `libbirdnet_cnn_fp32.so` | FP32 | 51.7 MB | **Currently active**; most compatible |
| `libbirdnet_cnn_fp16.so` | FP16 | 26.2 MB | Half-precision weights; untested on CPU backend |
| `libbirdnet_cnn_int8.so` | INT8 | 51.7 MB | INT8 without calibration → similar size to FP32 |
| `libbirdnet_int8_cnn_net.so` | INT8 | ~51 MB | Converted from `birdnet_int8_cnn.onnx` (different pipeline) |
| `libQnnCpu.so` | — | 6.8 MB | CPU backend |
| `libQnnGpu.so` | — | 4.7 MB | GPU backend (blocked) |
| `libQnnSystem.so` | — | 3.4 MB | Required by all backends |
| `fake_soc.so` | — | 71 KB | SoC ID intercept library |

---

## Quick Start

### Prerequisites

| Requirement | Where to get it |
|---|---|
| QAIRT SDK 2.39+ | [Qualcomm Developer Portal](https://developer.qualcomm.com/software/ai-engine-direct-sdk) |
| `aarch64-linux-gnu-gcc` | `apt install gcc-aarch64-linux-gnu` (for building fake_soc.so) |
| `zig` toolchain | [ziglang.org/download](https://ziglang.org/download/) (for cross-compiling Go binary) |

### Step 1 — Generate QNN Model Libraries

Model libraries are already deployed on the device. To regenerate from scratch:

```powershell
# On Windows with QAIRT SDK 2.39 (PowerShell):
cd C:\path\to\qairt\2.39.0.250926
.\bin\envsetup.ps1

# Split the BirdNET ONNX model at the mel spectrogram boundary:
python qnn-handoff/split_model.py  # produces birdnet_cnn.onnx

# Convert to QNN format (FP32 example):
qnn-onnx-converter `
    -i birdnet_cnn.onnx `
    --input_dim "model/MEL_SPEC1/Pow_1" 1,511,96 `
    --input_dim "model/MEL_SPEC2/Pow_1" 1,511,96 `
    -o birdnet_v24/fp32/birdnet_cnn_fp32

# Compile the model library for aarch64:
qnn-model-lib-generator `
    -m birdnet_v24/fp32/birdnet_cnn_fp32.cpp `
    -b birdnet_v24/fp32/birdnet_cnn_fp32.bin `
    -o birdnet_v24/fp32/ `
    --lib_targets aarch64-ubuntu-gcc9.4

# Output: birdnet_v24/fp32/aarch64-ubuntu-gcc9.4/libbirdnet_cnn.so
# Rename for deployment:
cp birdnet_v24/fp32/.../libbirdnet_cnn.so libbirdnet_cnn_fp32.so
```

### Step 2 — Deploy to Device

```bash
# Copy model and backend libraries:
scp libbirdnet_cnn_fp32.so arduino@192.168.86.27:/opt/qnn/
scp $QAIRT/lib/aarch64-oe-linux-gcc11.2/libQnnCpu.so arduino@192.168.86.27:/opt/qnn/
scp $QAIRT/lib/aarch64-oe-linux-gcc11.2/libQnnSystem.so arduino@192.168.86.27:/opt/qnn/

# Compile and deploy fake_soc.so:
aarch64-linux-gnu-gcc -shared -fPIC -o fake_soc.so \
    internal/inference/qnn/fake_soc.c -ldl
scp fake_soc.so arduino@192.168.86.27:/opt/qnn/
```

### Step 3 — Build birdnet-go with QNN Tag

```bash
# From the birdnet-go-fork root:

# Symlink the QAIRT headers (Windows PowerShell, run as Admin):
New-Item -ItemType Junction `
    -Path "vendor\qairt\include" `
    -Target "C:\path\to\qairt\2.39.0.250926\include"

# Cross-compile for aarch64 (using Zig as CC):
GOOS=linux GOARCH=arm64 CGO_ENABLED=1 \
CC="zig cc -target aarch64-linux-gnu" \
CXX="zig c++ -target aarch64-linux-gnu" \
CGO_CFLAGS="-I${SRCDIR}/vendor/qairt/include -I${SRCDIR}/vendor/qairt/include/QNN" \
CGO_LDFLAGS="-L${TFLITE_LIB_DIR} -ltensorflowlite_c" \
go build -mod=mod -tags onnx,qnn -o birdnet-go-linux-arm64 .

# Deploy:
scp birdnet-go-linux-arm64 arduino@192.168.86.27:/tmp/birdnet-go-new
ssh arduino@192.168.86.27 "echo PASSWORD | sudo -S sh -c '
    systemctl stop birdnet-go &&
    mv /tmp/birdnet-go-new /opt/birdnet-go/birdnet-go &&
    chmod +x /opt/birdnet-go/birdnet-go &&
    systemctl start birdnet-go
'"
```

> **Build note:** The vendor directory may be stale. Use `-mod=mod` to bypass
> vendor consistency checks.

### Step 4 — Configure birdnet-go

Edit `/etc/birdnet-go/config.yaml` directly (the web UI may reset these):

```yaml
birdnet:
  version: "2.4-cnn-fp32"    # maps to BirdNET_V2.4_CNN_FP32 registry entry
  qnnbackend: "cpu"           # libQnnCpu.so with fake_soc.so
  qnnlibdir: "/opt/qnn"
  qnnmodellibdir: "/opt/qnn"
```

**Available version strings:**

| Config `version` | Registry ID | Model file | Notes |
|---|---|---|---|
| `2.4-cnn-fp32` | `BirdNET_V2.4_CNN_FP32` | `libbirdnet_cnn_fp32.so` | ✅ Working |
| `2.4-cnn-fp16` | `BirdNET_V2.4_CNN_FP16` | `libbirdnet_cnn_fp16.so` | Untested |
| `2.4-cnn-int8` | `BirdNET_V2.4_CNN_INT8` | `libbirdnet_cnn_int8.so` | Untested |
| `2.4-int8-cnn` | `BirdNET_V2.4_INT8_CNN` | `libbirdnet_int8_cnn_net.so` | Untested |

### Step 5 — Configure systemd for fake_soc

```bash
# Add to /etc/systemd/system/birdnet-go.service under [Service]:
Environment=LD_PRELOAD=/opt/qnn/fake_soc.so
Environment=FAKE_SOC_ID=356

sudo systemctl daemon-reload && sudo systemctl restart birdnet-go
```

### Step 6 (Optional) — Generate a Context Binary on the Device

A context binary eliminates JIT compilation on every restart:

```bash
# On the device:
LD_LIBRARY_PATH=/opt/qnn \
LD_PRELOAD=/opt/qnn/fake_soc.so \
FAKE_SOC_ID=356 \
/opt/qnn/qnn-context-binary-generator \
    --model   /opt/qnn/libbirdnet_cnn_fp32.so \
    --backend /opt/qnn/libQnnCpu.so \
    --binary_file /opt/qnn/birdnet_cnn_fp32_cpu_context.bin
```

birdnet-go automatically prefers `<model>_<backend>_context.bin` over the
model library. With the above path it will find
`/opt/qnn/birdnet_cnn_fp32_cpu_context.bin` automatically.

---

## Verification

### Confirm QNN libraries are loaded

```bash
sudo cat /proc/$(pgrep birdnet-go)/maps | grep -E "qnn|birdnet_cnn"
# Expected output includes:
#   /opt/qnn/libbirdnet_cnn_fp32.so   (model, ~52MB mapped)
#   /opt/qnn/libQnnCpu.so             (backend)
#   /opt/qnn/libQnnSystem.so          (system library)
#   /opt/qnn/fake_soc.so              (LD_PRELOAD)
```

### Check inference is running

```bash
# CPU usage should be elevated (~120% on aarch64 × 4 cores):
top -bn1 -p $(pgrep birdnet-go)

# Detections appear in the database (SQLite):
python3 -c "
import sqlite3, datetime
conn = sqlite3.connect('/var/lib/birdnet-go/data/database/birdnet.db')
c = conn.cursor()
c.execute('SELECT COUNT(*) FROM detections')
print('Total detections:', c.fetchone()[0])
c.execute('SELECT detected_at, processing_time_ms FROM detections ORDER BY id DESC LIMIT 3')
for row in c.fetchall():
    dt = datetime.datetime.fromtimestamp(row[0], tz=datetime.timezone.utc)
    print(f'  {dt}  proc={row[1]}ms')
conn.close()
"
# Processing times ~1200–1400ms are consistent with QNN CPU inference
```

### Check journalctl for QNN startup

```bash
journalctl -u birdnet-go --since "5 minutes ago" | grep -E "fake_soc|QNN|qnn"
# Expected sequence:
#   [fake_soc] Loaded, will fake soc_id as: 356
#   🎯 INITIALIZING MODEL - QNN Supported: true, Backend: 'cpu'
#   🎯 ENTERING QNN PATH
#   [fake_soc] Intercepted open(/sys/devices/soc0/soc_id)...
# (no "🎯 QNN INIT FAILED" = success)
```

---

## GPU Backend — Adreno 702

### Current Blocker

Mesa Rusticl's OpenCL implementation for the FD702 (Adreno 702) does not
implement the `clGetDeviceImageInfoQCOM` extension that `libQnnGpu.so` requires.
The backend fails with:

```
clGetDeviceImageInfoQCOM not found in the OpenCL platform
```

**Required:** Qualcomm proprietary OpenCL stack (not Mesa Rusticl).

### Investigation Steps Taken

- `clinfo` confirms OpenCL 3.0 available on `FD702` via Mesa Rusticl
- `libQnnGpu.so` loads correctly (no dlopen error)
- `QnnBackend_Create` fails when calling `clGetDeviceImageInfoQCOM`
- Mesa Rusticl is open-source; the extension is Qualcomm-proprietary and
  would need to be added to Mesa

### Potential Future Paths

1. **Qualcomm proprietary OpenCL** — install `libOpenCL.so` from Qualcomm's
   Android/Yocto BSP if available for this board's kernel
2. **Mesa extension stub** — implement a no-op `clGetDeviceImageInfoQCOM` in
   Mesa Rusticl and rebuild (requires Mesa build from source)
3. **FP16 GPU model via context binary** — if proprietary OpenCL becomes
   available, the FP16 model (`libbirdnet_cnn_fp16.so`) would give ~2× speedup
   vs CPU FP32

---

## HTP (Hexagon DSP) — Current Blocker

The Hexagon 686 DSP is running (ADSP remoteproc is active, `fastrpc` kernel
module is loaded) but there is currently **no `/dev/fastrpc*` character device**
node on the device. The QNN HTP backend uses FastRPC to communicate with the
DSP, so without the device node it will fail to initialize.

**Possible fixes:**

1. **Kernel configuration** — check `CONFIG_QCOM_FASTRPC`:
   ```bash
   zcat /proc/config.gz | grep FASTRPC
   ```
2. **Manual `mknod`** (temporary test):
   ```bash
   sudo mknod /dev/adsprpc-smd c <major> <minor>
   # Find major/minor from: cat /proc/devices | grep fastrpc
   ```

---

## Remaining Work

| Priority | Task | Notes |
|---|---|---|
| Medium | **GPU backend** | Needs proprietary OpenCL or Mesa extension |
| Medium | **Web UI model selector** | Add Svelte 5 dropdown for `2.4-cnn-fp32/fp16/int8` |
| Low | **Context binary generation** | Pre-compile on-device for faster startup |
| Low | **INT8 CNN inference validation** | Test `libbirdnet_cnn_int8.so` accuracy vs FP32 |
| Low | **HTP backend** | Requires fastrpc device node kernel fix |
| Low | **"QNN model initialized" log routing** | Structured log not appearing in application.log; stdout 🎯 markers confirm success |

---

## Troubleshooting

### `backendCreate failed (rc=1006)`

SoC ID not in libQnnCpu.so whitelist. Ensure `fake_soc.so` is loaded:
```bash
journalctl -u birdnet-go | grep fake_soc
# If missing, check LD_PRELOAD in systemd service file
sudo systemctl cat birdnet-go | grep LD_PRELOAD
```

### `🎯 QNN INIT FAILED: ...model library not found`

Check model file naming. The model name comes from `ConfigAliases[0]` in the
model registry. For `version: 2.4-cnn-fp32`, the model name is `birdnet_cnn_fp32`,
so birdnet-go looks for `/opt/qnn/libbirdnet_cnn_fp32.so`:

```bash
ls -la /opt/qnn/lib*.so
```

### SIGSEGV in `composeGraphs`

Caused by loading the backend with `RTLD_LOCAL`. Verify the build uses `RTLD_GLOBAL`:
```bash
grep RTLD_GLOBAL internal/inference/qnn/qnn_backend.c
```

### Inference timing shows 0.0ms in web UI

This is a known display issue — the web UI's inference timing metric was designed
for TFLite and doesn't track QNN inference time. Verify QNN is running via:
1. Process maps: `sudo cat /proc/$(pgrep birdnet-go)/maps | grep qnn`
2. CPU usage: `top -p $(pgrep birdnet-go)` (expect ~100-120% on 4-core ARM)
3. Database: detections with `processing_time_ms` ~1200–1400ms

### Service crashes immediately after start (no fake_soc message)

Check if `fake_soc.so` is readable and correctly owned:
```bash
ls -la /opt/qnn/fake_soc.so
# Should be executable, readable by the birdnet-go service user
```

---

## File Reference

| File | Purpose |
|---|---|
| `internal/inference/qnn/qnn_backend.c` | QNN C API wrapper (`dlopen`-based); key fix: `RTLD_GLOBAL` |
| `internal/inference/qnn/qnn_backend.h` | C header for the QNN shim |
| `internal/inference/qnn/classifier.go` | Go QNN classifier (`-tags qnn`) |
| `internal/inference/qnn/melspec.go` | Mel spectrogram preprocessing (replaces ONNX DFT nodes) |
| `internal/inference/qnn/data/mel_fb_spec1.bin` | Pre-computed mel filterbank weights (spec1) |
| `internal/inference/qnn/data/mel_fb_spec2.bin` | Pre-computed mel filterbank weights (spec2) |
| `internal/inference/qnn/fake_soc.c` | SoC ID bypass library source |
| `internal/classifier/model_qnn.go` | BirdNET QNN initializer (`-tags qnn`) |
| `internal/classifier/model_qnn_stub.go` | No-op stub for standard builds |
| `internal/classifier/model_registry.go` | Model registry (CNN variants have `NumSpecies=6522`) |
| `vendor/qairt/include` | Symlink to QAIRT SDK `include/` (not committed; required at build time) |
