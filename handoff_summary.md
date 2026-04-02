# Handoff: BirdNET QNN Acceleration for Arduino Uno Q

This document summarizes the progress, current state, and next steps for enabling Qualcomm QNN (Neural Network) acceleration for BirdNET-Go on the Arduino Uno Q (QCM2290 SoC).

## 🎯 Current Status
- **Go Application**: Fully refactored to support QNN via CGO. Compiles and runs on the device.
- **Model Library**: Previously generated libraries were corrupted (Windows SDK bug). We have successfully generated a valid 50MB model library for the `birdnet_int8_cnn` model using the Linux QAIRT SDK in WSL.
- **GPU Backend Blocker**: The Adreno GPU backend (`libQnnGpu.so`) fails to initialize on the device because the current OpenCL provider (**Mesa Rusticl**) is missing the Qualcomm-proprietary symbol `clGetDeviceImageInfoQCOM`.
- **CPU Backend Blocker**: The CPU backend is correctly mapped but fails with `rc=1006` (backendCreate failed), possibly due to SDK version mismatches or internal op support issues.
- **Converter Blocker**: The `birdnet_int8_cnn.onnx` model uses `onnx_dft` (Discrete Fourier Transform), which the QAIRT 2.39 converter does not have a translation for.

## ✅ Accomplishments
1.  **CGO/C API Fixes**:
    - Cleaned up `internal/inference/qnn/classifier.go` and `qnn_backend.c`.
    - Resolved major API changes between QNN 1.x and 2.x (e.g., replacement of `graphGetTensors` with System Context metadata extraction).
2.  **Cross-Compilation**:
    - Established a working Zig-based cross-compilation pipeline for `aarch64-linux` with `-tags onnx,qnn`.
3.  **Deployment & Diagnostics**:
    - Built automated deployment and log-fetching scripts (`deploy_sequence2.sh`, `gen_context.sh`).
    - Confirmed via `strings` and `nm` that the Linux-compiled model library correctly embeds weight blobs.
4.  **Backend Flexibility**:
    - Added a `"cpu"` backend option to the Go application for testing and fallback, supplementing the existing `"gpu"` and `"htp"` options.
5.  **Model Inspection**:
    - Analyzed new user-provided models. `birdnet.onnx` (54MB) is a single-input float32 model that may be more compatible than the 2-input INT8 version.

## 🚧 Challenges & Blockers
### 1. GPU OpenCL Conflict
The QNN GPU backend requires specific Qualcomm extensions.
- **Status**: Mesa Rusticl is installed on the device but lacks proprietary extensions.
- **Path Forward**: Investigate if a Qualcomm-proprietary `libOpenCL.so` can be installed on the Debian-based system, OR determine if there is a QNN GPU library version that is more compatible with standard OpenCL.

### 2. Supported Operators (`onnx_dft`)
The `birdnet_int8_cnn.onnx` uses a DFT op which is not supported by the QNN converter (v2.39).
- **Status**: Conversion fails with `KeyError: 'No translation registered for op type onnx_dft.'`.
- **Path Forward**: Use the new `birdnet.onnx` (float32) and see if it avoids this op, or use a custom op package (complex).

### 3. Converter Environment
Setting up the QAIRT Python environment (especially the `qti` and `onnx` modules) has been fragile between Windows and WSL.
- **Status**: Windows venv is partially working but has path/collision issues with QAIRT's internal numpy.

## 📋 Next Steps
1.  **Convert New Model**: Attempt to convert the provided `birdnet.onnx` (float32) to QNN format using the Windows `qairt-converter` or the WSL script.
2.  **Fix CPU Backend**: Figure out why the CPU backend fails to create (`rc=1006`). This is a prerequisite to knowing if the model library itself is valid.
3.  **Generate Context Binary**: Once a model library can be loaded (even on CPU), run `qnn-context-binary-generator` on the device to create the `.bin` context for fast loading.
4.  **Check HTP (Hexagon DSP)**: The device has a DSP. If GPU is blocked by OpenCL, the HTP backend might be the ultimate "win" for performance.

## 📁 Key Files
- `internal/inference/qnn/qnn_backend.c`: The C bridge.
- `internal/inference/qnn/classifier.go`: The Go/CGO layer.
- `docs/gpu-acceleration.md`: Technical guide for the process.
- `deploy_sequence2.sh`: The main deployment script.
- `convert_birdnet.ps1`: The Windows-based conversion attempt script.
