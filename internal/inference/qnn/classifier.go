//go:build qnn

// Package qnn provides a BirdNET-Go Classifier backed by the Qualcomm QNN C
// API.  It supports two loading modes:
//
//  1. Context binary  — fastest start-up; produced by qnn-context-binary-generator
//     on the target device.  Device/driver-version specific.
//
//  2. Model library   — portable shared library produced by
//     qnn-onnx-converter + qnn-model-lib-generator on any Linux/Windows host
//     with the QAIRT SDK.  The QNN backend JIT-compiles the graph at first use.
//
// # Hardware
//
// Tested targets for the QCM2290 (Arduino Portenta X8 / Imola):
//
//   - GPU backend:   libQnnGpu.so   — Adreno 702 via OpenCL
//                    Requires libOpenCL.so.  Mesa Rusticl (FD702) may work;
//                    Qualcomm proprietary OpenCL gives best performance.
//   - HTP backend:   libQnnHtp.so   — Hexagon 686 (v68) DSP
//                    Requires /dev/fastrpc-adsp device node and
//                    libQnnHtpV68Stub.so + libQnnHtpPrepare.so.
//
// # Quick start
//
//  1. Generate the QNN model library on a Windows or Linux x86-64 machine:
//     (requires QAIRT SDK ≥ 2.39 and Python 3.10)
//
//     # Windows PowerShell (QAIRT SDK 2.39):
//     qairt-converter -d ONNX -i birdnet_int8_cnn.onnx `
//         --input_dim input_1 1,144000 `
//         -o birdnet_int8_cnn.dlc
//
//     # Linux x86-64 (alternative):
//     source qairt/2.39.0.250926/bin/envsetup.sh
//     qnn-onnx-converter \
//         -i birdnet_int8_cnn.onnx \
//         --input_dim input_1 1,144000 \
//         -o /tmp/birdnet_qnn/birdnet_int8_cnn.cpp
//     qnn-model-lib-generator \
//         -m /tmp/birdnet_qnn/birdnet_int8_cnn.cpp \
//         -b /tmp/birdnet_qnn/birdnet_int8_cnn.bin \
//         -o /tmp/birdnet_qnn/libs/ \
//         --lib_targets aarch64-oe-linux-gcc11.2
//
//  2. Deploy to the device:
//     scp /tmp/birdnet_qnn/libs/aarch64-oe-linux-gcc11.2/libmodel_net.so \
//         device:/opt/qnn/
//     scp qairt/2.39.0.250926/lib/aarch64-oe-linux-gcc11.2/libQnnGpu.so \
//         qairt/2.39.0.250926/lib/aarch64-oe-linux-gcc11.2/libQnnSystem.so \
//         device:/opt/qnn/
//
//  3. (Optional) Pre-compile context binary ON the device for faster start-up:
//     /opt/qnn/qnn-context-binary-generator \
//         --model /opt/qnn/libmodel_net.so \
//         --backend /opt/qnn/libQnnGpu.so \
//         --binary_file /opt/qnn/birdnet_gpu_context.bin
//
//  4. Configure birdnet-go (config.yaml):
//     birdnet:
//       version: "2.4-int8-cnn"
//       qnnbackend: "gpu"                        # or "htp"
//       qnnlibdir: "/opt/qnn"
//       qnnmodellibdir: "/opt/qnn"
//
//  5. Build with QNN tag:
//     go build -tags onnx,qnn -o birdnet-go .
package qnn

// #cgo CFLAGS: -std=c11
// #cgo LDFLAGS: -ldl
// #cgo CFLAGS: -I${SRCDIR}/../../../vendor/qairt/include
//
// The QNN SDK headers are expected at vendor/qairt/include inside the repo
// root.  Symlink or copy them from the QAIRT SDK before building with the
// qnn tag:
//
//   ln -s /path/to/qairt/2.39.0.250926/include vendor/qairt/include
//
// #include "qnn_backend.h"
// #include <stdlib.h>
import "C"
import (
	"fmt"
	"os"
	"path/filepath"
	"unsafe"
)

const errBufSize = 512

// ---------------------------------------------------------------------------
// Classifier
// ---------------------------------------------------------------------------

// Classifier implements inference.Classifier using the Qualcomm QNN C API.
type Classifier struct {
	session    *C.qnn_session_t
	numSpecies int
}

// LoadMode selects how the QNN model is loaded.
type LoadMode int

const (
	// LoadModeModelLib composes the graph from a model library at runtime.
	// More portable; slower first-run due to JIT compilation.
	LoadModeModelLib LoadMode = iota

	// LoadModeContextBinary loads a pre-compiled context binary.
	// Fastest start-up but device/driver-version specific.
	LoadModeContextBinary
)

// Options configures a QNN Classifier.
type Options struct {
	// Backend is the path to the QNN backend shared library,
	// e.g. "/opt/qnn/libQnnGpu.so" or "/opt/qnn/libQnnHtp.so".
	Backend string

	// SystemLib is the path to libQnnSystem.so.
	SystemLib string

	// ModelLib is the path to the compiled model .so (LoadModeModelLib).
	ModelLib string

	// ContextBinary is the path to a pre-compiled context binary
	// (LoadModeContextBinary).
	ContextBinary string

	// NumSpecies is the number of output classes (required).
	NumSpecies int
}

// DetectLoadMode infers the LoadMode from o.
func (o Options) DetectLoadMode() (LoadMode, error) {
	if o.ContextBinary != "" {
		return LoadModeContextBinary, nil
	}
	if o.ModelLib != "" {
		return LoadModeModelLib, nil
	}
	return 0, fmt.Errorf("qnn: Options must set either ContextBinary or ModelLib")
}

// NewClassifier creates a Classifier from the given options.
func NewClassifier(opts Options) (*Classifier, error) {
	if opts.NumSpecies <= 0 {
		return nil, fmt.Errorf("qnn: NumSpecies must be > 0")
	}

	mode, err := opts.DetectLoadMode()
	if err != nil {
		return nil, err
	}

	if _, err := os.Stat(opts.Backend); err != nil {
		return nil, fmt.Errorf("qnn: backend library not found at %q: %w", opts.Backend, err)
	}
	if _, err := os.Stat(opts.SystemLib); err != nil {
		return nil, fmt.Errorf("qnn: system library not found at %q: %w", opts.SystemLib, err)
	}

	var errBuf [errBufSize]C.char

	switch mode {
	case LoadModeContextBinary:
		if _, err := os.Stat(opts.ContextBinary); err != nil {
			return nil, fmt.Errorf("qnn: context binary not found at %q: %w",
				opts.ContextBinary, err)
		}
		data, err := os.ReadFile(opts.ContextBinary)
		if err != nil {
			return nil, fmt.Errorf("qnn: reading context binary: %w", err)
		}
		cBackend := C.CString(opts.Backend)
		cSystem := C.CString(opts.SystemLib)
		defer C.free(unsafe.Pointer(cBackend))
		defer C.free(unsafe.Pointer(cSystem))

		sess := C.qnn_create_from_context_binary(
			cBackend, cSystem,
			unsafe.Pointer(&data[0]), C.size_t(len(data)),
			&errBuf[0], errBufSize,
		)
		if sess == nil {
			return nil, fmt.Errorf("qnn: create from context binary: %s",
				C.GoString(&errBuf[0]))
		}
		return &Classifier{session: sess, numSpecies: opts.NumSpecies}, nil

	case LoadModeModelLib:
		if _, err := os.Stat(opts.ModelLib); err != nil {
			return nil, fmt.Errorf("qnn: model library not found at %q: %w",
				opts.ModelLib, err)
		}
		cBackend := C.CString(opts.Backend)
		cSystem := C.CString(opts.SystemLib)
		cModel := C.CString(opts.ModelLib)
		defer C.free(unsafe.Pointer(cBackend))
		defer C.free(unsafe.Pointer(cSystem))
		defer C.free(unsafe.Pointer(cModel))

		sess := C.qnn_create_from_model_lib(
			cBackend, cSystem, cModel,
			&errBuf[0], errBufSize,
		)
		if sess == nil {
			return nil, fmt.Errorf("qnn: create from model lib: %s",
				C.GoString(&errBuf[0]))
		}
		return &Classifier{session: sess, numSpecies: opts.NumSpecies}, nil
	}

	return nil, fmt.Errorf("qnn: unsupported load mode %d", mode)
}

// ---------------------------------------------------------------------------
// OptionsFromConfig builds Options from birdnet-go config fields.
//
//	backend     — "gpu" or "htp"
//	libDir      — directory containing libQnnGpu.so (or libQnnHtp.so) and
//	              libQnnSystem.so
//	modelLibDir — directory containing the compiled model .so; if it also
//	              contains a .bin file for the same model name, context binary
//	              mode is used automatically.
//	modelName   — base name used to locate files, e.g. "birdnet_int8_cnn"
// ---------------------------------------------------------------------------

// OptionsFromConfig resolves Options from path components.
func OptionsFromConfig(backend, libDir, modelLibDir, modelName string) (Options, error) {
	if libDir == "" {
		return Options{}, fmt.Errorf("qnn: QNNLibDir must be set")
	}
	if modelLibDir == "" {
		return Options{}, fmt.Errorf("qnn: QNNModelLibDir must be set")
	}

	// Determine backend library name.
	var backendLib string
	switch backend {
	case "gpu":
		backendLib = "libQnnGpu.so"
	case "htp":
		backendLib = "libQnnHtp.so"
	default:
		return Options{}, fmt.Errorf("qnn: unknown backend %q (valid: gpu, htp)", backend)
	}

	opts := Options{
		Backend:    filepath.Join(libDir, backendLib),
		SystemLib:  filepath.Join(libDir, "libQnnSystem.so"),
		NumSpecies: 0, // caller must set
	}

	// Prefer context binary (faster) if present.
	contextBin := filepath.Join(modelLibDir, modelName+"_"+backend+"_context.bin")
	if _, err := os.Stat(contextBin); err == nil {
		opts.ContextBinary = contextBin
		return opts, nil
	}

	// Fall back to model library (online compilation).
	modelLib := filepath.Join(modelLibDir, "lib"+modelName+"_net.so")
	if _, err := os.Stat(modelLib); err == nil {
		opts.ModelLib = modelLib
		return opts, nil
	}

	// Also accept the default qnn-model-lib-generator output name.
	modelLibDefault := filepath.Join(modelLibDir, "libmodel_net.so")
	if _, err := os.Stat(modelLibDefault); err == nil {
		opts.ModelLib = modelLibDefault
		return opts, nil
	}

	return Options{}, fmt.Errorf(
		"qnn: no model library or context binary found in %q for model %q "+
			"(looked for %s, %s, %s)",
		modelLibDir, modelName,
		filepath.Base(contextBin),
		filepath.Base(modelLib),
		filepath.Base(modelLibDefault))
}

// ---------------------------------------------------------------------------
// inference.Classifier interface
// ---------------------------------------------------------------------------

// Predict runs one inference pass on the provided audio samples.
// samples must contain exactly the number of elements the model expects
// (48 000 Hz × 3 s = 144 000 float32 values for BirdNET V2.4).
func (c *Classifier) Predict(samples []float32) ([]float32, error) {
	if c.session == nil {
		return nil, fmt.Errorf("qnn: classifier is closed")
	}

	inputCount := C.size_t(len(samples))
	outputCount := C.size_t(c.numSpecies)
	output := make([]float32, c.numSpecies)

	var errBuf [errBufSize]C.char

	rc := C.qnn_run_inference(
		c.session,
		(*C.float)(unsafe.Pointer(&samples[0])),
		inputCount,
		(*C.float)(unsafe.Pointer(&output[0])),
		outputCount,
		&errBuf[0], errBufSize,
	)
	if rc != 0 {
		return nil, fmt.Errorf("qnn inference: %s", C.GoString(&errBuf[0]))
	}
	return output, nil
}

// NumSpecies returns the number of species in the model output.
func (c *Classifier) NumSpecies() int { return c.numSpecies }

// Close releases all QNN resources and unloads the backend libraries.
func (c *Classifier) Close() {
	if c.session != nil {
		C.qnn_destroy_session(c.session)
		c.session = nil
	}
}
