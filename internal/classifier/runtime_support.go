package classifier

import (
	"strings"

	"github.com/tphakala/birdnet-go/internal/conf"
	"github.com/tphakala/birdnet-go/internal/errors"
)

// CompiledBackendSupport reports which inference backends are built into the binary.
type CompiledBackendSupport struct {
	TFLite bool `json:"tflite"`
	ONNX   bool `json:"onnx"`
	NCNN   bool `json:"ncnn"`
	QNN    bool `json:"qnn"`
}

// GetCompiledBackendSupport returns the backends compiled into the current binary.
func GetCompiledBackendSupport() CompiledBackendSupport {
	return CompiledBackendSupport{
		TFLite: isTFLiteSupported(),
		ONNX:   isONNXSupported(),
		NCNN:   isNCNNSupported(),
		QNN:    isQNNSupported(),
	}
}

// IsBackendCompiled reports whether a backend is built into the current binary.
func IsBackendCompiled(backend string) bool {
	switch strings.ToLower(strings.TrimSpace(backend)) {
	case "", "auto":
		return true
	case "tflite":
		return isTFLiteSupported()
	case "onnx":
		return isONNXSupported()
	case "ncnn":
		return isNCNNSupported()
	case "qnn":
		return isQNNSupported()
	default:
		return false
	}
}

// ResolveBirdNETModelInfoFromConfig resolves the effective model identity from BirdNET settings
// without constructing an interpreter.
func ResolveBirdNETModelInfoFromConfig(cfg conf.BirdNETConfig) (ModelInfo, error) {
	switch {
	case cfg.Version != "":
		info, ok := ResolveBirdNETVersion(cfg.Version)
		if !ok {
			return ModelInfo{}, errors.Newf("unknown BirdNET version: %s", cfg.Version).
				Component("classifier.runtime_support").
				Category(errors.CategoryModelInit).
				Context("version", cfg.Version).
				Build()
		}
		if cfg.ModelPath != "" {
			info.CustomPath = cfg.ModelPath
		}
		return info, nil
	case cfg.ModelPath != "":
		return DetermineModelInfo(cfg.ModelPath)
	default:
		info, ok := ModelRegistry[DefaultModelVersion]
		if !ok {
			return ModelInfo{}, errors.Newf("default model version %s not found in registry", DefaultModelVersion).
				Component("classifier.runtime_support").
				Category(errors.CategoryModelInit).
				Build()
		}
		return info, nil
	}
}

// DefaultBirdNETModelAvailable reports whether the default BirdNET V2.4 TFLite
// model can be resolved by this build, either from embedded assets or the
// standard filesystem discovery paths used by noembed builds.
func DefaultBirdNETModelAvailable() bool {
	if hasEmbeddedModels && len(modelData) > 0 {
		return true
	}

	_, _, err := tryLoadModelFromStandardPaths(DefaultBirdNETModelName, "BirdNET")
	return err == nil
}
