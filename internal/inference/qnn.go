//go:build qnn

package inference

import (
	qnnpkg "github.com/tphakala/birdnet-go/internal/inference/qnn"
)

// QNNClassifierOptions configures a QNN-accelerated classifier.
type QNNClassifierOptions struct {
	// Backend selects the QNN hardware backend: "gpu" or "htp".
	Backend string
	// LibDir is the directory containing the QNN backend shared libraries
	// (libQnnGpu.so or libQnnHtp.so, plus libQnnSystem.so).
	LibDir string
	// ModelLibDir is the directory containing the compiled model artifacts
	// (libmodel_net.so or a pre-compiled context binary).
	ModelLibDir string
	// ModelName is the base name used to locate model artifacts,
	// e.g. "birdnet_int8_cnn".
	ModelName string
	// Labels is the species label list.  Required.
	Labels []string
}

// NewQNNClassifier creates a Classifier backed by the Qualcomm QNN C API.
// Returns an error if the QNN libraries or model artifacts are not found.
func NewQNNClassifier(opts QNNClassifierOptions) (Classifier, error) {
	qnnOpts, err := qnnpkg.OptionsFromConfig(
		opts.Backend, opts.LibDir, opts.ModelLibDir, opts.ModelName)
	if err != nil {
		return nil, err
	}
	qnnOpts.NumSpecies = len(opts.Labels)

	c, err := qnnpkg.NewClassifier(qnnOpts)
	if err != nil {
		return nil, err
	}
	return c, nil
}
