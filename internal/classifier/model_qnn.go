//go:build qnn

package classifier

import (
	"time"

	"github.com/tphakala/birdnet-go/internal/errors"
	"github.com/tphakala/birdnet-go/internal/inference"
	"github.com/tphakala/birdnet-go/internal/logger"
)

// initializeQNNModel loads the model via the Qualcomm QNN C API.
//
// The backend (GPU or HTP), library directory, model library directory, and
// model name are taken from BirdNETConfig.QNN* fields.  If the QNN backend
// fails to load (e.g. libraries not deployed) the error is surfaced to the
// caller so it can decide whether to fall back to ONNX CPU inference.
func (bn *BirdNET) initializeQNNModel() error {
	start := time.Now()
	log := GetLogger()

	cfg := bn.Settings.BirdNET

	if cfg.QNNBackend == "" {
		return errors.Newf("qnn: QNNBackend is empty; set to 'gpu' or 'htp'").
			Category(errors.CategoryModelInit).
			ModelContext("", bn.ModelInfo.ID).
			Build()
	}

	opts := inference.QNNClassifierOptions{
		Backend:     cfg.QNNBackend,
		LibDir:      cfg.QNNLibDir,
		ModelLibDir: cfg.QNNModelLibDir,
		// Use the model's config alias as the base name for locating artifacts,
		// e.g. "birdnet_int8_cnn" → libmodel_net.so or birdnet_int8_cnn_gpu_context.bin
		ModelName: modelNameForQNN(bn.ModelInfo),
		Labels:    cfg.Labels,
	}

	classifier, err := inference.NewQNNClassifier(opts)
	if err != nil {
		return errors.New(err).
			Category(errors.CategoryModelInit).
			Context("qnn_backend", cfg.QNNBackend).
			Context("qnn_lib_dir", cfg.QNNLibDir).
			Context("qnn_model_lib_dir", cfg.QNNModelLibDir).
			ModelContext("", bn.ModelInfo.ID).
			Timing("qnn-model-init", time.Since(start)).
			Build()
	}

	bn.classifier = classifier

	log.Info("QNN model initialized",
		logger.String("model", bn.ModelInfo.ID),
		logger.String("backend", cfg.QNNBackend),
		logger.String("lib_dir", cfg.QNNLibDir),
		logger.Int("species", classifier.NumSpecies()),
		logger.Duration("init_time", time.Since(start)))

	return nil
}

// modelNameForQNN returns the base name used to locate QNN model artifacts.
// It picks the first ConfigAlias (which is the canonical snake_case model name,
// e.g. "birdnet_int8_cnn") or falls back to the lower-cased registry ID.
func modelNameForQNN(info ModelInfo) string {
	if len(info.ConfigAliases) > 0 {
		return info.ConfigAliases[0]
	}
	return info.ID
}

// isQNNSupported returns true when the binary is built with QNN support.
func isQNNSupported() bool { return true }
