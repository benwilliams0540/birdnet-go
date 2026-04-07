//go:build !notflite

package classifier

import (
	"runtime"
	"time"

	"github.com/tphakala/birdnet-go/internal/cpuspec"
	"github.com/tphakala/birdnet-go/internal/errors"
	"github.com/tphakala/birdnet-go/internal/inference"
	"github.com/tphakala/birdnet-go/internal/logger"
)

// initializeTFLiteModel loads and initializes a TFLite model as the classifier backend.
func (bn *BirdNET) initializeTFLiteModel() error {
	start := time.Now()

	modelData, err := bn.loadModel()
	if err != nil {
		return errors.New(err).
			Category(errors.CategoryModelLoad).
			ModelContext(bn.Settings.BirdNET.ModelPath, bn.ModelInfo.ID).
			Timing("model-load", time.Since(start)).
			Build()
	}

	log := GetLogger()
	classifier, threads, err := inference.NewTFLiteClassifier(modelData, inference.TFLiteClassifierOptions{
		Threads:    bn.Settings.BirdNET.Threads,
		UseXNNPACK: bn.Settings.BirdNET.UseXNNPACK,
		ErrorFunc: func(msg string) {
			log.Error("TFLite error", logger.String("message", msg))
		},
		WarnFunc: func(msg string) {
			log.Warn(msg, logger.String("tflite_download", "https://github.com/tphakala/tflite_c/releases/tag/v2.17.1"))
		},
	})
	if err != nil {
		return errors.New(err).
			Category(errors.CategoryModelInit).
			ModelContext(bn.Settings.BirdNET.ModelPath, bn.ModelInfo.ID).
			Context("model_size_mb", len(modelData)/1024/1024).
			Context("use_xnnpack", bn.Settings.BirdNET.UseXNNPACK).
			Timing("model-init", time.Since(start)).
			Build()
	}

	bn.classifier = classifier

	// Update the human-readable model version string for display when a custom
	// model path is provided.
	if bn.Settings.BirdNET.ModelPath != "" {
		bn.modelVersion = bn.Settings.BirdNET.ModelPath
	}

	// Log model initialization details
	if bn.Settings.BirdNET.Threads == 0 {
		spec := cpuspec.GetCPUSpec()
		if spec.PerformanceCores > 0 {
			log.Info("BirdNET model initialized",
				logger.String("model", bn.modelVersion),
				logger.Int("threads", threads),
				logger.Int("performance_cores", spec.PerformanceCores),
				logger.Int("total_cpus", runtime.NumCPU()))
		} else {
			log.Info("BirdNET model initialized",
				logger.String("model", bn.modelVersion),
				logger.Int("threads", threads),
				logger.Int("total_cpus", runtime.NumCPU()))
		}
	} else {
		log.Info("BirdNET model initialized",
			logger.String("model", bn.modelVersion),
			logger.Int("threads", threads),
			logger.Int("total_cpus", runtime.NumCPU()),
			logger.Bool("threads_configured", true))
	}
	return nil
}

// initializeTFLiteMetaModel loads and initializes a TFLite range filter model.
func (bn *BirdNET) initializeTFLiteMetaModel() error {
	start := time.Now()

	metaModelData, err := bn.getMetaModelData()
	if err != nil {
		return err
	}

	rangeFilter, err := inference.NewTFLiteRangeFilter(metaModelData, inference.TFLiteRangeFilterOptions{
		ErrorFunc: func(msg string) {
			GetLogger().Error("TFLite meta model error", logger.String("message", msg))
		},
	})
	if err != nil {
		return errors.New(err).
			Category(errors.CategoryModelInit).
			Context("model_type", "range_filter").
			Context("range_filter_model", bn.Settings.BirdNET.RangeFilter.Model).
			Timing("meta-model-init", time.Since(start)).
			Build()
	}

	bn.rangeFilter = rangeFilter
	return nil
}

// isTFLiteSupported returns true when the binary is built with TFLite support.
func isTFLiteSupported() bool { return true }
