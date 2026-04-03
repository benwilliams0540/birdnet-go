//go:build ncnn

package inference

import (
	ncnnpkg "github.com/tphakala/birdnet-go/internal/inference/ncnn"
)

// NCNNClassifierOptions configures an NCNN-accelerated classifier.
type NCNNClassifierOptions struct {
	// ModelDir is the directory containing birdnet_cnn.param and birdnet_cnn.bin.
	// These are produced by running onnx2ncnn on the BirdNET CNN sub-model.
	ModelDir string
	// Threads is the number of CPU threads. 0 = NCNN default (all cores).
	Threads int
	// UseVulkan enables Vulkan GPU acceleration.
	UseVulkan bool
}

// NewNCNNClassifier creates a Classifier backed by the NCNN inference engine.
// Returns an error if the model files are not found or cannot be loaded.
func NewNCNNClassifier(opts NCNNClassifierOptions) (Classifier, error) {
	return ncnnpkg.New(ncnnpkg.Options{
		ModelDir:  opts.ModelDir,
		Threads:   opts.Threads,
		UseVulkan: opts.UseVulkan,
	})
}
