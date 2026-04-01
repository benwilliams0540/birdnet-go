//go:build !qnn

package inference

import "fmt"

// QNNClassifierOptions is a placeholder for builds without the qnn tag.
type QNNClassifierOptions struct {
	Backend     string
	LibDir      string
	ModelLibDir string
	ModelName   string
	Labels      []string
}

// NewQNNClassifier always returns an error on non-qnn builds.
func NewQNNClassifier(_ QNNClassifierOptions) (Classifier, error) {
	return nil, fmt.Errorf(
		"QNN acceleration is not compiled in; rebuild with -tags qnn")
}
