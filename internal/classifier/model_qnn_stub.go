//go:build !qnn

package classifier

import "fmt"

// initializeQNNModel is a no-op stub for non-qnn builds.
func (bn *BirdNET) initializeQNNModel() error {
	return fmt.Errorf(
		"QNN acceleration is not compiled in; rebuild with -tags qnn")
}

// isQNNSupported returns false for non-qnn builds.
func isQNNSupported() bool { return false }
