//go:build !ncnn

package classifier

import "fmt"

// initializeNCNNModel is a no-op stub for non-ncnn builds.
func (bn *BirdNET) initializeNCNNModel() error {
	return fmt.Errorf("NCNN acceleration is not compiled in; rebuild with -tags ncnn")
}

// isNCNNSupported returns false for non-ncnn builds.
func isNCNNSupported() bool { return false }
