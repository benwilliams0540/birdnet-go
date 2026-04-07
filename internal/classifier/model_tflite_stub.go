//go:build notflite

package classifier

import (
	"github.com/tphakala/birdnet-go/internal/errors"
)

// initializeTFLiteModel is a stub for when TFLite is not supported.
func (bn *BirdNET) initializeTFLiteModel() error {
	return errors.Newf("TFLite classifier backend not available in this build").
		Category(errors.CategoryModelInit).
		Build()
}

// initializeTFLiteMetaModel is a stub for when TFLite is not supported.
func (bn *BirdNET) initializeTFLiteMetaModel() error {
	return errors.Newf("TFLite range filter backend not available in this build").
		Category(errors.CategoryModelInit).
		Build()
}

// isTFLiteSupported returns false when the binary is built without TFLite support.
func isTFLiteSupported() bool { return false }
