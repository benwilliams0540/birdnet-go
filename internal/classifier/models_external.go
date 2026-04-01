//go:build noembed

package classifier

// This file is included when building with -tags noembed
// It provides empty model data variables for a non-monolithic build
// where models must be loaded from external files.

// modelData is nil when models are not embedded
var modelData []byte

// metaModelDataV1 is nil when models are not embedded
var metaModelDataV1 []byte

// metaModelDataV2 is nil when models are not embedded
var metaModelDataV2 []byte

// int8ModelData is nil when models are not embedded
var int8ModelData []byte

// hasEmbeddedModels indicates whether models are embedded in the binary
// This is a var instead of const to allow test overrides
var hasEmbeddedModels = false

// GetEmbeddedONNXData always returns false for noembed builds.
func GetEmbeddedONNXData(_ string) ([]byte, bool) {
	return nil, false
}
