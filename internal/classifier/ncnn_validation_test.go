package classifier

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestInspectNCNNModelDirRequiresValidationMarker(t *testing.T) {
	t.Parallel()

	modelDir := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(modelDir, "birdnet_cnn.param"), []byte("param"), 0o600))
	require.NoError(t, os.WriteFile(filepath.Join(modelDir, "birdnet_cnn.bin"), []byte("bin"), 0o600))

	status := InspectNCNNModelDir(modelDir)

	assert.True(t, status.Found)
	assert.False(t, status.Validated)
	assert.False(t, NCNNModelDirReady(modelDir))
	assert.Equal(t, NCNNValidationMarkerName, status.ValidationMarker)
}

func TestInspectNCNNModelDirAcceptsValidatedArtifacts(t *testing.T) {
	t.Parallel()

	modelDir := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(modelDir, "birdnet_cnn.param"), []byte("param"), 0o600))
	require.NoError(t, os.WriteFile(filepath.Join(modelDir, "birdnet_cnn.bin"), []byte("bin"), 0o600))
	require.NoError(t, os.WriteFile(filepath.Join(modelDir, NCNNValidationMarkerName), []byte("validated"), 0o600))

	status := InspectNCNNModelDir(modelDir)

	assert.True(t, status.Found)
	assert.True(t, status.Validated)
	assert.True(t, NCNNModelDirReady(modelDir))
	assert.Equal(t, "birdnet_cnn.param", status.ParamFile)
	assert.Equal(t, "birdnet_cnn.bin", status.BinFile)
}

func TestInspectNCNNModelDirAcceptsValidatedSplitArtifacts(t *testing.T) {
	t.Parallel()

	modelDir := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(modelDir, "birdnet_cnn_only.param"), []byte("param"), 0o600))
	require.NoError(t, os.WriteFile(filepath.Join(modelDir, "birdnet_cnn_only.bin"), []byte("bin"), 0o600))
	require.NoError(t, os.WriteFile(filepath.Join(modelDir, NCNNValidationMarkerName), []byte("validated"), 0o600))

	status := InspectNCNNModelDir(modelDir)

	assert.True(t, status.Found)
	assert.True(t, status.Validated)
	assert.True(t, NCNNModelDirReady(modelDir))
	assert.Equal(t, "birdnet_cnn_only.param", status.ParamFile)
	assert.Equal(t, "birdnet_cnn_only.bin", status.BinFile)
}

func TestInspectNCNNModelDirAcceptsValidatedPNNXArtifacts(t *testing.T) {
	t.Parallel()

	modelDir := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(modelDir, "birdnet.pnnx.param"), []byte("param"), 0o600))
	require.NoError(t, os.WriteFile(filepath.Join(modelDir, "birdnet.pnnx.bin"), []byte("bin"), 0o600))
	require.NoError(t, os.WriteFile(filepath.Join(modelDir, NCNNValidationMarkerName), []byte("validated"), 0o600))

	status := InspectNCNNModelDir(modelDir)

	assert.True(t, status.Found)
	assert.True(t, status.Validated)
	assert.True(t, NCNNModelDirReady(modelDir))
	assert.Equal(t, "birdnet.pnnx.param", status.ParamFile)
	assert.Equal(t, "birdnet.pnnx.bin", status.BinFile)
}
