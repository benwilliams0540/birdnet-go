package ncnn

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWrapSplitModelParamInsertsBirdNETFrontendLayer(t *testing.T) {
	t.Parallel()

	input := strings.Join([]string{
		"7767517",
		"212 238",
		"Input in0 0 1 in0",
		"Convolution convrelu_0 1 1 in0 1 0=24",
		"Split splitncnn_0 1 2 1 2 3",
	}, "\n")

	got, err := wrapSplitModelParam(input)
	require.NoError(t, err)

	lines := strings.Split(strings.TrimSpace(got), "\n")
	require.Len(t, lines, 6)
	assert.Equal(t, "7767517", lines[0])
	assert.Equal(t, "213 239", lines[1])
	assert.Equal(t, "Input in0 0 1 in0", lines[2])
	assert.Equal(t, "BirdNETFrontend birdnet_frontend 1 1 in0 birdnet_frontend_out", lines[3])
	assert.Equal(t, "Convolution convrelu_0 1 1 birdnet_frontend_out 1 0=24", lines[4])
	assert.Equal(t, "Split splitncnn_0 1 2 1 2 3", lines[5])
}
