//go:build ncnn

package ncnn

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBirdNETFrontendCustomLayerMatchesGoReference(t *testing.T) {
	samples := make([]float32, splitAudioLen)
	for i := range samples {
		samples[i] = float32(math.Sin(2 * math.Pi * 440 * float64(i) / 48000.0))
	}

	want := ComputeSplitCNNInput(samples)
	got, err := computeBirdNETFrontendWithCustomLayer(samples)
	require.NoError(t, err)
	require.Len(t, got, len(want))

	maxDelta := 0.0
	for i := range want {
		delta := math.Abs(float64(want[i] - got[i]))
		if delta > maxDelta {
			maxDelta = delta
		}
	}

	assert.LessOrEqual(t, maxDelta, 1e-4)
}
