package ncnn

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestComputeSplitCNNInputProducesFiniteChannelFirstTensor(t *testing.T) {
	t.Parallel()

	samples := make([]float32, splitAudioLen)
	for i := range samples {
		samples[i] = float32(math.Sin(2 * math.Pi * 440 * float64(i) / 48000.0))
	}

	got := ComputeSplitCNNInput(samples)

	require.Len(t, got, splitChannels*splitBins*splitFrames)

	var minVal float32 = got[0]
	var maxVal float32 = got[0]
	for _, value := range got {
		assert.False(t, math.IsNaN(float64(value)))
		assert.False(t, math.IsInf(float64(value), 0))
		if value < minVal {
			minVal = value
		}
		if value > maxVal {
			maxVal = value
		}
	}

	assert.NotEqual(t, minVal, maxVal)
	assert.NotEqual(t, got[0], got[splitBins*splitFrames])
}

func TestComputeSplitCNNInputMatchesSourceONNXReferenceForSineWave(t *testing.T) {
	t.Parallel()

	samples := make([]float32, splitAudioLen)
	for i := range samples {
		samples[i] = float32(math.Sin(2 * math.Pi * 440 * float64(i) / 48000.0))
	}

	got := ComputeSplitCNNInput(samples)
	require.Len(t, got, splitChannels*splitBins*splitFrames)

	expectedFirst := []float64{
		-0.6511173248291016,
		-0.6515721678733826,
		-0.6527586579322815,
		-0.6522443294525146,
		-0.6514492630958557,
		-0.6509525179862976,
		-0.650650143623352,
		-0.6505247354507446,
	}
	for i, want := range expectedFirst {
		assert.InDelta(t, want, float64(got[i]), 5e-4)
	}

	channelOffset := splitBins * splitFrames
	expectedSecondChannel := []float64{
		-0.31200146675109863,
		-0.3171388506889343,
		-0.32058900594711304,
		-0.31767338514328003,
		-0.32294341921806335,
		-0.31632277369499207,
		-0.31460195779800415,
		-0.31548166275024414,
	}
	for i, want := range expectedSecondChannel {
		assert.InDelta(t, want, float64(got[channelOffset+i]), 2e-2)
	}
}
