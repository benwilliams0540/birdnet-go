package api

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLiveSpectrogramTapComputeColumn_Silence(t *testing.T) {
	tap := &liveSpectrogramTap{
		fftSize:    liveSpectrogramFFTSize,
		ring:       make([]float64, liveSpectrogramFFTSize),
		hann:       periodicHannFloat64(liveSpectrogramFFTSize),
		twiddle:    computeTwiddleFloat64(liveSpectrogramFFTSize),
		smoothed:   make([]float64, liveSpectrogramFFTSize/2),
		sampleRate: 48000,
	}

	bins := tap.computeColumn()
	require.Len(t, bins, liveSpectrogramFFTSize/2)

	for _, bin := range bins {
		assert.Zero(t, bin)
	}
}

func TestLiveSpectrogramTapComputeColumn_TonePeaksNearExpectedBin(t *testing.T) {
	const (
		sampleRate = 48000
		toneHz     = 3000.0
	)

	tap := &liveSpectrogramTap{
		fftSize:    liveSpectrogramFFTSize,
		ring:       make([]float64, liveSpectrogramFFTSize),
		hann:       periodicHannFloat64(liveSpectrogramFFTSize),
		twiddle:    computeTwiddleFloat64(liveSpectrogramFFTSize),
		smoothed:   make([]float64, liveSpectrogramFFTSize/2),
		sampleRate: sampleRate,
	}

	for i := range tap.ring {
		tap.ring[i] = 0.7 * math.Sin(2*math.Pi*toneHz*float64(i)/sampleRate)
	}

	bins := tap.computeColumn()
	require.Len(t, bins, liveSpectrogramFFTSize/2)

	peakIndex := 0
	peakValue := byte(0)
	for i, value := range bins {
		if value > peakValue {
			peakValue = value
			peakIndex = i
		}
	}

	expectedBin := int(math.Round(toneHz * liveSpectrogramFFTSize / sampleRate))
	assert.InDelta(t, expectedBin, peakIndex, 2)
	assert.Greater(t, int(peakValue), 0)
}
