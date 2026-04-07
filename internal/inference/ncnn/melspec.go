package ncnn

import (
	_ "embed"
	"math"
)

// The NCNN split model is derived from source_models/BirdNET_V2.4.onnx at the
// tensor boundary immediately after the spectrogram affine normalization and
// transpose. It therefore expects a single NCHW tensor shaped [1, 2, 96, 511].

//go:embed data/mel_fb_spec1.bin
var melFBSpec1Raw []byte

//go:embed data/mel_fb_spec2.bin
var melFBSpec2Raw []byte

const (
	splitAudioLen = 144_000
	splitFrames   = 511
	splitBins     = 96
	splitChannels = 2

	spec1FFT  = 2048
	spec1Hop  = 278
	spec1Bins = spec1FFT/2 + 1

	spec2FFT  = 1024
	spec2Hop  = 280
	spec2Bins = spec2FFT/2 + 1

	spec1PowExp = float64(0.22952409088611603)
	spec2PowExp = float64(0.1905273050069809)

	normEps = float32(9.999999974752427e-07)
	normSub = float32(0.5)
	normMul = float32(2.0)
)

var (
	melFB1 []float32
	melFB2 []float32
	hann1  []float32
	hann2  []float32

	twiddle1 []complex128
	twiddle2 []complex128

	// Channel-wise affine normalization constants taken from the source ONNX.
	channelScale = [splitChannels]float32{0.19752595, 3.152703}
	channelBias  = [splitChannels]float32{-0.65386057, -0.34679556}
)

func init() {
	melFB1 = rawBytesToFloat32(melFBSpec1Raw)
	melFB2 = rawBytesToFloat32(melFBSpec2Raw)
	hann1 = periodicHann(spec1FFT)
	hann2 = periodicHann(spec2FFT)
	twiddle1 = computeTwiddle(spec1FFT)
	twiddle2 = computeTwiddle(spec2FFT)
}

func rawBytesToFloat32(b []byte) []float32 {
	f := make([]float32, len(b)/4)
	for i := range f {
		u := uint32(b[i*4]) | uint32(b[i*4+1])<<8 | uint32(b[i*4+2])<<16 | uint32(b[i*4+3])<<24
		f[i] = math.Float32frombits(u)
	}
	return f
}

func periodicHann(n int) []float32 {
	w := make([]float32, n)
	scale := 2.0 * math.Pi / float64(n)
	for i := range w {
		w[i] = float32(0.5 * (1.0 - math.Cos(scale*float64(i))))
	}
	return w
}

func computeTwiddle(n int) []complex128 {
	t := make([]complex128, n/2)
	ang := -2.0 * math.Pi / float64(n)
	for k := range t {
		t[k] = complex(math.Cos(ang*float64(k)), math.Sin(ang*float64(k)))
	}
	return t
}

// ComputeSplitCNNInput reproduces the source ONNX preprocessing and returns the
// exact channel-first tensor expected by the split NCNN CNN model:
// [channel=2][mel=96][frame=511].
func ComputeSplitCNNInput(samples []float32) []float32 {
	norm := normalizeAudio(samples)

	const channelSize = splitFrames * splitBins

	spec1 := make([]float32, channelSize)
	spec2 := make([]float32, channelSize)

	stftMelSpec(norm, spec1FFT, spec1Hop, spec1Bins, hann1, melFB1, spec1PowExp, twiddle1, spec1)
	stftMelSpec(norm, spec2FFT, spec2Hop, spec2Bins, hann2, melFB2, spec2PowExp, twiddle2, spec2)

	out := make([]float32, splitChannels*channelSize)
	for frame := 0; frame < splitFrames; frame++ {
		frameBase := frame * splitBins
		for bin := 0; bin < splitBins; bin++ {
			// Nodes 39 and 40 in the source ONNX reverse the last mel axis before
			// the channel concat, so we must mirror that here.
			srcIndex := frameBase + (splitBins - 1 - bin)
			dstIndex := bin*splitFrames + frame
			out[dstIndex] = spec1[srcIndex]*channelScale[0] + channelBias[0]
			out[channelSize+dstIndex] = spec2[srcIndex]*channelScale[1] + channelBias[1]
		}
	}

	return out
}

func normalizeAudio(samples []float32) []float32 {
	if len(samples) == 0 {
		return samples
	}
	minVal := samples[0]
	for _, s := range samples[1:] {
		if s < minVal {
			minVal = s
		}
	}
	maxSub := float32(0)
	for _, s := range samples {
		if v := s - minVal; v > maxSub {
			maxSub = v
		}
	}
	denom := maxSub + normEps
	norm := make([]float32, len(samples))
	for i, s := range samples {
		x := (s - minVal) / denom
		norm[i] = (x - normSub) * normMul
	}
	return norm
}

func stftMelSpec(
	signal []float32,
	fftSize, hop, freqBins int,
	hann []float32,
	fb []float32,
	powExp float64,
	twiddle []complex128,
	dst []float32,
) {
	buf := make([]complex128, fftSize)
	realBins := make([]float32, freqBins)
	mel := make([]float32, splitBins)

	sigLen := len(signal)

	for frame := 0; frame < splitFrames; frame++ {
		start := frame * hop

		for i := 0; i < fftSize; i++ {
			si := start + i
			if si < sigLen {
				buf[i] = complex(float64(signal[si])*float64(hann[i]), 0)
			} else {
				buf[i] = 0
			}
		}

		fftInPlace(buf, twiddle)

		for k := 0; k < freqBins; k++ {
			realBins[k] = float32(real(buf[k]))
		}

		for j := range mel {
			mel[j] = 0
		}
		for k := 0; k < freqBins; k++ {
			v := realBins[k]
			if v == 0 {
				continue
			}
			base := k * splitBins
			for j := 0; j < splitBins; j++ {
				mel[j] += v * fb[base+j]
			}
		}

		base := frame * splitBins
		for j, m := range mel {
			// Match the source ONNX graph: mel projection first, then square,
			// then apply the learned power-compression exponent.
			power := m * m
			dst[base+j] = float32(math.Pow(float64(power), powExp))
		}
	}
}

func fftInPlace(x []complex128, twiddle []complex128) {
	n := len(x)

	j := 0
	for i := 1; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			x[i], x[j] = x[j], x[i]
		}
	}

	for length := 2; length <= n; length <<= 1 {
		half := length / 2
		step := n / length
		for i := 0; i < n; i += length {
			for k := 0; k < half; k++ {
				u := x[i+k]
				v := x[i+k+half] * twiddle[k*step]
				x[i+k] = u + v
				x[i+k+half] = u - v
			}
		}
	}
}
