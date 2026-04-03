//go:build ncnn

package ncnn

// melspec.go — audio-to-mel-spectrogram preprocessing for the NCNN CNN model.
//
// Identical computation to internal/inference/qnn/melspec.go.
// The mel filterbank matrices are embedded from data/ (same files used by QNN).
//
// # Output format
//
// ComputeMelSpectrograms returns a flat []float32 of length 2×511×96 = 98112
// in planar channel-first layout:
//
//	[0 … 49055]    SPEC1 (fft=2048, hop=278) — channel 0
//	[49056 … 98111] SPEC2 (fft=1024, hop=280) — channel 1
//
// Each channel is stored in row-major [H=511, W=96] order, matching what
// NCNN expects for ncnn_mat_create_3d(W=96, H=511, C=2) with planar data.

import (
	_ "embed"
	"math"
)

//go:embed data/mel_fb_spec1.bin
var melFBSpec1Raw []byte // [1025 × 96] float32 little-endian

//go:embed data/mel_fb_spec2.bin
var melFBSpec2Raw []byte // [513 × 96] float32 little-endian

const (
	melAudioLen  = 144_000
	melNumFrames = 511
	melNumBins   = 96

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
	melFB1   []float32
	melFB2   []float32
	hann1    []float32
	hann2    []float32
	twiddle1 []complex128
	twiddle2 []complex128
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

// ComputeMelSpectrograms computes the two-channel mel spectrogram from raw audio.
//
// samples must contain exactly 144 000 float32 values (48 kHz × 3 s).
//
// Returns a flat []float32 of length 98 112 in planar format:
//   - [0:49056]    SPEC1 → NCNN channel 0
//   - [49056:98112] SPEC2 → NCNN channel 1
func ComputeMelSpectrograms(samples []float32) []float32 {
	norm := normalizeAudio(samples)

	const halfLen = melNumFrames * melNumBins
	out := make([]float32, 2*halfLen)

	stftMelSpec(norm, spec1FFT, spec1Hop, spec1Bins, hann1, melFB1, spec1PowExp, twiddle1, out[:halfLen])
	stftMelSpec(norm, spec2FFT, spec2Hop, spec2Bins, hann2, melFB2, spec2PowExp, twiddle2, out[halfLen:])

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
	reSq := make([]float32, freqBins)
	mel := make([]float32, melNumBins)

	sigLen := len(signal)

	for frame := range melNumFrames {
		start := frame * hop

		for i := range fftSize {
			si := start + i
			if si < sigLen {
				buf[i] = complex(float64(signal[si])*float64(hann[i]), 0)
			} else {
				buf[i] = 0
			}
		}

		fftInPlace(buf, twiddle)

		for k := range freqBins {
			re := float32(real(buf[k]))
			reSq[k] = re * re
		}

		for j := range mel {
			mel[j] = 0
		}
		for k := range freqBins {
			r := reSq[k]
			if r == 0 {
				continue
			}
			base := k * melNumBins
			for j := range melNumBins {
				mel[j] += r * fb[base+j]
			}
		}

		base := frame * melNumBins
		for j, m := range mel {
			dst[base+j] = float32(math.Pow(float64(m), powExp))
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
			for k := range half {
				u := x[i+k]
				v := x[i+k+half] * twiddle[k*step]
				x[i+k] = u + v
				x[i+k+half] = u - v
			}
		}
	}
}
