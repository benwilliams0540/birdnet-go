//go:build qnn

package qnn

// melspec.go — audio-to-mel-spectrogram preprocessing for the QNN CNN model.
//
// The original birdnet.onnx embeds preprocessing inside the ONNX graph (nodes
// 0–38).  Because QAIRT 2.39 cannot translate the ONNX DFT operator, we split
// the model at the mel-spectrogram output boundary and reimplement the
// preprocessing here using only the Go standard library.
//
// # Computation (per path)
//
//   raw audio [144000]
//     → normalize (min/max scaling)
//     → frame (overlapping windows)
//     → multiply by Hann window
//     → N-point radix-2 FFT
//     → take Re(FFT[k]) for k = 0..N/2   ← real part only (confirmed by onnxruntime trace)
//     → square: Re(FFT[k])²
//     → mel filterbank matmul
//     → power compression
//     → [511, 96]
//
// # Key implementation note — real part not magnitude
//
// A common assumption is that mel spectrograms use power spectrum |FFT|².
// birdnet.onnx uses only Re(FFT[k])²: the ONNX DFT node outputs shape
// [batch, frames, freq, 2], and the subsequent Slice takes axis=-1 start=0
// end=1 — extracting only the real component.  This was verified by running
// the original preprocessing through onnxruntime: mean abs error vs Re(FFT)
// was ~3e-6, vs |FFT| was ~1.06.

import (
	_ "embed"
	"math"
)

// Embedded mel filterbank matrices extracted from birdnet.onnx.
// Layout: [freqBins × numMelBins] float32 little-endian row-major.
//
//go:embed data/mel_fb_spec1.bin
var melFBSpec1Raw []byte // [1025 × 96] = 393 600 bytes

//go:embed data/mel_fb_spec2.bin
var melFBSpec2Raw []byte // [513 × 96] = 196 992 bytes

const (
	// melAudioLen is the number of float32 audio samples expected (48 kHz × 3 s).
	melAudioLen = 144_000

	// Number of time frames and mel frequency bins in each spectrogram.
	melNumFrames = 511
	melNumBins   = 96

	// SPEC1 — long STFT path → CNN input tensor 0.
	spec1FFT  = 2048
	spec1Hop  = 278
	spec1Bins = spec1FFT/2 + 1 // 1025 onesided frequency bins

	// SPEC2 — short STFT path → CNN input tensor 1.
	spec2FFT  = 1024
	spec2Hop  = 280
	spec2Bins = spec2FFT/2 + 1 // 513 onesided frequency bins

	// Power compression exponents (from model initializers).
	spec1PowExp = float64(0.22952409088611603)
	spec2PowExp = float64(0.1905273050069809)

	// Audio normalization constants (from model initializers).
	normEps = float32(9.999999974752427e-07)
	normSub = float32(0.5)
	normMul = float32(2.0)
)

// Package-level pre-computed tables; initialised once at startup.
var (
	melFB1 []float32 // [spec1Bins × melNumBins] row-major
	melFB2 []float32 // [spec2Bins × melNumBins] row-major
	hann1  []float32 // [spec1FFT] periodic Hann window
	hann2  []float32 // [spec2FFT] periodic Hann window

	// Pre-computed twiddle factors for the two fixed FFT sizes.
	twiddle1 []complex128 // [spec1FFT/2]
	twiddle2 []complex128 // [spec2FFT/2]
)

func init() {
	melFB1 = rawBytesToFloat32(melFBSpec1Raw)
	melFB2 = rawBytesToFloat32(melFBSpec2Raw)
	hann1 = periodicHann(spec1FFT)
	hann2 = periodicHann(spec2FFT)
	twiddle1 = computeTwiddle(spec1FFT)
	twiddle2 = computeTwiddle(spec2FFT)
}

// rawBytesToFloat32 reinterprets little-endian float32 bytes.
func rawBytesToFloat32(b []byte) []float32 {
	f := make([]float32, len(b)/4)
	for i := range f {
		u := uint32(b[i*4]) | uint32(b[i*4+1])<<8 | uint32(b[i*4+2])<<16 | uint32(b[i*4+3])<<24
		f[i] = math.Float32frombits(u)
	}
	return f
}

// periodicHann returns an N-point periodic Hann window:
// w[n] = 0.5 × (1 − cos(2π·n/N))
func periodicHann(n int) []float32 {
	w := make([]float32, n)
	scale := 2.0 * math.Pi / float64(n)
	for i := range w {
		w[i] = float32(0.5 * (1.0 - math.Cos(scale*float64(i))))
	}
	return w
}

// computeTwiddle pre-computes the complex twiddle factors W_N^k = e^{-2πi·k/N}
// for k = 0..N/2-1.
func computeTwiddle(n int) []complex128 {
	t := make([]complex128, n/2)
	ang := -2.0 * math.Pi / float64(n)
	for k := range t {
		t[k] = complex(math.Cos(ang*float64(k)), math.Sin(ang*float64(k)))
	}
	return t
}

// ComputeMelSpectrograms computes the two mel spectrograms from raw audio samples.
//
// samples must contain exactly melAudioLen (144 000) float32 values (48 kHz PCM,
// normalised to the range roughly −1…+1, as produced by the standard birdnet-go
// audio pipeline).
//
// Returns a flat []float32 of length 2 × melNumFrames × melNumBins = 98 112:
//
//	[0 … 49055]    SPEC1 (fft=2048, hop=278) → CNN input tensor 0
//	[49056 … 98111] SPEC2 (fft=1024, hop=280) → CNN input tensor 1
func ComputeMelSpectrograms(samples []float32) []float32 {
	norm := normalizeAudio(samples)

	const halfLen = melNumFrames * melNumBins
	out := make([]float32, 2*halfLen)

	stftMelSpec(norm, spec1FFT, spec1Hop, spec1Bins, hann1, melFB1, spec1PowExp, twiddle1, out[:halfLen])
	stftMelSpec(norm, spec2FFT, spec2Hop, spec2Bins, hann2, melFB2, spec2PowExp, twiddle2, out[halfLen:])

	return out
}

// normalizeAudio applies the model's input normalisation:
//
//	x_sub   = x − min(x)
//	denom   = max(x_sub) + ε
//	x_norm  = x_sub / denom
//	x_final = (x_norm − 0.5) × 2
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

// stftMelSpec computes one STFT mel spectrogram path.
// dst receives melNumFrames × melNumBins float32 values.
func stftMelSpec(
	signal []float32,
	fftSize, hop, freqBins int,
	hann []float32,
	fb []float32, // [freqBins × melNumBins] row-major
	powExp float64,
	twiddle []complex128,
	dst []float32,
) {
	buf := make([]complex128, fftSize)
	reSq := make([]float32, freqBins) // Re(FFT[k])²
	mel := make([]float32, melNumBins)

	sigLen := len(signal)

	for frame := range melNumFrames {
		start := frame * hop

		// Build complex input: real = Hann-windowed sample, imag = 0.
		for i := range fftSize {
			si := start + i
			if si < sigLen {
				buf[i] = complex(float64(signal[si])*float64(hann[i]), 0)
			} else {
				buf[i] = 0
			}
		}

		// In-place radix-2 Cooley–Tukey FFT.
		fftInPlace(buf, twiddle)

		// Re(FFT[k])² for the one-sided spectrum.
		for k := range freqBins {
			re := float32(real(buf[k]))
			reSq[k] = re * re
		}

		// Mel filterbank: mel[j] = Σ_k reSq[k] × fb[k·melNumBins + j]
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

		// Power compression + write output.
		base := frame * melNumBins
		for j, m := range mel {
			dst[base+j] = float32(math.Pow(float64(m), powExp))
		}
	}
}

// fftInPlace performs an in-place radix-2 DIT (decimation-in-time) FFT.
// n must be a power of 2.  twiddle[k] = exp(-2πi·k/n) for k=0..n/2-1.
func fftInPlace(x []complex128, twiddle []complex128) {
	n := len(x)

	// Bit-reversal permutation.
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

	// Butterfly stages.
	for length := 2; length <= n; length <<= 1 {
		half := length / 2
		step := n / length // stride in twiddle table
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
