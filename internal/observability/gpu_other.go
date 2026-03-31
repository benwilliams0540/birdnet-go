//go:build !linux

package observability

type gpuTransStat struct {
	freqTimeMs map[uint64]uint64
	total      uint64
}

func findGPUDevfreqPath() (devfreqPath string, minFreqHz uint64) {
	return "", 0
}

func readDevfreqFreq(_ string, _ string) uint64 { return 0 }

func readGPUTransStat(_ string) (gpuTransStat, bool) { return gpuTransStat{}, false }

func gpuUtilizationPercent(_, _ gpuTransStat, _ uint64) float64 { return 0 }
