//go:build linux

package observability

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const (
	gpuDevfreqBasePath = "/sys/class/devfreq"
)

// gpuTransStat holds a snapshot of per-frequency cumulative time from trans_stat.
type gpuTransStat struct {
	// freqTimeMs maps frequency (Hz) -> cumulative milliseconds spent at that frequency
	freqTimeMs map[uint64]uint64
	// total is the sum of all frequency times
	total uint64
}

// findGPUDevfreqPath searches /sys/class/devfreq for a GPU device entry.
// Recognises "gpu", "kgsl", "adreno", "mali" in the device name.
// Returns the devfreq path and the minimum frequency, or empty string if not found.
func findGPUDevfreqPath() (devfreqPath string, minFreqHz uint64) {
	entries, err := os.ReadDir(gpuDevfreqBasePath)
	if err != nil {
		return "", 0
	}

	gpuKeywords := []string{"gpu", "kgsl", "adreno", "mali"}

	for _, entry := range entries {
		name := strings.ToLower(entry.Name())
		for _, kw := range gpuKeywords {
			if strings.Contains(name, kw) {
				path := filepath.Join(gpuDevfreqBasePath, entry.Name())

				// Read modalias or name for further confirmation (best-effort)
				devicePath := filepath.Join(path, "device")
				if _, err := os.Stat(devicePath); err == nil {
					modalias, _ := os.ReadFile(filepath.Join(devicePath, "modalias")) //nolint:gosec // system path
					if len(modalias) > 0 && !strings.Contains(strings.ToLower(string(modalias)), "gpu") {
						// Skip non-GPU devices that matched by name coincidence
						continue
					}
				}

				minHz := readDevfreqFreq(path, "min_freq")
				return path, minHz
			}
		}
	}
	return "", 0
}

// readDevfreqFreq reads a frequency file (min_freq, max_freq, cur_freq) from a devfreq path.
func readDevfreqFreq(devfreqPath, filename string) uint64 {
	data, err := os.ReadFile(filepath.Join(devfreqPath, filename)) //nolint:gosec // system path
	if err != nil {
		return 0
	}
	val, err := strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
	if err != nil {
		return 0
	}
	return val
}

// readGPUTransStat parses the trans_stat file from a devfreq path into a gpuTransStat snapshot.
// The file format has one row per frequency with a trailing time-in-state column (ms).
// Example:
//
//	     From  :   To
//	           : 355200000 537600000 ... time(ms)
//	* 355200000:         0 ...             496916
//	  537600000:         3 ...                168
func readGPUTransStat(devfreqPath string) (gpuTransStat, bool) {
	data, err := os.ReadFile(filepath.Join(devfreqPath, "trans_stat")) //nolint:gosec // system path
	if err != nil {
		return gpuTransStat{}, false
	}

	snap := gpuTransStat{freqTimeMs: make(map[uint64]uint64)}
	lines := strings.Split(string(data), "\n")

	for _, line := range lines {
		line = strings.TrimSpace(strings.TrimPrefix(line, "*"))
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "From") || strings.HasPrefix(line, ":") {
			continue
		}

		// Format: "<freq>: <count> ... <time_ms>"
		colonIdx := strings.Index(line, ":")
		if colonIdx < 0 {
			continue
		}

		freqStr := strings.TrimSpace(line[:colonIdx])
		freq, err := strconv.ParseUint(freqStr, 10, 64)
		if err != nil {
			continue
		}

		// The last whitespace-separated field is time_ms
		fields := strings.Fields(line[colonIdx+1:])
		if len(fields) == 0 {
			continue
		}
		timeMs, err := strconv.ParseUint(fields[len(fields)-1], 10, 64)
		if err != nil {
			continue
		}

		snap.freqTimeMs[freq] = timeMs
		snap.total += timeMs
	}

	if len(snap.freqTimeMs) == 0 {
		return gpuTransStat{}, false
	}
	return snap, true
}

// gpuUtilizationPercent computes GPU utilization between two trans_stat snapshots.
// Utilization = fraction of time spent NOT at the minimum frequency.
// Returns 0 if delta total is zero (no time has elapsed).
func gpuUtilizationPercent(prev, curr gpuTransStat, minFreqHz uint64) float64 {
	if curr.total <= prev.total {
		return 0
	}
	deltaTotal := curr.total - prev.total
	prevMinTime := prev.freqTimeMs[minFreqHz]
	currMinTime := curr.freqTimeMs[minFreqHz]

	var deltaMinTime uint64
	if currMinTime >= prevMinTime {
		deltaMinTime = currMinTime - prevMinTime
	}

	busyTime := deltaTotal - deltaMinTime
	return float64(busyTime) / float64(deltaTotal) * 100.0
}
