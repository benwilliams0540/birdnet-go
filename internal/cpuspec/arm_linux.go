//go:build linux && arm64

package cpuspec

import (
	"bufio"
	"os"
	"strings"
)

// Known ARM CPU implementers
const (
	armImplementerQualcomm = "0x51"
	armImplementerARM      = "0x41"
	armImplementerBroadcom = "0x42"
)

// Known ARM CPU parts (from Qualcomm)
var armPartNames = map[string]string{
	// Qualcomm Kryo series
	"0x800": "Kryo-2xx-Gold",   // Cortex-A73 class
	"0x801": "Kryo-2xx-Silver", // Cortex-A53 class
	"0x802": "Kryo-3xx-Gold",   // Cortex-A75 class
	"0x803": "Kryo-3xx-Silver", // Cortex-A55 class
	"0x804": "Kryo-V2",         // Cortex-A55 class (QRB2210/QCM2290)
	"0x805": "Kryo-4xx-Gold",   // Cortex-A76 class

	// ARM standard cores (when implementer is 0x41)
	"0xd03": "Cortex-A53",
	"0xd05": "Cortex-A55",
	"0xd07": "Cortex-A57",
	"0xd08": "Cortex-A72",
	"0xd09": "Cortex-A73",
	"0xd0a": "Cortex-A75",
	"0xd0b": "Cortex-A76",
	"0xd0d": "Cortex-A77",
	"0xd40": "Neoverse-V1",
	"0xd41": "Cortex-A78",
	"0xd44": "Cortex-X1",
	"0xd46": "Cortex-A510",
	"0xd47": "Cortex-A710",
	"0xd48": "Cortex-X2",
	"0xd4d": "Cortex-A715",
	"0xd4e": "Cortex-X3",
	"0xd80": "Cortex-A520",
	"0xd81": "Cortex-A720",
	"0xd82": "Cortex-X4",

	// Broadcom
	"0x516": "BCM2712", // Raspberry Pi 5
}

// enrichARMSpec reads /proc/cpuinfo on ARM Linux to detect CPU details and features.
func enrichARMSpec(spec *CPUSpec) {
	f, err := os.Open("/proc/cpuinfo")
	if err != nil {
		return
	}
	defer f.Close()

	var implementer, part, features string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if k, v, ok := parseCpuInfoLine(line); ok {
			switch k {
			case "CPU implementer":
				if implementer == "" {
					implementer = v
				}
			case "CPU part":
				if part == "" {
					part = v
				}
			case "Features":
				if features == "" {
					features = v
				}
			}
		}
		// Stop after first processor block — all cores are the same on Uno Q
		if implementer != "" && part != "" && features != "" {
			break
		}
	}

	// Detect manufacturer
	if implementer == armImplementerQualcomm {
		spec.IsQualcomm = true
	}

	// Look up part name
	if name, ok := armPartNames[part]; ok {
		spec.ARMPartName = name
		// If brand name is empty (common on ARM), set it
		if spec.BrandName == "" {
			if spec.IsQualcomm {
				spec.BrandName = "Qualcomm " + name
			} else {
				spec.BrandName = "ARM " + name
			}
		}
	}

	// Parse feature flags
	featureSet := parseFeatures(features)
	spec.HasDotProduct = featureSet["asimddp"]
	spec.HasI8MM = featureSet["i8mm"]
	spec.HasSVE = featureSet["sve"]

	spec.Architecture = "aarch64"
}

// parseCpuInfoLine splits a /proc/cpuinfo line into key and value.
func parseCpuInfoLine(line string) (key, value string, ok bool) {
	parts := strings.SplitN(line, ":", 2)
	if len(parts) != 2 {
		return "", "", false
	}
	return strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1]), true
}

// parseFeatures splits the Features line into a set for O(1) lookup.
func parseFeatures(features string) map[string]bool {
	set := make(map[string]bool)
	for _, f := range strings.Fields(features) {
		set[f] = true
	}
	return set
}
