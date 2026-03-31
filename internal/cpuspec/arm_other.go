//go:build !(linux && arm64)

package cpuspec

// enrichARMSpec is a no-op on non-ARM-Linux platforms.
func enrichARMSpec(_ *CPUSpec) {}
