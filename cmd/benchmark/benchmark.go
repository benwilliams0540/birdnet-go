package benchmark

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"github.com/tphakala/birdnet-go/internal/classifier"
	"github.com/tphakala/birdnet-go/internal/conf"
)

type options struct {
	backend         string
	version         string
	modelPath       string
	labelPath       string
	onnxRuntimePath string
	ncnnModelDir    string
	ncnnUseVulkan   bool
	useXNNPACK      bool
	compareXNNPACK  bool
	duration        time.Duration
	warmup          int
}

func Command(settings *conf.Settings) *cobra.Command {
	opts := options{
		backend:         normalizeBackendLabel(settings.BirdNET.Backend),
		version:         settings.BirdNET.Version,
		modelPath:       settings.BirdNET.ModelPath,
		labelPath:       settings.BirdNET.LabelPath,
		onnxRuntimePath: settings.BirdNET.ONNXRuntimePath,
		ncnnModelDir:    settings.BirdNET.NCNNModelDir,
		ncnnUseVulkan:   settings.BirdNET.NCNNUseVulkan,
		useXNNPACK:      settings.BirdNET.UseXNNPACK,
		compareXNNPACK:  shouldCompareXNNPACK(settings.BirdNET.Backend),
		duration:        30 * time.Second,
		warmup:          3,
	}
	if opts.version == "" {
		opts.version = "2.4"
	}

	cmd := &cobra.Command{
		Use:   "benchmark",
		Short: "Run BirdNET inference benchmark",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runBenchmark(settings, opts)
		},
	}

	cmd.Flags().StringVar(&opts.backend, "backend", opts.backend, "Inference backend to use: auto, tflite, onnx, ncnn")
	cmd.Flags().StringVar(&opts.version, "version", opts.version, "BirdNET model version (for example: 2.4, 2.4-int8, 2.4-int8-cnn)")
	cmd.Flags().StringVar(&opts.modelPath, "model-path", opts.modelPath, "Explicit classifier model path (.tflite or .onnx)")
	cmd.Flags().StringVar(&opts.labelPath, "label-path", opts.labelPath, "Explicit label file path")
	cmd.Flags().StringVar(&opts.onnxRuntimePath, "onnx-runtime-path", opts.onnxRuntimePath, "Path to the ONNX Runtime shared library")
	cmd.Flags().StringVar(&opts.ncnnModelDir, "ncnn-model-dir", opts.ncnnModelDir, "Directory containing validated NCNN artifacts")
	cmd.Flags().BoolVar(&opts.ncnnUseVulkan, "ncnn-use-vulkan", opts.ncnnUseVulkan, "Enable Vulkan for the NCNN backend")
	cmd.Flags().BoolVar(&opts.useXNNPACK, "use-xnnpack", opts.useXNNPACK, "Enable XNNPACK for the TFLite backend")
	cmd.Flags().BoolVar(&opts.compareXNNPACK, "compare-xnnpack", opts.compareXNNPACK, "Run paired TFLite benchmarks with and without XNNPACK")
	cmd.Flags().DurationVar(&opts.duration, "duration", opts.duration, "How long to benchmark each run")
	cmd.Flags().IntVar(&opts.warmup, "warmup", opts.warmup, "How many warmup inferences to run before timing")

	return cmd
}

func runBenchmark(baseSettings *conf.Settings, opts options) error {
	if opts.duration <= 0 {
		return fmt.Errorf("--duration must be greater than 0")
	}
	if opts.warmup < 0 {
		return fmt.Errorf("--warmup must be greater than or equal to 0")
	}

	settingsCopy := cloneSettings(baseSettings)
	applyOptions(&settingsCopy, opts)

	if opts.compareXNNPACK && !shouldCompareXNNPACK(settingsCopy.BirdNET.Backend) {
		fmt.Printf("XNNPACK comparison is only meaningful for the TFLite backend; running a single configured benchmark instead.\n")
		opts.compareXNNPACK = false
	}

	if !opts.compareXNNPACK {
		label := "Configured"
		if settingsCopy.BirdNET.UseXNNPACK {
			label = "Configured+XNN"
		}
		fmt.Printf("Benchmarking backend=%s version=%s\n", normalizeBackendLabel(settingsCopy.BirdNET.Backend), settingsCopy.BirdNET.Version)
		return runSingleBenchmark(&settingsCopy, label, opts.duration, opts.warmup)
	}

	var xnnpackResults, standardResults benchmarkResults
	xnnpackSettings := cloneSettings(&settingsCopy)
	standardSettings := cloneSettings(&settingsCopy)

	fmt.Println("Testing with XNNPACK delegate:")
	xnnpackSettings.BirdNET.UseXNNPACK = true
	if err := runInferenceBenchmark(&xnnpackSettings, opts.duration, opts.warmup, &xnnpackResults); err != nil {
		fmt.Printf("XNNPACK benchmark failed: %v\n", err)
	}

	fmt.Println("\nTesting standard CPU inference:")
	standardSettings.BirdNET.UseXNNPACK = false
	if err := runInferenceBenchmark(&standardSettings, opts.duration, opts.warmup, &standardResults); err != nil {
		return fmt.Errorf("standard CPU inference benchmark failed: %w", err)
	}

	fmt.Printf("Results:\n")
	fmt.Printf("Method         Inference Time   Throughput\n")
	fmt.Printf("─────────────  ───────────────  ──────────────────────\n")

	if standardResults.totalInferences > 0 {
		fmt.Printf("Standard       %6.1f ms         %6.2f inferences/sec\n",
			float64(standardResults.avgTime.Milliseconds()),
			standardResults.inferencesPerSecond)
	} else {
		fmt.Printf("Standard       Failed\n")
	}

	if xnnpackResults.totalInferences > 0 {
		fmt.Printf("XNNPACK        %6.1f ms         %6.2f inferences/sec\n",
			float64(xnnpackResults.avgTime.Milliseconds()),
			xnnpackResults.inferencesPerSecond)
	} else {
		fmt.Printf("XNNPACK        Failed\n")
	}
	fmt.Printf("─────────────  ───────────────  ──────────────────────\n")

	if xnnpackResults.totalInferences > 0 && standardResults.totalInferences > 0 {
		speedImprovement := (float64(standardResults.avgTime.Milliseconds()) -
			float64(xnnpackResults.avgTime.Milliseconds())) /
			float64(standardResults.avgTime.Milliseconds()) * 100

		fmt.Printf("\nXNNPACK speed improvement: %.1f%%\n", speedImprovement)
		rating, description := getPerformanceRating(float64(xnnpackResults.avgTime.Milliseconds()))
		fmt.Printf("System Rating: %s, %s\n", rating, description)
	}

	return nil
}

func runSingleBenchmark(settings *conf.Settings, label string, duration time.Duration, warmup int) error {
	var results benchmarkResults
	if err := runInferenceBenchmark(settings, duration, warmup, &results); err != nil {
		return err
	}

	fmt.Printf("Results:\n")
	fmt.Printf("Method         Inference Time   Throughput\n")
	fmt.Printf("─────────────  ───────────────  ──────────────────────\n")
	fmt.Printf("%-13s %6.1f ms         %6.2f inferences/sec\n",
		label,
		float64(results.avgTime.Milliseconds()),
		results.inferencesPerSecond,
	)
	fmt.Printf("─────────────  ───────────────  ──────────────────────\n")

	rating, description := getPerformanceRating(float64(results.avgTime.Milliseconds()))
	fmt.Printf("\nSystem Rating: %s, %s\n", rating, description)
	return nil
}

type benchmarkResults struct {
	totalInferences     int
	avgTime             time.Duration
	inferencesPerSecond float64
}

func runInferenceBenchmark(settings *conf.Settings, duration time.Duration, warmup int, results *benchmarkResults) error {
	bn, err := classifier.NewOrchestrator(settings)
	if err != nil {
		return fmt.Errorf("failed to initialize BirdNET: %w", err)
	}
	defer bn.Delete()

	sampleSize := int(bn.ModelInfo.Spec.ClipLength.Seconds() * float64(bn.ModelInfo.Spec.SampleRate))
	silentChunk := make([]float32, sampleSize)

	for i := 0; i < warmup; i++ {
		if _, err := bn.Predict(context.Background(), [][]float32{silentChunk}); err != nil {
			return fmt.Errorf("warmup prediction failed: %w", err)
		}
	}

	startTime := time.Now()
	var totalInferences int
	var totalDuration time.Duration

	fmt.Printf("Running benchmark for %s...\n", duration)

	for time.Since(startTime) < duration {
		inferenceStart := time.Now()
		_, err := bn.Predict(context.Background(), [][]float32{silentChunk})
		if err != nil {
			return fmt.Errorf("prediction failed: %w", err)
		}
		inferenceTime := time.Since(inferenceStart)
		totalDuration += inferenceTime
		totalInferences++

		if totalInferences%10 == 0 {
			avgTime := totalDuration / time.Duration(totalInferences)
			fmt.Printf("\rInferences: %d, Average time: %dms", totalInferences, avgTime.Milliseconds())
		}
	}
	fmt.Println()

	if totalInferences == 0 {
		return fmt.Errorf("benchmark produced no inferences")
	}

	results.totalInferences = totalInferences
	results.avgTime = totalDuration / time.Duration(totalInferences)
	results.inferencesPerSecond = float64(totalInferences) / duration.Seconds()

	return nil
}

func cloneSettings(base *conf.Settings) conf.Settings {
	settingsCopy := *base
	settingsCopy.BirdNET.RangeFilter.Species = slices.Clone(base.BirdNET.RangeFilter.Species)
	return settingsCopy
}

func applyOptions(settings *conf.Settings, opts options) {
	if opts.backend == "auto" {
		settings.BirdNET.Backend = ""
	} else {
		settings.BirdNET.Backend = strings.TrimSpace(opts.backend)
	}
	settings.BirdNET.Version = strings.TrimSpace(opts.version)
	settings.BirdNET.UseXNNPACK = opts.useXNNPACK

	if strings.TrimSpace(opts.modelPath) != "" {
		settings.BirdNET.ModelPath = strings.TrimSpace(opts.modelPath)
	}
	if strings.TrimSpace(opts.labelPath) != "" {
		settings.BirdNET.LabelPath = strings.TrimSpace(opts.labelPath)
	}
	if strings.TrimSpace(opts.onnxRuntimePath) != "" {
		settings.BirdNET.ONNXRuntimePath = strings.TrimSpace(opts.onnxRuntimePath)
	}
	if strings.TrimSpace(opts.ncnnModelDir) != "" {
		settings.BirdNET.NCNNModelDir = strings.TrimSpace(opts.ncnnModelDir)
	}
	settings.BirdNET.NCNNUseVulkan = opts.ncnnUseVulkan
}

func normalizeBackendLabel(backend string) string {
	trimmed := strings.TrimSpace(strings.ToLower(backend))
	if trimmed == "" {
		return "auto"
	}
	return trimmed
}

func shouldCompareXNNPACK(backend string) bool {
	normalized := normalizeBackendLabel(backend)
	return normalized == "auto" || normalized == "tflite"
}

func getPerformanceRating(inferenceTime float64) (rating, description string) {
	switch {
	case inferenceTime > 3000:
		return "Failed", "System is too slow for BirdNET-Go real-time detection"
	case inferenceTime > 2000:
		return "Very Poor", "System is too slow for reliable operation"
	case inferenceTime > 1000:
		return "Poor", "System may struggle with real-time detection"
	case inferenceTime > 500:
		return "Decent", "System should handle real-time detection"
	case inferenceTime > 200:
		return "Good", "System will perform well"
	case inferenceTime > 100:
		return "Very Good", "System will perform very well"
	case inferenceTime > 20:
		return "Excellent", "System will perform excellently"
	default:
		return "Superb", "System will perform exceptionally well"
	}
}
