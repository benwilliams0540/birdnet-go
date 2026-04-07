package compareaudio

import (
	"context"
	"encoding/binary"
	"encoding/json"
	stdErrors "errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"github.com/spf13/cobra"
	"github.com/tphakala/birdnet-go/internal/audiocore/readfile"
	iresample "github.com/tphakala/birdnet-go/internal/audiocore/resample"
	"github.com/tphakala/birdnet-go/internal/classifier"
	"github.com/tphakala/birdnet-go/internal/conf"
	"github.com/tphakala/birdnet-go/internal/datastore"
)

var errChunkLimitReached = stdErrors.New("compare-audio chunk limit reached")

type options struct {
	audioPath       string
	backend         string
	version         string
	modelPath       string
	labelPath       string
	onnxRuntimePath string
	ncnnModelDir    string
	ncnnUseVulkan   bool
	topK            int
	summaryK        int
	maxChunks       int
	jsonOutput      bool
	summaryOnly     bool
}

type predictionResult struct {
	Label          string  `json:"label"`
	ScientificName string  `json:"scientificName,omitempty"`
	CommonName     string  `json:"commonName,omitempty"`
	Confidence     float32 `json:"confidence"`
}

type chunkResult struct {
	Index        int                `json:"index"`
	StartSeconds float64            `json:"startSeconds"`
	EndSeconds   float64            `json:"endSeconds"`
	EOF          bool               `json:"eof"`
	Predictions  []predictionResult `json:"predictions"`
}

type aggregateResult struct {
	Label          string  `json:"label"`
	ScientificName string  `json:"scientificName,omitempty"`
	CommonName     string  `json:"commonName,omitempty"`
	MaxConfidence  float32 `json:"maxConfidence"`
	AvgConfidence  float32 `json:"avgConfidence"`
	Chunks         int     `json:"chunks"`
}

type compareSummary struct {
	AudioPath            string            `json:"audioPath"`
	Backend              string            `json:"backend"`
	Version              string            `json:"version"`
	ModelPath            string            `json:"modelPath,omitempty"`
	LabelPath            string            `json:"labelPath,omitempty"`
	ModelID              string            `json:"modelId"`
	AudioSampleRate      int               `json:"audioSampleRate"`
	ModelSampleRate      int               `json:"modelSampleRate"`
	ClipDurationSeconds  float64           `json:"clipDurationSeconds"`
	OverlapSeconds       float64           `json:"overlapSeconds"`
	TotalDurationSeconds float64           `json:"totalDurationSeconds"`
	EstimatedChunkCount  int               `json:"estimatedChunkCount"`
	ProcessedChunkCount  int               `json:"processedChunkCount"`
	TopSpeciesByMax      []aggregateResult `json:"topSpeciesByMax"`
}

type compareReport struct {
	Summary compareSummary `json:"summary"`
	Chunks  []chunkResult  `json:"chunks,omitempty"`
}

type aggregateState struct {
	label          string
	scientificName string
	commonName     string
	maxConfidence  float32
	total          float64
	chunks         int
}

// Command creates the compare-audio command.
func Command(settings *conf.Settings) *cobra.Command {
	opts := options{
		backend:         "auto",
		version:         settings.BirdNET.Version,
		onnxRuntimePath: settings.BirdNET.ONNXRuntimePath,
		ncnnModelDir:    settings.BirdNET.NCNNModelDir,
		ncnnUseVulkan:   settings.BirdNET.NCNNUseVulkan,
		topK:            5,
		summaryK:        10,
	}
	if opts.version == "" {
		opts.version = "2.4"
	}

	cmd := &cobra.Command{
		Use:   "compare-audio",
		Short: "Run BirdNET inference on an audio file and summarize the predictions",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runCompareAudio(settings, opts)
		},
	}

	cmd.Flags().StringVar(&opts.audioPath, "audio", "", "Path to a WAV or FLAC file to analyze")
	cmd.Flags().StringVar(&opts.backend, "backend", opts.backend, "Inference backend to use: auto, tflite, onnx, ncnn")
	cmd.Flags().StringVar(&opts.version, "version", opts.version, "BirdNET model version (for example: 2.4, 2.4-int8, 2.4-int8-cnn)")
	cmd.Flags().StringVar(&opts.modelPath, "model-path", "", "Explicit classifier model path (.tflite or .onnx)")
	cmd.Flags().StringVar(&opts.labelPath, "label-path", "", "Explicit label file path")
	cmd.Flags().StringVar(&opts.onnxRuntimePath, "onnx-runtime-path", opts.onnxRuntimePath, "Path to the ONNX Runtime shared library")
	cmd.Flags().StringVar(&opts.ncnnModelDir, "ncnn-model-dir", opts.ncnnModelDir, "Directory containing validated NCNN artifacts")
	cmd.Flags().BoolVar(&opts.ncnnUseVulkan, "ncnn-use-vulkan", opts.ncnnUseVulkan, "Enable Vulkan for the NCNN backend")
	cmd.Flags().IntVar(&opts.topK, "top-k", opts.topK, "Number of predictions to keep per chunk")
	cmd.Flags().IntVar(&opts.summaryK, "summary-k", opts.summaryK, "Number of species to keep in the final summary")
	cmd.Flags().IntVar(&opts.maxChunks, "max-chunks", 0, "Maximum number of chunks to analyze (0 = all chunks)")
	cmd.Flags().BoolVar(&opts.jsonOutput, "json", false, "Emit machine-readable JSON")
	cmd.Flags().BoolVar(&opts.summaryOnly, "summary-only", false, "Only emit the summary, not per-chunk results")

	return cmd
}

func runCompareAudio(baseSettings *conf.Settings, opts options) error {
	if strings.TrimSpace(opts.audioPath) == "" {
		return fmt.Errorf("--audio is required")
	}
	if opts.topK <= 0 {
		return fmt.Errorf("--top-k must be greater than 0")
	}
	if opts.summaryK <= 0 {
		return fmt.Errorf("--summary-k must be greater than 0")
	}

	settingsCopy := cloneSettings(baseSettings)
	applyOptions(&settingsCopy, opts)

	modelInfo, err := classifier.ResolveBirdNETModelInfoFromConfig(settingsCopy.BirdNET)
	if err != nil {
		return err
	}

	audioInfo, err := readfile.GetAudioInfo(opts.audioPath)
	if err != nil {
		return err
	}

	orchestrator, err := classifier.NewOrchestrator(&settingsCopy)
	if err != nil {
		return err
	}
	defer orchestrator.Delete()

	targetSampleRate := modelInfo.Spec.SampleRate
	clipDurationSeconds := modelInfo.Spec.ClipLength.Seconds()
	chunkSize := int(clipDurationSeconds * float64(targetSampleRate))
	resampleFn, closeResampler, err := buildResampleFunc(audioInfo.SampleRate, targetSampleRate)
	if err != nil {
		return err
	}
	if closeResampler != nil {
		defer closeResampler()
	}

	report := compareReport{
		Summary: compareSummary{
			AudioPath:            opts.audioPath,
			Backend:              normalizeBackendLabel(settingsCopy.BirdNET.Backend),
			Version:              settingsCopy.BirdNET.Version,
			ModelPath:            settingsCopy.BirdNET.ModelPath,
			LabelPath:            settingsCopy.BirdNET.LabelPath,
			ModelID:              orchestrator.ModelInfo.ID,
			AudioSampleRate:      audioInfo.SampleRate,
			ModelSampleRate:      targetSampleRate,
			ClipDurationSeconds:  clipDurationSeconds,
			OverlapSeconds:       settingsCopy.BirdNET.Overlap,
			TotalDurationSeconds: float64(audioInfo.TotalSamples) / float64(audioInfo.SampleRate),
		},
	}
	report.Summary.EstimatedChunkCount = estimateChunkCount(audioInfo, targetSampleRate, settingsCopy.BirdNET.Overlap)

	stepSamples := int((clipDurationSeconds - settingsCopy.BirdNET.Overlap) * float64(targetSampleRate))
	if stepSamples <= 0 {
		return fmt.Errorf("invalid overlap %.3f for clip duration %.3f", settingsCopy.BirdNET.Overlap, clipDurationSeconds)
	}

	chunkIndex := 0
	aggregates := make(map[string]*aggregateState)
	callback := func(samples []float32, isEOF bool) error {
		if len(samples) == 0 {
			return nil
		}
		if opts.maxChunks > 0 && chunkIndex >= opts.maxChunks {
			return errChunkLimitReached
		}

		results, predictErr := orchestrator.Predict(context.Background(), [][]float32{samples})
		if predictErr != nil {
			return predictErr
		}
		results = limitResults(results, opts.topK)

		startSeconds := float64(chunkIndex*stepSamples) / float64(targetSampleRate)
		endSeconds := startSeconds + clipDurationSeconds

		predictions := make([]predictionResult, 0, len(results))
		for _, result := range results {
			prediction := enrichPrediction(orchestrator, result)
			predictions = append(predictions, prediction)
			updateAggregate(aggregates, prediction)
		}

		if !opts.summaryOnly {
			report.Chunks = append(report.Chunks, chunkResult{
				Index:        chunkIndex,
				StartSeconds: startSeconds,
				EndSeconds:   endSeconds,
				EOF:          isEOF,
				Predictions:  predictions,
			})
		}

		chunkIndex++
		return nil
	}

	if err := streamAudioFile(opts.audioPath, chunkSize, settingsCopy.BirdNET.Overlap, targetSampleRate, resampleFn, callback); err != nil && !stdErrors.Is(err, errChunkLimitReached) {
		return err
	}

	report.Summary.ProcessedChunkCount = chunkIndex
	report.Summary.TopSpeciesByMax = finalizeAggregates(aggregates, opts.summaryK)

	if opts.jsonOutput {
		encoder := json.NewEncoder(os.Stdout)
		encoder.SetIndent("", "  ")
		return encoder.Encode(report)
	}

	printSummary(report)
	if !opts.summaryOnly {
		printChunks(report.Chunks)
	}

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

func estimateChunkCount(info readfile.AudioInfo, targetSampleRate int, overlap float64) int {
	duration := float64(info.TotalSamples) / float64(info.SampleRate)
	totalSamples := int(duration * float64(targetSampleRate))
	return readfile.GetTotalChunks(targetSampleRate, totalSamples, overlap)
}

func buildResampleFunc(sourceRate, targetRate int) (func([]float32, int, int) ([]float32, error), func() error, error) {
	resampler, err := iresample.NewResampler(sourceRate, targetRate)
	if err != nil {
		return nil, nil, err
	}
	if resampler == nil {
		return nil, nil, nil
	}

	resampleFn := func(samples []float32, _, _ int) ([]float32, error) {
		pcm := float32ToPCM16(samples)
		resampledPCM, resampleErr := resampler.ResampleInto(pcm)
		if resampleErr != nil {
			return nil, resampleErr
		}
		return pcm16ToFloat32(resampledPCM), nil
	}

	return resampleFn, resampler.Close, nil
}

func float32ToPCM16(samples []float32) []byte {
	pcm := make([]byte, len(samples)*2)
	for i, sample := range samples {
		value := math.Max(-1.0, math.Min(1.0, float64(sample)))
		scaled := int16(value * 32767.0)
		binary.LittleEndian.PutUint16(pcm[i*2:], uint16(scaled)) //nolint:gosec // intentional PCM bit reinterpretation
	}
	return pcm
}

func pcm16ToFloat32(pcm []byte) []float32 {
	samples := make([]float32, len(pcm)/2)
	for i := range samples {
		value := int16(binary.LittleEndian.Uint16(pcm[i*2:])) //nolint:gosec // intentional PCM bit reinterpretation
		samples[i] = float32(value) / 32768.0
	}
	return samples
}

func streamAudioFile(
	audioPath string,
	chunkSize int,
	overlap float64,
	targetSampleRate int,
	resampleFn func([]float32, int, int) ([]float32, error),
	callback readfile.AudioChunkCallback,
) error {
	file, err := os.Open(audioPath) //nolint:gosec // audioPath is a user-supplied CLI path
	if err != nil {
		return err
	}
	defer func() {
		_ = file.Close()
	}()

	switch strings.ToLower(filepath.Ext(audioPath)) {
	case readfile.ExtWAV:
		return readfile.ReadWAVBuffered(file, chunkSize, overlap, targetSampleRate, resampleFn, callback)
	case readfile.ExtFLAC:
		return readfile.ReadFLACBuffered(file, chunkSize, overlap, targetSampleRate, resampleFn, callback)
	default:
		return fmt.Errorf("unsupported audio format: %s", filepath.Ext(audioPath))
	}
}

func limitResults(results []datastore.Results, topK int) []datastore.Results {
	if len(results) <= topK {
		return results
	}
	return results[:topK]
}

func enrichPrediction(orchestrator *classifier.Orchestrator, result datastore.Results) predictionResult {
	scientificName, commonName := orchestrator.GetSpeciesWithScientificAndCommonName(result.Species)
	if scientificName == "" && commonName == "" {
		scientificName = result.Species
	}

	return predictionResult{
		Label:          result.Species,
		ScientificName: scientificName,
		CommonName:     commonName,
		Confidence:     result.Confidence,
	}
}

func updateAggregate(aggregates map[string]*aggregateState, prediction predictionResult) {
	entry, ok := aggregates[prediction.Label]
	if !ok {
		aggregates[prediction.Label] = &aggregateState{
			label:          prediction.Label,
			scientificName: prediction.ScientificName,
			commonName:     prediction.CommonName,
			maxConfidence:  prediction.Confidence,
			total:          float64(prediction.Confidence),
			chunks:         1,
		}
		return
	}

	if prediction.Confidence > entry.maxConfidence {
		entry.maxConfidence = prediction.Confidence
	}
	entry.total += float64(prediction.Confidence)
	entry.chunks++
}

func finalizeAggregates(aggregates map[string]*aggregateState, summaryK int) []aggregateResult {
	results := make([]aggregateResult, 0, len(aggregates))
	for _, aggregate := range aggregates {
		results = append(results, aggregateResult{
			Label:          aggregate.label,
			ScientificName: aggregate.scientificName,
			CommonName:     aggregate.commonName,
			MaxConfidence:  aggregate.maxConfidence,
			AvgConfidence:  float32(aggregate.total / float64(aggregate.chunks)),
			Chunks:         aggregate.chunks,
		})
	}

	sort.Slice(results, func(i, j int) bool {
		if results[i].MaxConfidence != results[j].MaxConfidence {
			return results[i].MaxConfidence > results[j].MaxConfidence
		}
		if results[i].AvgConfidence != results[j].AvgConfidence {
			return results[i].AvgConfidence > results[j].AvgConfidence
		}
		return results[i].Label < results[j].Label
	})

	if len(results) > summaryK {
		results = results[:summaryK]
	}
	return results
}

func printSummary(report compareReport) {
	summary := report.Summary
	fmt.Printf("Audio:    %s\n", summary.AudioPath)
	fmt.Printf("Backend:  %s\n", summary.Backend)
	fmt.Printf("Version:  %s\n", summary.Version)
	fmt.Printf("Model ID: %s\n", summary.ModelID)
	if summary.ModelPath != "" {
		fmt.Printf("Model:    %s\n", summary.ModelPath)
	}
	if summary.LabelPath != "" {
		fmt.Printf("Labels:   %s\n", summary.LabelPath)
	}
	fmt.Printf("Audio SR: %d Hz\n", summary.AudioSampleRate)
	fmt.Printf("Model SR: %d Hz\n", summary.ModelSampleRate)
	fmt.Printf("Chunks:   %d/%d processed\n", summary.ProcessedChunkCount, summary.EstimatedChunkCount)
	fmt.Printf("Duration: %.2fs\n", summary.TotalDurationSeconds)
	fmt.Printf("\nTop species by max confidence:\n")
	for i, species := range summary.TopSpeciesByMax {
		name := species.ScientificName
		if species.CommonName != "" {
			name = fmt.Sprintf("%s (%s)", species.CommonName, species.ScientificName)
		}
		if name == "" {
			name = species.Label
		}
		fmt.Printf(
			"  %2d. %-48s max=%.3f avg=%.3f chunks=%d\n",
			i+1,
			name,
			species.MaxConfidence,
			species.AvgConfidence,
			species.Chunks,
		)
	}
}

func printChunks(chunks []chunkResult) {
	if len(chunks) == 0 {
		return
	}
	fmt.Printf("\nPer-chunk predictions:\n")
	for _, chunk := range chunks {
		fmt.Printf("  Chunk %d [%.2fs - %.2fs]\n", chunk.Index+1, chunk.StartSeconds, chunk.EndSeconds)
		for _, prediction := range chunk.Predictions {
			name := prediction.ScientificName
			if prediction.CommonName != "" {
				name = fmt.Sprintf("%s (%s)", prediction.CommonName, prediction.ScientificName)
			}
			if name == "" {
				name = prediction.Label
			}
			fmt.Printf("    - %-48s %.3f\n", name, prediction.Confidence)
		}
	}
}
