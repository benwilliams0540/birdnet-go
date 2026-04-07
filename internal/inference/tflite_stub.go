//go:build !notflite

package inference

import (
	"fmt"
	"runtime"

	"github.com/tphakala/birdnet-go/internal/cpuspec"
	tflitelib "github.com/tphakala/go-tflite"
	"github.com/tphakala/go-tflite/delegates/xnnpack"
)

// LogFunc is a callback for logging messages from the inference backend.
type LogFunc func(msg string)

// TFLiteClassifierOptions configures the TFLite species classifier.
type TFLiteClassifierOptions struct {
	Threads    int
	UseXNNPACK bool
	ErrorFunc  LogFunc
	WarnFunc   LogFunc
}

type tfliteClassifier struct {
	interpreter *tflitelib.Interpreter
	numSpecies  int
}

// NewTFLiteClassifier creates a Classifier backed by a TensorFlow Lite model.
// Returns the classifier and the resolved thread count for logging.
func NewTFLiteClassifier(modelData []byte, opts TFLiteClassifierOptions) (Classifier, int, error) {
	model := tflitelib.NewModel(modelData)
	if model == nil {
		return nil, 0, fmt.Errorf("cannot create TFLite model from data (%d bytes)", len(modelData))
	}

	threads := determineTFLiteThreadCount(opts.Threads)
	options := tflitelib.NewInterpreterOptions()

	if opts.UseXNNPACK {
		delegate := xnnpack.New(xnnpack.DelegateOptions{
			NumThreads: int32(max(1, threads-1)), //nolint:gosec // bounded by CPU count
		})
		if delegate == nil {
			if opts.WarnFunc != nil {
				opts.WarnFunc("Failed to create XNNPACK delegate, falling back to default CPU")
			}
			options.SetNumThread(threads)
		} else {
			options.AddDelegate(delegate)
			options.SetNumThread(1)
		}
	} else {
		options.SetNumThread(threads)
	}

	options.SetErrorReporter(func(msg string, _ any) {
		if opts.ErrorFunc != nil {
			opts.ErrorFunc(msg)
		}
	}, nil)

	interpreter := tflitelib.NewInterpreter(model, options)
	if interpreter == nil {
		return nil, 0, fmt.Errorf("cannot create TFLite interpreter")
	}
	if status := interpreter.AllocateTensors(); status != tflitelib.OK {
		return nil, 0, fmt.Errorf("TFLite tensor allocation failed")
	}

	outputTensor := interpreter.GetOutputTensor(0)
	if outputTensor == nil {
		return nil, 0, fmt.Errorf("cannot get output tensor from TFLite model")
	}

	runtime.GC()

	return &tfliteClassifier{
		interpreter: interpreter,
		numSpecies:  outputTensor.Dim(outputTensor.NumDims() - 1),
	}, threads, nil
}

func (c *tfliteClassifier) Predict(samples []float32) ([]float32, error) {
	inputTensor := c.interpreter.GetInputTensor(0)
	if inputTensor == nil {
		return nil, fmt.Errorf("cannot get input tensor")
	}

	inputSlice := inputTensor.Float32s()
	if len(samples) != len(inputSlice) {
		return nil, fmt.Errorf("input size mismatch: expected %d samples, got %d", len(inputSlice), len(samples))
	}
	copy(inputSlice, samples)

	if status := c.interpreter.Invoke(); status != tflitelib.OK {
		return nil, fmt.Errorf("TFLite invoke failed: %v", status)
	}

	outputTensor := c.interpreter.GetOutputTensor(0)
	if outputTensor == nil {
		return nil, fmt.Errorf("cannot get output tensor")
	}
	predictions := make([]float32, c.numSpecies)
	copy(predictions, outputTensor.Float32s()[:c.numSpecies])
	return predictions, nil
}

func (c *tfliteClassifier) NumSpecies() int {
	return c.numSpecies
}

func (c *tfliteClassifier) Close() {
	c.interpreter = nil
}

// TFLiteRangeFilterOptions configures the TFLite range filter.
type TFLiteRangeFilterOptions struct {
	ErrorFunc LogFunc
}

type tfliteRangeFilter struct {
	interpreter *tflitelib.Interpreter
	numSpecies  int
}

const rangeFilterInputSize = 3

// NewTFLiteRangeFilter creates a RangeFilter backed by a TensorFlow Lite meta model.
func NewTFLiteRangeFilter(modelData []byte, opts TFLiteRangeFilterOptions) (RangeFilter, error) {
	model := tflitelib.NewModel(modelData)
	if model == nil {
		return nil, fmt.Errorf("cannot create TFLite range filter model from data (%d bytes)", len(modelData))
	}

	options := tflitelib.NewInterpreterOptions()
	options.SetNumThread(1)
	options.SetErrorReporter(func(msg string, _ any) {
		if opts.ErrorFunc != nil {
			opts.ErrorFunc(msg)
		}
	}, nil)

	interpreter := tflitelib.NewInterpreter(model, options)
	if interpreter == nil {
		return nil, fmt.Errorf("cannot create TFLite range filter interpreter")
	}
	if status := interpreter.AllocateTensors(); status != tflitelib.OK {
		return nil, fmt.Errorf("TFLite range filter tensor allocation failed: %v", status)
	}

	outputTensor := interpreter.GetOutputTensor(0)
	if outputTensor == nil {
		return nil, fmt.Errorf("cannot get output tensor from TFLite range filter model")
	}

	runtime.GC()

	return &tfliteRangeFilter{
		interpreter: interpreter,
		numSpecies:  outputTensor.Dim(outputTensor.NumDims() - 1),
	}, nil
}

func (r *tfliteRangeFilter) Predict(latitude, longitude, week float32) ([]float32, error) {
	input := r.interpreter.GetInputTensor(0)
	if input == nil {
		return nil, fmt.Errorf("cannot get range filter input tensor")
	}

	float32s := input.Float32s()
	if len(float32s) < rangeFilterInputSize {
		return nil, fmt.Errorf("range filter input tensor too small: need %d, have %d", rangeFilterInputSize, len(float32s))
	}

	float32s[0] = latitude
	float32s[1] = longitude
	float32s[2] = week

	if status := r.interpreter.Invoke(); status != tflitelib.OK {
		return nil, fmt.Errorf("TFLite range filter invoke failed: %v", status)
	}

	output := r.interpreter.GetOutputTensor(0)
	if output == nil {
		return nil, fmt.Errorf("cannot get range filter output tensor")
	}
	scores := make([]float32, r.numSpecies)
	copy(scores, output.Float32s()[:r.numSpecies])
	return scores, nil
}

func (r *tfliteRangeFilter) NumSpecies() int {
	return r.numSpecies
}

func (r *tfliteRangeFilter) Close() {
	r.interpreter = nil
}

func determineTFLiteThreadCount(configuredThreads int) int {
	systemCPUCount := runtime.NumCPU()
	if configuredThreads == 0 {
		spec := cpuspec.GetCPUSpec()
		optimalThreads := spec.GetOptimalThreadCount()
		if optimalThreads > 0 {
			return min(optimalThreads, systemCPUCount)
		}
		return systemCPUCount
	}
	if configuredThreads > systemCPUCount {
		return systemCPUCount
	}
	return configuredThreads
}
