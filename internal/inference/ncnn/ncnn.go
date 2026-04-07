//go:build ncnn

// Package ncnn provides NCNN-based inference for bird species classification.
package ncnn

/*
#ifndef NCNN_VULKAN
#define NCNN_VULKAN 1
#endif
#include <stdlib.h>
#include "ncnn/c_api.h"

// Define has_vulkan here to be absolutely sure we check the macro visibility
int has_vulkan_support() {
#ifdef NCNN_VULKAN
    return 1;
#else
    return 0;
#endif
}
*/
import "C"
import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"unsafe"
)

type inferenceMode int

const (
	modeUnknown  inferenceMode = 0
	modeRawAudio inferenceMode = 1 // PNNX style: 1x144000 raw samples (full graph or wrapped split model)
	modeSplitCNN inferenceMode = 2 // Split V2.4 CNN: 1x2x96x511 channel-first tensor
)

const (
	inputLenRaw       = 144000
	splitModelInputW  = splitFrames
	splitModelInputH  = splitBins
	splitModelInputC  = splitChannels
	numSpeciesDefault = 6522
)

type Classifier struct {
	net         C.ncnn_net_t
	mode        inferenceMode
	inputBlob   string
	outputBlob  string
	numChannels int
	numSpecies  int
	useVulkan   bool
	debug       bool
	debugBlob   string
}

// Options configures an NCNN classifier.
type Options struct {
	ModelDir  string
	Threads   int
	UseVulkan bool
}

// New creates a new NCNN Classifier.
func New(opts Options) (*Classifier, error) {
	net := C.ncnn_net_create()
	registerBirdNETCustomLayers(net)

	ncnnOpt := C.ncnn_option_create()
	if opts.Threads > 0 {
		C.ncnn_option_set_num_threads(ncnnOpt, C.int(opts.Threads))
	}

	if opts.UseVulkan {
		if int(C.has_vulkan_support()) == 1 {
			C.ncnn_option_set_use_vulkan_compute(ncnnOpt, 1)
			C.ncnn_net_set_vulkan_device(net, 0)
		} else {
			fmt.Printf("NCNN WARN: Vulkan requested but has_vulkan_support=0. Falling back to CPU.\n")
			opts.UseVulkan = false
		}
	}

	C.ncnn_net_set_option(net, ncnnOpt)
	C.ncnn_option_destroy(ncnnOpt)

	var paramPath, binPath, paramFile string
	patterns := [][]string{
		{"birdnet.pnnx.param", "birdnet.pnnx.bin"},
		{"birdnet_cnn_only.param", "birdnet_cnn_only.bin"},
		{"birdnet_cnn.param", "birdnet_cnn.bin"},
		{"birdnet_v2_cnn_sim.ncnn.param", "birdnet_v2_cnn_sim.ncnn.bin"},
		{"model.param", "model.bin"},
	}

	var loaded bool
	var wrappedSplitModel bool
	for _, p := range patterns {
		paramFile = p[0]
		paramPath = filepath.Join(opts.ModelDir, p[0])
		binPath = filepath.Join(opts.ModelDir, p[1])
		wrappedSplitModel = false

		fmt.Printf("NCNN PROBING: %s\n", paramPath)
		var retParam C.int
		if shouldWrapSplitModelParam(paramFile) {
			wrappedParam, err := buildBirdNETFrontendWrappedParamFromFile(paramPath)
			if err != nil {
				fmt.Printf("NCNN ERROR: wrap_param(%s) failed: %v\n", paramFile, err)
				continue
			}
			cParam := C.CString(wrappedParam)
			retParam = C.ncnn_net_load_param_memory(net, cParam)
			C.free(unsafe.Pointer(cParam))
			if retParam == 0 {
				wrappedSplitModel = true
				fmt.Printf("NCNN INFO: using BirdNETFrontend custom layer wrapper for %s\n", paramFile)
			}
		} else {
			cParam := C.CString(paramPath)
			retParam = C.ncnn_net_load_param(net, cParam)
			C.free(unsafe.Pointer(cParam))
		}
		if retParam != 0 {
			fmt.Printf("NCNN ERROR: load_param(%s) failed: %v\n", paramFile, retParam)
			continue
		}

		cBin := C.CString(binPath)
		retBin := C.ncnn_net_load_model(net, cBin)
		C.free(unsafe.Pointer(cBin))
		if retBin != 0 {
			fmt.Printf("NCNN ERROR: load_model(%s) failed: %v\n", p[1], retBin)
			continue
		}

		loaded = true
		break
	}

	if !loaded {
		C.ncnn_net_destroy(net)
		return nil, fmt.Errorf("failed to load ncnn model from %s", opts.ModelDir)
	}

	c := &Classifier{
		net:        net,
		numSpecies: numSpeciesDefault,
		useVulkan:  opts.UseVulkan,
		debug:      os.Getenv("BIRDNET_NCNN_DEBUG") != "",
		debugBlob:  os.Getenv("BIRDNET_NCNN_DEBUG_BLOB"),
	}

	if paramFile == "birdnet.pnnx.param" || wrappedSplitModel {
		c.mode = modeRawAudio
		c.inputBlob = "in0"
		c.outputBlob = "out0"
	} else {
		c.mode = modeSplitCNN
		c.outputBlob = "Identity"
	}

	inputCount := int(C.ncnn_net_get_input_count(net))
	if inputCount != 1 {
		C.ncnn_net_destroy(net)
		return nil, fmt.Errorf("unsupported ncnn model input count %d for %s; regenerate birdnet_cnn_only.param/bin", inputCount, paramFile)
	}
	outputCount := int(C.ncnn_net_get_output_count(net))
	if outputCount < 1 {
		C.ncnn_net_destroy(net)
		return nil, fmt.Errorf("ncnn model %s has no outputs", paramFile)
	}
	c.inputBlob = C.GoString(C.ncnn_net_get_input_name(net, 0))
	c.outputBlob = C.GoString(C.ncnn_net_get_output_name(net, 0))

	fmt.Printf("NCNN LOADED: Model=%s, Mode=%d, Input=%s, Output=%s, Vulkan=%v\n",
		paramFile, c.mode, c.inputBlob, c.outputBlob, c.useVulkan)

	return c, nil
}

// Predict runs inference.
func (c *Classifier) Predict(samples []float32) ([]float32, error) {
	if len(samples) != inputLenRaw {
		return nil, fmt.Errorf("bad audio size: %d", len(samples))
	}

	var inMat C.ncnn_mat_t
	if c.mode == modeRawAudio {
		inMat = C.ncnn_mat_create_1d(C.int(inputLenRaw), nil)
		copy(unsafe.Slice((*float32)(C.ncnn_mat_get_data(inMat)), inputLenRaw), samples)
	} else {
		cnnInput := ComputeSplitCNNInput(samples)
		inMat = C.ncnn_mat_create_3d(C.int(splitModelInputW), C.int(splitModelInputH), C.int(splitModelInputC), nil)
		channelSize := splitModelInputW * splitModelInputH
		for channel := 0; channel < splitModelInputC; channel++ {
			dst := unsafe.Slice(
				(*float32)(C.ncnn_mat_get_channel_data(inMat, C.int(channel))),
				channelSize,
			)
			srcStart := channel * channelSize
			copy(dst, cnnInput[srcStart:srcStart+channelSize])
		}
		if c.debug {
			logTensorStats("NCNN DEBUG input", matInfoFromMat(inMat), cnnInput)
		}
	}

	ex := C.ncnn_extractor_create(c.net)
	defer C.ncnn_extractor_destroy(ex)

	cInputName := C.CString(c.inputBlob)
	defer C.free(unsafe.Pointer(cInputName))
	C.ncnn_extractor_input(ex, cInputName, inMat)
	C.ncnn_mat_destroy(inMat)

	var outMat C.ncnn_mat_t
	cOutputName := C.CString(c.outputBlob)
	defer C.free(unsafe.Pointer(cOutputName))
	if ret := C.ncnn_extractor_extract(ex, cOutputName, &outMat); ret != 0 {
		return nil, fmt.Errorf("ncnn extract failed: %d", int(ret))
	}
	defer C.ncnn_mat_destroy(outMat)

	predictions := flattenMatFloat32(outMat)
	if c.debug {
		logTensorStats("NCNN DEBUG output", matInfoFromMat(outMat), predictions)
	}
	if c.debugBlob != "" {
		logDebugBlob(ex, c.debugBlob)
	}

	var nanCount int
	for i, v := range predictions {
		if v != v || v > 1e15 || v < -1e15 {
			nanCount++
			predictions[i] = 0
		}
	}
	if nanCount > 0 {
		fmt.Printf("NCNN ERROR: Blocked %d NaNs/Infs\n", nanCount)
		return nil, fmt.Errorf("ncnn numerical instability")
	}

	return predictions, nil
}

func (c *Classifier) NumSpecies() int { return c.numSpecies }
func (c *Classifier) Close()          { C.ncnn_net_destroy(c.net) }
func (c *Classifier) Name() string    { return "ncnn-vulkan" }

type matInfo struct {
	dims     int
	w        int
	h        int
	d        int
	c        int
	elempack int
	cstep    int
}

func matInfoFromMat(mat C.ncnn_mat_t) matInfo {
	return matInfo{
		dims:     int(C.ncnn_mat_get_dims(mat)),
		w:        int(C.ncnn_mat_get_w(mat)),
		h:        int(C.ncnn_mat_get_h(mat)),
		d:        int(C.ncnn_mat_get_d(mat)),
		c:        int(C.ncnn_mat_get_c(mat)),
		elempack: int(C.ncnn_mat_get_elempack(mat)),
		cstep:    int(C.ncnn_mat_get_cstep(mat)),
	}
}

func flattenMatFloat32(mat C.ncnn_mat_t) []float32 {
	info := matInfoFromMat(mat)
	if info.elempack <= 0 {
		info.elempack = 1
	}

	switch info.dims {
	case 1:
		total := info.w * info.elempack
		out := make([]float32, total)
		copy(out, unsafe.Slice((*float32)(C.ncnn_mat_get_data(mat)), total))
		return out
	case 2:
		total := info.w * info.h * info.elempack
		out := make([]float32, total)
		copy(out, unsafe.Slice((*float32)(C.ncnn_mat_get_data(mat)), total))
		return out
	case 3, 4:
		perChannel := info.w * info.h * max(1, info.d) * info.elempack
		total := perChannel * info.c
		out := make([]float32, total)
		for channel := 0; channel < info.c; channel++ {
			src := unsafe.Slice(
				(*float32)(C.ncnn_mat_get_channel_data(mat, C.int(channel))),
				perChannel,
			)
			copy(out[channel*perChannel:(channel+1)*perChannel], src)
		}
		return out
	default:
		return nil
	}
}

func logTensorStats(prefix string, info matInfo, values []float32) {
	if len(values) == 0 {
		fmt.Printf("%s dims=%d w=%d h=%d d=%d c=%d elempack=%d cstep=%d len=0\n",
			prefix, info.dims, info.w, info.h, info.d, info.c, info.elempack, info.cstep)
		return
	}

	minValue := float32(math.MaxFloat32)
	maxValue := float32(-math.MaxFloat32)
	var sum float64
	for _, value := range values {
		if value < minValue {
			minValue = value
		}
		if value > maxValue {
			maxValue = value
		}
		sum += float64(value)
	}

	previewCount := min(8, len(values))
	fmt.Printf(
		"%s dims=%d w=%d h=%d d=%d c=%d elempack=%d cstep=%d len=%d min=%0.6f max=%0.6f mean=%0.6f first=%v\n",
		prefix,
		info.dims,
		info.w,
		info.h,
		info.d,
		info.c,
		info.elempack,
		info.cstep,
		len(values),
		minValue,
		maxValue,
		sum/float64(len(values)),
		values[:previewCount],
	)
}

func logDebugBlob(ex C.ncnn_extractor_t, blobName string) {
	cBlobName := C.CString(blobName)
	defer C.free(unsafe.Pointer(cBlobName))

	var blobMat C.ncnn_mat_t
	ret := C.ncnn_extractor_extract(ex, cBlobName, &blobMat)
	if ret != 0 {
		fmt.Printf("NCNN DEBUG blob=%q extract failed: %d\n", blobName, int(ret))
		return
	}
	defer C.ncnn_mat_destroy(blobMat)

	values := flattenMatFloat32(blobMat)
	logTensorStats(fmt.Sprintf("NCNN DEBUG blob=%q", blobName), matInfoFromMat(blobMat), values)
}
