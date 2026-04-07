//go:build ncnn

package ncnn

/*
#include <stdlib.h>
#include "ncnn/c_api.h"

void birdnet_ncnn_register_custom_layers(ncnn_net_t net);
void birdnet_ncnn_frontend_set_filterbanks(const unsigned char* spec1, int spec1_len, const unsigned char* spec2, int spec2_len);
int birdnet_ncnn_frontend_compute(const float* samples, int sample_count, float* out, int out_count);
*/
import "C"

import (
	"fmt"
	"unsafe"
)

var (
	melFBSpec1C unsafe.Pointer
	melFBSpec2C unsafe.Pointer
)

func init() {
	melFBSpec1C = C.CBytes(melFBSpec1Raw)
	melFBSpec2C = C.CBytes(melFBSpec2Raw)

	C.birdnet_ncnn_frontend_set_filterbanks(
		(*C.uchar)(melFBSpec1C),
		C.int(len(melFBSpec1Raw)),
		(*C.uchar)(melFBSpec2C),
		C.int(len(melFBSpec2Raw)),
	)
}

func registerBirdNETCustomLayers(net C.ncnn_net_t) {
	C.birdnet_ncnn_register_custom_layers(net)
}

func computeBirdNETFrontendWithCustomLayer(samples []float32) ([]float32, error) {
	if len(samples) != splitAudioLen {
		return nil, fmt.Errorf("bad audio size: %d", len(samples))
	}

	out := make([]float32, splitChannels*splitBins*splitFrames)
	status := C.birdnet_ncnn_frontend_compute(
		(*C.float)(unsafe.Pointer(&samples[0])),
		C.int(len(samples)),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(len(out)),
	)
	if status != 0 {
		return nil, fmt.Errorf("BirdNETFrontend custom layer compute failed: %d", int(status))
	}

	return out, nil
}
