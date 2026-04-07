package classifier

import (
	"os"
	"path/filepath"
	"strings"
)

// NCNNValidationMarkerName is written alongside NCNN artifacts only after
// they have passed parity checks against the baseline BirdNET outputs.
const NCNNValidationMarkerName = "birdnet-go.ncnn-validated"

// NCNNModelPatterns lists recognized NCNN classifier file pairs.
var NCNNModelPatterns = [][]string{
	{"birdnet.pnnx.param", "birdnet.pnnx.bin"},
	{"birdnet_cnn_only.param", "birdnet_cnn_only.bin"},
	{"birdnet_cnn.param", "birdnet_cnn.bin"},
	{"birdnet_v2_cnn_sim.ncnn.param", "birdnet_v2_cnn_sim.ncnn.bin"},
	{"model.param", "model.bin"},
}

// NCNNModelDirStatus describes whether a directory contains recognized and
// validated NCNN artifacts that BirdNET-Go is willing to use.
type NCNNModelDirStatus struct {
	Dir              string
	ParamFile        string
	BinFile          string
	ValidationMarker string
	Found            bool
	Validated        bool
}

// InspectNCNNModelDir reports whether dir contains a recognized NCNN model
// pair and the validation marker required for production use.
func InspectNCNNModelDir(dir string) NCNNModelDirStatus {
	trimmedDir := strings.TrimSpace(dir)
	status := NCNNModelDirStatus{
		Dir:              trimmedDir,
		ValidationMarker: NCNNValidationMarkerName,
	}
	if trimmedDir == "" {
		return status
	}

	for _, pair := range NCNNModelPatterns {
		paramPath := filepath.Join(trimmedDir, pair[0])
		binPath := filepath.Join(trimmedDir, pair[1])
		if _, err := os.Stat(paramPath); err == nil {
			if _, err := os.Stat(binPath); err == nil {
				status.Found = true
				status.ParamFile = pair[0]
				status.BinFile = pair[1]
				break
			}
		}
	}
	if !status.Found {
		return status
	}

	validationPath := filepath.Join(trimmedDir, NCNNValidationMarkerName)
	if _, err := os.Stat(validationPath); err == nil {
		status.Validated = true
	}

	return status
}

// NCNNModelDirReady reports whether dir contains both recognized NCNN model
// files and the explicit validation marker required for use.
func NCNNModelDirReady(dir string) bool {
	status := InspectNCNNModelDir(dir)
	return status.Found && status.Validated
}
