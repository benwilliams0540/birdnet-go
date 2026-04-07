package ncnn

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const (
	birdNETFrontendEnvVar     = "BIRDNET_NCNN_USE_FRONTEND_LAYER"
	birdNETFrontendLayerType  = "BirdNETFrontend"
	birdNETFrontendLayerName  = "birdnet_frontend"
	birdNETFrontendOutputBlob = "birdnet_frontend_out"
)

func shouldWrapSplitModelParam(paramFile string) bool {
	if filepath.Base(paramFile) != "birdnet_cnn_only.param" {
		return false
	}

	value := strings.TrimSpace(strings.ToLower(os.Getenv(birdNETFrontendEnvVar)))
	switch value {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func buildBirdNETFrontendWrappedParamFromFile(paramPath string) (string, error) {
	data, err := os.ReadFile(paramPath)
	if err != nil {
		return "", err
	}
	return wrapSplitModelParam(string(data))
}

func wrapSplitModelParam(paramData string) (string, error) {
	normalized := strings.ReplaceAll(paramData, "\r\n", "\n")
	normalized = strings.TrimSpace(normalized)
	lines := strings.Split(normalized, "\n")
	if len(lines) < 4 {
		return "", fmt.Errorf("invalid NCNN param: expected at least 4 lines")
	}

	headerFields := strings.Fields(lines[1])
	if len(headerFields) != 2 {
		return "", fmt.Errorf("invalid NCNN param header %q", lines[1])
	}

	layerCount, err := strconv.Atoi(headerFields[0])
	if err != nil {
		return "", fmt.Errorf("invalid NCNN param layer count: %w", err)
	}
	blobCount, err := strconv.Atoi(headerFields[1])
	if err != nil {
		return "", fmt.Errorf("invalid NCNN param blob count: %w", err)
	}

	inputFields := strings.Fields(lines[2])
	if len(inputFields) < 5 || inputFields[0] != "Input" {
		return "", fmt.Errorf("invalid NCNN input line %q", lines[2])
	}
	inputBlob := inputFields[len(inputFields)-1]

	renamed := make([]string, 0, len(lines)+1)
	renamed = append(renamed, lines[0])
	renamed = append(renamed, fmt.Sprintf("%d %d", layerCount+1, blobCount+1))
	renamed = append(renamed, lines[2])
	renamed = append(renamed, fmt.Sprintf("%s %s 1 1 %s %s", birdNETFrontendLayerType, birdNETFrontendLayerName, inputBlob, birdNETFrontendOutputBlob))

	for _, line := range lines[3:] {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		renamed = append(renamed, rewriteBlobName(line, inputBlob, birdNETFrontendOutputBlob))
	}

	return strings.Join(renamed, "\n") + "\n", nil
}

func rewriteBlobName(line, from, to string) string {
	fields := strings.Fields(line)
	for i := range fields {
		if fields[i] == from {
			fields[i] = to
		}
	}
	return strings.Join(fields, " ")
}
