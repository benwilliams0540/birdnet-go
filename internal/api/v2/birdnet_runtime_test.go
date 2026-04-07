package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/labstack/echo/v4"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tphakala/birdnet-go/internal/classifier"
	"github.com/tphakala/birdnet-go/internal/conf"
)

func TestBuildBirdNETVersionCapabilitiesExcludesUnavailableBackends(t *testing.T) {
	t.Parallel()

	versions := buildBirdNETVersionCapabilities(conf.BirdNETConfig{})
	require.NotEmpty(t, versions)

	for _, version := range versions {
		assert.Empty(t, version.ValidBackends)
	}
}

func TestAssessBirdNETChangeRejectsInvalidONNXSelection(t *testing.T) {
	t.Parallel()

	assessment := assessBirdNETChange(
		conf.BirdNETConfig{},
		conf.BirdNETConfig{
			Version: "2.4",
			Backend: "onnx",
		},
	)

	assert.False(t, assessment.Valid)
	assert.Equal(t, birdnetChangeModeInvalid, assessment.ChangeMode)
	assert.Contains(t, assessment.Reason, "explicit .onnx model path")
}

func TestAssessBirdNETChangeRequiresRestartWhenBackendMissingFromBuild(t *testing.T) {
	t.Parallel()

	assessment := assessBirdNETChange(
		conf.BirdNETConfig{},
		conf.BirdNETConfig{
			Version:   "2.4-int8-cnn",
			Backend:   "onnx",
			ModelPath: "/tmp/birdnet_int8_cnn.onnx",
		},
	)

	assert.True(t, assessment.Valid)
	assert.True(t, assessment.RestartRequired)
	assert.Equal(t, birdnetChangeModeRestartRequired, assessment.ChangeMode)
	assert.Equal(t, "onnx", assessment.EffectiveBackend)
	assert.Contains(t, assessment.Reason, "not built with ONNX support")
}

func TestClassifyBirdNETConfigChangeBackendFamilySwitchRequiresRestart(t *testing.T) {
	t.Parallel()

	oldCfg := conf.BirdNETConfig{
		Version:   "2.4-int8-cnn",
		Backend:   "onnx",
		ModelPath: "/tmp/birdnet_int8_cnn.onnx",
	}

	newCfg := oldCfg
	newCfg.Backend = "ncnn"
	newCfg.NCNNModelDir = "/tmp/ncnn"

	decision := classifyBirdNETConfigChange(oldCfg, newCfg)

	assert.True(t, decision.changed)
	assert.Equal(t, birdnetBackendRestartReason, decision.restartReason)
	assert.Empty(t, decision.action)
}

func TestClassifyBirdNETConfigChangeSameBackendCanHotReload(t *testing.T) {
	t.Parallel()

	oldCfg := conf.BirdNETConfig{
		Version:   "2.4",
		Backend:   "tflite",
		ModelPath: "/tmp/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite",
		Locale:    "en",
	}

	newCfg := oldCfg
	newCfg.Locale = "fi"

	decision := classifyBirdNETConfigChange(oldCfg, newCfg)

	assert.True(t, decision.changed)
	assert.Equal(t, SignalReloadModel, decision.action)
	assert.Empty(t, decision.restartReason)
}

func TestGetBirdNETCapabilitiesReportsRequestedRuntimeState(t *testing.T) {
	t.Parallel()

	e := echo.New()
	controller := &Controller{
		Echo:     e,
		Settings: getTestSettings(t),
	}

	req := httptest.NewRequest(
		http.MethodGet,
		"/api/v2/system/birdnet/capabilities?backend=onnx&version=2.4-int8-cnn&modelPath=/tmp/birdnet_int8_cnn.onnx",
		nil,
	)
	rec := httptest.NewRecorder()
	ctx := e.NewContext(req, rec)

	err := controller.GetBirdNETCapabilities(ctx)
	require.NoError(t, err)
	require.Equal(t, http.StatusOK, rec.Code)

	var response birdnetCapabilitiesResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &response))

	assert.Equal(t, "onnx", response.RequestedChange.RequestedBackend)
	assert.Equal(t, "onnx", response.RequestedChange.EffectiveBackend)
	assert.True(t, response.RequestedChange.Valid)
	assert.True(t, response.RequestedChange.RestartRequired)
	assert.Equal(t, birdnetChangeModeRestartRequired, response.RequestedChange.ChangeMode)
	assert.Contains(t, response.RequestedChange.Reason, "not built with ONNX support")
}

func TestValidateBirdNETConfigSelectionAcceptsScannedNCNNModelDir(t *testing.T) {
	t.Parallel()

	modelDir := t.TempDir()
	paramPath := filepath.Join(modelDir, "birdnet_cnn.param")
	binPath := filepath.Join(modelDir, "birdnet_cnn.bin")

	require.NoError(t, os.WriteFile(paramPath, []byte("param"), 0o600))
	require.NoError(t, os.WriteFile(binPath, []byte("bin"), 0o600))
	require.NoError(t, os.WriteFile(filepath.Join(modelDir, classifier.NCNNValidationMarkerName), []byte("validated"), 0o600))

	err := validateBirdNETConfigSelection(conf.BirdNETConfig{
		Version:      "2.4-int8-cnn",
		Backend:      "ncnn",
		ModelPath:    "/tmp/birdnet_int8_cnn.onnx",
		NCNNModelDir: modelDir,
	})

	require.NoError(t, err)
}

func TestValidateBirdNETConfigSelectionAcceptsSplitNCNNModelDir(t *testing.T) {
	t.Parallel()

	modelDir := t.TempDir()
	paramPath := filepath.Join(modelDir, "birdnet_cnn_only.param")
	binPath := filepath.Join(modelDir, "birdnet_cnn_only.bin")

	require.NoError(t, os.WriteFile(paramPath, []byte("param"), 0o600))
	require.NoError(t, os.WriteFile(binPath, []byte("bin"), 0o600))
	require.NoError(t, os.WriteFile(filepath.Join(modelDir, classifier.NCNNValidationMarkerName), []byte("validated"), 0o600))

	err := validateBirdNETConfigSelection(conf.BirdNETConfig{
		Version:      "2.4-int8-cnn",
		Backend:      "ncnn",
		ModelPath:    "/tmp/birdnet_int8_cnn.onnx",
		NCNNModelDir: modelDir,
	})

	require.NoError(t, err)
}

func TestValidateBirdNETConfigSelectionRejectsUnvalidatedNCNNModelDir(t *testing.T) {
	t.Parallel()

	modelDir := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(modelDir, "birdnet_cnn.param"), []byte("param"), 0o600))
	require.NoError(t, os.WriteFile(filepath.Join(modelDir, "birdnet_cnn.bin"), []byte("bin"), 0o600))

	err := validateBirdNETConfigSelection(conf.BirdNETConfig{
		Version:      "2.4-int8-cnn",
		Backend:      "ncnn",
		ModelPath:    "/tmp/birdnet_int8_cnn.onnx",
		NCNNModelDir: modelDir,
	})

	require.Error(t, err)
	assert.Contains(t, err.Error(), classifier.NCNNValidationMarkerName)
}

func TestBuildBirdNETVersionCapabilitiesExcludesUnvalidatedNCNNBackend(t *testing.T) {
	t.Parallel()

	modelDir := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(modelDir, "birdnet_cnn.param"), []byte("param"), 0o600))
	require.NoError(t, os.WriteFile(filepath.Join(modelDir, "birdnet_cnn.bin"), []byte("bin"), 0o600))

	versions := buildBirdNETVersionCapabilities(conf.BirdNETConfig{
		Version:      "2.4-int8-cnn",
		ModelPath:    "/tmp/birdnet_int8_cnn.onnx",
		NCNNModelDir: modelDir,
	})

	var version24 birdnetVersionCapability
	for _, version := range versions {
		if version.Value == "2.4-int8-cnn" {
			version24 = version
			break
		}
	}

	assert.NotContains(t, version24.ValidBackends, "ncnn")
}
