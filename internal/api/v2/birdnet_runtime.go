package api

import (
	"fmt"
	"net/http"
	"slices"
	"strings"

	"github.com/labstack/echo/v4"
	"github.com/tphakala/birdnet-go/internal/classifier"
	"github.com/tphakala/birdnet-go/internal/conf"
	"github.com/tphakala/birdnet-go/internal/restart"
)

const (
	birdnetChangeModeNone            = "none"
	birdnetChangeModeHotReload       = "hot_reload"
	birdnetChangeModeRestartRequired = "restart_required"
	birdnetChangeModeInvalid         = "invalid"

	birdnetModelRestartReason   = "BirdNET model selection changed"
	birdnetBackendRestartReason = "BirdNET inference backend changed"
	birdnetRuntimeRestartReason = "BirdNET runtime dependency changed"
)

var birdnetVersionLabels = map[string]string{
	"2.4":          "BirdNET Global 6K V2.4",
	"2.4-int8":     "BirdNET Global 6K V2.4 INT8 (ONNX)",
	"2.4-int8-cnn": "BirdNET Global 6K V2.4 INT8 CNN (ONNX)",
	"3.0":          "BirdNET+ V3.0 Preview 3 Global 11K",
}

var birdnetVersionOrder = []string{"2.4", "2.4-int8", "2.4-int8-cnn", "3.0"}

type birdnetBackendCapability struct {
	Value    string `json:"value"`
	Label    string `json:"label"`
	Compiled bool   `json:"compiled"`
}

type birdnetVersionCapability struct {
	Value         string   `json:"value"`
	Label         string   `json:"label"`
	ValidBackends []string `json:"validBackends"`
}

type birdnetChangeAssessment struct {
	RequestedBackend string `json:"requestedBackend"`
	EffectiveBackend string `json:"effectiveBackend,omitempty"`
	Version          string `json:"version"`
	ModelID          string `json:"modelId,omitempty"`
	Valid            bool   `json:"valid"`
	Reason           string `json:"reason,omitempty"`
	ChangeMode       string `json:"changeMode"`
	RestartRequired  bool   `json:"restartRequired"`
}

type birdnetCapabilitiesResponse struct {
	CompiledBackends []string                   `json:"compiledBackends"`
	Backends         []birdnetBackendCapability `json:"backends"`
	Versions         []birdnetVersionCapability `json:"versions"`
	RequestedChange  birdnetChangeAssessment    `json:"requestedChange"`
	RestartReasons   []string                   `json:"restartReasons,omitempty"`
}

type birdnetChangeDecision struct {
	changed       bool
	action        string
	restartReason string
}

// GetBirdNETCapabilities reports compiled backends, valid backend/model combinations,
// and whether the requested configuration can hot-reload or requires a restart.
func (c *Controller) GetBirdNETCapabilities(ctx echo.Context) error {
	settings := c.getSettingsOrFallback()
	if settings == nil {
		return c.HandleError(ctx, fmt.Errorf("settings not initialized"), "Failed to inspect BirdNET capabilities", http.StatusInternalServerError)
	}

	requested := requestedBirdNETConfigFromQuery(settings.BirdNET, ctx)
	assessment := assessBirdNETChange(settings.BirdNET, requested)

	response := birdnetCapabilitiesResponse{
		CompiledBackends: compiledBirdNETBackends(),
		Backends: []birdnetBackendCapability{
			{Value: "auto", Label: "Auto", Compiled: true},
			{Value: "tflite", Label: "TFLite", Compiled: classifier.IsBackendCompiled("tflite")},
			{Value: "onnx", Label: "ONNX", Compiled: classifier.IsBackendCompiled("onnx")},
			{Value: "ncnn", Label: "NCNN", Compiled: classifier.IsBackendCompiled("ncnn")},
			{Value: "qnn", Label: "QNN", Compiled: classifier.IsBackendCompiled("qnn")},
		},
		Versions:        buildBirdNETVersionCapabilities(requested),
		RequestedChange: assessment,
		RestartReasons:  restart.GetRestartReasons(),
	}

	return ctx.JSON(http.StatusOK, response)
}

func compiledBirdNETBackends() []string {
	support := classifier.GetCompiledBackendSupport()
	backends := make([]string, 0, 4)
	if support.TFLite {
		backends = append(backends, "tflite")
	}
	if support.ONNX {
		backends = append(backends, "onnx")
	}
	if support.NCNN {
		backends = append(backends, "ncnn")
	}
	if support.QNN {
		backends = append(backends, "qnn")
	}
	return backends
}

func buildBirdNETVersionCapabilities(base conf.BirdNETConfig) []birdnetVersionCapability {
	versions := make([]birdnetVersionCapability, 0, len(birdnetVersionOrder))
	for _, version := range birdnetVersionOrder {
		cfg := base
		cfg.Version = version
		cfg.ModelPath = strings.TrimSpace(cfg.ModelPath)

		validBackends := birdnetAvailableBackends(cfg, true)
		if len(validBackends) > 0 {
			validBackends = append([]string{"auto"}, validBackends...)
		}

		versions = append(versions, birdnetVersionCapability{
			Value:         version,
			Label:         birdnetVersionLabels[version],
			ValidBackends: validBackends,
		})
	}
	return versions
}

func requestedBirdNETConfigFromQuery(base conf.BirdNETConfig, ctx echo.Context) conf.BirdNETConfig {
	cfg := base
	query := ctx.QueryParams()

	if query.Has("backend") {
		cfg.Backend = normalizeBirdNETBackend(query.Get("backend"))
	}
	if query.Has("version") {
		cfg.Version = strings.TrimSpace(query.Get("version"))
	}
	if query.Has("modelPath") {
		cfg.ModelPath = strings.TrimSpace(query.Get("modelPath"))
	}
	if query.Has("labelPath") {
		cfg.LabelPath = strings.TrimSpace(query.Get("labelPath"))
	}
	if query.Has("onnxRuntimePath") {
		cfg.ONNXRuntimePath = strings.TrimSpace(query.Get("onnxRuntimePath"))
	}
	if query.Has("ncnnModelDir") {
		cfg.NCNNModelDir = strings.TrimSpace(query.Get("ncnnModelDir"))
	}
	if query.Has("qnnBackend") {
		cfg.QNNBackend = strings.TrimSpace(query.Get("qnnBackend"))
	}
	if query.Has("qnnLibDir") {
		cfg.QNNLibDir = strings.TrimSpace(query.Get("qnnLibDir"))
	}
	if query.Has("qnnModelLibDir") {
		cfg.QNNModelLibDir = strings.TrimSpace(query.Get("qnnModelLibDir"))
	}

	return cfg
}

func normalizeBirdNETBackend(backend string) string {
	normalized := strings.ToLower(strings.TrimSpace(backend))
	if normalized == "auto" {
		return ""
	}
	return normalized
}

func hasModelExtension(path, ext string) bool {
	return strings.HasSuffix(strings.ToLower(strings.TrimSpace(path)), ext)
}

func birdnetTFLiteSourceAvailable(cfg conf.BirdNETConfig, info classifier.ModelInfo) bool {
	return hasModelExtension(cfg.ModelPath, ".tflite") ||
		(cfg.ModelPath == "" && info.ID == classifier.DefaultModelVersion && classifier.DefaultBirdNETModelAvailable())
}

func birdnetONNXSourceAvailable(cfg conf.BirdNETConfig, info classifier.ModelInfo) bool {
	if hasModelExtension(cfg.ModelPath, ".onnx") {
		return true
	}
	if !info.IsONNX {
		return false
	}
	_, ok := classifier.GetEmbeddedONNXData(info.ID)
	return ok
}

func birdnetNCNNSourceAvailable(cfg conf.BirdNETConfig, info classifier.ModelInfo) bool {
	dir := strings.TrimSpace(cfg.NCNNModelDir)
	if dir == "" || info.DetectionVersion != "2.4" {
		return false
	}

	return classifier.NCNNModelDirReady(dir)
}

func birdnetBackendCompatible(cfg conf.BirdNETConfig, backend string) (bool, string) {
	info, err := classifier.ResolveBirdNETModelInfoFromConfig(cfg)
	if err != nil {
		return false, err.Error()
	}

	switch normalizeBirdNETBackend(backend) {
	case "tflite":
		if !birdnetTFLiteSourceAvailable(cfg, info) {
			return false, "TFLite backend requires a TFLite model source"
		}
		return true, ""
	case "onnx":
		if !birdnetONNXSourceAvailable(cfg, info) {
			return false, "ONNX backend requires an embedded INT8 ONNX model or an explicit .onnx model path"
		}
		return true, ""
	case "ncnn":
		if !birdnetNCNNSourceAvailable(cfg, info) {
			return false, fmt.Sprintf(
				"NCNN backend requires a BirdNET V2.4 model family selection, recognized NCNN model files, and validation marker %s",
				classifier.NCNNValidationMarkerName,
			)
		}
		return true, ""
	case "qnn":
		if strings.TrimSpace(cfg.QNNBackend) == "" || strings.TrimSpace(cfg.QNNLibDir) == "" || strings.TrimSpace(cfg.QNNModelLibDir) == "" {
			return false, "QNN backend requires backend, library, and model library directories"
		}
		if !strings.Contains(info.ID, "CNN") {
			return false, "QNN backend requires a CNN-style BirdNET model"
		}
		return true, ""
	default:
		return false, "unknown BirdNET backend"
	}
}

func birdnetAvailableBackends(cfg conf.BirdNETConfig, compiledOnly bool) []string {
	backends := make([]string, 0, 4)
	for _, backend := range []string{"ncnn", "qnn", "onnx", "tflite"} {
		if compiledOnly && !classifier.IsBackendCompiled(backend) {
			continue
		}
		if ok, _ := birdnetBackendCompatible(cfg, backend); ok {
			backends = append(backends, backend)
		}
	}
	return backends
}

func determineAutomaticBirdNETBackend(cfg conf.BirdNETConfig) (string, string) {
	runtimeBackends := birdnetAvailableBackends(cfg, true)
	if len(runtimeBackends) > 0 {
		return runtimeBackends[0], ""
	}

	configBackends := birdnetAvailableBackends(cfg, false)
	if len(configBackends) == 0 {
		return "", "No supported inference backend is available for the selected BirdNET model"
	}

	return "", fmt.Sprintf(
		"The selected BirdNET model is valid, but this binary was not built with a compatible backend. Compatible backends: %s",
		strings.Join(configBackends, ", "),
	)
}

func validateBirdNETConfigSelection(cfg conf.BirdNETConfig) error {
	requestedBackend := normalizeBirdNETBackend(cfg.Backend)
	if requestedBackend != "" {
		if ok, reason := birdnetBackendCompatible(cfg, requestedBackend); !ok {
			return fmt.Errorf("%s", reason)
		}
		return nil
	}

	if len(birdnetAvailableBackends(cfg, false)) == 0 {
		return fmt.Errorf("No supported inference backend is available for the selected BirdNET model")
	}

	return nil
}

func birdnetRelevantConfigsChanged(oldCfg, newCfg conf.BirdNETConfig) bool {
	return oldCfg.Locale != newCfg.Locale ||
		oldCfg.Threads != newCfg.Threads ||
		oldCfg.ModelPath != newCfg.ModelPath ||
		oldCfg.LabelPath != newCfg.LabelPath ||
		oldCfg.UseXNNPACK != newCfg.UseXNNPACK ||
		oldCfg.Version != newCfg.Version ||
		normalizeBirdNETBackend(oldCfg.Backend) != normalizeBirdNETBackend(newCfg.Backend) ||
		oldCfg.ONNXRuntimePath != newCfg.ONNXRuntimePath ||
		oldCfg.NCNNModelDir != newCfg.NCNNModelDir ||
		oldCfg.NCNNUseVulkan != newCfg.NCNNUseVulkan ||
		oldCfg.QNNBackend != newCfg.QNNBackend ||
		oldCfg.QNNLibDir != newCfg.QNNLibDir ||
		oldCfg.QNNModelLibDir != newCfg.QNNModelLibDir
}

func sameBirdNETIdentity(oldInfo, newInfo classifier.ModelInfo) bool {
	return oldInfo.ID == newInfo.ID && oldInfo.CustomPath == newInfo.CustomPath
}

func configuredBirdNETBackendFamily(cfg conf.BirdNETConfig) string {
	requestedBackend := normalizeBirdNETBackend(cfg.Backend)
	if requestedBackend != "" {
		return requestedBackend
	}

	backends := birdnetAvailableBackends(cfg, false)
	if len(backends) == 0 {
		return ""
	}

	return backends[0]
}

func birdnetRequestedBackendLabel(cfg conf.BirdNETConfig) string {
	backend := normalizeBirdNETBackend(cfg.Backend)
	if backend == "" {
		return "auto"
	}
	return backend
}

func birdnetONNXRuntimeChanged(oldCfg, newCfg conf.BirdNETConfig) bool {
	return oldCfg.ONNXRuntimePath != newCfg.ONNXRuntimePath
}

func birdnetQNNRuntimeChanged(oldCfg, newCfg conf.BirdNETConfig) bool {
	return oldCfg.QNNBackend != newCfg.QNNBackend ||
		oldCfg.QNNLibDir != newCfg.QNNLibDir ||
		oldCfg.QNNModelLibDir != newCfg.QNNModelLibDir
}

func birdnetHotReloadRelevantChanged(oldCfg, newCfg conf.BirdNETConfig, backendFamily string) bool {
	if oldCfg.Locale != newCfg.Locale ||
		oldCfg.Threads != newCfg.Threads ||
		oldCfg.LabelPath != newCfg.LabelPath ||
		oldCfg.UseXNNPACK != newCfg.UseXNNPACK ||
		normalizeBirdNETBackend(oldCfg.Backend) != normalizeBirdNETBackend(newCfg.Backend) {
		return true
	}

	switch backendFamily {
	case "onnx":
		return oldCfg.ModelPath != newCfg.ModelPath
	case "ncnn":
		return oldCfg.ModelPath != newCfg.ModelPath ||
			oldCfg.NCNNModelDir != newCfg.NCNNModelDir ||
			oldCfg.NCNNUseVulkan != newCfg.NCNNUseVulkan
	case "qnn":
		return oldCfg.ModelPath != newCfg.ModelPath
	default:
		return oldCfg.ModelPath != newCfg.ModelPath
	}
}

func classifyBirdNETConfigChange(oldCfg, newCfg conf.BirdNETConfig) birdnetChangeDecision {
	if !birdnetRelevantConfigsChanged(oldCfg, newCfg) {
		return birdnetChangeDecision{}
	}

	oldInfo, oldErr := classifier.ResolveBirdNETModelInfoFromConfig(oldCfg)
	newInfo, newErr := classifier.ResolveBirdNETModelInfoFromConfig(newCfg)
	if oldErr == nil && newErr == nil && !sameBirdNETIdentity(oldInfo, newInfo) {
		return birdnetChangeDecision{
			changed:       true,
			restartReason: birdnetModelRestartReason,
		}
	}

	oldBackend := configuredBirdNETBackendFamily(oldCfg)
	newBackend := configuredBirdNETBackendFamily(newCfg)
	if oldBackend != newBackend {
		return birdnetChangeDecision{
			changed:       true,
			restartReason: birdnetBackendRestartReason,
		}
	}

	switch newBackend {
	case "onnx":
		if birdnetONNXRuntimeChanged(oldCfg, newCfg) {
			return birdnetChangeDecision{
				changed:       true,
				restartReason: birdnetRuntimeRestartReason,
			}
		}
	case "qnn":
		if birdnetQNNRuntimeChanged(oldCfg, newCfg) {
			return birdnetChangeDecision{
				changed:       true,
				restartReason: birdnetRuntimeRestartReason,
			}
		}
	}

	if !birdnetHotReloadRelevantChanged(oldCfg, newCfg, newBackend) {
		return birdnetChangeDecision{changed: true}
	}

	return birdnetChangeDecision{
		changed: true,
		action:  "reload_birdnet",
	}
}

func assessBirdNETChange(currentCfg, requestedCfg conf.BirdNETConfig) birdnetChangeAssessment {
	assessment := birdnetChangeAssessment{
		RequestedBackend: birdnetRequestedBackendLabel(requestedCfg),
		Version:          requestedCfg.Version,
		Valid:            true,
		ChangeMode:       birdnetChangeModeNone,
	}

	info, err := classifier.ResolveBirdNETModelInfoFromConfig(requestedCfg)
	if err != nil {
		assessment.Valid = false
		assessment.ChangeMode = birdnetChangeModeInvalid
		assessment.Reason = err.Error()
		return assessment
	}
	assessment.ModelID = info.ID

	if err := validateBirdNETConfigSelection(requestedCfg); err != nil {
		assessment.Valid = false
		assessment.ChangeMode = birdnetChangeModeInvalid
		assessment.Reason = err.Error()
		return assessment
	}

	requestedBackend := normalizeBirdNETBackend(requestedCfg.Backend)
	if requestedBackend != "" {
		assessment.EffectiveBackend = requestedBackend
		if !classifier.IsBackendCompiled(requestedBackend) {
			assessment.ChangeMode = birdnetChangeModeRestartRequired
			assessment.RestartRequired = true
			assessment.Reason = fmt.Sprintf(
				"%s backend is selected, but this binary was not built with %s support",
				strings.ToUpper(requestedBackend),
				strings.ToUpper(requestedBackend),
			)
			return assessment
		}
	} else {
		effective, reason := determineAutomaticBirdNETBackend(requestedCfg)
		if effective == "" {
			assessment.ChangeMode = birdnetChangeModeRestartRequired
			assessment.RestartRequired = true
			assessment.Reason = reason
			return assessment
		}
		assessment.EffectiveBackend = effective
	}

	change := classifyBirdNETConfigChange(currentCfg, requestedCfg)
	switch {
	case !change.changed:
		assessment.ChangeMode = birdnetChangeModeNone
	case change.restartReason != "":
		assessment.ChangeMode = birdnetChangeModeRestartRequired
		assessment.RestartRequired = true
		assessment.Reason = change.restartReason
	case change.action == "reload_birdnet":
		assessment.ChangeMode = birdnetChangeModeHotReload
	}

	return assessment
}

func replaceSettingsRestartReasons(reasons []string) {
	unique := make([]string, 0, len(reasons))
	for _, reason := range reasons {
		trimmed := strings.TrimSpace(reason)
		if trimmed == "" || slices.Contains(unique, trimmed) {
			continue
		}
		unique = append(unique, trimmed)
	}
	restart.ReplaceReasons(unique)
}
