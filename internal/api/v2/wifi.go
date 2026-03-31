//go:build linux

// internal/api/v2/wifi.go — WiFi management API endpoints (Linux/nmcli).
package api

import (
	"bytes"
	"net/http"
	"os/exec"
	"sort"
	"strconv"
	"strings"

	"github.com/labstack/echo/v4"
	"github.com/tphakala/birdnet-go/internal/logger"
)

// WiFi-related constants.
const (
	// nmcliPath is the absolute path to the nmcli binary.
	nmcliPath = "/usr/bin/nmcli"

	// wifiInterface is the wireless interface name on the device.
	wifiInterface = "wlan0"

	// hotspotSSID is the SSID used when creating the provisioning hotspot.
	hotspotSSID = "BirdNET-Q Setup"

	// hotspotPassword is the WPA2 passphrase for the provisioning hotspot.
	hotspotPassword = "birdnetq1"

	// hotspotIP is the default gateway IP when the hotspot is active.
	hotspotIP = "10.42.0.1"

	// maxSSIDLength is the maximum allowed SSID length per IEEE 802.11.
	maxSSIDLength = 32

	// minPasswordLength is the minimum WPA2-Personal passphrase length.
	minPasswordLength = 8

	// maxPasswordLength is the maximum WPA2-Personal passphrase length.
	maxPasswordLength = 63
)

// WiFiStatusResponse contains the current WiFi connection state.
type WiFiStatusResponse struct {
	// State is the NetworkManager general state (e.g. "connected", "disconnected").
	State string `json:"state"`
	// Connectivity is the NM connectivity level (e.g. "full", "none").
	Connectivity string `json:"connectivity"`
	// ActiveConnection is the name of the currently active WiFi connection, if any.
	ActiveConnection string `json:"active_connection,omitempty"`
}

// WiFiNetwork represents a single visible wireless network.
type WiFiNetwork struct {
	// SSID is the network name.
	SSID string `json:"ssid"`
	// Signal is the signal strength (0–100).
	Signal int `json:"signal"`
	// Security indicates the security type (e.g. "WPA2", "--" for open).
	Security string `json:"security"`
}

// WiFiScanResponse contains the list of scanned networks.
type WiFiScanResponse struct {
	// Networks is the deduplicated list sorted by signal strength descending.
	Networks []WiFiNetwork `json:"networks"`
}

// WiFiConnectRequest is the body of a connect request.
type WiFiConnectRequest struct {
	// SSID is the target network name (required, 1–32 chars).
	SSID string `json:"ssid"`
	// Password is the WPA passphrase (8–63 chars, or empty for open networks).
	Password string `json:"password"`
}

// HotspotStatusResponse reports whether a hotspot is currently active.
type HotspotStatusResponse struct {
	// Active indicates whether a hotspot connection is currently up.
	Active bool `json:"active"`
	// Name is the NM connection name of the active hotspot, if any.
	Name string `json:"name,omitempty"`
}

// HotspotStartResponse is returned when the hotspot is successfully started.
type HotspotStartResponse struct {
	// Success indicates the hotspot was started.
	Success bool `json:"success"`
	// SSID is the hotspot network name.
	SSID string `json:"ssid"`
	// Password is the hotspot WPA passphrase.
	Password string `json:"password"`
	// IP is the device's IP address on the hotspot network.
	IP string `json:"ip"`
}

// initWifiRoutes registers all WiFi management API endpoints.
func (c *Controller) initWifiRoutes() {
	c.logInfoIfEnabled("Initializing WiFi routes")

	// Status is public — needed by the provisioning page before auth is configured.
	c.Group.GET("/wifi/status", c.GetWiFiStatus)

	// All other WiFi endpoints require authentication.
	wifiGroup := c.Group.Group("/wifi", c.authMiddleware)
	wifiGroup.GET("/scan", c.ScanWiFiNetworks)
	wifiGroup.POST("/connect", c.ConnectToWiFi)
	wifiGroup.GET("/hotspot/status", c.GetHotspotStatus)
	wifiGroup.POST("/hotspot/start", c.StartHotspot)
	wifiGroup.POST("/hotspot/stop", c.StopHotspot)
}

// GetWiFiStatus returns the current WiFi state using nmcli.
// GET /api/v2/wifi/status — public endpoint.
func (c *Controller) GetWiFiStatus(ctx echo.Context) error {
	// Fetch general state and connectivity.
	out, err := exec.CommandContext(ctx.Request().Context(),
		nmcliPath, "-t", "-f", "STATE,CONNECTIVITY", "general", "status",
	).Output()
	if err != nil {
		return c.HandleError(ctx, err, "failed to query WiFi status", http.StatusInternalServerError)
	}

	state, connectivity := parseNmcliGeneralStatus(string(out))

	// Fetch the SSID of the active connection on the wireless interface.
	activeName := ""
	conOut, err := exec.CommandContext(ctx.Request().Context(),
		nmcliPath, "-t", "-f", "ACTIVE,SSID", "dev", "wifi",
	).Output()
	if err == nil {
		activeName = parseActiveSSID(string(conOut))
	}

	c.logInfoIfEnabled("WiFi status queried",
		logger.String("state", state),
		logger.String("connectivity", connectivity),
		logger.String("active_connection", activeName),
	)

	return ctx.JSON(http.StatusOK, WiFiStatusResponse{
		State:            state,
		Connectivity:     connectivity,
		ActiveConnection: activeName,
	})
}

// ScanWiFiNetworks returns available wireless networks.
// GET /api/v2/wifi/scan — requires auth.
func (c *Controller) ScanWiFiNetworks(ctx echo.Context) error {
	out, err := exec.CommandContext(ctx.Request().Context(),
		nmcliPath, "-t", "-f", "SSID,SIGNAL,SECURITY", "dev", "wifi", "list", "--rescan", "yes",
	).Output()
	if err != nil {
		return c.HandleError(ctx, err, "failed to scan WiFi networks", http.StatusInternalServerError)
	}

	networks := parseNmcliWifiList(string(out))

	c.logInfoIfEnabled("WiFi scan completed",
		logger.Int("networks_found", len(networks)),
	)

	return ctx.JSON(http.StatusOK, WiFiScanResponse{Networks: networks})
}

// ConnectToWiFi attempts to connect to the specified WiFi network.
// POST /api/v2/wifi/connect — requires auth.
func (c *Controller) ConnectToWiFi(ctx echo.Context) error {
	var req WiFiConnectRequest
	if err := ctx.Bind(&req); err != nil {
		return c.HandleError(ctx, err, "invalid request body", http.StatusBadRequest)
	}

	// Validate SSID.
	if req.SSID == "" {
		return c.HandleError(ctx, nil, "ssid is required", http.StatusBadRequest)
	}
	if len(req.SSID) > maxSSIDLength {
		return c.HandleError(ctx, nil, "ssid must not exceed 32 characters", http.StatusBadRequest)
	}

	// Validate password (empty is allowed for open networks).
	if req.Password != "" && (len(req.Password) < minPasswordLength || len(req.Password) > maxPasswordLength) {
		return c.HandleError(ctx, nil, "password must be 8–63 characters, or empty for open networks", http.StatusBadRequest)
	}

	c.logInfoIfEnabled("WiFi connect requested", logger.String("ssid", req.SSID))

	reqCtx := ctx.Request().Context()

	// First attempt: bring up a saved connection profile by name.
	upErr := exec.CommandContext(reqCtx, //nolint:gosec // SSID comes from validated user input, not shell expansion
		nmcliPath, "con", "up", "id", req.SSID,
	).Run()

	if upErr == nil {
		c.logInfoIfEnabled("Connected to saved WiFi network", logger.String("ssid", req.SSID))
		return ctx.JSON(http.StatusOK, map[string]any{
			"success": true,
			"message": "Connected to " + req.SSID,
		})
	}

	// Second attempt: connect to a new network.
	args := []string{"dev", "wifi", "connect", req.SSID, "ifname", wifiInterface}
	if req.Password != "" {
		args = append(args, "password", req.Password)
	}

	var stderr bytes.Buffer
	cmd := exec.CommandContext(reqCtx, nmcliPath, args...) //nolint:gosec // args are validated above
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		c.logInfoIfEnabled("WiFi connect failed",
			logger.String("ssid", req.SSID),
			logger.String("stderr", stderr.String()),
		)
		return c.HandleError(ctx, err, "failed to connect to "+req.SSID+": "+strings.TrimSpace(stderr.String()), http.StatusInternalServerError)
	}

	c.logInfoIfEnabled("Connected to WiFi network", logger.String("ssid", req.SSID))
	return ctx.JSON(http.StatusOK, map[string]any{
		"success": true,
		"message": "Connected to " + req.SSID,
	})
}

// GetHotspotStatus returns whether an access-point hotspot is currently active.
// GET /api/v2/wifi/hotspot/status — requires auth.
func (c *Controller) GetHotspotStatus(ctx echo.Context) error {
	name, active, err := findActiveHotspot(ctx)
	if err != nil {
		return c.HandleError(ctx, err, "failed to query hotspot status", http.StatusInternalServerError)
	}
	return ctx.JSON(http.StatusOK, HotspotStatusResponse{Active: active, Name: name})
}

// StartHotspot creates and activates a WiFi access-point hotspot.
// POST /api/v2/wifi/hotspot/start — requires auth.
func (c *Controller) StartHotspot(ctx echo.Context) error {
	var stderr bytes.Buffer
	cmd := exec.CommandContext(ctx.Request().Context(), //nolint:gosec // constants only, no user input
		nmcliPath, "dev", "wifi", "hotspot",
		"ifname", wifiInterface,
		"ssid", hotspotSSID,
		"password", hotspotPassword,
	)
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return c.HandleError(ctx, err, "failed to start hotspot: "+strings.TrimSpace(stderr.String()), http.StatusInternalServerError)
	}

	c.logInfoIfEnabled("WiFi hotspot started",
		logger.String("ssid", hotspotSSID),
		logger.String("ip", hotspotIP),
	)

	return ctx.JSON(http.StatusOK, HotspotStartResponse{
		Success:  true,
		SSID:     hotspotSSID,
		Password: hotspotPassword,
		IP:       hotspotIP,
	})
}

// StopHotspot deactivates the active WiFi hotspot.
// POST /api/v2/wifi/hotspot/stop — requires auth.
func (c *Controller) StopHotspot(ctx echo.Context) error {
	name, active, err := findActiveHotspot(ctx)
	if err != nil {
		return c.HandleError(ctx, err, "failed to query hotspot status", http.StatusInternalServerError)
	}

	if !active {
		return ctx.JSON(http.StatusOK, map[string]any{
			"success": false,
			"message": "no active hotspot found",
		})
	}

	var stderr bytes.Buffer
	cmd := exec.CommandContext(ctx.Request().Context(), //nolint:gosec // name comes from nmcli output, not user input
		nmcliPath, "con", "down", name,
	)
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return c.HandleError(ctx, err, "failed to stop hotspot: "+strings.TrimSpace(stderr.String()), http.StatusInternalServerError)
	}

	c.logInfoIfEnabled("WiFi hotspot stopped", logger.String("connection", name))

	return ctx.JSON(http.StatusOK, map[string]any{
		"success": true,
		"message": "hotspot stopped",
	})
}

// =============================================================================
// nmcli output parsers
// =============================================================================

// parseNmcliGeneralStatus parses the two-field output of:
// nmcli -t -f STATE,CONNECTIVITY general status
// Returns (state, connectivity).
func parseNmcliGeneralStatus(output string) (state, connectivity string) {
	line := strings.TrimSpace(strings.SplitN(output, "\n", 2)[0])
	parts := strings.SplitN(line, ":", 2)
	if len(parts) == 2 { //nolint:mnd // two fields expected
		return parts[0], parts[1]
	}
	return line, ""
}

// parseActiveSSID parses the output of:
// nmcli -t -f ACTIVE,SSID dev wifi
// and returns the SSID of the currently active WiFi connection.
func parseActiveSSID(output string) string {
	for line := range strings.SplitSeq(output, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		// Format: ACTIVE:SSID — split on first unescaped colon.
		colonIdx := strings.Index(line, ":")
		if colonIdx < 0 {
			continue
		}
		if line[:colonIdx] != "yes" {
			continue
		}
		// Unescape \: → : in the SSID.
		ssid := strings.ReplaceAll(line[colonIdx+1:], `\:`, ":")
		if ssid != "" {
			return ssid
		}
	}
	return ""
}

// parseNmcliWifiList parses the output of:
// nmcli -t -f SSID,SIGNAL,SECURITY dev wifi list --rescan yes
//
// nmcli escapes literal colons in SSID as "\:" so we must unescape after splitting.
// The output has three fields per line; we split on the last two colons working
// right-to-left to handle SSIDs that contain colons.
func parseNmcliWifiList(output string) []WiFiNetwork {
	seen := make(map[string]int) // ssid → index in result
	networks := make([]WiFiNetwork, 0)

	for line := range strings.SplitSeq(output, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Find the last two colons that are NOT escaped.
		// Fields: SSID:SIGNAL:SECURITY
		ssid, signal, security, ok := splitNmcliWifiLine(line)
		if !ok {
			continue
		}

		if ssid == "" {
			continue
		}

		sig, err := strconv.Atoi(signal)
		if err != nil {
			sig = 0
		}

		if idx, exists := seen[ssid]; exists {
			// Keep the entry with the strongest signal.
			if sig > networks[idx].Signal {
				networks[idx].Signal = sig
			}
			continue
		}

		seen[ssid] = len(networks)
		networks = append(networks, WiFiNetwork{
			SSID:     ssid,
			Signal:   sig,
			Security: security,
		})
	}

	// Sort by signal strength descending.
	sort.Slice(networks, func(i, j int) bool {
		return networks[i].Signal > networks[j].Signal
	})

	return networks
}

// splitNmcliWifiLine splits a single nmcli wifi list line (SSID:SIGNAL:SECURITY)
// where SSID may contain escaped colons (\:).
// Returns (ssid, signal, security, ok).
func splitNmcliWifiLine(line string) (ssid, signal, security string, ok bool) {
	// Find the last unescaped colon (security field boundary).
	lastColon := findLastUnescapedColon(line)
	if lastColon < 0 {
		return "", "", "", false
	}
	security = line[lastColon+1:]
	rest := line[:lastColon]

	// Find the second-to-last unescaped colon (signal field boundary).
	sigColon := findLastUnescapedColon(rest)
	if sigColon < 0 {
		return "", "", "", false
	}
	signal = rest[sigColon+1:]
	rawSSID := rest[:sigColon]

	// Unescape \: → : in the SSID.
	ssid = strings.ReplaceAll(rawSSID, `\:`, ":")
	return ssid, signal, security, true
}

// findLastUnescapedColon returns the index of the last colon that is not
// preceded by a backslash, or -1 if none is found.
func findLastUnescapedColon(s string) int {
	for i := len(s) - 1; i >= 0; i-- {
		if s[i] == ':' {
			if i == 0 || s[i-1] != '\\' {
				return i
			}
		}
	}
	return -1
}

// findActiveHotspot checks whether a WiFi access-point hotspot is currently active.
// It uses two nmcli calls: first to list active wifi connections, then to check each
// one's 802-11-wireless.mode property. Returns the connection name, whether it is an
// AP hotspot, and any error.
func findActiveHotspot(ctx echo.Context) (name string, active bool, err error) {
	// Step 1: list active connections, keep only wifi type.
	out, err := exec.CommandContext(ctx.Request().Context(),
		nmcliPath, "-t", "-f", "NAME,TYPE", "con", "show", "--active",
	).Output()
	if err != nil {
		return "", false, err
	}

	// Collect active wifi connection names.
	var wifiNames []string
	for line := range strings.SplitSeq(string(out), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		// Format: NAME:TYPE — split on last colon so names with colons work.
		idx := strings.LastIndex(line, ":")
		if idx < 0 {
			continue
		}
		connName := line[:idx]
		connType := line[idx+1:]
		if connType == "802-11-wireless" {
			wifiNames = append(wifiNames, connName)
		}
	}

	// Step 2: for each active wifi connection, check its wireless mode.
	for _, connName := range wifiNames {
		modeOut, modeErr := exec.CommandContext(ctx.Request().Context(), //nolint:gosec // connName from nmcli output
			nmcliPath, "-g", "802-11-wireless.mode", "con", "show", connName,
		).Output()
		if modeErr != nil {
			continue
		}
		if strings.TrimSpace(string(modeOut)) == "ap" {
			return connName, true, nil
		}
	}
	return "", false, nil
}
