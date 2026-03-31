//go:build !linux

// internal/api/v2/wifi_stub.go — stub WiFi routes for non-Linux platforms.
// All endpoints return HTTP 501 Not Implemented because nmcli is Linux-only.
package api

import (
	"net/http"

	"github.com/labstack/echo/v4"
)

// initWifiRoutes registers stub WiFi endpoints that return 501 on non-Linux platforms.
func (c *Controller) initWifiRoutes() {
	c.logInfoIfEnabled("Initializing WiFi routes (stub — nmcli not available on this platform)")

	notImplemented := func(ctx echo.Context) error {
		return ctx.JSON(http.StatusNotImplemented, map[string]any{
			"success": false,
			"message": "WiFi management is only supported on Linux",
		})
	}

	c.Group.GET("/wifi/status", notImplemented)

	wifiGroup := c.Group.Group("/wifi", c.authMiddleware)
	wifiGroup.GET("/scan", notImplemented)
	wifiGroup.POST("/connect", notImplemented)
	wifiGroup.GET("/hotspot/status", notImplemented)
	wifiGroup.POST("/hotspot/start", notImplemented)
	wifiGroup.POST("/hotspot/stop", notImplemented)
}
