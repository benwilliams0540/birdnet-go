// Package api provides the HTTP server and route handlers for BirdNET-Go.
package api

import (
	"net/http"

	"github.com/labstack/echo/v4"
)

// wifiProvisionHTML is the self-contained captive-portal provisioning page served at /wifi-setup.
// It has no external dependencies — all CSS and JS are inlined.
const wifiProvisionHTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BirdNET-Q WiFi Setup</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #1a1d23;
    --surface: #23272f;
    --border: #333844;
    --accent: #4a9eff;
    --accent-hover: #6bb3ff;
    --text: #e2e8f0;
    --text-muted: #8a95a3;
    --success: #34d399;
    --error: #f87171;
    --radius: 10px;
  }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1rem;
  }
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    width: 100%;
    max-width: 420px;
    padding: 2rem;
  }
  .logo {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.5rem;
  }
  .logo svg { flex-shrink: 0; }
  .logo-text { font-size: 1.25rem; font-weight: 700; letter-spacing: -0.02em; }
  .logo-sub { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.1rem; }
  h1 { font-size: 1.1rem; font-weight: 600; margin-bottom: 1.5rem; color: var(--text); }
  label { display: block; font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.35rem; }
  .field { margin-bottom: 1rem; }
  select, input[type="text"], input[type="password"] {
    width: 100%;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-size: 0.95rem;
    padding: 0.55rem 0.75rem;
    outline: none;
    transition: border-color 0.15s;
  }
  select:focus, input:focus { border-color: var(--accent); }
  .pw-wrap { position: relative; }
  .pw-wrap input { padding-right: 2.5rem; }
  .pw-toggle {
    position: absolute;
    right: 0.6rem;
    top: 50%;
    transform: translateY(-50%);
    background: none;
    border: none;
    cursor: pointer;
    color: var(--text-muted);
    padding: 0.2rem;
    display: flex;
    align-items: center;
  }
  .pw-toggle:hover { color: var(--text); }
  #manual-ssid-wrap { display: none; margin-top: 0.5rem; }
  .row-buttons { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
  button.primary {
    flex: 1;
    background: var(--accent);
    border: none;
    border-radius: 6px;
    color: #fff;
    cursor: pointer;
    font-size: 0.95rem;
    font-weight: 600;
    padding: 0.65rem 1rem;
    transition: background 0.15s;
  }
  button.primary:hover { background: var(--accent-hover); }
  button.primary:disabled { opacity: 0.5; cursor: not-allowed; }
  button.secondary {
    background: var(--border);
    border: none;
    border-radius: 6px;
    color: var(--text);
    cursor: pointer;
    font-size: 0.85rem;
    padding: 0.65rem 0.9rem;
    transition: background 0.15s;
  }
  button.secondary:hover { background: #444b59; }
  button.secondary:disabled { opacity: 0.5; cursor: not-allowed; }
  .spinner {
    display: none;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.9rem;
    color: var(--text-muted);
    margin-bottom: 1rem;
  }
  .spinner.visible { display: flex; }
  .spin {
    width: 18px; height: 18px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .message { border-radius: 6px; font-size: 0.9rem; padding: 0.75rem 1rem; margin-bottom: 1rem; display: none; }
  .message.visible { display: block; }
  .message.success { background: #0d2e22; border: 1px solid var(--success); color: var(--success); }
  .message.error { background: #2e0d0d; border: 1px solid var(--error); color: var(--error); }
  .note { font-size: 0.78rem; color: var(--text-muted); line-height: 1.5; border-top: 1px solid var(--border); padding-top: 1rem; margin-top: 0.5rem; }
  .signal { font-size: 0.8rem; color: var(--text-muted); margin-left: 0.4rem; }
</style>
</head>
<body>
<div class="card">
  <div class="logo">
    <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <circle cx="16" cy="16" r="15" stroke="#4a9eff" stroke-width="2"/>
      <path d="M10 20 Q13 10 16 14 Q19 18 22 8" stroke="#34d399" stroke-width="2" stroke-linecap="round" fill="none"/>
      <circle cx="16" cy="22" r="2.5" fill="#4a9eff"/>
    </svg>
    <div>
      <div class="logo-text">BirdNET-Q</div>
      <div class="logo-sub">WiFi Setup</div>
    </div>
  </div>

  <h1>Connect to your network</h1>

  <div class="field">
    <label for="network-select">Available networks</label>
    <select id="network-select">
      <option value="">Scanning…</option>
    </select>
    <div id="manual-ssid-wrap">
      <input type="text" id="manual-ssid" placeholder="Network name (SSID)" maxlength="32" autocomplete="off" autocorrect="off" spellcheck="false">
    </div>
  </div>

  <div class="field">
    <label for="password">Password</label>
    <div class="pw-wrap">
      <input type="password" id="password" placeholder="Leave empty for open networks" autocomplete="current-password">
      <button type="button" class="pw-toggle" id="pw-toggle" aria-label="Show password">
        <svg id="eye-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>
        </svg>
      </button>
    </div>
  </div>

  <div class="row-buttons">
    <button type="button" class="primary" id="connect-btn">Connect</button>
    <button type="button" class="secondary" id="rescan-btn" title="Rescan networks">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
      </svg>
      Rescan
    </button>
  </div>

  <div class="spinner" id="spinner"><div class="spin"></div><span>Connecting, please wait…</span></div>
  <div class="message" id="msg"></div>

  <p class="note">After connecting, the BirdNET-Q hotspot will disappear. Reconnect your device to your home WiFi to continue.</p>
</div>

<script>
(function () {
  'use strict';

  const networkSelect = document.getElementById('network-select');
  const manualWrap = document.getElementById('manual-ssid-wrap');
  const manualSSID = document.getElementById('manual-ssid');
  const passwordInput = document.getElementById('password');
  const pwToggle = document.getElementById('pw-toggle');
  const eyeIcon = document.getElementById('eye-icon');
  const connectBtn = document.getElementById('connect-btn');
  const rescanBtn = document.getElementById('rescan-btn');
  const spinner = document.getElementById('spinner');
  const msg = document.getElementById('msg');

  const EYE_OPEN = '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>';
  const EYE_CLOSED = '<path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/>';

  function signalBars(signal) {
    if (signal >= 75) return '\u25cf\u25cf\u25cf\u25cf';
    if (signal >= 50) return '\u25cf\u25cf\u25cf\u25cb';
    if (signal >= 25) return '\u25cf\u25cf\u25cb\u25cb';
    return '\u25cf\u25cb\u25cb\u25cb';
  }

  function setMessage(text, type) {
    msg.textContent = text;
    msg.className = 'message visible ' + type;
  }

  function clearMessage() {
    msg.className = 'message';
    msg.textContent = '';
  }

  function setLoading(loading) {
    spinner.className = loading ? 'spinner visible' : 'spinner';
    connectBtn.disabled = loading;
    rescanBtn.disabled = loading;
    networkSelect.disabled = loading;
  }

  function getSSID() {
    const val = networkSelect.value;
    if (val === '__other__') return manualSSID.value.trim();
    return val;
  }

  async function scanNetworks() {
    rescanBtn.disabled = true;
    networkSelect.innerHTML = '<option value="">Scanning\u2026</option>';
    clearMessage();

    try {
      const resp = await fetch('/api/v2/wifi/scan');
      if (!resp.ok) throw new Error('Scan failed (' + resp.status + ')');
      const data = await resp.json();
      const nets = data.networks || [];

      networkSelect.innerHTML = '';
      if (nets.length === 0) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = 'No networks found';
        networkSelect.appendChild(opt);
      } else {
        nets.forEach(function (n) {
          const opt = document.createElement('option');
          opt.value = n.ssid;
          opt.textContent = n.ssid + '  ' + signalBars(n.signal);
          networkSelect.appendChild(opt);
        });
      }

      const other = document.createElement('option');
      other.value = '__other__';
      other.textContent = 'Other network\u2026';
      networkSelect.appendChild(other);
    } catch (err) {
      setMessage('Could not scan networks: ' + err.message, 'error');
    } finally {
      rescanBtn.disabled = false;
    }
  }

  networkSelect.addEventListener('change', function () {
    manualWrap.style.display = networkSelect.value === '__other__' ? 'block' : 'none';
    clearMessage();
  });

  pwToggle.addEventListener('click', function () {
    const isPassword = passwordInput.type === 'password';
    passwordInput.type = isPassword ? 'text' : 'password';
    eyeIcon.innerHTML = isPassword ? EYE_CLOSED : EYE_OPEN;
  });

  rescanBtn.addEventListener('click', scanNetworks);

  connectBtn.addEventListener('click', async function () {
    clearMessage();
    const ssid = getSSID();
    if (!ssid) {
      setMessage('Please select or enter a network name.', 'error');
      return;
    }
    const password = passwordInput.value;

    setLoading(true);
    try {
      const resp = await fetch('/api/v2/wifi/connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ssid: ssid, password: password }),
      });
      const data = await resp.json();
      if (resp.ok && data.success) {
        setMessage(
          'Connected! The device will now join \u201c' + ssid + '\u201d. ' +
          'You may need to reconnect your device to your home network.',
          'success'
        );
      } else {
        setMessage(data.message || 'Connection failed.', 'error');
      }
    } catch (err) {
      setMessage('Network error: ' + err.message, 'error');
    } finally {
      setLoading(false);
    }
  });

  // Kick off initial scan on page load.
  scanNetworks();
})();
</script>
</body>
</html>`

// RegisterWifiProvisionRoutes registers the captive portal WiFi setup page and
// captive portal detection redirect routes on the given Echo instance.
//
// Captive portal detection URLs for major OS clients are redirected to /wifi-setup
// so that devices automatically open the provisioning page when joining the hotspot.
func RegisterWifiProvisionRoutes(e *echo.Echo) {
	// Main provisioning page.
	e.GET("/wifi-setup", serveWifiProvisionPage)

	// Captive portal detection redirects.
	// Android: generates a 204 No Content check.
	e.GET("/generate_204", redirectToWifiSetup)
	// Apple (iOS/macOS): hotspot detection page.
	e.GET("/hotspot-detect.html", redirectToWifiSetup)
	// Windows: NCSI and connection test.
	e.GET("/connecttest.txt", redirectToWifiSetup)
	e.GET("/ncsi.txt", redirectToWifiSetup)
	// Firefox OS / Firefox captive portal detection.
	e.GET("/success.txt", redirectToWifiSetup)
}

// serveWifiProvisionPage returns the self-contained WiFi setup HTML page.
func serveWifiProvisionPage(c echo.Context) error {
	return c.HTML(http.StatusOK, wifiProvisionHTML)
}

// redirectToWifiSetup redirects captive-portal detection probes to /wifi-setup.
func redirectToWifiSetup(c echo.Context) error {
	return c.Redirect(http.StatusFound, "/wifi-setup")
}
