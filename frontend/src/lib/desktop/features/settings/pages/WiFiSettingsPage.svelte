<!--
  WiFi Settings Page Component

  Purpose: WiFi connection management and hotspot control for BirdNET-Go.

  Features:
  - Connection tab: scan for networks, connect with password, show signal strength
  - Hotspot tab: start/stop hotspot, display credentials

  Props: None - action-based page with no settings store

  @component
-->
<script lang="ts">
  import SettingsSection from '$lib/desktop/features/settings/components/SettingsSection.svelte';
  import SettingsTabs from '$lib/desktop/features/settings/components/SettingsTabs.svelte';
  import type { TabDefinition } from '$lib/desktop/features/settings/components/SettingsTabs.svelte';
  import {
    Wifi,
    Radio,
    RefreshCw,
    Eye,
    EyeOff,
    CircleCheck,
    XCircle,
    Info,
  } from '@lucide/svelte';
  import { t } from '$lib/i18n';
  import { loggers } from '$lib/utils/logger';
  import { api } from '$lib/utils/api';

  const logger = loggers.settings;

  // Tab state
  let activeTab = $state('connection');

  // Tab definitions
  let tabs = $derived<TabDefinition[]>([
    {
      id: 'connection',
      label: t('settings.wifi.tabs.connection'),
      icon: Wifi,
      content: connectionTabContent,
      hasChanges: false,
    },
    {
      id: 'hotspot',
      label: t('settings.wifi.tabs.hotspot'),
      icon: Radio,
      content: hotspotTabContent,
      hasChanges: false,
    },
  ]);

  // --- API response types ---

  interface WiFiStatusResponse {
    state: string;
    connectivity: string;
    active_connection?: string;
  }

  interface WiFiNetwork {
    ssid: string;
    signal: number;
    security: string;
  }

  interface WiFiScanResponse {
    networks: WiFiNetwork[];
  }

  interface HotspotStatusResponse {
    active: boolean;
    name?: string;
  }

  interface HotspotStartResponse {
    success: boolean;
    ssid: string;
    password: string;
    ip: string;
  }

  // --- Connection tab state ---

  let wifiStatus = $state<WiFiStatusResponse | null>(null);
  let wifiStatusError = $state('');

  let networks = $state<WiFiNetwork[]>([]);
  let scanning = $state(false);
  let scanError = $state('');

  let selectedSSID = $state('');
  let useManualSSID = $state(false);
  let manualSSID = $state('');
  let password = $state('');
  let showPassword = $state(false);

  let connecting = $state(false);
  let connectMessage = $state('');
  let connectMessageType = $state<'success' | 'error'>('success');

  // Derived: effective SSID to connect to
  let effectiveSSID = $derived(useManualSSID ? manualSSID : selectedSSID);

  // Signal strength bars (4 levels)
  function signalBars(signal: number): string {
    if (signal >= 75) return '▂▄▆█';
    if (signal >= 50) return '▂▄▆ ';
    if (signal >= 25) return '▂▄  ';
    return '▂   ';
  }

  // Load initial WiFi status
  $effect(() => {
    loadWifiStatus();
  });

  async function loadWifiStatus() {
    wifiStatusError = '';
    try {
      const data = await api.get<WiFiStatusResponse>('/api/v2/wifi/status');
      wifiStatus = data;
    } catch (error) {
      logger.error('Failed to load WiFi status:', error);
      wifiStatusError = t('settings.wifi.errors.statusFailed');
    }
  }

  async function scanNetworks() {
    scanning = true;
    scanError = '';
    networks = [];
    try {
      const data = await api.get<WiFiScanResponse>('/api/v2/wifi/scan');
      networks = data.networks ?? [];
    } catch (error) {
      logger.error('Failed to scan WiFi networks:', error);
      scanError = t('settings.wifi.errors.scanFailed');
    } finally {
      scanning = false;
    }
  }

  async function connectToNetwork() {
    if (!effectiveSSID) return;
    connecting = true;
    connectMessage = '';
    try {
      await api.post('/api/v2/wifi/connect', { ssid: effectiveSSID, password });
      connectMessage = t('settings.wifi.connect.success');
      connectMessageType = 'success';
      // Refresh status after connecting
      await loadWifiStatus();
    } catch (error) {
      logger.error('Failed to connect to WiFi network:', error);
      connectMessage = t('settings.wifi.errors.connectFailed');
      connectMessageType = 'error';
    } finally {
      connecting = false;
    }
  }

  // --- Hotspot tab state ---

  let hotspotStatus = $state<HotspotStatusResponse | null>(null);
  let hotspotStatusError = $state('');

  let hotspotStarting = $state(false);
  let hotspotStopping = $state(false);
  let hotspotActionMessage = $state('');
  let hotspotActionType = $state<'success' | 'error'>('success');

  // Credentials revealed after starting hotspot
  let hotspotCredentials = $state<HotspotStartResponse | null>(null);

  $effect(() => {
    loadHotspotStatus();
  });

  async function loadHotspotStatus() {
    hotspotStatusError = '';
    try {
      const data = await api.get<HotspotStatusResponse>('/api/v2/wifi/hotspot/status');
      hotspotStatus = data;
    } catch (error) {
      logger.error('Failed to load hotspot status:', error);
      hotspotStatusError = t('settings.wifi.errors.hotspotStatusFailed');
    }
  }

  async function startHotspot() {
    hotspotStarting = true;
    hotspotActionMessage = '';
    hotspotCredentials = null;
    try {
      const data = await api.post<HotspotStartResponse>('/api/v2/wifi/hotspot/start', {});
      hotspotCredentials = data;
      hotspotActionMessage = '';
      await loadHotspotStatus();
    } catch (error) {
      logger.error('Failed to start hotspot:', error);
      hotspotActionMessage = t('settings.wifi.errors.hotspotStartFailed');
      hotspotActionType = 'error';
    } finally {
      hotspotStarting = false;
    }
  }

  async function stopHotspot() {
    hotspotStopping = true;
    hotspotActionMessage = '';
    hotspotCredentials = null;
    try {
      await api.post('/api/v2/wifi/hotspot/stop', {});
      hotspotActionMessage = '';
      await loadHotspotStatus();
    } catch (error) {
      logger.error('Failed to stop hotspot:', error);
      hotspotActionMessage = t('settings.wifi.errors.hotspotStopFailed');
      hotspotActionType = 'error';
    } finally {
      hotspotStopping = false;
    }
  }
</script>

{#snippet connectionTabContent()}
  <div class="space-y-6">
    <!-- WiFi Status Card -->
    <SettingsSection
      title={t('settings.wifi.status.title')}
      defaultOpen={true}
    >
      {#if wifiStatusError}
        <div
          class="flex items-center gap-2 py-2 px-3 rounded-lg text-sm bg-[color-mix(in_srgb,var(--color-error)_15%,transparent)] text-[var(--color-error)]"
          role="alert"
        >
          <XCircle class="size-4 shrink-0" />
          <span>{wifiStatusError}</span>
        </div>
      {:else if wifiStatus}
        <dl class="grid grid-cols-2 gap-x-6 gap-y-2 text-sm max-w-sm">
          <dt class="text-[var(--color-base-content)]/70">{t('settings.wifi.status.network')}</dt>
          <dd class="font-medium">
            {#if wifiStatus.active_connection}
              {wifiStatus.active_connection}
            {:else}
              <span class="opacity-50">&mdash;</span>
            {/if}
          </dd>

          <dt class="text-[var(--color-base-content)]/70">{t('settings.wifi.status.title')}</dt>
          <dd>
            {#if wifiStatus.state === 'connected'}
              <span class="inline-flex items-center gap-1 text-[var(--color-success)]">
                <CircleCheck class="size-4 shrink-0" />
                {t('settings.wifi.status.connected')}
              </span>
            {:else}
              <span class="inline-flex items-center gap-1 text-[var(--color-base-content)]/60">
                <XCircle class="size-4 shrink-0" />
                {t('settings.wifi.status.disconnected')}
              </span>
            {/if}
          </dd>

          <dt class="text-[var(--color-base-content)]/70">{t('settings.wifi.status.connectivity')}</dt>
          <dd class="capitalize">{wifiStatus.connectivity}</dd>
        </dl>
      {:else}
        <p class="text-sm text-[var(--color-base-content)]/60">{t('common.loading')}</p>
      {/if}
    </SettingsSection>

    <!-- Scan for Networks -->
    <SettingsSection
      title={t('settings.wifi.connect.title')}
      defaultOpen={true}
    >
      <div class="space-y-4">
        <!-- Scan Button -->
        <div class="flex items-center gap-3">
          <button
            onclick={scanNetworks}
            disabled={scanning}
            class="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md cursor-pointer transition-all bg-[var(--color-base-200)] text-[var(--color-base-content)] border border-[var(--border-200)] hover:bg-[var(--color-base-300)] disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <RefreshCw class={scanning ? 'size-4 animate-spin' : 'size-4'} />
            {scanning ? t('settings.wifi.scan.scanning') : t('settings.wifi.scan.button')}
          </button>
        </div>

        {#if scanError}
          <div
            class="flex items-center gap-2 py-2 px-3 rounded-lg text-sm bg-[color-mix(in_srgb,var(--color-error)_15%,transparent)] text-[var(--color-error)]"
            role="alert"
          >
            <XCircle class="size-4 shrink-0" />
            <span>{scanError}</span>
          </div>
        {/if}

        <!-- Network List -->
        {#if networks.length > 0}
          <div class="space-y-1">
            {#each networks as network (network.ssid)}
              <button
                onclick={() => { selectedSSID = network.ssid; useManualSSID = false; }}
                class="flex items-center justify-between w-full px-3 py-2 rounded-md text-sm transition-colors duration-150 border"
                class:border-[var(--color-primary)]={selectedSSID === network.ssid && !useManualSSID}
                class:bg-[color-mix(in_srgb,var(--color-primary)_10%,transparent)]={selectedSSID === network.ssid && !useManualSSID}
                class:border-[var(--border-200)]={selectedSSID !== network.ssid || useManualSSID}
                class:hover:bg-[var(--color-base-200)]={selectedSSID !== network.ssid || useManualSSID}
              >
                <span class="font-medium">{network.ssid}</span>
                <span class="font-mono text-xs tracking-widest text-[var(--color-base-content)]/60" aria-label="{network.signal}%">
                  {signalBars(network.signal)}
                </span>
              </button>
            {/each}

            <!-- Other network option -->
            <button
              onclick={() => { useManualSSID = true; selectedSSID = ''; }}
              class="flex items-center w-full px-3 py-2 rounded-md text-sm transition-colors duration-150 border"
              class:border-[var(--color-primary)]={useManualSSID}
              class:bg-[color-mix(in_srgb,var(--color-primary)_10%,transparent)]={useManualSSID}
              class:border-[var(--border-200)]={!useManualSSID}
              class:hover:bg-[var(--color-base-200)]={!useManualSSID}
            >
              {t('settings.wifi.scan.otherNetwork')}
            </button>
          </div>
        {:else if !scanning && networks.length === 0}
          <!-- Show "Other network" even without scan results -->
          <div class="space-y-1">
            <button
              onclick={() => { useManualSSID = true; selectedSSID = ''; }}
              class="flex items-center w-full px-3 py-2 rounded-md text-sm transition-colors duration-150 border"
              class:border-[var(--color-primary)]={useManualSSID}
              class:bg-[color-mix(in_srgb,var(--color-primary)_10%,transparent)]={useManualSSID}
              class:border-[var(--border-200)]={!useManualSSID}
              class:hover:bg-[var(--color-base-200)]={!useManualSSID}
            >
              {t('settings.wifi.scan.otherNetwork')}
            </button>
          </div>
        {/if}

        <!-- Manual SSID input -->
        {#if useManualSSID}
          <div>
            <label class="block text-sm mb-1" for="manualSSID">
              {t('settings.wifi.scan.manualSSID')}
            </label>
            <input
              id="manualSSID"
              type="text"
              bind:value={manualSSID}
              placeholder={t('settings.wifi.scan.manualSSID')}
              class="block w-full max-w-sm px-3 py-1.5 text-sm bg-[var(--color-base-100)] text-[var(--color-base-content)] border border-[var(--border-200)] rounded-md transition-all focus:outline-none focus:border-[var(--color-primary)] focus:ring-2 focus:ring-[var(--color-primary)]/10"
            />
          </div>
        {/if}

        <!-- Password field -->
        {#if effectiveSSID || useManualSSID}
          <div>
            <label class="block text-sm mb-1" for="wifiPassword">
              {t('settings.wifi.connect.password')}
            </label>
            <div class="relative max-w-sm">
              <input
                id="wifiPassword"
                type={showPassword ? 'text' : 'password'}
                bind:value={password}
                placeholder={t('settings.wifi.connect.password')}
                class="block w-full pr-10 px-3 py-1.5 text-sm bg-[var(--color-base-100)] text-[var(--color-base-content)] border border-[var(--border-200)] rounded-md transition-all focus:outline-none focus:border-[var(--color-primary)] focus:ring-2 focus:ring-[var(--color-primary)]/10"
              />
              <button
                type="button"
                onclick={() => (showPassword = !showPassword)}
                class="absolute right-2 top-1/2 -translate-y-1/2 text-[var(--color-base-content)]/60 hover:text-[var(--color-base-content)] transition-colors"
                aria-label={showPassword ? t('settings.wifi.connect.hidePassword') : t('settings.wifi.connect.showPassword')}
              >
                {#if showPassword}
                  <EyeOff class="size-4" />
                {:else}
                  <Eye class="size-4" />
                {/if}
              </button>
            </div>
          </div>
        {/if}

        <!-- Connect button -->
        <div class="flex items-center gap-3">
          <button
            onclick={connectToNetwork}
            disabled={connecting || !effectiveSSID}
            class="inline-flex items-center justify-center gap-2 px-4 py-2 text-sm font-medium rounded-md cursor-pointer transition-all bg-[var(--color-primary)] text-[var(--color-primary-content)] border border-[var(--color-primary)] hover:bg-[var(--color-primary-hover)] disabled:opacity-50 disabled:cursor-not-allowed focus-visible:outline-2 focus-visible:outline-[var(--color-primary)] focus-visible:outline-offset-2"
          >
            {#if connecting}
              <span
                class="inline-block w-4 h-4 border-2 border-[var(--color-primary-content)]/30 border-t-[var(--color-primary-content)] rounded-full animate-spin"
              ></span>
              {t('settings.wifi.connect.connecting')}
            {:else}
              <Wifi class="size-4" />
              {t('settings.wifi.connect.button')}
            {/if}
          </button>
        </div>

        <!-- Connect feedback -->
        {#if connectMessage}
          <div
            class="flex items-center gap-2 py-2 px-3 rounded-lg text-sm"
            class:bg-[color-mix(in_srgb,var(--color-success)_15%,transparent)]={connectMessageType === 'success'}
            class:text-[var(--color-success)]={connectMessageType === 'success'}
            class:bg-[color-mix(in_srgb,var(--color-error)_15%,transparent)]={connectMessageType === 'error'}
            class:text-[var(--color-error)]={connectMessageType === 'error'}
            role="status"
            aria-live="polite"
          >
            {#if connectMessageType === 'success'}
              <CircleCheck class="size-4 shrink-0" />
            {:else}
              <XCircle class="size-4 shrink-0" />
            {/if}
            <span>{connectMessage}</span>
          </div>
        {/if}
      </div>
    </SettingsSection>
  </div>
{/snippet}

{#snippet hotspotTabContent()}
  <div class="space-y-6">
    <!-- Hotspot Status Card -->
    <SettingsSection
      title={t('settings.wifi.hotspot.status')}
      defaultOpen={true}
    >
      {#if hotspotStatusError}
        <div
          class="flex items-center gap-2 py-2 px-3 rounded-lg text-sm bg-[color-mix(in_srgb,var(--color-error)_15%,transparent)] text-[var(--color-error)]"
          role="alert"
        >
          <XCircle class="size-4 shrink-0" />
          <span>{hotspotStatusError}</span>
        </div>
      {:else if hotspotStatus}
        <dl class="grid grid-cols-2 gap-x-6 gap-y-2 text-sm max-w-sm mb-4">
          <dt class="text-[var(--color-base-content)]/70">{t('settings.wifi.hotspot.status')}</dt>
          <dd>
            {#if hotspotStatus.active}
              <span class="inline-flex items-center gap-1 text-[var(--color-success)]">
                <CircleCheck class="size-4 shrink-0" />
                {t('settings.wifi.hotspot.active')}
              </span>
            {:else}
              <span class="text-[var(--color-base-content)]/60">
                {t('settings.wifi.hotspot.inactive')}
              </span>
            {/if}
          </dd>

          {#if hotspotStatus.active && hotspotStatus.name}
            <dt class="text-[var(--color-base-content)]/70">{t('settings.wifi.hotspot.ssid')}</dt>
            <dd class="font-medium">{hotspotStatus.name}</dd>
          {/if}
        </dl>

        <!-- Start / Stop button -->
        {#if hotspotStatus.active}
          <button
            onclick={stopHotspot}
            disabled={hotspotStopping}
            class="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md cursor-pointer transition-all bg-[var(--color-error)] text-[var(--color-error-content)] hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {#if hotspotStopping}
              <span
                class="inline-block w-4 h-4 border-2 border-[var(--color-error-content)]/30 border-t-[var(--color-error-content)] rounded-full animate-spin"
              ></span>
              {t('settings.wifi.hotspot.stopping')}
            {:else}
              <Radio class="size-4" />
              {t('settings.wifi.hotspot.stop')}
            {/if}
          </button>
        {:else}
          <button
            onclick={startHotspot}
            disabled={hotspotStarting}
            class="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md cursor-pointer transition-all bg-[var(--color-primary)] text-[var(--color-primary-content)] hover:bg-[var(--color-primary-hover)] disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {#if hotspotStarting}
              <span
                class="inline-block w-4 h-4 border-2 border-[var(--color-primary-content)]/30 border-t-[var(--color-primary-content)] rounded-full animate-spin"
              ></span>
              {t('settings.wifi.hotspot.starting')}
            {:else}
              <Radio class="size-4" />
              {t('settings.wifi.hotspot.start')}
            {/if}
          </button>
        {/if}
      {:else}
        <p class="text-sm text-[var(--color-base-content)]/60">{t('common.loading')}</p>
      {/if}

      <!-- Hotspot action feedback -->
      {#if hotspotActionMessage}
        <div
          class="mt-3 flex items-center gap-2 py-2 px-3 rounded-lg text-sm bg-[color-mix(in_srgb,var(--color-error)_15%,transparent)] text-[var(--color-error)]"
          role="alert"
          aria-live="polite"
        >
          <XCircle class="size-4 shrink-0" />
          <span>{hotspotActionMessage}</span>
        </div>
      {/if}

      <!-- Credentials revealed after start -->
      {#if hotspotCredentials}
        <div class="mt-4 rounded-lg bg-[var(--color-base-200)] p-4 space-y-2">
          <dl class="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
            <dt class="text-[var(--color-base-content)]/70">{t('settings.wifi.hotspot.ssid')}</dt>
            <dd class="font-mono font-medium">{hotspotCredentials.ssid}</dd>

            <dt class="text-[var(--color-base-content)]/70">{t('settings.wifi.hotspot.password')}</dt>
            <dd class="font-mono font-medium">{hotspotCredentials.password}</dd>

            <dt class="text-[var(--color-base-content)]/70">{t('settings.wifi.hotspot.ip')}</dt>
            <dd class="font-mono font-medium">{hotspotCredentials.ip}</dd>
          </dl>
        </div>
      {/if}
    </SettingsSection>

    <!-- Informational note + setup link -->
    <SettingsSection
      title={t('settings.wifi.hotspot.title')}
      defaultOpen={true}
    >
      <div class="space-y-3">
        <div
          class="flex items-start gap-2 text-sm text-[var(--color-base-content)]/80"
        >
          <Info class="size-4 shrink-0 mt-0.5" />
          <span>{t('settings.wifi.hotspot.info')}</span>
        </div>

        <a
          href="/wifi-setup"
          target="_blank"
          rel="noopener noreferrer"
          class="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md transition-all bg-[var(--color-base-200)] text-[var(--color-base-content)] border border-[var(--border-200)] hover:bg-[var(--color-base-300)]"
        >
          <Wifi class="size-4" />
          {t('settings.wifi.hotspot.setupLink')}
        </a>
      </div>
    </SettingsSection>
  </div>
{/snippet}

<!-- Main Content -->
<main class="settings-page-content" aria-label="WiFi settings configuration">
  <SettingsTabs {tabs} bind:activeTab showActions={false} />
</main>
