<script lang="ts">
  /**
   * MiniSpectrogram — Compact live spectrogram widget for the dashboard
   *
   * Displays a real-time scrolling waterfall spectrogram of the first available
   * audio source. Manages its own HLS.js connection, heartbeat, and access control.
   *
   * Key design:
   * - Uses useSpectrogramAnalyser composable (no inline Web Audio graph)
   * - Source discovery via SSE audio-level stream
   * - Silent by default with speaker toggle
   * - Manual start/stop with localStorage persistence
   * - Respects appState.liveSpectrogram for auto-start
   */

  import Hls from 'hls.js';
  import ReconnectingEventSource from 'reconnecting-eventsource';
  import { onMount } from 'svelte';

  import { Volume, Volume1, Volume2, VolumeX, Play, Square, Tag } from '@lucide/svelte';
  import { t } from '$lib/i18n';
  import { appState, hasLiveAudioAccess } from '$lib/stores/appState.svelte';
  import { HLS_AUDIO_CONFIG } from '$lib/desktop/components/ui/hls-config';
  import { useSpectrogramAnalyser } from '$lib/utils/useSpectrogramAnalyser.svelte';
  import SpectrogramCanvas from '$lib/desktop/components/media/SpectrogramCanvas.svelte';
  import { fetchWithCSRF } from '$lib/utils/api';
  import { buildAppUrl } from '$lib/utils/urlHelpers';
  import { generateSessionId } from '$lib/utils/session';
  import { loggers } from '$lib/utils/logger';
  import type { ColorMapName } from '$lib/utils/spectrogramColorMaps';
  import type { PendingDetection } from '$lib/types/pending.types';
  import type { OverlayLabel, QueuedLabel } from '$lib/utils/detectionOverlay';
  import {
    diffPendingSnapshot,
    shouldDedup,
    promoteFromQueue,
    nextYSlot,
    computeWallClockAtPlayhead,
    STALE_DEDUP_PRUNE_SECONDS,
    LABEL_LEAD_IN_SECONDS,
  } from '$lib/utils/detectionOverlay';

  const logger = loggers.audio;
  const STORAGE_KEY = 'birdnet-spectrogram-active';
  const FFT_SIZE = 1024;
  const HEARTBEAT_INTERVAL = 20000;
  const SOURCE_DISCOVERY_TIMEOUT = 5000;
  const ZERO_SIGNAL_CHECK_INTERVAL_MS = 500;
  const ZERO_SIGNAL_GRACE_MS = 3000;
  /** How often (ms) to poll for label promotion and pruning */
  const LABEL_POLL_INTERVAL_MS = 200;
  /** Maximum label age (ms) before pruning from overlay */
  const LABEL_MAX_AGE_MS = 60000;

  const sessionId = generateSessionId();

  // Props
  let { pendingDetections = [] }: { pendingDetections?: PendingDetection[] } = $props();

  // Volume/gain presets: muted → 0dB → +6dB → +12dB
  const GAIN_PRESETS = [
    { db: -Infinity, audio: false, labelKey: 'spectrogram.gain.muted' },
    { db: 0, audio: true, labelKey: 'spectrogram.gain.level', value: '0' },
    { db: 6, audio: true, labelKey: 'spectrogram.gain.level', value: '+6' },
    { db: 12, audio: true, labelKey: 'spectrogram.gain.level', value: '+12' },
  ] as const;

  // Local state
  let isActive = $state(false);
  let isConnecting = $state(false);
  let gainPresetIndex = $state(0);
  let showDetectionLabels = $state(true);
  let liveSignalUnavailable = $state(false);

  const gainLabel = $derived.by(() => {
    // eslint-disable-next-line security/detect-object-injection -- gainPresetIndex is a numeric index bounded by GAIN_PRESETS.length
    const preset = GAIN_PRESETS[gainPresetIndex];
    return 'value' in preset ? t(preset.labelKey, { value: preset.value }) : t(preset.labelKey);
  });
  let colorMap = $state<ColorMapName>('inferno');
  let frequencyRange = $state<[number, number]>([0, 15000]);

  // Stream token state
  let activeStreamToken: string | null = null;

  // HLS + audio refs
  let hls: Hls | null = null;
  let audioElement = $state<HTMLAudioElement | null>(null);
  let heartbeatTimer: ReturnType<typeof globalThis.setInterval> | null = null;
  let abortController: AbortController | null = null;
  let activeSourceId = $state<string | null>(null);
  let zeroSignalSinceMs = 0;

  // Detection overlay state
  let overlayLabels = $state<OverlayLabel[]>([]);
  let labelQueue: QueuedLabel[] = [];
  let prevSnapshot: PendingDetection[] = [];
  let lastSeenSpecies = new Map<string, number>();
  let slotCounter = 0;
  const MAX_OVERLAY_SLOTS = 4;

  // Initialize composable during component init (must be at top level for $effect cleanup)
  const spectro = useSpectrogramAnalyser({ fftSize: FFT_SIZE, audioOutput: false });

  function isWebKitBrowser(): boolean {
    const ua = globalThis.navigator?.userAgent ?? '';
    const vendor = globalThis.navigator?.vendor ?? '';
    return (
      /Apple/i.test(vendor) && /Safari/i.test(ua) && !/Chrome|Chromium|CriOS|FxiOS|EdgiOS/i.test(ua)
    );
  }

  const liveSignalUnavailableMessage = $derived.by(() => {
    if (!liveSignalUnavailable) {
      return '';
    }

    if (isWebKitBrowser()) {
      return 'Live spectrogram is unavailable in Safari/WebKit for HLS audio streams. Audio and detections still work, but the graph currently requires Chrome, Brave, or Firefox.';
    }

    return 'Live spectrogram data is not reaching the browser audio analyser even though the stream is connected.';
  });

  function shouldAutoStart(): boolean {
    if (appState.liveSpectrogram) return true;
    try {
      return globalThis.localStorage?.getItem(STORAGE_KEY) === 'true';
    } catch {
      return false;
    }
  }

  function persistToggleState(active: boolean) {
    try {
      if (active) {
        globalThis.localStorage?.setItem(STORAGE_KEY, 'true');
      } else {
        globalThis.localStorage?.removeItem(STORAGE_KEY);
      }
    } catch {
      /* localStorage not available */
    }
  }

  /**
   * Discover the first available audio source via the SSE audio-level stream.
   * Returns the source ID, or null if none found within the timeout.
   * Accepts an AbortSignal so callers can cancel discovery mid-flight.
   */
  async function discoverFirstSource(signal: AbortSignal): Promise<string | null> {
    return new Promise(resolve => {
      if (signal.aborted) {
        resolve(null);
        return;
      }

      const sse = new ReconnectingEventSource(buildAppUrl('/api/v2/streams/audio-level'), {
        max_retry_time: 30000,
        withCredentials: false,
      });

      // Abort listener: clean up SSE and timeout if caller cancels
      const onAbort = () => {
        globalThis.clearTimeout(timeout);
        sse.close();
        resolve(null);
      };

      const timeout = globalThis.setTimeout(() => {
        signal?.removeEventListener('abort', onAbort);
        sse.close();
        resolve(null);
      }, SOURCE_DISCOVERY_TIMEOUT);

      signal.addEventListener('abort', onAbort, { once: true });

      sse.onmessage = (event: globalThis.MessageEvent) => {
        try {
          const data = JSON.parse(event.data) as {
            type?: string;
            levels?: Record<string, unknown>;
          };
          if (data.type === 'audio-level' && data.levels) {
            const sourceIds = Object.keys(data.levels);
            if (sourceIds.length > 0) {
              globalThis.clearTimeout(timeout);
              signal?.removeEventListener('abort', onAbort);
              sse.close();
              resolve(sourceIds[0]);
            }
          }
        } catch {
          /* ignore parse errors */
        }
      };

      sse.onerror = () => {
        /* ReconnectingEventSource handles reconnection */
      };
    });
  }

  async function startElementPlayback(
    mediaElement: HTMLAudioElement,
    options?: { bootstrapMuted?: boolean }
  ): Promise<void> {
    const bootstrapMuted = options?.bootstrapMuted ?? false;
    const previousMuted = mediaElement.muted;

    if (bootstrapMuted) {
      mediaElement.muted = true;
    }

    try {
      await mediaElement.play();
    } finally {
      // Use the Web Audio graph, not the HTMLMediaElement mute flag, as the
      // long-lived source of truth for silent monitoring. Keeping the element
      // muted can cause Safari to feed silence into the analyser.
      mediaElement.muted = false;

      // Preserve the previous mute state if playback never started and we
      // weren't intentionally bootstrapping muted autoplay.
      if (!bootstrapMuted && mediaElement.paused) {
        mediaElement.muted = previousMuted;
      }
    }
  }

  async function start() {
    if (isActive || isConnecting) return;
    isConnecting = true;
    void spectro.prime();

    // Abort any previous in-flight operation
    abortController?.abort();
    const controller = new AbortController();
    abortController = controller;
    const { signal } = controller;

    try {
      const sourceId = await discoverFirstSource(signal);
      if (signal.aborted || !sourceId) {
        if (!signal.aborted) isConnecting = false;
        return;
      }

      const encodedSourceId = encodeURIComponent(sourceId);

      // Capture source before await so stop() can clean up if the request
      // is aborted after the server processes it
      activeSourceId = sourceId;

      const data = await fetchWithCSRF<{
        status: string;
        stream_token: string;
        playlist_url: string;
        playlist_ready: boolean;
      }>(`/api/v2/streams/hls/${encodedSourceId}/start`, {
        method: 'POST',
        signal,
        body: { session_id: sessionId },
      });

      if (signal.aborted) return;

      activeStreamToken = data.stream_token;
      const hlsUrl = buildAppUrl(data.playlist_url);

      audioElement = new globalThis.Audio();
      audioElement.crossOrigin = 'anonymous';
      audioElement.setAttribute('playsinline', 'true');
      // Use muted playback only as an autoplay bootstrap. Once playback starts,
      // the Web Audio output gain node controls whether speakers are silent.
      audioElement.muted = true;

      if (Hls.isSupported()) {
        hls = new Hls(HLS_AUDIO_CONFIG);
        hls.loadSource(hlsUrl);
        hls.attachMedia(audioElement);

        hls.on(Hls.Events.MANIFEST_PARSED, async () => {
          if (signal.aborted) return;
          if (audioElement) {
            await spectro.connect(audioElement);
          }
          if (signal.aborted) {
            spectro.disconnect();
            return;
          }

          if (audioElement) {
            try {
              await startElementPlayback(audioElement, { bootstrapMuted: true });
            } catch {
              /* autoplay blocked — spectrogram can recover on user gesture */
            }
          }

          if (signal.aborted) return;

          startHeartbeat(activeStreamToken!);
          isActive = true;
          isConnecting = false;
          persistToggleState(true);
        });

        hls.on(Hls.Events.ERROR, (_event, data) => {
          if (signal.aborted) return;
          if (data.fatal) {
            logger.error('MiniSpectrogram: fatal HLS error', {
              type: data.type,
              details: data.details,
            });
            stop();
          }
        });
      } else if (audioElement.canPlayType('application/vnd.apple.mpegurl')) {
        // Native HLS (Safari / iOS)
        audioElement.src = hlsUrl;
        await spectro.connect(audioElement);
        if (signal.aborted) {
          spectro.disconnect();
          return;
        }

        try {
          await startElementPlayback(audioElement, { bootstrapMuted: true });
        } catch {
          /* autoplay blocked */
        }

        if (signal.aborted || !audioElement) return;

        startHeartbeat(activeStreamToken!);
        isActive = true;
        isConnecting = false;
        persistToggleState(true);
      } else {
        // Browser supports neither HLS.js nor native HLS — tear down
        logger.warn('MiniSpectrogram: browser does not support HLS');
        stop();
        return;
      }
    } catch (error) {
      if (signal.aborted) return;
      logger.error('MiniSpectrogram: failed to start', error);
      isConnecting = false;
    }
  }

  function startHeartbeat(token: string) {
    stopHeartbeat();
    const sendHeartbeat = async () => {
      try {
        await fetchWithCSRF('/api/v2/streams/hls/heartbeat', {
          method: 'POST',
          body: { stream_token: token, session_id: sessionId },
        });
      } catch {
        /* ignore heartbeat failures */
      }
    };
    sendHeartbeat();
    heartbeatTimer = globalThis.setInterval(sendHeartbeat, HEARTBEAT_INTERVAL);
  }

  function stopHeartbeat() {
    if (heartbeatTimer) {
      globalThis.clearInterval(heartbeatTimer);
      heartbeatTimer = null;
    }
  }

  // stopRuntime tears down the stream without clearing localStorage persistence.
  // Used by $effect cleanup so reactive re-runs don't erase the user's play preference.
  function stopRuntime() {
    abortController?.abort();
    abortController = null;

    // Send explicit stop for immediate server-side client removal
    if (activeSourceId) {
      const encodedSourceId = encodeURIComponent(activeSourceId);
      fetchWithCSRF(`/api/v2/streams/hls/${encodedSourceId}/stop`, {
        method: 'POST',
        keepalive: true,
        body: { session_id: sessionId },
      }).catch(() => {});
      activeSourceId = null;
    }

    // Send disconnect heartbeat as fallback
    if (activeStreamToken) {
      fetchWithCSRF('/api/v2/streams/hls/heartbeat?disconnect=true', {
        method: 'POST',
        keepalive: true,
        body: { stream_token: activeStreamToken, session_id: sessionId },
      }).catch(() => {});
      activeStreamToken = null;
    }

    stopHeartbeat();
    spectro.disconnect();

    if (hls) {
      hls.destroy();
      hls = null;
    }
    if (audioElement) {
      audioElement.pause();
      audioElement.removeAttribute('src');
      audioElement = null;
    }

    // Clear detection overlay state
    overlayLabels = [];
    labelQueue = [];
    prevSnapshot = [];
    lastSeenSpecies.clear();
    slotCounter = 0;
    zeroSignalSinceMs = 0;
    liveSignalUnavailable = false;

    isActive = false;
    isConnecting = false;
  }

  // stop tears down the stream AND clears the user's auto-start preference.
  // Used by explicit user actions (stop button, fatal errors).
  function stop() {
    stopRuntime();
    persistToggleState(false);
  }

  function cycleVolume() {
    gainPresetIndex = (gainPresetIndex + 1) % GAIN_PRESETS.length;
    // eslint-disable-next-line security/detect-object-injection -- gainPresetIndex is a numeric index bounded by modulo
    const preset = GAIN_PRESETS[gainPresetIndex];
    if (audioElement) {
      if (audioElement.paused) {
        audioElement.play().catch(() => {
          // Ignore autoplay errors here; the spectrogram can still resume on a
          // later user gesture via the deferred AudioContext resume handler.
        });
      }
    }
    spectro.setAudioOutput(preset.audio);
    // Always update gain -- when muted (db: -Infinity), this resets the
    // spectrogram visualization to 0 dB instead of leaving it at max gain.
    spectro.setGain(preset.audio ? preset.db : 0);
  }

  // Diff incoming pending detections and queue new labels.
  // Always update prevSnapshot — even when pendingDetections is empty — so
  // stale species are cleared after detections stop.
  $effect(() => {
    if (!activeSourceId) return;
    if (pendingDetections.length > 0) {
      const newDetections = diffPendingSnapshot(prevSnapshot, pendingDetections, activeSourceId);
      const nowUnix = Date.now() / 1000;
      for (const det of newDetections) {
        if (shouldDedup(det.species, nowUnix, lastSeenSpecies)) continue;
        lastSeenSpecies.set(det.species, nowUnix);
        const { slot, next } = nextYSlot(slotCounter, MAX_OVERLAY_SLOTS);
        slotCounter = next;
        labelQueue.push({
          text: det.species,
          firstDetected: (det.audioCapturedAt ?? det.firstDetected) - LABEL_LEAD_IN_SECONDS,
          ySlot: slot,
        });
      }
    }
    prevSnapshot = [...pendingDetections];
  });

  // Promote queued detection labels when playhead catches up.
  // Prefers hls.playingDate (wall-clock time interpolated from
  // #EXT-X-PROGRAM-DATE-TIME tags). Falls back to client wall-clock
  // for native HLS (Safari/iOS) where hls.js is not used.
  $effect(() => {
    if (!audioElement) return;

    const interval = globalThis.setInterval(() => {
      if (!audioElement) return;

      const now = globalThis.performance.now();
      const nowUnix = Date.now() / 1000;

      const wallClockAtPlayhead = computeWallClockAtPlayhead(
        audioElement,
        hls?.playingDate ?? null,
        nowUnix
      );

      // Promote queued labels when playhead is available
      if (wallClockAtPlayhead > 0 && labelQueue.length > 0) {
        const { promoted, remaining } = promoteFromQueue(labelQueue, wallClockAtPlayhead, now);
        if (promoted.length > 0) {
          labelQueue = remaining;
          overlayLabels = [...overlayLabels, ...promoted];
        }
      }

      // Prune labels older than LABEL_MAX_AGE_MS
      const cutoff = now - LABEL_MAX_AGE_MS;
      overlayLabels = overlayLabels.filter(l => l.birthTime >= cutoff);

      // Prune stale dedup entries
      for (const [species, time] of lastSeenSpecies) {
        if (nowUnix - time > STALE_DEDUP_PRUNE_SECONDS) {
          lastSeenSpecies.delete(species);
        }
      }
    }, LABEL_POLL_INTERVAL_MS);

    return () => globalThis.clearInterval(interval);
  });

  $effect(() => {
    if (!audioElement || !spectro.analyser || !spectro.isActive || !isActive) {
      zeroSignalSinceMs = 0;
      liveSignalUnavailable = false;
      return;
    }

    const probe = new Uint8Array(spectro.analyser.frequencyBinCount);
    const interval = globalThis.setInterval(() => {
      if (!audioElement || !spectro.analyser) return;

      const mediaAdvancing =
        !audioElement.paused && audioElement.currentTime > 0.5 && audioElement.readyState >= 2;
      if (!mediaAdvancing) {
        zeroSignalSinceMs = 0;
        liveSignalUnavailable = false;
        return;
      }

      spectro.analyser.getByteFrequencyData(probe);
      let peak = 0;
      for (let i = 0; i < probe.length; i++) {
        // eslint-disable-next-line security/detect-object-injection -- loop index is bounded by typed array length
        if (probe[i] > peak) peak = probe[i];
      }

      if (peak > 0) {
        zeroSignalSinceMs = 0;
        liveSignalUnavailable = false;
        return;
      }

      if (zeroSignalSinceMs === 0) {
        zeroSignalSinceMs = Date.now();
        return;
      }

      if (Date.now() - zeroSignalSinceMs >= ZERO_SIGNAL_GRACE_MS) {
        liveSignalUnavailable = true;
      }
    }, ZERO_SIGNAL_CHECK_INTERVAL_MS);

    return () => globalThis.clearInterval(interval);
  });

  // Use onMount (NOT $effect) for auto-start. This is fire-once initialization:
  // - $effect caused effect_update_depth_exceeded because start() reads $state
  //   variables (isActive, isConnecting) which become tracked dependencies,
  //   creating a cleanup→restart infinite loop.
  // - The "re-run on auth change" use case ($effect) is not worth the complexity;
  //   if a user logs in after mount, they can click the play button.
  onMount(() => {
    if (hasLiveAudioAccess() && shouldAutoStart()) {
      start();
    }
    return () => stop();
  });
</script>

{#if hasLiveAudioAccess()}
  <div
    class="overflow-hidden rounded-2xl border border-border-100 bg-[var(--color-base-100)] shadow-sm"
  >
    <div
      class="flex items-center justify-between border-b border-[var(--color-base-200)] px-6 py-4"
    >
      <h3 class="font-semibold">{t('spectrogram.dashboard.toggle')}</h3>
      <div class="flex items-center gap-1">
        {#if isActive}
          <button
            type="button"
            onclick={() => {
              showDetectionLabels = !showDetectionLabels;
            }}
            class="rounded p-1 transition-colors {showDetectionLabels
              ? 'bg-[var(--color-primary)]/20 text-[var(--color-primary)]'
              : 'text-[var(--color-base-content)]/60 hover:bg-[var(--color-base-200)]'}"
            aria-label={t('spectrogram.labels.toggle')}
            aria-pressed={showDetectionLabels}
            title={t('spectrogram.labels.toggle')}
          >
            <Tag class="size-4" />
          </button>
          <button
            onclick={cycleVolume}
            class="rounded p-1 transition-colors hover:bg-[var(--color-base-200)]"
            aria-label={gainLabel}
            title={gainLabel}
          >
            {#if gainPresetIndex === 0}
              <VolumeX class="size-4" />
            {:else if gainPresetIndex === 1}
              <Volume class="size-4" />
            {:else if gainPresetIndex === 2}
              <Volume1 class="size-4" />
            {:else}
              <Volume2 class="size-4" />
            {/if}
          </button>
          <button
            onclick={stop}
            class="rounded p-1 transition-colors hover:bg-[var(--color-base-200)]"
            aria-label={t('media.audio.stop')}
            title={t('media.audio.stop')}
          >
            <Square class="size-4" />
          </button>
        {:else}
          <button
            onclick={start}
            class="rounded p-1 transition-colors hover:bg-[var(--color-base-200)]"
            disabled={isConnecting}
            aria-label={t('media.audio.play')}
            title={t('media.audio.play')}
          >
            <Play class="size-4" />
          </button>
        {/if}
      </div>
    </div>

    {#if (isActive || isConnecting) && spectro.isActive}
      <div class="relative h-28 w-full">
        <SpectrogramCanvas
          analyser={spectro.analyser}
          frequencyData={spectro.frequencyData}
          sampleRate={spectro.sampleRate}
          fftSize={FFT_SIZE}
          {frequencyRange}
          {colorMap}
          isActive={spectro.isActive}
          overlayLabels={showDetectionLabels ? overlayLabels : []}
          overlayFontSize={9}
          className="h-28 w-full"
        />
        {#if liveSignalUnavailable}
          <div
            class="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/75 px-4 text-center"
          >
            <p class="max-w-md text-xs leading-5 text-[var(--color-base-content)]/80">
              {liveSignalUnavailableMessage}
            </p>
          </div>
        {/if}
      </div>
    {:else if isActive || isConnecting}
      <div class="flex h-28 items-center justify-center bg-black">
        <div
          class="h-5 w-5 animate-spin rounded-full border-2 border-[var(--color-primary)] border-t-transparent"
        ></div>
      </div>
    {/if}
  </div>
{/if}
