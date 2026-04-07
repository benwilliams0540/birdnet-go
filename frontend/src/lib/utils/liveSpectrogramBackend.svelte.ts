import ReconnectingEventSource from 'reconnecting-eventsource';

import { buildAppUrl } from '$lib/utils/urlHelpers';
import { loggers } from '$lib/utils/logger';

const logger = loggers.audio;

const DEFAULT_FFT_SIZE = 1024;
const DEFAULT_SAMPLE_RATE = 48000;
const BYTE_MAX = 255;

interface LiveSpectrogramMessage {
  type?: string;
  sourceId?: string;
  sampleRate?: number;
  fftSize?: number;
  bins?: string;
}

function decodeBase64ToUint8Array(encoded: string): Uint8Array<ArrayBuffer> {
  const binary = globalThis.atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes as Uint8Array<ArrayBuffer>;
}

function clampByte(value: number): number {
  if (!Number.isFinite(value) || value <= 0) {
    return 0;
  }
  if (value >= BYTE_MAX) {
    return BYTE_MAX;
  }
  return value;
}

function gainDbToScale(db: number): number {
  return 10 ** (db / 20);
}

export function isWebKitLiveSpectrogramBrowser(): boolean {
  const ua = globalThis.navigator?.userAgent ?? '';
  const vendor = globalThis.navigator?.vendor ?? '';
  return /Apple/i.test(vendor) && /Safari/i.test(ua) && !/Chrome|Chromium|CriOS|FxiOS|EdgiOS/i.test(ua);
}

export function useLiveSpectrogramBackend(options?: { gainDb?: number }) {
  let eventSource: ReconnectingEventSource | null = null;
  let latestBins = new Uint8Array(DEFAULT_FFT_SIZE / 2) as Uint8Array<ArrayBuffer>;
  let frameToken = 0;

  let analyser = $state<AnalyserNode | null>(null);
  let frequencyData = $state<Uint8Array<ArrayBuffer>>(new Uint8Array(DEFAULT_FFT_SIZE / 2));
  let fftSize = $state(DEFAULT_FFT_SIZE);
  let sampleRate = $state(DEFAULT_SAMPLE_RATE);
  let isActive = $state(false);
  let lastFrameAtMs = $state(0);
  let gainDb = $state(options?.gainDb ?? 0);

  const analyserLike = {
    get frequencyBinCount() {
      return latestBins.length;
    },
    get frameToken() {
      return frameToken;
    },
    getByteFrequencyData(target: Uint8Array<ArrayBuffer>) {
      const scale = gainDbToScale(gainDb);
      const length = Math.min(target.length, latestBins.length);
      for (let i = 0; i < length; i++) {
        target[i] = clampByte(Math.round(latestBins[i] * scale));
      }
      for (let i = length; i < target.length; i++) {
        target[i] = 0;
      }
    },
  } as unknown as AnalyserNode;

  analyser = analyserLike;

  function disconnect(): void {
    eventSource?.close();
    eventSource = null;
    isActive = false;
    lastFrameAtMs = 0;
    latestBins = new Uint8Array(fftSize / 2) as Uint8Array<ArrayBuffer>;
  }

  function connect(sourceId: string): void {
    disconnect();

    eventSource = new ReconnectingEventSource(
      buildAppUrl(`/api/v2/streams/live-spectrogram?source=${encodeURIComponent(sourceId)}`),
      {
        max_retry_time: 30000,
        withCredentials: false,
      }
    );

    eventSource.onmessage = event => {
      try {
        const message = JSON.parse(event.data) as LiveSpectrogramMessage;
        if (message.type !== 'live-spectrogram' || !message.bins) {
          return;
        }

        const decodedBins = decodeBase64ToUint8Array(message.bins);
        latestBins = decodedBins;

        const nextFFTSize = message.fftSize ?? decodedBins.length * 2;
        if (nextFFTSize !== fftSize || frequencyData.length !== decodedBins.length) {
          fftSize = nextFFTSize;
          frequencyData = new Uint8Array(decodedBins.length);
        }

        if (message.sampleRate) {
          sampleRate = message.sampleRate;
        }

        frameToken += 1;
        lastFrameAtMs = Date.now();
        isActive = true;
      } catch (error) {
        logger.warn('Failed to parse live spectrogram event', error);
      }
    };

    eventSource.onerror = () => {
      logger.debug('Live spectrogram SSE error, will auto-reconnect');
    };
  }

  function setGain(db: number): void {
    gainDb = db;
  }

  $effect(() => {
    return () => disconnect();
  });

  return {
    get analyser() {
      return analyser;
    },
    get frequencyData() {
      return frequencyData;
    },
    get fftSize() {
      return fftSize;
    },
    get sampleRate() {
      return sampleRate;
    },
    get isActive() {
      return isActive;
    },
    get lastFrameAtMs() {
      return lastFrameAtMs;
    },
    connect,
    disconnect,
    setGain,
  };
}
