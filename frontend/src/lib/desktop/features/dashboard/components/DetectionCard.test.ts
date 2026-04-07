import { render, screen, cleanup } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { writable } from 'svelte/store';
import DetectionCard from './DetectionCard.svelte';
import type { Detection } from '$lib/types/detection.types';

const mockLoader = {
  start: vi.fn(),
  stop: vi.fn(),
  destroy: vi.fn(),
  handleImageLoad: vi.fn(),
  handleImageError: vi.fn(),
  showSpinner: false,
  isQueued: false,
  isGenerating: false,
  error: false,
  spectrogramUrl: '',
  state: 'idle',
};

vi.mock('./ConfidenceBadge.svelte', () => ({
  default: vi.fn(() => ({ $set: vi.fn(), $destroy: vi.fn(), $on: vi.fn() })),
}));
vi.mock('./WeatherBadge.svelte', () => ({
  default: vi.fn(() => ({ $set: vi.fn(), $destroy: vi.fn(), $on: vi.fn() })),
}));
vi.mock('./MoonBadge.svelte', () => ({
  default: vi.fn(() => ({ $set: vi.fn(), $destroy: vi.fn(), $on: vi.fn() })),
}));
vi.mock('./PlayOverlay.svelte', () => ({
  default: vi.fn(() => ({ $set: vi.fn(), $destroy: vi.fn(), $on: vi.fn() })),
}));
vi.mock('./SpeciesInfoBar.svelte', () => ({
  default: vi.fn(() => ({ $set: vi.fn(), $destroy: vi.fn(), $on: vi.fn() })),
}));
vi.mock('./CardActionMenu.svelte', () => ({
  default: vi.fn(() => ({ $set: vi.fn(), $destroy: vi.fn(), $on: vi.fn() })),
}));
vi.mock('./AudioSettingsButton.svelte', () => ({
  default: vi.fn(() => ({ $set: vi.fn(), $destroy: vi.fn(), $on: vi.fn() })),
}));
vi.mock('$lib/utils/spectrogramLoader.svelte', () => ({
  createSpectrogramLoader: vi.fn(() => mockLoader),
}));
vi.mock('$lib/stores/settings', () => ({
  dashboardSettings: writable({ defaultAudioGain: 0 }),
}));

const baseDetection: Detection = {
  id: 42,
  date: '2026-04-07',
  time: '15:22:00',
  timestamp: '2026-04-07T15:22:00-04:00',
  source: 'Front Yard',
  beginTime: '2026-04-07T15:21:57-04:00',
  endTime: '2026-04-07T15:22:00-04:00',
  speciesCode: 'norcar',
  scientificName: 'Cardinalis cardinalis',
  commonName: 'Northern Cardinal',
  confidence: 0.91,
  verified: 'unverified',
  locked: false,
};

describe('DetectionCard', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  it('shows a clear empty-media message when no clip is available', () => {
    render(DetectionCard, {
      props: {
        detection: baseDetection,
      },
    });

    expect(screen.getByText('No recording available')).toBeInTheDocument();
    expect(screen.getByText('No spectrogram available')).toBeInTheDocument();
  });

  it('does not show the empty-media message when a clip is available', () => {
    render(DetectionCard, {
      props: {
        detection: {
          ...baseDetection,
          clipName: 'northern_cardinal.wav',
        },
      },
    });

    expect(screen.queryByText('No recording available')).not.toBeInTheDocument();
    expect(screen.queryByText('No spectrogram available')).not.toBeInTheDocument();
  });

  it('keeps the spectrogram image mounted while the loading spinner is visible', () => {
    mockLoader.spectrogramUrl = '/api/v2/spectrogram/42?size=md&raw=true';
    mockLoader.showSpinner = true;
    mockLoader.state = 'loading';

    const { container } = render(DetectionCard, {
      props: {
        detection: {
          ...baseDetection,
          clipName: 'northern_cardinal.wav',
        },
      },
    });

    const image = container.querySelector('img.spectrogram-image');
    expect(image).toBeInTheDocument();
    expect(screen.getByRole('img', { name: 'Spectrogram for Northern Cardinal' })).toBeInTheDocument();
  });
});
