package api

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/labstack/echo/v4"
	"github.com/tphakala/birdnet-go/internal/audiocore"
	"github.com/tphakala/birdnet-go/internal/conf"
	"github.com/tphakala/birdnet-go/internal/logger"
	"github.com/tphakala/birdnet-go/internal/privacy"
)

const (
	liveSpectrogramEndpoint = "/api/v2/streams/live-spectrogram"

	liveSpectrogramFFTSize           = 1024
	liveSpectrogramChannelBuffer     = 8
	liveSpectrogramSubscriberBuffer  = 32
	liveSpectrogramHeartbeatInterval = 10 * time.Second
	liveSpectrogramColumnsPerSecond  = 60

	liveSpectrogramMinDecibels  = -100.0
	liveSpectrogramMaxDecibels  = -30.0
	liveSpectrogramMagnitudeEps = 1e-12
	liveSpectrogramSmoothing    = 0.8
	liveSpectrogramByteMax      = 255
)

type liveSpectrogramEnvelope struct {
	Type       string `json:"type"`
	SourceID   string `json:"sourceId"`
	SampleRate int    `json:"sampleRate"`
	FFTSize    int    `json:"fftSize"`
	Bins       string `json:"bins"`
}

type liveSpectrogramManager struct {
	mu   sync.Mutex
	taps map[string]*liveSpectrogramTap
}

type liveSpectrogramTap struct {
	sourceID     string
	sampleRate   int
	fftSize      int
	hopSamples   int64
	consumerID   string
	consumer     *liveSpectrogramConsumer
	cleanupRoute func()

	subscribers   map[chan []byte]struct{}
	subscribersMu sync.RWMutex

	ring         []float64
	ringPos      int
	totalSamples int64
	nextEmitAt   int64
	hann         []float64
	twiddle      []complex128
	smoothed     []float64
}

type liveSpectrogramConsumer struct {
	id        string
	rate      int
	depth     int
	channels  int
	ch        chan []byte
	closed    atomic.Bool
	closeOnce sync.Once
}

var liveSpectrogramMgr = &liveSpectrogramManager{
	taps: make(map[string]*liveSpectrogramTap),
}

func (c *Controller) initLiveSpectrogramRoutes() {
	c.Group.GET("/streams/live-spectrogram", c.StreamLiveSpectrogram, c.publicLiveAudioAuth)
}

func (c *Controller) StreamLiveSpectrogram(ctx echo.Context) error {
	if c.engine == nil {
		return c.HandleError(ctx, nil, "Live spectrogram stream is not available", http.StatusServiceUnavailable)
	}

	sourceID := ctx.QueryParam("source")
	if sourceID == "" {
		return c.HandleError(ctx, nil, "Missing required source parameter", http.StatusBadRequest)
	}

	if _, ok := c.engine.Registry().Get(sourceID); !ok {
		return c.HandleError(ctx, nil, "Audio source not found", http.StatusNotFound)
	}

	subscription, cleanup, err := liveSpectrogramMgr.subscribe(c, sourceID)
	if err != nil {
		GetLogger().Warn("failed to subscribe live spectrogram stream",
			logger.String("source_id", privacy.SanitizeRTSPUrl(sourceID)),
			logger.Error(err),
		)
		return c.HandleError(ctx, nil, "Live spectrogram stream is not available", http.StatusServiceUnavailable)
	}
	defer cleanup()

	response := ctx.Response()
	response.Header().Set(echo.HeaderContentType, "text/event-stream")
	response.Header().Set(echo.HeaderCacheControl, "no-cache, no-transform")
	response.Header().Set(echo.HeaderConnection, "keep-alive")
	response.Header().Set("X-Accel-Buffering", "no")
	response.WriteHeader(http.StatusOK)
	response.Flush()

	heartbeat := time.NewTicker(liveSpectrogramHeartbeatInterval)
	defer heartbeat.Stop()

	for {
		select {
		case <-ctx.Request().Context().Done():
			return nil
		case payload := <-subscription:
			if len(payload) == 0 {
				continue
			}
			if _, err := fmt.Fprintf(response, "data: %s\n\n", payload); err != nil {
				return nil
			}
			response.Flush()
		case <-heartbeat.C:
			if _, err := response.Write([]byte(": ping\n\n")); err != nil {
				return nil
			}
			response.Flush()
		}
	}
}

func (m *liveSpectrogramManager) subscribe(c *Controller, sourceID string) (chan []byte, func(), error) {
	subscriber := make(chan []byte, liveSpectrogramSubscriberBuffer)

	m.mu.Lock()
	tap, ok := m.taps[sourceID]
	if !ok {
		var err error
		tap, err = newLiveSpectrogramTap(c, sourceID)
		if err != nil {
			m.mu.Unlock()
			return nil, nil, err
		}
		m.taps[sourceID] = tap
	}
	tap.addSubscriber(subscriber)
	m.mu.Unlock()

	cleanup := func() {
		var tapToClose *liveSpectrogramTap

		m.mu.Lock()
		tap, ok := m.taps[sourceID]
		if ok {
			if tap.removeSubscriber(subscriber) == 0 {
				delete(m.taps, sourceID)
				tapToClose = tap
			}
		}
		m.mu.Unlock()

		if tapToClose != nil {
			tapToClose.Close()
		}
	}

	return subscriber, cleanup, nil
}

func newLiveSpectrogramTap(c *Controller, sourceID string) (*liveSpectrogramTap, error) {
	source, ok := c.engine.Registry().Get(sourceID)
	if !ok {
		return nil, fmt.Errorf("source %s not found", sourceID)
	}

	sampleRate := source.SampleRate
	if c.Settings.WebServer.LiveStream.SampleRate > 0 {
		sampleRate = c.Settings.WebServer.LiveStream.SampleRate
	}
	if sampleRate <= 0 {
		sampleRate = hlsDefaultSampleRate
	}

	hopSamples := sampleRate / liveSpectrogramColumnsPerSecond
	if hopSamples <= 0 {
		hopSamples = 1
	}

	consumerID := fmt.Sprintf("live_spectrogram_%s_%s", privacy.SanitizeStreamUrl(sourceID), uuid.New().String()[:8])
	consumer := &liveSpectrogramConsumer{
		id:       consumerID,
		rate:     sampleRate,
		depth:    conf.BitDepth,
		channels: 1,
		ch:       make(chan []byte, liveSpectrogramChannelBuffer),
	}

	tap := &liveSpectrogramTap{
		sourceID:    sourceID,
		sampleRate:  sampleRate,
		fftSize:     liveSpectrogramFFTSize,
		hopSamples:  int64(hopSamples),
		consumerID:  consumerID,
		consumer:    consumer,
		subscribers: make(map[chan []byte]struct{}),
		ring:        make([]float64, liveSpectrogramFFTSize),
		hann:        periodicHannFloat64(liveSpectrogramFFTSize),
		twiddle:     computeTwiddleFloat64(liveSpectrogramFFTSize),
		smoothed:    make([]float64, liveSpectrogramFFTSize/2),
		nextEmitAt:  liveSpectrogramFFTSize,
	}

	if err := c.engine.Router().AddRoute(sourceID, consumer, source.SampleRate); err != nil {
		return nil, fmt.Errorf("add live spectrogram route: %w", err)
	}

	tap.cleanupRoute = func() {
		c.engine.Router().RemoveRoute(sourceID, consumerID)
	}

	go tap.run()

	GetLogger().Debug("registered live spectrogram route",
		logger.String("source_id", privacy.SanitizeRTSPUrl(sourceID)),
		logger.String("consumer_id", consumerID),
		logger.Int("sample_rate", sampleRate),
	)

	return tap, nil
}

func (t *liveSpectrogramTap) addSubscriber(ch chan []byte) {
	t.subscribersMu.Lock()
	t.subscribers[ch] = struct{}{}
	t.subscribersMu.Unlock()
}

func (t *liveSpectrogramTap) removeSubscriber(ch chan []byte) int {
	t.subscribersMu.Lock()
	delete(t.subscribers, ch)
	count := len(t.subscribers)
	t.subscribersMu.Unlock()
	return count
}

func (t *liveSpectrogramTap) Close() {
	if t.cleanupRoute != nil {
		t.cleanupRoute()
	}
}

func (t *liveSpectrogramTap) run() {
	for data := range t.consumer.ch {
		t.processPCM(data)
	}
}

func (t *liveSpectrogramTap) processPCM(data []byte) {
	if len(data) < 2 {
		return
	}

	for i := 0; i+1 < len(data); i += 2 {
		sample := int16(uint16(data[i]) | uint16(data[i+1])<<8)
		t.pushSample(float64(sample) / 32768.0)
	}
}

func (t *liveSpectrogramTap) pushSample(sample float64) {
	t.ring[t.ringPos] = sample
	t.ringPos = (t.ringPos + 1) % t.fftSize
	t.totalSamples++

	if t.totalSamples < int64(t.fftSize) {
		return
	}

	for t.totalSamples >= t.nextEmitAt {
		payload, err := t.buildPayload()
		if err == nil {
			t.broadcast(payload)
		}
		t.nextEmitAt += t.hopSamples
	}
}

func (t *liveSpectrogramTap) buildPayload() ([]byte, error) {
	bins := t.computeColumn()
	encodedBins := base64.StdEncoding.EncodeToString(bins)

	envelope := liveSpectrogramEnvelope{
		Type:       "live-spectrogram",
		SourceID:   t.sourceID,
		SampleRate: t.sampleRate,
		FFTSize:    t.fftSize,
		Bins:       encodedBins,
	}

	return json.Marshal(envelope)
}

func (t *liveSpectrogramTap) broadcast(payload []byte) {
	t.subscribersMu.RLock()
	for ch := range t.subscribers {
		select {
		case ch <- payload:
		default:
		}
	}
	t.subscribersMu.RUnlock()
}

func (t *liveSpectrogramTap) computeColumn() []byte {
	window := make([]complex128, t.fftSize)
	for i := range t.fftSize {
		sourceIndex := (t.ringPos + i) % t.fftSize
		window[i] = complex(t.ring[sourceIndex]*t.hann[i], 0)
	}

	fftInPlaceFloat64(window, t.twiddle)

	bins := make([]byte, t.fftSize/2)
	for i := range bins {
		magnitude := math.Hypot(real(window[i]), imag(window[i]))
		normalizedMagnitude := magnitude / (float64(t.fftSize) / 2)
		db := 20 * math.Log10(normalizedMagnitude+liveSpectrogramMagnitudeEps)
		normalized := normalizeLiveSpectrogramDecibels(db)
		smoothed := t.smoothed[i]*liveSpectrogramSmoothing + normalized*(1-liveSpectrogramSmoothing)
		t.smoothed[i] = smoothed
		bins[i] = byte(math.Round(smoothed * liveSpectrogramByteMax))
	}

	return bins
}

func normalizeLiveSpectrogramDecibels(db float64) float64 {
	if db <= liveSpectrogramMinDecibels {
		return 0
	}
	if db >= liveSpectrogramMaxDecibels {
		return 1
	}
	return (db - liveSpectrogramMinDecibels) / (liveSpectrogramMaxDecibels - liveSpectrogramMinDecibels)
}

func (c *liveSpectrogramConsumer) ID() string { return c.id }

func (c *liveSpectrogramConsumer) SampleRate() int { return c.rate }

func (c *liveSpectrogramConsumer) BitDepth() int { return c.depth }

func (c *liveSpectrogramConsumer) Channels() int { return c.channels }

func (c *liveSpectrogramConsumer) Write(frame audiocore.AudioFrame) error { //nolint:gocritic // interface requires value parameter
	if c.closed.Load() {
		return audiocore.ErrConsumerClosed
	}

	copied := make([]byte, len(frame.Data))
	copy(copied, frame.Data)

	select {
	case c.ch <- copied:
	default:
		select {
		case <-c.ch:
		default:
		}
		select {
		case c.ch <- copied:
		default:
		}
	}

	return nil
}

func (c *liveSpectrogramConsumer) Close() error {
	c.closed.Store(true)
	c.closeOnce.Do(func() {
		close(c.ch)
	})
	return nil
}

func periodicHannFloat64(n int) []float64 {
	window := make([]float64, n)
	scale := 2 * math.Pi / float64(n)
	for i := range window {
		window[i] = 0.5 * (1 - math.Cos(scale*float64(i)))
	}
	return window
}

func computeTwiddleFloat64(n int) []complex128 {
	twiddle := make([]complex128, n/2)
	angle := -2 * math.Pi / float64(n)
	for i := range twiddle {
		twiddle[i] = complex(math.Cos(angle*float64(i)), math.Sin(angle*float64(i)))
	}
	return twiddle
}

func fftInPlaceFloat64(data []complex128, twiddle []complex128) {
	size := len(data)

	j := 0
	for i := 1; i < size; i++ {
		bit := size >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			data[i], data[j] = data[j], data[i]
		}
	}

	for length := 2; length <= size; length <<= 1 {
		half := length >> 1
		step := size / length
		for start := 0; start < size; start += length {
			for i := range half {
				even := data[start+i]
				odd := data[start+i+half] * twiddle[i*step]
				data[start+i] = even + odd
				data[start+i+half] = even - odd
			}
		}
	}
}

var _ audiocore.AudioConsumer = (*liveSpectrogramConsumer)(nil)
