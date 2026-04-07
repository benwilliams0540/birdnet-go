package api

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func clearSpectrogramQueue() {
	spectrogramQueue.Range(func(key, _ any) bool {
		spectrogramQueue.Delete(key)
		return true
	})
}

func TestFinalizeQueueStatus_RetainsFailures(t *testing.T) {
	t.Parallel()

	clearSpectrogramQueue()
	t.Cleanup(clearSpectrogramQueue)

	controller := &Controller{}
	spectrogramKey := "test-failed-status"

	controller.initializeQueueStatus(spectrogramKey)
	controller.finalizeQueueStatus(spectrogramKey, errors.New("boom"))

	statusValue, ok := spectrogramQueue.Load(spectrogramKey)
	require.True(t, ok, "failed status should remain available for polling clients")

	status, ok := statusValue.(*SpectrogramQueueStatus)
	require.True(t, ok, "queue entry should have the expected status type")

	snapshot := status.Get()
	assert.Equal(t, spectrogramStatusFailed, snapshot["status"])
	assert.Equal(t, "Generation failed: boom", snapshot["message"])
}

func TestFinalizeQueueStatus_DropsOperationalErrorsImmediately(t *testing.T) {
	t.Parallel()

	clearSpectrogramQueue()
	t.Cleanup(clearSpectrogramQueue)

	controller := &Controller{}
	spectrogramKey := "test-operational-error"

	controller.initializeQueueStatus(spectrogramKey)
	controller.finalizeQueueStatus(spectrogramKey, context.Canceled)

	_, ok := spectrogramQueue.Load(spectrogramKey)
	assert.False(t, ok, "operational errors should not be retained as hard failures")
}
