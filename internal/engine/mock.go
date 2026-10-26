package engine

import (
	"context"
	"time"

	"github.com/llm-bench/internal/config"
)

// MockRunner is a deterministic Runner used in tests and dry runs.
type MockRunner struct {
	ModelName       string
	ResponseText    string
	PromptTokens    int
	GeneratedTokens int
	TTFT            time.Duration
	Duration        time.Duration
	Err             error
	BlockUntil      <-chan struct{}
}

// Name returns the mock model name.
func (m *MockRunner) Name() string { return m.ModelName }

// Run returns the preconfigured result, optionally blocking until released or
// the context is cancelled.
func (m *MockRunner) Run(ctx context.Context, prompt string, limits config.ResourceLimits) (*BenchmarkResult, error) {
	if m.BlockUntil != nil {
		select {
		case <-m.BlockUntil:
		case <-ctx.Done():
			return nil, wrapTimeout(m.ModelName, ctx.Err())
		}
	}
	if m.Err != nil {
		return nil, m.Err
	}
	res := &BenchmarkResult{
		ModelName:       m.ModelName,
		Device:          limits.Device,
		PromptTokens:    m.PromptTokens,
		GeneratedTokens: m.GeneratedTokens,
		Duration:        m.Duration,
		TTFT:            m.TTFT,
		ResponseText:    m.ResponseText,
	}
	Finalize(res)
	return res, nil
}
