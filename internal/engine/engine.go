// Package engine defines the Runner abstraction and concrete implementations
// for local (Ollama, exec) and remote (OpenAI-compatible) LLM backends.
package engine

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/llm-bench/internal/config"
)

// Status values reported for a benchmark run.
const (
	StatusSuccess = "Success"
	StatusTimeout = "Timeout"
	StatusOOM     = "OOM"
	StatusError   = "Error"
)

// Sentinel errors used to classify failures.
var (
	ErrTimeout = errors.New("execution timed out")
	ErrOOM     = errors.New("out of memory")
)

// BenchmarkResult captures the measured outcome of a single model run.
type BenchmarkResult struct {
	ModelName       string
	Device          config.DeviceTarget
	PromptTokens    int
	GeneratedTokens int
	Duration        time.Duration
	TTFT            time.Duration
	TokensPerSec    float64
	ResponseText    string
	Status          string
	Error           error
}

// Runner executes a prompt against a single model.
type Runner interface {
	Run(ctx context.Context, prompt string, limits config.ResourceLimits) (*BenchmarkResult, error)
	Name() string
}

// TokensPerSecond computes generation throughput from token count and the
// generation window (total duration minus time to first token).
func TokensPerSecond(generated int, duration, ttft time.Duration) float64 {
	genWindow := duration - ttft
	if generated <= 0 || genWindow <= 0 {
		return 0
	}
	return float64(generated) / genWindow.Seconds()
}

// Finalize fills derived fields and normalizes the status of a result.
func Finalize(res *BenchmarkResult) {
	if res == nil {
		return
	}
	res.TokensPerSec = TokensPerSecond(res.GeneratedTokens, res.Duration, res.TTFT)
	if res.Status == "" {
		if res.Error != nil {
			res.Status = StatusError
		} else {
			res.Status = StatusSuccess
		}
	}
}

// ClassifyError maps an error to a status string.
func ClassifyError(err error) string {
	switch {
	case err == nil:
		return StatusSuccess
	case errors.Is(err, context.DeadlineExceeded), errors.Is(err, ErrTimeout):
		return StatusTimeout
	case errors.Is(err, ErrOOM):
		return StatusOOM
	default:
		return StatusError
	}
}

// EstimateTokens provides a coarse token estimate (~4 chars per token) used
// when a backend does not report prompt token counts.
func EstimateTokens(text string) int {
	if text == "" {
		return 0
	}
	n := len(text) / 4
	if n < 1 {
		n = 1
	}
	return n
}

// wrapTimeout converts a context deadline error into ErrTimeout with context.
func wrapTimeout(model string, err error) error {
	if errors.Is(err, context.DeadlineExceeded) {
		return fmt.Errorf("model %q: %w", model, ErrTimeout)
	}
	return err
}
