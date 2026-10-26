package engine

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/llm-bench/internal/config"
)

func TestTokensPerSecond(t *testing.T) {
	cases := []struct {
		name      string
		generated int
		duration  time.Duration
		ttft      time.Duration
		want      float64
	}{
		{"typical", 100, 2 * time.Second, 500 * time.Millisecond, 100.0 / 1.5},
		{"zero tokens", 0, time.Second, 0, 0},
		{"zero window", 10, time.Second, time.Second, 0},
		{"negative window", 10, time.Second, 2 * time.Second, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := TokensPerSecond(c.generated, c.duration, c.ttft)
			if diff := got - c.want; diff > 1e-9 || diff < -1e-9 {
				t.Errorf("TokensPerSecond = %f, want %f", got, c.want)
			}
		})
	}
}

func TestFinalize(t *testing.T) {
	res := &BenchmarkResult{
		GeneratedTokens: 50,
		Duration:        2 * time.Second,
		TTFT:            500 * time.Millisecond,
	}
	Finalize(res)
	if res.Status != StatusSuccess {
		t.Errorf("Status = %q, want %q", res.Status, StatusSuccess)
	}
	if res.TokensPerSec <= 0 {
		t.Errorf("TokensPerSec = %f, want > 0", res.TokensPerSec)
	}

	errRes := &BenchmarkResult{Error: errors.New("boom")}
	Finalize(errRes)
	if errRes.Status != StatusError {
		t.Errorf("Status = %q, want %q", errRes.Status, StatusError)
	}
}

func TestClassifyError(t *testing.T) {
	cases := []struct {
		err  error
		want string
	}{
		{nil, StatusSuccess},
		{context.DeadlineExceeded, StatusTimeout},
		{ErrTimeout, StatusTimeout},
		{ErrOOM, StatusOOM},
		{errors.New("other"), StatusError},
	}
	for _, c := range cases {
		if got := ClassifyError(c.err); got != c.want {
			t.Errorf("ClassifyError(%v) = %q, want %q", c.err, got, c.want)
		}
	}
}

func TestEstimateTokens(t *testing.T) {
	if got := EstimateTokens(""); got != 0 {
		t.Errorf("EstimateTokens(\"\") = %d, want 0", got)
	}
	if got := EstimateTokens("a"); got != 1 {
		t.Errorf("EstimateTokens(\"a\") = %d, want 1", got)
	}
	if got := EstimateTokens("12345678"); got != 2 {
		t.Errorf("EstimateTokens(8 chars) = %d, want 2", got)
	}
}

func TestMockRunnerCPU(t *testing.T) {
	m := &MockRunner{
		ModelName:       "mock-cpu",
		ResponseText:    "hello",
		PromptTokens:    10,
		GeneratedTokens: 20,
		Duration:        2 * time.Second,
		TTFT:            500 * time.Millisecond,
	}
	limits := config.ResourceLimits{Device: config.DeviceCPU, CPUThreads: 4, MaxTokens: 64, NumCtx: 512, Timeout: time.Second}
	res, err := m.Run(context.Background(), "prompt", limits)
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if res.Device != config.DeviceCPU {
		t.Errorf("Device = %q, want cpu", res.Device)
	}
	if res.Status != StatusSuccess {
		t.Errorf("Status = %q, want Success", res.Status)
	}
	if res.TokensPerSec <= 0 {
		t.Errorf("TokensPerSec = %f, want > 0", res.TokensPerSec)
	}
}

func TestMockRunnerGPU(t *testing.T) {
	m := &MockRunner{ModelName: "mock-gpu", GeneratedTokens: 10, Duration: time.Second}
	limits := config.ResourceLimits{Device: config.DeviceGPU, CPUThreads: 4, GPULayers: -1, MaxTokens: 64, NumCtx: 512, Timeout: time.Second}
	res, err := m.Run(context.Background(), "prompt", limits)
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if res.Device != config.DeviceGPU {
		t.Errorf("Device = %q, want gpu", res.Device)
	}
}

func TestMockRunnerError(t *testing.T) {
	m := &MockRunner{ModelName: "mock-err", Err: ErrOOM}
	limits := config.ResourceLimits{Device: config.DeviceCPU, CPUThreads: 1, MaxTokens: 1, NumCtx: 1, Timeout: time.Second}
	if _, err := m.Run(context.Background(), "p", limits); !errors.Is(err, ErrOOM) {
		t.Errorf("expected ErrOOM, got %v", err)
	}
}

func TestMockRunnerTimeout(t *testing.T) {
	block := make(chan struct{})
	defer close(block)
	m := &MockRunner{ModelName: "mock-timeout", BlockUntil: block}
	limits := config.ResourceLimits{Device: config.DeviceCPU, CPUThreads: 1, MaxTokens: 1, NumCtx: 1, Timeout: time.Second}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	_, err := m.Run(ctx, "p", limits)
	if !errors.Is(err, ErrTimeout) {
		t.Errorf("expected ErrTimeout, got %v", err)
	}
}
