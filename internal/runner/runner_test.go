package runner

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/llm-bench/internal/config"
	"github.com/llm-bench/internal/engine"
)

func cpuLimits(timeout time.Duration) config.ResourceLimits {
	return config.ResourceLimits{
		Device:     config.DeviceCPU,
		CPUThreads: 2,
		GPULayers:  0,
		MaxTokens:  64,
		NumCtx:     512,
		Timeout:    timeout,
	}
}

func TestOrchestratorRunsAllJobs(t *testing.T) {
	jobs := []Job{
		{Runner: &engine.MockRunner{ModelName: "a", GeneratedTokens: 10, Duration: time.Second, TTFT: 100 * time.Millisecond}, Limits: cpuLimits(time.Second)},
		{Runner: &engine.MockRunner{ModelName: "b", GeneratedTokens: 20, Duration: time.Second, TTFT: 100 * time.Millisecond}, Limits: cpuLimits(time.Second)},
		{Runner: &engine.MockRunner{ModelName: "c", GeneratedTokens: 30, Duration: time.Second, TTFT: 100 * time.Millisecond}, Limits: cpuLimits(time.Second)},
	}
	o := New(2)
	results, err := o.Run(context.Background(), "prompt", jobs)
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if len(results) != 3 {
		t.Fatalf("results = %d, want 3", len(results))
	}
	for i, want := range []string{"a", "b", "c"} {
		if results[i].ModelName != want {
			t.Errorf("results[%d].ModelName = %q, want %q", i, results[i].ModelName, want)
		}
		if results[i].Status != engine.StatusSuccess {
			t.Errorf("results[%d].Status = %q, want Success", i, results[i].Status)
		}
	}
}

func TestOrchestratorRespectsConcurrency(t *testing.T) {
	var active int32
	var maxActive int32
	block := make(chan struct{})

	runner := &countingRunner{
		name: "counter",
		onStart: func() {
			cur := atomic.AddInt32(&active, 1)
			for {
				prev := atomic.LoadInt32(&maxActive)
				if cur <= prev || atomic.CompareAndSwapInt32(&maxActive, prev, cur) {
					break
				}
			}
			<-block
			atomic.AddInt32(&active, -1)
		},
	}

	jobs := make([]Job, 6)
	for i := range jobs {
		jobs[i] = Job{Runner: runner, Limits: cpuLimits(5 * time.Second)}
	}

	o := New(2)
	done := make(chan struct{})
	go func() {
		_, _ = o.Run(context.Background(), "p", jobs)
		close(done)
	}()

	time.Sleep(50 * time.Millisecond)
	close(block)
	<-done

	if got := atomic.LoadInt32(&maxActive); got > 2 {
		t.Errorf("max concurrent = %d, want <= 2", got)
	}
}

func TestOrchestratorTimeout(t *testing.T) {
	block := make(chan struct{})
	defer close(block)
	jobs := []Job{
		{Runner: &engine.MockRunner{ModelName: "slow", BlockUntil: block}, Limits: cpuLimits(20 * time.Millisecond)},
	}
	o := New(1)
	results, err := o.Run(context.Background(), "p", jobs)
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if results[0].Status != engine.StatusTimeout {
		t.Errorf("Status = %q, want Timeout", results[0].Status)
	}
	if results[0].Error == nil {
		t.Error("expected error to be captured")
	}
}

func TestOrchestratorOOMDoesNotAbort(t *testing.T) {
	jobs := []Job{
		{Runner: &engine.MockRunner{ModelName: "oom", Err: engine.ErrOOM}, Limits: cpuLimits(time.Second)},
		{Runner: &engine.MockRunner{ModelName: "ok", GeneratedTokens: 5, Duration: time.Second}, Limits: cpuLimits(time.Second)},
	}
	o := New(2)
	results, err := o.Run(context.Background(), "p", jobs)
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if results[0].Status != engine.StatusOOM {
		t.Errorf("results[0].Status = %q, want OOM", results[0].Status)
	}
	if results[1].Status != engine.StatusSuccess {
		t.Errorf("results[1].Status = %q, want Success", results[1].Status)
	}
}

func TestOrchestratorCancelledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	jobs := []Job{
		{Runner: &engine.MockRunner{ModelName: "a"}, Limits: cpuLimits(time.Second)},
	}
	o := New(1)
	results, err := o.Run(ctx, "p", jobs)
	if err == nil {
		t.Error("expected interruption error")
	}
	if len(results) != 1 || results[0].Status != engine.StatusError {
		t.Errorf("unexpected results: %+v", results)
	}
}

func TestOrchestratorEmpty(t *testing.T) {
	o := New(1)
	results, err := o.Run(context.Background(), "p", nil)
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if results != nil {
		t.Errorf("results = %v, want nil", results)
	}
}

func TestSortResults(t *testing.T) {
	results := []*engine.BenchmarkResult{
		{ModelName: "slow", Status: engine.StatusSuccess, TokensPerSec: 1},
		{ModelName: "fail", Status: engine.StatusError},
		{ModelName: "fast", Status: engine.StatusSuccess, TokensPerSec: 9},
	}
	SortResults(results)
	if results[0].ModelName != "fast" || results[2].ModelName != "fail" {
		t.Errorf("unexpected order: %s ... %s", results[0].ModelName, results[2].ModelName)
	}
}

func TestClassifyWrappedTimeout(t *testing.T) {
	err := errors.Join(engine.ErrTimeout)
	if got := engine.ClassifyError(err); got != engine.StatusTimeout {
		t.Errorf("ClassifyError = %q, want Timeout", got)
	}
}

type countingRunner struct {
	name    string
	onStart func()
}

func (c *countingRunner) Name() string { return c.name }

func (c *countingRunner) Run(ctx context.Context, prompt string, limits config.ResourceLimits) (*engine.BenchmarkResult, error) {
	c.onStart()
	return &engine.BenchmarkResult{
		ModelName:       c.name,
		Device:          limits.Device,
		GeneratedTokens: 1,
		Duration:        time.Second,
		Status:          engine.StatusSuccess,
	}, nil
}
