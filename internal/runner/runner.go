// Package runner orchestrates concurrent benchmark execution across models.
package runner

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/llm-bench/internal/config"
	"github.com/llm-bench/internal/engine"
)

// Job pairs a runner with the resource limits it should execute under.
type Job struct {
	Runner engine.Runner
	Limits config.ResourceLimits
}

// Orchestrator executes jobs concurrently with a bounded worker pool.
type Orchestrator struct {
	Concurrency int
}

// New creates an Orchestrator. Concurrency below 1 is clamped to 1.
func New(concurrency int) *Orchestrator {
	if concurrency < 1 {
		concurrency = 1
	}
	return &Orchestrator{Concurrency: concurrency}
}

// Run executes all jobs and returns results ordered by input position. A
// failing job never aborts the batch; failures are captured per result.
func (o *Orchestrator) Run(ctx context.Context, prompt string, jobs []Job) ([]*engine.BenchmarkResult, error) {
	if len(jobs) == 0 {
		return nil, nil
	}

	results := make([]*engine.BenchmarkResult, len(jobs))
	var wg sync.WaitGroup
	sem := make(chan struct{}, o.Concurrency)

	for i, job := range jobs {
		wg.Add(1)
		go func(idx int, j Job) {
			defer wg.Done()
			if err := ctx.Err(); err != nil {
				results[idx] = cancelledResult(j, err)
				return
			}
			select {
			case sem <- struct{}{}:
				defer func() { <-sem }()
			case <-ctx.Done():
				results[idx] = cancelledResult(j, ctx.Err())
				return
			}
			results[idx] = o.runOne(ctx, j, prompt)
		}(i, job)
	}
	wg.Wait()

	if err := ctx.Err(); err != nil {
		return results, fmt.Errorf("benchmark interrupted: %w", err)
	}
	return results, nil
}

func (o *Orchestrator) runOne(ctx context.Context, job Job, prompt string) *engine.BenchmarkResult {
	runCtx, cancel := context.WithTimeout(ctx, job.Limits.Timeout)
	defer cancel()

	start := time.Now()
	res, err := job.Runner.Run(runCtx, prompt, job.Limits)
	elapsed := time.Since(start)

	if err != nil {
		return failedResult(job, err, elapsed)
	}
	if res == nil {
		return failedResult(job, fmt.Errorf("runner returned nil result"), elapsed)
	}
	if res.ModelName == "" {
		res.ModelName = job.Runner.Name()
	}
	if res.Device == "" {
		res.Device = job.Limits.Device
	}
	if res.Duration <= 0 {
		res.Duration = elapsed
	}
	engine.Finalize(res)
	return res
}

func failedResult(job Job, err error, elapsed time.Duration) *engine.BenchmarkResult {
	return &engine.BenchmarkResult{
		ModelName: job.Runner.Name(),
		Device:    job.Limits.Device,
		Duration:  elapsed,
		Status:    engine.ClassifyError(err),
		Error:     err,
	}
}

func cancelledResult(job Job, err error) *engine.BenchmarkResult {
	return &engine.BenchmarkResult{
		ModelName: job.Runner.Name(),
		Device:    job.Limits.Device,
		Status:    engine.StatusError,
		Error:     fmt.Errorf("cancelled before start: %w", err),
	}
}

// SortResults orders results by throughput, successes first.
func SortResults(results []*engine.BenchmarkResult) {
	sort.SliceStable(results, func(i, j int) bool {
		si := results[i].Status == engine.StatusSuccess
		sj := results[j].Status == engine.StatusSuccess
		if si != sj {
			return si
		}
		return results[i].TokensPerSec > results[j].TokensPerSec
	})
}
