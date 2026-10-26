package engine

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
	"time"

	"github.com/llm-bench/internal/config"
)

// ExecRunner executes a local CLI binary, passing the prompt on stdin.
type ExecRunner struct {
	modelName string
	command   string
}

// NewExecRunner constructs a runner that shells out to command.
func NewExecRunner(modelName, command string) *ExecRunner {
	return &ExecRunner{modelName: modelName, command: command}
}

// Name returns the display name of the model.
func (e *ExecRunner) Name() string { return e.modelName }

// Run executes the command, capturing stdout as the response text.
func (e *ExecRunner) Run(ctx context.Context, prompt string, limits config.ResourceLimits) (*BenchmarkResult, error) {
	if strings.TrimSpace(e.command) == "" {
		return nil, fmt.Errorf("exec runner %q: empty command", e.modelName)
	}

	fields := strings.Fields(e.command)
	cmd := exec.CommandContext(ctx, fields[0], fields[1:]...)
	cmd.Stdin = strings.NewReader(prompt)

	start := time.Now()
	out, err := cmd.Output()
	duration := time.Since(start)

	if err != nil {
		if ctx.Err() != nil {
			return nil, wrapTimeout(e.modelName, fmt.Errorf("exec %q: %w", e.command, ctx.Err()))
		}
		return nil, fmt.Errorf("exec %q: %w", e.command, err)
	}

	text := string(out)
	res := &BenchmarkResult{
		ModelName:       e.modelName,
		Device:          limits.Device,
		PromptTokens:    EstimateTokens(prompt),
		GeneratedTokens: EstimateTokens(text),
		Duration:        duration,
		ResponseText:    text,
	}
	Finalize(res)
	return res, nil
}
