// Package cmd wires the CLI flags, configuration, orchestration, and output.
package cmd

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/llm-bench/internal/config"
	"github.com/llm-bench/internal/engine"
	"github.com/llm-bench/internal/logger"
	"github.com/llm-bench/internal/printer"
	"github.com/llm-bench/internal/runner"
)

// NewRootCommand builds the root cobra command.
func NewRootCommand() *cobra.Command {
	var flags config.FlagValues

	cmd := &cobra.Command{
		Use:           "llm-bench",
		Short:         "Benchmark a prompt across multiple LLM models concurrently",
		SilenceUsage:  true,
		SilenceErrors: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			return runBenchmark(cmd, flags)
		},
	}

	f := cmd.Flags()
	f.StringVarP(&flags.Project, "project", "P", "", "project workspace name for artifact logging")
	f.StringVarP(&flags.PromptFile, "prompt-file", "p", "", "path to the prompt file (required)")
	f.StringVarP(&flags.Models, "models", "m", "", "comma-separated list of models")
	f.StringVarP(&flags.ModelsDir, "models-dir", "d", "", "directory to scan for model weights/configs")
	f.StringVarP(&flags.ConfigPath, "config", "c", "config.yaml", "path to configuration file")
	f.IntVar(&flags.Concurrency, "concurrency", 0, "max parallel model runs (default: CPU cores)")
	f.StringVar(&flags.Format, "format", "", "output format: table, json, markdown")
	f.BoolVar(&flags.NoColor, "no-color", false, "disable ANSI colors")
	f.StringVar(&flags.Device, "device", "", "compute backend: cpu or gpu")
	f.IntVar(&flags.CPUThreads, "cpu-threads", 0, "CPU thread count for local engines")
	f.IntVar(&flags.GPULayers, "gpu-layers", 0, "GPU layers to offload (-1 for all)")
	f.IntVar(&flags.MaxTokens, "max-tokens", 0, "max generated tokens per run")
	f.IntVar(&flags.NumCtx, "num-ctx", 0, "context window size per run")
	f.DurationVar(&flags.Timeout, "timeout", 0, "per-model execution timeout")
	f.StringVar(&flags.LogDir, "log-dir", "", "directory for project artifacts")
	f.StringVar(&flags.OllamaURL, "ollama-url", "", "Ollama base URL")

	return cmd
}

func runBenchmark(cmd *cobra.Command, flags config.FlagValues) error {
	file, err := config.LoadFile(flags.ConfigPath, true)
	if err != nil {
		return err
	}
	opts, err := config.Resolve(file, flags)
	if err != nil {
		return fmt.Errorf("resolve configuration: %w", err)
	}

	promptBytes, err := os.ReadFile(opts.PromptFile)
	if err != nil {
		return fmt.Errorf("read prompt file %q: %w", opts.PromptFile, err)
	}
	prompt := strings.TrimSpace(string(promptBytes))
	if prompt == "" {
		return fmt.Errorf("prompt file %q is empty", opts.PromptFile)
	}

	factory := engine.NewFactory(opts)
	modelNames := opts.Models
	if len(modelNames) == 0 {
		for _, m := range opts.ModelDefs {
			modelNames = append(modelNames, m.Name)
		}
	}

	jobs := make([]runner.Job, 0, len(modelNames))
	for _, name := range modelNames {
		r, err := factory.Build(name)
		if err != nil {
			return err
		}
		limits, err := limitsFor(opts, name)
		if err != nil {
			return err
		}
		jobs = append(jobs, runner.Job{Runner: r, Limits: limits})
	}

	ctx, stop := runner.WithSignalContext(context.Background())
	defer stop()

	started := time.Now()
	orch := runner.New(opts.Concurrency)
	results, runErr := orch.Run(ctx, prompt, jobs)
	finished := time.Now()

	out := cmd.OutOrStdout()
	p := printer.New(out, opts.Format, opts.NoColor)
	if err := p.Print(results); err != nil {
		return err
	}

	if opts.Project != "" {
		path, err := writeArtifact(opts, prompt, results, started, finished)
		if err != nil {
			return err
		}
		fmt.Fprintf(out, "\nArtifacts written to %s\n", path)
	}

	return runErr
}

func limitsFor(opts *config.Options, name string) (config.ResourceLimits, error) {
	for _, m := range opts.ModelDefs {
		if m.Name == name {
			return opts.ForModel(m)
		}
	}
	return opts.ForModel(config.ModelConfig{Name: name})
}

func writeArtifact(opts *config.Options, prompt string, results []*engine.BenchmarkResult, started, finished time.Time) (string, error) {
	meta := logger.RunMetadata{
		Project:     opts.Project,
		PromptFile:  opts.PromptFile,
		Prompt:      prompt,
		Device:      string(opts.Device),
		CPUThreads:  opts.CPUThreads,
		GPULayers:   opts.GPULayers,
		MaxTokens:   opts.MaxTokens,
		NumCtx:      opts.NumCtx,
		Timeout:     opts.Timeout.String(),
		Concurrency: opts.Concurrency,
		StartedAt:   started.UTC(),
		FinishedAt:  finished.UTC(),
	}
	l := logger.New(opts.LogDir)
	return l.Write(opts.Project, meta, results)
}
