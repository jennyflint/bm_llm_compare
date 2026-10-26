// Package logger persists benchmark artifacts to project-scoped log files.
package logger

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/llm-bench/internal/engine"
)

// RunMetadata describes the context of a benchmark run.
type RunMetadata struct {
	Project     string    `json:"project"`
	PromptFile  string    `json:"prompt_file"`
	Prompt      string    `json:"prompt"`
	Device      string    `json:"device"`
	CPUThreads  int       `json:"cpu_threads"`
	GPULayers   int       `json:"gpu_layers"`
	MaxTokens   int       `json:"max_tokens"`
	NumCtx      int       `json:"num_ctx"`
	Timeout     string    `json:"timeout"`
	Concurrency int       `json:"concurrency"`
	StartedAt   time.Time `json:"started_at"`
	FinishedAt  time.Time `json:"finished_at"`
}

// ResultRecord is the serializable form of a benchmark result.
type ResultRecord struct {
	ModelName       string  `json:"model_name"`
	Device          string  `json:"device"`
	PromptTokens    int     `json:"prompt_tokens"`
	GeneratedTokens int     `json:"generated_tokens"`
	DurationMS      int64   `json:"duration_ms"`
	TTFTMS          int64   `json:"ttft_ms"`
	TokensPerSec    float64 `json:"tokens_per_sec"`
	ResponseText    string  `json:"response_text"`
	Status          string  `json:"status"`
	Error           string  `json:"error,omitempty"`
}

// Artifact is the full JSON document written to disk.
type Artifact struct {
	Metadata RunMetadata    `json:"metadata"`
	Results  []ResultRecord `json:"results"`
}

// Logger writes artifacts under a base directory.
type Logger struct {
	BaseDir string
	Now     func() time.Time
}

// New creates a Logger rooted at baseDir.
func New(baseDir string) *Logger {
	if baseDir == "" {
		baseDir = "logs"
	}
	return &Logger{BaseDir: baseDir, Now: time.Now}
}

var (
	unsafeChars = regexp.MustCompile(`[^a-zA-Z0-9._-]+`)
	dotRuns     = regexp.MustCompile(`\.{2,}`)
	dashRuns    = regexp.MustCompile(`-{2,}`)
)

// SanitizeProject converts a project name into a filesystem-safe token,
// collapsing unsafe characters and eliminating path traversal sequences.
func SanitizeProject(name string) string {
	cleaned := unsafeChars.ReplaceAllString(strings.TrimSpace(name), "-")
	cleaned = dotRuns.ReplaceAllString(cleaned, "-")
	cleaned = dashRuns.ReplaceAllString(cleaned, "-")
	cleaned = strings.Trim(cleaned, "-.")
	if cleaned == "" {
		return "default"
	}
	return cleaned
}

// PathFor returns the artifact path for a project at the given time.
func (l *Logger) PathFor(project string, ts time.Time) string {
	name := SanitizeProject(project)
	stamp := ts.UTC().Format("20060102T150405Z")
	return filepath.Join(l.BaseDir, fmt.Sprintf("%s_%s.json", name, stamp))
}

// EnsureDir creates the base directory if it does not exist.
func (l *Logger) EnsureDir() error {
	if err := os.MkdirAll(l.BaseDir, 0o755); err != nil {
		return fmt.Errorf("create log dir %q: %w", l.BaseDir, err)
	}
	return nil
}

// ToRecord converts an engine result into a serializable record.
func ToRecord(r *engine.BenchmarkResult) ResultRecord {
	rec := ResultRecord{
		ModelName:       r.ModelName,
		Device:          string(r.Device),
		PromptTokens:    r.PromptTokens,
		GeneratedTokens: r.GeneratedTokens,
		DurationMS:      r.Duration.Milliseconds(),
		TTFTMS:          r.TTFT.Milliseconds(),
		TokensPerSec:    r.TokensPerSec,
		ResponseText:    r.ResponseText,
		Status:          r.Status,
	}
	if r.Error != nil {
		rec.Error = r.Error.Error()
	}
	return rec
}

// Write serializes the artifact to the project log file and returns its path.
func (l *Logger) Write(project string, meta RunMetadata, results []*engine.BenchmarkResult) (string, error) {
	if err := l.EnsureDir(); err != nil {
		return "", err
	}
	records := make([]ResultRecord, 0, len(results))
	for _, r := range results {
		if r == nil {
			continue
		}
		records = append(records, ToRecord(r))
	}
	artifact := Artifact{Metadata: meta, Results: records}

	data, err := json.MarshalIndent(artifact, "", "  ")
	if err != nil {
		return "", fmt.Errorf("marshal artifact: %w", err)
	}

	path := l.PathFor(project, l.Now())
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return "", fmt.Errorf("write artifact %q: %w", path, err)
	}
	return path, nil
}
