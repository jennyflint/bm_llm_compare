package logger

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/llm-bench/internal/config"
	"github.com/llm-bench/internal/engine"
)

func TestSanitizeProject(t *testing.T) {
	cases := map[string]string{
		"mistral-vs-qwen": "mistral-vs-qwen",
		"my project/../x": "my-project-x",
		"  ":              "default",
		"":                "default",
		"a b c":           "a-b-c",
	}
	for in, want := range cases {
		if got := SanitizeProject(in); got != want {
			t.Errorf("SanitizeProject(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestPathFor(t *testing.T) {
	l := New("logs")
	ts := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	got := l.PathFor("my project", ts)
	want := filepath.Join("logs", "my-project_20260102T030405Z.json")
	if got != want {
		t.Errorf("PathFor = %q, want %q", got, want)
	}
}

func TestEnsureDirCreatesNested(t *testing.T) {
	base := filepath.Join(t.TempDir(), "a", "b", "logs")
	l := New(base)
	if err := l.EnsureDir(); err != nil {
		t.Fatalf("EnsureDir error: %v", err)
	}
	info, err := os.Stat(base)
	if err != nil {
		t.Fatalf("stat dir: %v", err)
	}
	if !info.IsDir() {
		t.Error("expected directory")
	}
}

func TestWriteCreatesArtifact(t *testing.T) {
	base := t.TempDir()
	l := New(base)
	fixed := time.Date(2026, 5, 6, 7, 8, 9, 0, time.UTC)
	l.Now = func() time.Time { return fixed }

	results := []*engine.BenchmarkResult{
		{
			ModelName:       "llama",
			Device:          config.DeviceCPU,
			PromptTokens:    10,
			GeneratedTokens: 20,
			Duration:        2 * time.Second,
			TTFT:            500 * time.Millisecond,
			TokensPerSec:    13.33,
			ResponseText:    "hello",
			Status:          engine.StatusSuccess,
		},
		{
			ModelName: "qwen",
			Device:    config.DeviceGPU,
			Status:    engine.StatusOOM,
			Error:     errors.New("out of memory"),
		},
		nil,
	}
	meta := RunMetadata{Project: "proj", Prompt: "hi", Device: "cpu"}

	path, err := l.Write("proj", meta, results)
	if err != nil {
		t.Fatalf("Write error: %v", err)
	}
	if !strings.HasSuffix(path, "proj_20260506T070809Z.json") {
		t.Errorf("unexpected path %q", path)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read artifact: %v", err)
	}
	var art Artifact
	if err := json.Unmarshal(data, &art); err != nil {
		t.Fatalf("unmarshal artifact: %v", err)
	}
	if art.Metadata.Project != "proj" {
		t.Errorf("metadata project = %q", art.Metadata.Project)
	}
	if len(art.Results) != 2 {
		t.Fatalf("results len = %d, want 2 (nil skipped)", len(art.Results))
	}
	if art.Results[0].ModelName != "llama" || art.Results[0].TokensPerSec != 13.33 {
		t.Errorf("unexpected first result: %+v", art.Results[0])
	}
	if art.Results[1].Status != engine.StatusOOM || art.Results[1].Error != "out of memory" {
		t.Errorf("unexpected second result: %+v", art.Results[1])
	}
}

func TestWriteInvalidDir(t *testing.T) {
	dir := t.TempDir()
	blocker := filepath.Join(dir, "file")
	if err := os.WriteFile(blocker, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	l := New(filepath.Join(blocker, "logs"))
	if _, err := l.Write("p", RunMetadata{}, nil); err == nil {
		t.Error("expected error when log dir cannot be created")
	}
}

func TestToRecord(t *testing.T) {
	r := &engine.BenchmarkResult{
		ModelName:       "m",
		Device:          config.DeviceGPU,
		PromptTokens:    3,
		GeneratedTokens: 4,
		Duration:        1500 * time.Millisecond,
		TTFT:            250 * time.Millisecond,
		TokensPerSec:    3.2,
		ResponseText:    "text",
		Status:          engine.StatusSuccess,
	}
	rec := ToRecord(r)
	if rec.DurationMS != 1500 || rec.TTFTMS != 250 {
		t.Errorf("durations = %d/%d, want 1500/250", rec.DurationMS, rec.TTFTMS)
	}
	if rec.Device != "gpu" {
		t.Errorf("Device = %q, want gpu", rec.Device)
	}
	if rec.Error != "" {
		t.Errorf("Error = %q, want empty", rec.Error)
	}
}
