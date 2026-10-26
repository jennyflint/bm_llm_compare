package cmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRootCommandRequiresPromptFile(t *testing.T) {
	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--models", "llama3"})
	cmd.SetOut(&bytes.Buffer{})
	cmd.SetErr(&bytes.Buffer{})
	if err := cmd.Execute(); err == nil {
		t.Error("expected error when prompt file missing")
	}
}

func TestRootCommandMissingPromptFileOnDisk(t *testing.T) {
	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--models", "llama3", "--prompt-file", filepath.Join(t.TempDir(), "nope.txt")})
	cmd.SetOut(&bytes.Buffer{})
	cmd.SetErr(&bytes.Buffer{})
	if err := cmd.Execute(); err == nil {
		t.Error("expected error for missing prompt file on disk")
	}
}

func TestRootCommandEmptyPromptFile(t *testing.T) {
	dir := t.TempDir()
	promptPath := filepath.Join(dir, "prompt.txt")
	if err := os.WriteFile(promptPath, []byte("   \n"), 0o644); err != nil {
		t.Fatal(err)
	}
	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--models", "llama3", "--prompt-file", promptPath})
	cmd.SetOut(&bytes.Buffer{})
	cmd.SetErr(&bytes.Buffer{})
	if err := cmd.Execute(); err == nil {
		t.Error("expected error for empty prompt file")
	}
}

func TestRootCommandInvalidFormat(t *testing.T) {
	dir := t.TempDir()
	promptPath := filepath.Join(dir, "prompt.txt")
	if err := os.WriteFile(promptPath, []byte("hello"), 0o644); err != nil {
		t.Fatal(err)
	}
	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--models", "llama3", "--prompt-file", promptPath, "--format", "xml"})
	cmd.SetOut(&bytes.Buffer{})
	cmd.SetErr(&bytes.Buffer{})
	if err := cmd.Execute(); err == nil {
		t.Error("expected error for invalid format")
	}
}

func TestRootCommandExecRunnerWritesArtifact(t *testing.T) {
	dir := t.TempDir()
	promptPath := filepath.Join(dir, "prompt.txt")
	if err := os.WriteFile(promptPath, []byte("hello world"), 0o644); err != nil {
		t.Fatal(err)
	}
	configPath := filepath.Join(dir, "config.yaml")
	cfg := `
defaults:
  device: cpu
  cpu_threads: 2
  max_tokens: 16
  num_ctx: 256
  timeout: 10s
  format: json
models:
  - name: echoer
    engine: exec
    command: cat
`
	if err := os.WriteFile(configPath, []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	logDir := filepath.Join(dir, "logs")

	var out bytes.Buffer
	cmd := NewRootCommand()
	cmd.SetOut(&out)
	cmd.SetErr(&out)
	cmd.SetArgs([]string{
		"--config", configPath,
		"--prompt-file", promptPath,
		"--models", "echoer",
		"--project", "test-project",
		"--log-dir", logDir,
		"--no-color",
	})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute error: %v\noutput: %s", err, out.String())
	}

	entries, err := os.ReadDir(logDir)
	if err != nil {
		t.Fatalf("read log dir: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("log files = %d, want 1", len(entries))
	}
	if !strings.HasPrefix(entries[0].Name(), "test-project_") {
		t.Errorf("unexpected log file name %q", entries[0].Name())
	}

	data, err := os.ReadFile(filepath.Join(logDir, entries[0].Name()))
	if err != nil {
		t.Fatal(err)
	}
	var artifact struct {
		Metadata struct {
			Project string `json:"project"`
			Device  string `json:"device"`
		} `json:"metadata"`
		Results []struct {
			ModelName string `json:"model_name"`
			Status    string `json:"status"`
		} `json:"results"`
	}
	if err := json.Unmarshal(data, &artifact); err != nil {
		t.Fatalf("unmarshal artifact: %v", err)
	}
	if artifact.Metadata.Project != "test-project" {
		t.Errorf("project = %q", artifact.Metadata.Project)
	}
	if artifact.Metadata.Device != "cpu" {
		t.Errorf("device = %q, want cpu", artifact.Metadata.Device)
	}
	if len(artifact.Results) != 1 || artifact.Results[0].ModelName != "echoer" {
		t.Errorf("unexpected results: %+v", artifact.Results)
	}
	if artifact.Results[0].Status != "Success" {
		t.Errorf("status = %q, want Success", artifact.Results[0].Status)
	}
}
