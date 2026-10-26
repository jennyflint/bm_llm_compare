package config

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestParseDeviceTarget(t *testing.T) {
	cases := map[string]DeviceTarget{
		"":      DeviceCPU,
		"cpu":   DeviceCPU,
		"CPU":   DeviceCPU,
		" gpu ": DeviceGPU,
	}
	for in, want := range cases {
		got, err := ParseDeviceTarget(in)
		if err != nil {
			t.Fatalf("ParseDeviceTarget(%q) error: %v", in, err)
		}
		if got != want {
			t.Errorf("ParseDeviceTarget(%q) = %q, want %q", in, got, want)
		}
	}
	if _, err := ParseDeviceTarget("tpu"); err == nil {
		t.Error("expected error for invalid device")
	}
}

func TestParseOutputFormat(t *testing.T) {
	for _, in := range []string{"", "table", "json", "markdown", "JSON"} {
		if _, err := ParseOutputFormat(in); err != nil {
			t.Errorf("ParseOutputFormat(%q) unexpected error: %v", in, err)
		}
	}
	if _, err := ParseOutputFormat("xml"); err == nil {
		t.Error("expected error for invalid format")
	}
}

func TestResolveCPUForcesZeroGPULayers(t *testing.T) {
	opts, err := Resolve(&File{}, FlagValues{
		PromptFile: "prompt.txt",
		Models:     "llama3, qwen2",
		Device:     "cpu",
		CPUThreads: 4,
		GPULayers:  20,
	})
	if err != nil {
		t.Fatalf("Resolve error: %v", err)
	}
	if opts.GPULayers != 0 {
		t.Errorf("CPU mode GPULayers = %d, want 0", opts.GPULayers)
	}
	if opts.CPUThreads != 4 {
		t.Errorf("CPUThreads = %d, want 4", opts.CPUThreads)
	}
	if len(opts.Models) != 2 || opts.Models[0] != "llama3" || opts.Models[1] != "qwen2" {
		t.Errorf("Models = %v, want [llama3 qwen2]", opts.Models)
	}
}

func TestResolveGPUDefaultsAllLayers(t *testing.T) {
	opts, err := Resolve(&File{}, FlagValues{
		PromptFile: "prompt.txt",
		Models:     "llama3",
		Device:     "gpu",
	})
	if err != nil {
		t.Fatalf("Resolve error: %v", err)
	}
	if opts.GPULayers != -1 {
		t.Errorf("GPU mode default GPULayers = %d, want -1", opts.GPULayers)
	}
}

func TestResolveGPUExplicitLayers(t *testing.T) {
	opts, err := Resolve(&File{}, FlagValues{
		PromptFile: "prompt.txt",
		Models:     "llama3",
		Device:     "gpu",
		GPULayers:  12,
	})
	if err != nil {
		t.Fatalf("Resolve error: %v", err)
	}
	if opts.GPULayers != 12 {
		t.Errorf("GPULayers = %d, want 12", opts.GPULayers)
	}
}

func TestResolveRequiresPromptFile(t *testing.T) {
	if _, err := Resolve(&File{}, FlagValues{Models: "llama3"}); err == nil {
		t.Error("expected error when prompt file missing")
	}
}

func TestResolveRequiresModels(t *testing.T) {
	if _, err := Resolve(&File{}, FlagValues{PromptFile: "p.txt"}); err == nil {
		t.Error("expected error when no models specified")
	}
}

func TestResolveDefaults(t *testing.T) {
	opts, err := Resolve(&File{}, FlagValues{PromptFile: "p.txt", Models: "m"})
	if err != nil {
		t.Fatalf("Resolve error: %v", err)
	}
	if opts.Timeout != DefaultTimeout {
		t.Errorf("Timeout = %s, want %s", opts.Timeout, DefaultTimeout)
	}
	if opts.LogDir != "logs" {
		t.Errorf("LogDir = %q, want logs", opts.LogDir)
	}
	if opts.Format != FormatTable {
		t.Errorf("Format = %q, want table", opts.Format)
	}
	if opts.Concurrency < 1 {
		t.Errorf("Concurrency = %d, want >= 1", opts.Concurrency)
	}
}

func TestLoadFileAndResolve(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	content := `
defaults:
  device: gpu
  cpu_threads: 8
  gpu_layers: 30
  max_tokens: 256
  num_ctx: 4096
  timeout: 45s
  concurrency: 3
  format: json
models:
  - name: mistral
    engine: ollama
    model_id: mistral:7b
    cpu_threads: 2
`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	file, err := LoadFile(path, false)
	if err != nil {
		t.Fatalf("LoadFile error: %v", err)
	}
	opts, err := Resolve(file, FlagValues{PromptFile: "p.txt", Models: "mistral"})
	if err != nil {
		t.Fatalf("Resolve error: %v", err)
	}
	if opts.Device != DeviceGPU {
		t.Errorf("Device = %q, want gpu", opts.Device)
	}
	if opts.GPULayers != 30 {
		t.Errorf("GPULayers = %d, want 30", opts.GPULayers)
	}
	if opts.Timeout != 45*time.Second {
		t.Errorf("Timeout = %s, want 45s", opts.Timeout)
	}
	if opts.Concurrency != 3 {
		t.Errorf("Concurrency = %d, want 3", opts.Concurrency)
	}
	if opts.Format != FormatJSON {
		t.Errorf("Format = %q, want json", opts.Format)
	}

	limits, err := opts.ForModel(file.Models[0])
	if err != nil {
		t.Fatalf("ForModel error: %v", err)
	}
	if limits.CPUThreads != 2 {
		t.Errorf("per-model CPUThreads = %d, want 2", limits.CPUThreads)
	}
	if limits.GPULayers != 30 {
		t.Errorf("per-model GPULayers = %d, want 30", limits.GPULayers)
	}
}

func TestLoadFileMissingAllowed(t *testing.T) {
	file, err := LoadFile(filepath.Join(t.TempDir(), "nope.yaml"), true)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if file == nil {
		t.Fatal("expected non-nil file")
	}
}

func TestLoadFileMissingNotAllowed(t *testing.T) {
	if _, err := LoadFile(filepath.Join(t.TempDir(), "nope.yaml"), false); err == nil {
		t.Error("expected error for missing file")
	}
}

func TestLoadFileInvalidYAML(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.yaml")
	if err := os.WriteFile(path, []byte("defaults: [oops"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadFile(path, false); err == nil {
		t.Error("expected parse error")
	}
}
