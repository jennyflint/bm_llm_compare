package config

import (
	"testing"
	"time"
)

func TestResourceLimitsValidate(t *testing.T) {
	valid := ResourceLimits{
		Device:     DeviceCPU,
		CPUThreads: 4,
		GPULayers:  0,
		MaxTokens:  128,
		NumCtx:     1024,
		Timeout:    time.Second,
	}
	if err := valid.Validate(); err != nil {
		t.Fatalf("valid limits rejected: %v", err)
	}

	bad := []ResourceLimits{
		{Device: "tpu", CPUThreads: 1, MaxTokens: 1, NumCtx: 1, Timeout: time.Second},
		{Device: DeviceCPU, CPUThreads: 0, MaxTokens: 1, NumCtx: 1, Timeout: time.Second},
		{Device: DeviceCPU, CPUThreads: 1, MaxTokens: 0, NumCtx: 1, Timeout: time.Second},
		{Device: DeviceCPU, CPUThreads: 1, MaxTokens: 1, NumCtx: 0, Timeout: time.Second},
		{Device: DeviceCPU, CPUThreads: 1, MaxTokens: 1, NumCtx: 1, Timeout: 0},
		{Device: DeviceCPU, CPUThreads: 1, GPULayers: 5, MaxTokens: 1, NumCtx: 1, Timeout: time.Second},
	}
	for i, l := range bad {
		if err := l.Validate(); err == nil {
			t.Errorf("case %d: expected validation error", i)
		}
	}
}

func TestForModelCPUOverridesGPULayers(t *testing.T) {
	opts := &Options{
		Device:     DeviceCPU,
		CPUThreads: 8,
		GPULayers:  0,
		MaxTokens:  256,
		NumCtx:     2048,
		Timeout:    30 * time.Second,
	}
	limits, err := opts.ForModel(ModelConfig{Name: "m", GPULayers: 40})
	if err != nil {
		t.Fatalf("ForModel error: %v", err)
	}
	if limits.GPULayers != 0 {
		t.Errorf("CPU mode GPULayers = %d, want 0", limits.GPULayers)
	}
}

func TestForModelTimeoutOverride(t *testing.T) {
	opts := &Options{
		Device:     DeviceGPU,
		CPUThreads: 8,
		GPULayers:  -1,
		MaxTokens:  256,
		NumCtx:     2048,
		Timeout:    30 * time.Second,
	}
	limits, err := opts.ForModel(ModelConfig{Name: "m", Timeout: "5s"})
	if err != nil {
		t.Fatalf("ForModel error: %v", err)
	}
	if limits.Timeout != 5*time.Second {
		t.Errorf("Timeout = %s, want 5s", limits.Timeout)
	}
}

func TestExplicitFlagsOverrideModelConfig(t *testing.T) {
	file := &File{
		Models: []ModelConfig{{
			Name:       "mistral",
			CPUThreads: 8,
			MaxTokens:  512,
			NumCtx:     4096,
			Timeout:    "120s",
		}},
	}
	opts, err := Resolve(file, FlagValues{
		PromptFile: "p.txt",
		Models:     "mistral",
		Device:     "gpu",
		CPUThreads: 4,
		MaxTokens:  256,
		NumCtx:     8192,
		Timeout:    30 * time.Second,
	})
	if err != nil {
		t.Fatalf("Resolve error: %v", err)
	}
	limits, err := opts.ForModel(file.Models[0])
	if err != nil {
		t.Fatalf("ForModel error: %v", err)
	}
	if limits.CPUThreads != 4 {
		t.Errorf("CPUThreads = %d, want 4 (CLI flag wins)", limits.CPUThreads)
	}
	if limits.MaxTokens != 256 {
		t.Errorf("MaxTokens = %d, want 256 (CLI flag wins)", limits.MaxTokens)
	}
	if limits.NumCtx != 8192 {
		t.Errorf("NumCtx = %d, want 8192 (CLI flag wins)", limits.NumCtx)
	}
	if limits.Timeout != 30*time.Second {
		t.Errorf("Timeout = %s, want 30s (CLI flag wins)", limits.Timeout)
	}
}

func TestModelConfigAppliesWhenFlagsAbsent(t *testing.T) {
	file := &File{
		Models: []ModelConfig{{
			Name:       "mistral",
			CPUThreads: 8,
			MaxTokens:  512,
			NumCtx:     4096,
		}},
	}
	opts, err := Resolve(file, FlagValues{PromptFile: "p.txt", Models: "mistral", Device: "gpu"})
	if err != nil {
		t.Fatalf("Resolve error: %v", err)
	}
	limits, err := opts.ForModel(file.Models[0])
	if err != nil {
		t.Fatalf("ForModel error: %v", err)
	}
	if limits.CPUThreads != 8 || limits.MaxTokens != 512 || limits.NumCtx != 4096 {
		t.Errorf("model config not applied: %+v", limits)
	}
}

func TestForModelInvalidTimeout(t *testing.T) {
	opts := &Options{
		Device:     DeviceGPU,
		CPUThreads: 8,
		GPULayers:  -1,
		MaxTokens:  256,
		NumCtx:     2048,
		Timeout:    30 * time.Second,
	}
	if _, err := opts.ForModel(ModelConfig{Name: "m", Timeout: "not-a-duration"}); err == nil {
		t.Error("expected error for invalid timeout")
	}
}
