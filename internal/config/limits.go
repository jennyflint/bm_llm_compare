package config

import (
	"fmt"
	"time"
)

// ResourceLimits controls the execution backend and guardrails for one run.
type ResourceLimits struct {
	Device     DeviceTarget
	CPUThreads int
	GPULayers  int
	MaxTokens  int
	NumCtx     int
	Timeout    time.Duration
}

// Validate checks that the limits are internally consistent.
func (r ResourceLimits) Validate() error {
	if _, err := ParseDeviceTarget(string(r.Device)); err != nil {
		return err
	}
	if r.CPUThreads < 1 {
		return fmt.Errorf("cpu threads must be >= 1, got %d", r.CPUThreads)
	}
	if r.MaxTokens < 1 {
		return fmt.Errorf("max tokens must be >= 1, got %d", r.MaxTokens)
	}
	if r.NumCtx < 1 {
		return fmt.Errorf("num_ctx must be >= 1, got %d", r.NumCtx)
	}
	if r.Timeout <= 0 {
		return fmt.Errorf("timeout must be positive, got %s", r.Timeout)
	}
	if r.Device == DeviceCPU && r.GPULayers != 0 {
		return fmt.Errorf("cpu device requires gpu layers 0, got %d", r.GPULayers)
	}
	return nil
}

// ForModel builds ResourceLimits for a model, applying per-model overrides on
// top of the global options.
func (o *Options) ForModel(m ModelConfig) (ResourceLimits, error) {
	limits := ResourceLimits{
		Device:     o.Device,
		CPUThreads: o.CPUThreads,
		GPULayers:  o.GPULayers,
		MaxTokens:  o.MaxTokens,
		NumCtx:     o.NumCtx,
		Timeout:    o.Timeout,
	}
	if m.CPUThreads > 0 && !o.explicit.CPUThreads {
		limits.CPUThreads = m.CPUThreads
	}
	if m.GPULayers != 0 && !o.explicit.GPULayers {
		limits.GPULayers = m.GPULayers
	}
	if m.MaxTokens > 0 && !o.explicit.MaxTokens {
		limits.MaxTokens = m.MaxTokens
	}
	if m.NumCtx > 0 && !o.explicit.NumCtx {
		limits.NumCtx = m.NumCtx
	}
	if m.Timeout != "" && !o.explicit.Timeout {
		parsed, err := time.ParseDuration(m.Timeout)
		if err != nil {
			return ResourceLimits{}, fmt.Errorf("parse timeout for model %q: %w", m.Name, err)
		}
		limits.Timeout = parsed
	}
	if limits.Device == DeviceCPU {
		limits.GPULayers = 0
	}
	if err := limits.Validate(); err != nil {
		return ResourceLimits{}, fmt.Errorf("model %q limits: %w", m.Name, err)
	}
	return limits, nil
}
