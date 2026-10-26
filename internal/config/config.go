// Package config defines the benchmark configuration model, loads it from a
// YAML file, and merges CLI flag overrides on top of the file values.
package config

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// DeviceTarget selects the compute backend used for a model run.
type DeviceTarget string

const (
	DeviceCPU DeviceTarget = "cpu"
	DeviceGPU DeviceTarget = "gpu"
)

// ParseDeviceTarget validates and normalizes a device string.
func ParseDeviceTarget(raw string) (DeviceTarget, error) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", string(DeviceCPU):
		return DeviceCPU, nil
	case string(DeviceGPU):
		return DeviceGPU, nil
	default:
		return "", fmt.Errorf("invalid device %q: expected %q or %q", raw, DeviceCPU, DeviceGPU)
	}
}

// OutputFormat selects the report serializer.
type OutputFormat string

const (
	FormatTable    OutputFormat = "table"
	FormatJSON     OutputFormat = "json"
	FormatMarkdown OutputFormat = "markdown"
)

// ParseOutputFormat validates and normalizes an output format string.
func ParseOutputFormat(raw string) (OutputFormat, error) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", string(FormatTable):
		return FormatTable, nil
	case string(FormatJSON):
		return FormatJSON, nil
	case string(FormatMarkdown):
		return FormatMarkdown, nil
	default:
		return "", fmt.Errorf("invalid format %q: expected table, json or markdown", raw)
	}
}

// EngineKind identifies the runner implementation used for a model.
type EngineKind string

const (
	EngineOllama EngineKind = "ollama"
	EngineOpenAI EngineKind = "openai"
	EngineExec   EngineKind = "exec"
)

// ModelConfig describes a single model entry from the configuration file.
type ModelConfig struct {
	Name        string     `yaml:"name"`
	Engine      EngineKind `yaml:"engine"`
	Endpoint    string     `yaml:"endpoint"`
	ModelID     string     `yaml:"model_id"`
	APIKeyEnv   string     `yaml:"api_key_env"`
	Command     string     `yaml:"command"`
	CPUThreads  int        `yaml:"cpu_threads"`
	GPULayers   int        `yaml:"gpu_layers"`
	MaxTokens   int        `yaml:"max_tokens"`
	NumCtx      int        `yaml:"num_ctx"`
	Timeout     string     `yaml:"timeout"`
	Temperature float64    `yaml:"temperature"`
}

// Defaults holds global fallback values applied to every model run.
type Defaults struct {
	Device      DeviceTarget `yaml:"device"`
	CPUThreads  int          `yaml:"cpu_threads"`
	GPULayers   int          `yaml:"gpu_layers"`
	MaxTokens   int          `yaml:"max_tokens"`
	NumCtx      int          `yaml:"num_ctx"`
	Timeout     string       `yaml:"timeout"`
	Concurrency int          `yaml:"concurrency"`
	Format      OutputFormat `yaml:"format"`
	NoColor     bool         `yaml:"no_color"`
	LogDir      string       `yaml:"log_dir"`
	OllamaURL   string       `yaml:"ollama_url"`
}

// File is the on-disk configuration schema.
type File struct {
	Defaults Defaults      `yaml:"defaults"`
	Models   []ModelConfig `yaml:"models"`
}

// Options is the fully resolved runtime configuration.
type Options struct {
	Project     string
	PromptFile  string
	Models      []string
	ModelsDir   string
	ConfigPath  string
	Concurrency int
	Format      OutputFormat
	NoColor     bool
	Device      DeviceTarget
	CPUThreads  int
	GPULayers   int
	MaxTokens   int
	NumCtx      int
	Timeout     time.Duration
	LogDir      string
	OllamaURL   string
	ModelDefs   []ModelConfig

	explicit explicitFlags
}

// explicitFlags records which limit values were set directly on the CLI so
// they take precedence over per-model configuration entries.
type explicitFlags struct {
	CPUThreads bool
	GPULayers  bool
	MaxTokens  bool
	NumCtx     bool
	Timeout    bool
}

// DefaultTimeout is used when neither flags nor config specify one.
const DefaultTimeout = 120 * time.Second

// LoadFile reads and parses a YAML configuration file. A missing file is not
// an error when allowMissing is true; the returned File is then empty.
func LoadFile(path string, allowMissing bool) (*File, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) && allowMissing {
			return &File{}, nil
		}
		return nil, fmt.Errorf("read config %q: %w", path, err)
	}
	var f File
	if err := yaml.Unmarshal(data, &f); err != nil {
		return nil, fmt.Errorf("parse config %q: %w", path, err)
	}
	return &f, nil
}

// FlagValues carries raw CLI flag values. Zero values mean "not provided".
type FlagValues struct {
	Project     string
	PromptFile  string
	Models      string
	ModelsDir   string
	ConfigPath  string
	Concurrency int
	Format      string
	NoColor     bool
	Device      string
	CPUThreads  int
	GPULayers   int
	MaxTokens   int
	NumCtx      int
	Timeout     time.Duration
	LogDir      string
	OllamaURL   string
}

// Resolve merges file defaults with CLI flags, with flags taking precedence.
func Resolve(file *File, flags FlagValues) (*Options, error) {
	if file == nil {
		file = &File{}
	}
	d := file.Defaults

	opts := &Options{
		Project:    flags.Project,
		PromptFile: flags.PromptFile,
		ModelsDir:  flags.ModelsDir,
		ConfigPath: flags.ConfigPath,
		ModelDefs:  file.Models,
		explicit: explicitFlags{
			CPUThreads: flags.CPUThreads != 0,
			GPULayers:  flags.GPULayers != 0,
			MaxTokens:  flags.MaxTokens != 0,
			NumCtx:     flags.NumCtx != 0,
			Timeout:    flags.Timeout > 0,
		},
	}

	opts.Models = splitModels(flags.Models)

	opts.Concurrency = firstNonZero(flags.Concurrency, d.Concurrency, runtime.NumCPU())
	if opts.Concurrency < 1 {
		return nil, fmt.Errorf("concurrency must be >= 1, got %d", opts.Concurrency)
	}

	rawFormat := flags.Format
	if rawFormat == "" {
		rawFormat = string(d.Format)
	}
	format, err := ParseOutputFormat(rawFormat)
	if err != nil {
		return nil, err
	}
	opts.Format = format

	opts.NoColor = flags.NoColor || d.NoColor

	rawDevice := flags.Device
	if rawDevice == "" {
		rawDevice = string(d.Device)
	}
	device, err := ParseDeviceTarget(rawDevice)
	if err != nil {
		return nil, err
	}
	opts.Device = device

	opts.CPUThreads = firstNonZero(flags.CPUThreads, d.CPUThreads, runtime.NumCPU())
	if opts.CPUThreads < 1 {
		return nil, fmt.Errorf("cpu threads must be >= 1, got %d", opts.CPUThreads)
	}

	opts.GPULayers = resolveGPULayers(device, flags.GPULayers, d.GPULayers)
	opts.MaxTokens = firstNonZero(flags.MaxTokens, d.MaxTokens, 512)
	opts.NumCtx = firstNonZero(flags.NumCtx, d.NumCtx, 2048)

	opts.Timeout = flags.Timeout
	if opts.Timeout <= 0 {
		if d.Timeout != "" {
			parsed, perr := time.ParseDuration(d.Timeout)
			if perr != nil {
				return nil, fmt.Errorf("parse default timeout %q: %w", d.Timeout, perr)
			}
			opts.Timeout = parsed
		}
	}
	if opts.Timeout <= 0 {
		opts.Timeout = DefaultTimeout
	}

	opts.LogDir = firstNonEmpty(flags.LogDir, d.LogDir, "logs")
	opts.OllamaURL = firstNonEmpty(flags.OllamaURL, d.OllamaURL, "http://localhost:11434")

	if opts.PromptFile == "" {
		return nil, errors.New("prompt file is required (--prompt-file)")
	}
	if len(opts.Models) == 0 && len(opts.ModelDefs) == 0 {
		return nil, errors.New("no models specified (--models or config models)")
	}
	return opts, nil
}

// resolveGPULayers normalizes GPU layer offload: CPU mode always forces zero
// offload, GPU mode defaults to -1 (all layers) when unspecified.
func resolveGPULayers(device DeviceTarget, flagVal, defVal int) int {
	if device == DeviceCPU {
		return 0
	}
	if flagVal != 0 {
		return flagVal
	}
	if defVal != 0 {
		return defVal
	}
	return -1
}

func splitModels(raw string) []string {
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if v := strings.TrimSpace(p); v != "" {
			out = append(out, v)
		}
	}
	return out
}

func firstNonZero(values ...int) int {
	for _, v := range values {
		if v != 0 {
			return v
		}
	}
	return 0
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}
