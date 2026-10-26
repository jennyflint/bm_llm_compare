package engine

import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/llm-bench/internal/config"
)

// Factory builds runners for models, resolving config entries and scanning a
// models directory for local weights.
type Factory struct {
	Options *config.Options
	Client  *http.Client
}

// NewFactory constructs a Factory bound to resolved options.
func NewFactory(opts *config.Options) *Factory {
	return &Factory{Options: opts, Client: &http.Client{}}
}

// Build returns a Runner for the named model. Config entries take precedence;
// otherwise the model is treated as an Ollama model.
func (f *Factory) Build(name string) (Runner, error) {
	def, found := f.lookup(name)
	if !found {
		def = config.ModelConfig{Name: name, Engine: config.EngineOllama}
	}

	switch def.Engine {
	case config.EngineOllama, "":
		endpoint := f.Options.OllamaURL
		if def.Endpoint != "" {
			endpoint = def.Endpoint
		}
		return NewOllamaRunner(def.Name, def.ModelID, endpoint, f.Client), nil
	case config.EngineOpenAI:
		return NewOpenAIRunner(def.Name, def.ModelID, def.Endpoint, def.APIKeyEnv, f.Client), nil
	case config.EngineExec:
		return NewExecRunner(def.Name, def.Command), nil
	default:
		return nil, fmt.Errorf("model %q: unknown engine %q", name, def.Engine)
	}
}

// BuildAll constructs runners for every requested model.
func (f *Factory) BuildAll(names []string) ([]Runner, error) {
	runners := make([]Runner, 0, len(names))
	for _, name := range names {
		r, err := f.Build(name)
		if err != nil {
			return nil, err
		}
		runners = append(runners, r)
	}
	return runners, nil
}

// ScanModelsDir lists model files found in the configured models directory.
func (f *Factory) ScanModelsDir() ([]string, error) {
	dir := f.Options.ModelsDir
	if dir == "" {
		return nil, nil
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("scan models dir %q: %w", dir, err)
	}
	var models []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(e.Name()))
		if ext == ".gguf" || ext == ".bin" || ext == ".yaml" || ext == ".json" {
			models = append(models, e.Name())
		}
	}
	return models, nil
}

func (f *Factory) lookup(name string) (config.ModelConfig, bool) {
	for _, m := range f.Options.ModelDefs {
		if m.Name == name {
			return m, true
		}
	}
	return config.ModelConfig{}, false
}
