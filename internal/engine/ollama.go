package engine

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/llm-bench/internal/config"
)

// OllamaRunner talks to the Ollama HTTP API (/api/generate).
type OllamaRunner struct {
	modelName string
	modelID   string
	endpoint  string
	client    *http.Client
}

// NewOllamaRunner constructs an Ollama runner. modelID defaults to modelName.
func NewOllamaRunner(modelName, modelID, endpoint string, client *http.Client) *OllamaRunner {
	if modelID == "" {
		modelID = modelName
	}
	if endpoint == "" {
		endpoint = "http://localhost:11434"
	}
	if client == nil {
		client = &http.Client{}
	}
	return &OllamaRunner{
		modelName: modelName,
		modelID:   modelID,
		endpoint:  strings.TrimRight(endpoint, "/"),
		client:    client,
	}
}

// Name returns the display name of the model.
func (o *OllamaRunner) Name() string { return o.modelName }

type ollamaRequest struct {
	Model   string        `json:"model"`
	Prompt  string        `json:"prompt"`
	Stream  bool          `json:"stream"`
	Options ollamaOptions `json:"options"`
}

type ollamaOptions struct {
	NumGPU     int `json:"num_gpu"`
	NumThread  int `json:"num_thread"`
	NumPredict int `json:"num_predict"`
	NumCtx     int `json:"num_ctx"`
}

type ollamaResponse struct {
	Response        string `json:"response"`
	PromptEvalCount int    `json:"prompt_eval_count"`
	EvalCount       int    `json:"eval_count"`
	EvalDuration    int64  `json:"eval_duration"`
	Error           string `json:"error"`
}

// Run executes the prompt against Ollama, honoring device and guardrails.
func (o *OllamaRunner) Run(ctx context.Context, prompt string, limits config.ResourceLimits) (*BenchmarkResult, error) {
	reqBody := ollamaRequest{
		Model:  o.modelID,
		Prompt: prompt,
		Stream: false,
		Options: ollamaOptions{
			NumGPU:     limits.GPULayers,
			NumThread:  limits.CPUThreads,
			NumPredict: limits.MaxTokens,
			NumCtx:     limits.NumCtx,
		},
	}
	payload, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal ollama request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.endpoint+"/api/generate", bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("build ollama request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	start := time.Now()
	resp, err := o.client.Do(req)
	if err != nil {
		return nil, wrapTimeout(o.modelName, fmt.Errorf("ollama request: %w", err))
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read ollama response: %w", err)
	}
	duration := time.Since(start)

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama returned status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var parsed ollamaResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, fmt.Errorf("decode ollama response: %w", err)
	}
	if parsed.Error != "" {
		return nil, fmt.Errorf("ollama error: %s", parsed.Error)
	}

	res := &BenchmarkResult{
		ModelName:       o.modelName,
		Device:          limits.Device,
		PromptTokens:    parsed.PromptEvalCount,
		GeneratedTokens: parsed.EvalCount,
		Duration:        duration,
		ResponseText:    parsed.Response,
	}
	if res.PromptTokens == 0 {
		res.PromptTokens = EstimateTokens(prompt)
	}
	if parsed.EvalDuration > 0 {
		res.TTFT = duration - time.Duration(parsed.EvalDuration)
		if res.TTFT < 0 {
			res.TTFT = 0
		}
	}
	Finalize(res)
	return res, nil
}
