package engine

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/llm-bench/internal/config"
)

// OpenAIRunner talks to any OpenAI-compatible /v1/chat/completions endpoint.
type OpenAIRunner struct {
	modelName string
	modelID   string
	endpoint  string
	apiKey    string
	client    *http.Client
}

// NewOpenAIRunner constructs an OpenAI-compatible runner. The API key is read
// from the environment variable named by apiKeyEnv, if provided.
func NewOpenAIRunner(modelName, modelID, endpoint, apiKeyEnv string, client *http.Client) *OpenAIRunner {
	if modelID == "" {
		modelID = modelName
	}
	if endpoint == "" {
		endpoint = "http://localhost:8080"
	}
	if client == nil {
		client = &http.Client{}
	}
	apiKey := ""
	if apiKeyEnv != "" {
		apiKey = os.Getenv(apiKeyEnv)
	}
	return &OpenAIRunner{
		modelName: modelName,
		modelID:   modelID,
		endpoint:  strings.TrimRight(endpoint, "/"),
		apiKey:    apiKey,
		client:    client,
	}
}

// Name returns the display name of the model.
func (o *OpenAIRunner) Name() string { return o.modelName }

type openAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIRequest struct {
	Model       string          `json:"model"`
	Messages    []openAIMessage `json:"messages"`
	MaxTokens   int             `json:"max_tokens"`
	Temperature float64         `json:"temperature"`
	Stream      bool            `json:"stream"`
}

type openAIResponse struct {
	Choices []struct {
		Message openAIMessage `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// Run executes the prompt against the remote endpoint.
func (o *OpenAIRunner) Run(ctx context.Context, prompt string, limits config.ResourceLimits) (*BenchmarkResult, error) {
	reqBody := openAIRequest{
		Model:       o.modelID,
		Messages:    []openAIMessage{{Role: "user", Content: prompt}},
		MaxTokens:   limits.MaxTokens,
		Temperature: 0.7,
		Stream:      false,
	}
	payload, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal openai request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.endpoint+"/v1/chat/completions", bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("build openai request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if o.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+o.apiKey)
	}

	start := time.Now()
	resp, err := o.client.Do(req)
	if err != nil {
		return nil, wrapTimeout(o.modelName, fmt.Errorf("openai request: %w", err))
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read openai response: %w", err)
	}
	duration := time.Since(start)

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("openai returned status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var parsed openAIResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, fmt.Errorf("decode openai response: %w", err)
	}
	if parsed.Error != nil && parsed.Error.Message != "" {
		return nil, fmt.Errorf("openai error: %s", parsed.Error.Message)
	}
	if len(parsed.Choices) == 0 {
		return nil, fmt.Errorf("openai returned no choices")
	}

	text := parsed.Choices[0].Message.Content
	res := &BenchmarkResult{
		ModelName:       o.modelName,
		Device:          limits.Device,
		PromptTokens:    parsed.Usage.PromptTokens,
		GeneratedTokens: parsed.Usage.CompletionTokens,
		Duration:        duration,
		ResponseText:    text,
	}
	if res.PromptTokens == 0 {
		res.PromptTokens = EstimateTokens(prompt)
	}
	if res.GeneratedTokens == 0 {
		res.GeneratedTokens = EstimateTokens(text)
	}
	Finalize(res)
	return res, nil
}
