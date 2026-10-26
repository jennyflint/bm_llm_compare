package engine

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/llm-bench/internal/config"
)

func TestOpenAIRunnerSuccess(t *testing.T) {
	var captured openAIRequest
	var authHeader string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		authHeader = r.Header.Get("Authorization")
		_ = json.NewDecoder(r.Body).Decode(&captured)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"hello world"}}],"usage":{"prompt_tokens":5,"completion_tokens":2}}`))
	}))
	defer srv.Close()

	t.Setenv("TEST_API_KEY", "secret")
	runner := NewOpenAIRunner("remote", "gpt-x", srv.URL, "TEST_API_KEY", srv.Client())
	limits := config.ResourceLimits{Device: config.DeviceGPU, CPUThreads: 1, MaxTokens: 64, NumCtx: 1024, Timeout: time.Second}
	res, err := runner.Run(context.Background(), "hi", limits)
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if authHeader != "Bearer secret" {
		t.Errorf("Authorization = %q, want Bearer secret", authHeader)
	}
	if captured.MaxTokens != 64 {
		t.Errorf("max_tokens = %d, want 64", captured.MaxTokens)
	}
	if res.ResponseText != "hello world" {
		t.Errorf("ResponseText = %q", res.ResponseText)
	}
	if res.PromptTokens != 5 || res.GeneratedTokens != 2 {
		t.Errorf("tokens = %d/%d, want 5/2", res.PromptTokens, res.GeneratedTokens)
	}
}

func TestOpenAIRunnerErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"bad key"}}`))
	}))
	defer srv.Close()

	runner := NewOpenAIRunner("remote", "", srv.URL, "", srv.Client())
	limits := config.ResourceLimits{Device: config.DeviceGPU, CPUThreads: 1, MaxTokens: 1, NumCtx: 1, Timeout: time.Second}
	if _, err := runner.Run(context.Background(), "hi", limits); err == nil {
		t.Error("expected error on 401")
	}
}

func TestOpenAIRunnerNoChoices(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[]}`))
	}))
	defer srv.Close()

	runner := NewOpenAIRunner("remote", "", srv.URL, "", srv.Client())
	limits := config.ResourceLimits{Device: config.DeviceGPU, CPUThreads: 1, MaxTokens: 1, NumCtx: 1, Timeout: time.Second}
	if _, err := runner.Run(context.Background(), "hi", limits); err == nil {
		t.Error("expected error when no choices returned")
	}
}
