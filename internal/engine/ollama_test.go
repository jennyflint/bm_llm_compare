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

func TestOllamaRunnerPropagatesLimits(t *testing.T) {
	var captured ollamaRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/generate" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
			t.Errorf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"response":"hi there","prompt_eval_count":7,"eval_count":3,"eval_duration":1000000000}`))
	}))
	defer srv.Close()

	runner := NewOllamaRunner("llama", "llama:7b", srv.URL, srv.Client())
	limits := config.ResourceLimits{
		Device:     config.DeviceCPU,
		CPUThreads: 6,
		GPULayers:  0,
		MaxTokens:  128,
		NumCtx:     4096,
		Timeout:    5 * time.Second,
	}
	res, err := runner.Run(context.Background(), "hello", limits)
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}

	if captured.Options.NumGPU != 0 {
		t.Errorf("num_gpu = %d, want 0", captured.Options.NumGPU)
	}
	if captured.Options.NumThread != 6 {
		t.Errorf("num_thread = %d, want 6", captured.Options.NumThread)
	}
	if captured.Options.NumPredict != 128 {
		t.Errorf("num_predict = %d, want 128", captured.Options.NumPredict)
	}
	if captured.Options.NumCtx != 4096 {
		t.Errorf("num_ctx = %d, want 4096", captured.Options.NumCtx)
	}
	if captured.Model != "llama:7b" {
		t.Errorf("model = %q, want llama:7b", captured.Model)
	}
	if res.ResponseText != "hi there" {
		t.Errorf("ResponseText = %q", res.ResponseText)
	}
	if res.PromptTokens != 7 || res.GeneratedTokens != 3 {
		t.Errorf("tokens = %d/%d, want 7/3", res.PromptTokens, res.GeneratedTokens)
	}
	if res.Status != StatusSuccess {
		t.Errorf("Status = %q, want Success", res.Status)
	}
}

func TestOllamaRunnerGPUOffload(t *testing.T) {
	var captured ollamaRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&captured)
		_, _ = w.Write([]byte(`{"response":"ok","eval_count":1}`))
	}))
	defer srv.Close()

	runner := NewOllamaRunner("llama", "", srv.URL, srv.Client())
	limits := config.ResourceLimits{Device: config.DeviceGPU, CPUThreads: 2, GPULayers: -1, MaxTokens: 10, NumCtx: 512, Timeout: time.Second}
	if _, err := runner.Run(context.Background(), "p", limits); err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if captured.Options.NumGPU != -1 {
		t.Errorf("num_gpu = %d, want -1", captured.Options.NumGPU)
	}
}

func TestOllamaRunnerHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("model not found"))
	}))
	defer srv.Close()

	runner := NewOllamaRunner("llama", "", srv.URL, srv.Client())
	limits := config.ResourceLimits{Device: config.DeviceCPU, CPUThreads: 1, MaxTokens: 1, NumCtx: 1, Timeout: time.Second}
	if _, err := runner.Run(context.Background(), "p", limits); err == nil {
		t.Error("expected error on 500 response")
	}
}

func TestOllamaRunnerErrorField(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"error":"out of memory"}`))
	}))
	defer srv.Close()

	runner := NewOllamaRunner("llama", "", srv.URL, srv.Client())
	limits := config.ResourceLimits{Device: config.DeviceCPU, CPUThreads: 1, MaxTokens: 1, NumCtx: 1, Timeout: time.Second}
	if _, err := runner.Run(context.Background(), "p", limits); err == nil {
		t.Error("expected error from error field")
	}
}
