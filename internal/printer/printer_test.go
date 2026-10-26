package printer

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/llm-bench/internal/config"
	"github.com/llm-bench/internal/engine"
)

func sampleResults() []*engine.BenchmarkResult {
	return []*engine.BenchmarkResult{
		{
			ModelName:       "llama",
			Device:          config.DeviceCPU,
			Status:          engine.StatusSuccess,
			PromptTokens:    10,
			GeneratedTokens: 100,
			Duration:        2 * time.Second,
			TTFT:            500 * time.Millisecond,
			TokensPerSec:    66.67,
			ResponseText:    "the quick brown fox jumps over the lazy dog",
		},
		{
			ModelName: "qwen",
			Device:    config.DeviceGPU,
			Status:    engine.StatusTimeout,
			Error:     engine.ErrTimeout,
		},
	}
}

func TestPreview(t *testing.T) {
	if got := Preview("hello   world", 20); got != "hello world" {
		t.Errorf("Preview = %q, want 'hello world'", got)
	}
	if got := Preview("abcdefghij", 5); got != "ab..." {
		t.Errorf("Preview = %q, want 'ab...'", got)
	}
	if got := Preview("abc", 10); got != "abc" {
		t.Errorf("Preview = %q, want 'abc'", got)
	}
}

func TestPrintJSON(t *testing.T) {
	var buf bytes.Buffer
	p := New(&buf, config.FormatJSON, true)
	if err := p.Print(sampleResults()); err != nil {
		t.Fatalf("Print error: %v", err)
	}
	var rows []jsonRow
	if err := json.Unmarshal(buf.Bytes(), &rows); err != nil {
		t.Fatalf("unmarshal json output: %v", err)
	}
	if len(rows) != 2 {
		t.Fatalf("rows = %d, want 2", len(rows))
	}
	if rows[0].ModelName != "llama" || rows[0].TokensPerSec != 66.67 {
		t.Errorf("unexpected row: %+v", rows[0])
	}
	if rows[1].Status != engine.StatusTimeout {
		t.Errorf("status = %q, want Timeout", rows[1].Status)
	}
}

func TestPrintMarkdown(t *testing.T) {
	var buf bytes.Buffer
	p := New(&buf, config.FormatMarkdown, true)
	if err := p.Print(sampleResults()); err != nil {
		t.Fatalf("Print error: %v", err)
	}
	out := buf.String()
	if !strings.Contains(out, "| Model | Device | Status |") {
		t.Errorf("missing markdown header:\n%s", out)
	}
	if !strings.Contains(out, "| llama | cpu | Success |") {
		t.Errorf("missing llama row:\n%s", out)
	}
	if !strings.Contains(out, "| qwen | gpu | Timeout |") {
		t.Errorf("missing qwen row:\n%s", out)
	}
}

func TestPrintTable(t *testing.T) {
	var buf bytes.Buffer
	p := New(&buf, config.FormatTable, true)
	if err := p.Print(sampleResults()); err != nil {
		t.Fatalf("Print error: %v", err)
	}
	out := buf.String()
	for _, want := range []string{"MODEL", "DEVICE", "STATUS", "TOK/SEC", "llama", "qwen", "Success", "Timeout"} {
		if !strings.Contains(out, want) {
			t.Errorf("table output missing %q:\n%s", want, out)
		}
	}
}

func TestFastestModel(t *testing.T) {
	results := sampleResults()
	if got := fastestModel(results); got != "llama" {
		t.Errorf("fastestModel = %q, want llama", got)
	}
}

func TestSortByThroughput(t *testing.T) {
	results := []*engine.BenchmarkResult{
		{ModelName: "slow", Status: engine.StatusSuccess, TokensPerSec: 5},
		{ModelName: "err", Status: engine.StatusError},
		{ModelName: "fast", Status: engine.StatusSuccess, TokensPerSec: 50},
	}
	SortByThroughput(results)
	if results[0].ModelName != "fast" || results[1].ModelName != "slow" || results[2].ModelName != "err" {
		t.Errorf("unexpected order: %s, %s, %s", results[0].ModelName, results[1].ModelName, results[2].ModelName)
	}
}
