// Package printer renders benchmark results as colored tables, JSON, or
// Markdown.
package printer

import (
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/fatih/color"
	"github.com/llm-bench/internal/config"
	"github.com/llm-bench/internal/engine"
)

// Printer writes formatted reports to an output stream.
type Printer struct {
	Out     io.Writer
	Format  config.OutputFormat
	NoColor bool
}

// New constructs a Printer.
func New(out io.Writer, format config.OutputFormat, noColor bool) *Printer {
	if noColor {
		color.NoColor = true
	}
	return &Printer{Out: out, Format: format, NoColor: noColor}
}

// Print dispatches to the configured serializer.
func (p *Printer) Print(results []*engine.BenchmarkResult) error {
	switch p.Format {
	case config.FormatJSON:
		return p.printJSON(results)
	case config.FormatMarkdown:
		return p.printMarkdown(results)
	default:
		return p.printTable(results)
	}
}

type jsonRow struct {
	ModelName       string  `json:"model_name"`
	Device          string  `json:"device"`
	Status          string  `json:"status"`
	PromptTokens    int     `json:"prompt_tokens"`
	GeneratedTokens int     `json:"generated_tokens"`
	TTFTMS          int64   `json:"ttft_ms"`
	DurationMS      int64   `json:"duration_ms"`
	TokensPerSec    float64 `json:"tokens_per_sec"`
	Preview         string  `json:"preview"`
	Error           string  `json:"error,omitempty"`
}

func (p *Printer) printJSON(results []*engine.BenchmarkResult) error {
	rows := make([]jsonRow, 0, len(results))
	for _, r := range results {
		if r == nil {
			continue
		}
		row := jsonRow{
			ModelName:       r.ModelName,
			Device:          string(r.Device),
			Status:          r.Status,
			PromptTokens:    r.PromptTokens,
			GeneratedTokens: r.GeneratedTokens,
			TTFTMS:          r.TTFT.Milliseconds(),
			DurationMS:      r.Duration.Milliseconds(),
			TokensPerSec:    round2(r.TokensPerSec),
			Preview:         Preview(r.ResponseText, 80),
		}
		if r.Error != nil {
			row.Error = r.Error.Error()
		}
		rows = append(rows, row)
	}
	enc := json.NewEncoder(p.Out)
	enc.SetIndent("", "  ")
	if err := enc.Encode(rows); err != nil {
		return fmt.Errorf("encode json report: %w", err)
	}
	return nil
}

func (p *Printer) printMarkdown(results []*engine.BenchmarkResult) error {
	var b strings.Builder
	b.WriteString("| Model | Device | Status | TTFT | Duration | Tokens/sec | Preview |\n")
	b.WriteString("| --- | --- | --- | --- | --- | --- | --- |\n")
	for _, r := range results {
		if r == nil {
			continue
		}
		fmt.Fprintf(&b, "| %s | %s | %s | %s | %s | %.2f | %s |\n",
			escapeMD(r.ModelName),
			r.Device,
			r.Status,
			formatDuration(r.TTFT),
			formatDuration(r.Duration),
			r.TokensPerSec,
			escapeMD(Preview(r.ResponseText, 60)),
		)
	}
	if _, err := io.WriteString(p.Out, b.String()); err != nil {
		return fmt.Errorf("write markdown report: %w", err)
	}
	return nil
}

func (p *Printer) printTable(results []*engine.BenchmarkResult) error {
	header := color.New(color.FgCyan, color.Bold)
	fastest := fastestModel(results)

	tw := tabwriter.NewWriter(p.Out, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, header.Sprint("MODEL\tDEVICE\tSTATUS\tTTFT\tDURATION\tTOK/SEC\tPREVIEW"))

	for _, r := range results {
		if r == nil {
			continue
		}
		status := p.colorStatus(r.Status)
		model := color.New(color.FgCyan, color.Bold).Sprint(r.ModelName)
		tps := fmt.Sprintf("%.2f", r.TokensPerSec)
		if r.ModelName == fastest && r.Status == engine.StatusSuccess {
			tps = color.New(color.FgGreen, color.Bold).Sprint(tps)
		} else if r.Status == engine.StatusSuccess && r.TokensPerSec > 0 && r.TokensPerSec < 5 {
			tps = color.New(color.FgYellow).Sprint(tps)
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
			model,
			r.Device,
			status,
			formatDuration(r.TTFT),
			formatDuration(r.Duration),
			tps,
			Preview(r.ResponseText, 40),
		)
	}
	if err := tw.Flush(); err != nil {
		return fmt.Errorf("flush table: %w", err)
	}
	return nil
}

func (p *Printer) colorStatus(status string) string {
	switch status {
	case engine.StatusSuccess:
		return color.New(color.FgGreen).Sprint(status)
	case engine.StatusTimeout:
		return color.New(color.FgRed, color.Bold).Sprint(status)
	case engine.StatusOOM:
		return color.New(color.FgRed, color.Bold).Sprint(status)
	default:
		return color.New(color.FgRed).Sprint(status)
	}
}

// fastestModel returns the model with the highest throughput among successes.
func fastestModel(results []*engine.BenchmarkResult) string {
	best := ""
	bestTPS := -1.0
	for _, r := range results {
		if r == nil || r.Status != engine.StatusSuccess {
			continue
		}
		if r.TokensPerSec > bestTPS {
			bestTPS = r.TokensPerSec
			best = r.ModelName
		}
	}
	return best
}

// Preview truncates text to max runes, collapsing whitespace.
func Preview(text string, max int) string {
	clean := strings.Join(strings.Fields(text), " ")
	runes := []rune(clean)
	if len(runes) <= max {
		return clean
	}
	if max <= 3 {
		return string(runes[:max])
	}
	return string(runes[:max-3]) + "..."
}

func formatDuration(d time.Duration) string {
	if d <= 0 {
		return "-"
	}
	return d.Round(time.Millisecond).String()
}

func round2(v float64) float64 {
	return float64(int64(v*100+0.5)) / 100
}

func escapeMD(s string) string {
	return strings.ReplaceAll(s, "|", "\\|")
}

// SortByThroughput orders results descending by tokens/sec, successes first.
func SortByThroughput(results []*engine.BenchmarkResult) {
	sort.SliceStable(results, func(i, j int) bool {
		si := results[i].Status == engine.StatusSuccess
		sj := results[j].Status == engine.StatusSuccess
		if si != sj {
			return si
		}
		return results[i].TokensPerSec > results[j].TokensPerSec
	})
}
