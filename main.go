// Command llm-bench benchmarks a prompt across multiple LLM models.
package main

import (
	"fmt"
	"os"

	"github.com/llm-bench/cmd"
)

func main() {
	if err := cmd.NewRootCommand().Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}
