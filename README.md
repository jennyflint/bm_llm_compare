# llm-bench

A modular, production-ready CLI tool that runs a single prompt across multiple
local or remote LLM models concurrently, measures detailed performance
benchmarks, enforces hardware/resource guardrails, and exports colored reports
plus project-scoped JSON artifacts.

## Features

- Concurrent execution across models with a bounded worker pool (`--concurrency`).
- Compute backend targeting: `cpu` (forces `num_gpu: 0` + `num_thread`) or `gpu`
  (layer offload via `num_gpu`, `-1` = all layers).
- Resource guardrails: `num_ctx`, `max_tokens`, per-model timeout, and fail-safe
  handling of timeouts / OOM / errors without aborting the batch.
- Runners: Ollama HTTP API, OpenAI-compatible endpoints, and local `exec`.
- Output formats: colored `table`, `json`, `markdown`; `--no-color` for clean logs.
- Project artifact logging to `logs/<project>_<timestamp>.json`.
- Graceful shutdown on `SIGINT` / `SIGTERM`.
- Docker multi-stage build, docker-compose with host Ollama access, and
  cross-compilation for Linux and Windows.

## Project layout

```
cmd/                 CLI wiring (cobra flags, orchestration)
internal/config      Config model, YAML loading, flag merging, resource limits
internal/engine      Runner interface + Ollama / OpenAI / exec / mock runners
internal/runner      Concurrent orchestrator, signal handling
internal/logger      Project artifact persistence
internal/printer     Table / JSON / Markdown serializers
main.go              Entrypoint
```

## Build

```bash
make build           # bin/llm-bench
make build-linux     # static linux/amd64 binary
make build-windows   # bin/llm-bench.exe (GOOS=windows GOARCH=amd64)
make test-race       # go test -v -race ./...
```

## Usage

```bash
llm-bench \
  --prompt-file prompts/prompt.txt \
  --models mistral,qwen2.5 \
  --device cpu \
  --cpu-threads 8 \
  --concurrency 2 \
  --project mistral-vs-qwen \
  --format table
```

### Flags

| Flag | Short | Description |
| --- | --- | --- |
| `--project` | `-P` | Project workspace name for artifact logging |
| `--prompt-file` | `-p` | Path to the prompt file (required) |
| `--models` | `-m` | Comma-separated model list |
| `--models-dir` | `-d` | Directory to scan for model weights/configs |
| `--config` | `-c` | Config file path (default `config.yaml`) |
| `--concurrency` | | Max parallel runs (default: CPU cores) |
| `--format` | | `table`, `json`, `markdown` |
| `--no-color` | | Disable ANSI colors |
| `--device` | | `cpu` or `gpu` |
| `--cpu-threads` | | CPU thread count for local engines |
| `--gpu-layers` | | GPU layers to offload (`-1` = all) |
| `--max-tokens` | | Max generated tokens per run |
| `--num-ctx` | | Context window per run |
| `--timeout` | | Per-model execution timeout |
| `--log-dir` | | Artifact directory (default `logs`) |
| `--ollama-url` | | Ollama base URL |

CLI flags take precedence over `config.yaml` values, including per-model entries.

## Configuration

See [`config.yaml`](config.yaml) for a complete example. Precedence is:
CLI flags → per-model config → global defaults → built-in defaults.

## Docker

```bash
docker compose run --rm llm-bench
```

The compose file mounts `config.yaml`, `prompts/`, `models/`, and `logs/`, and
maps `host.docker.internal` so the container can reach Ollama on the host.

## Artifacts

With `--project` set, each run writes `logs/<project>_<timestamp>.json`
containing run metadata (device, limits, concurrency, timings) and per-model
results (tokens, TTFT, throughput, status, raw response).
