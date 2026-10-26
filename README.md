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

Build without Make:

```bash
go build -o bin/llm-bench .
go run . --help
```

## Quick start

```bash
# 1. Create a prompt file
mkdir -p prompts
printf 'Explain the difference between TCP and UDP in three sentences.\n' > prompts/prompt.txt

# 2. Make sure Ollama is running and the models are pulled
ollama serve &
ollama pull mistral:7b
ollama pull qwen2.5:7b

# 3. Run the benchmark
./bin/llm-bench --prompt-file prompts/prompt.txt --models mistral,qwen2.5
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

### Complete command examples

Run with the bundled [`config.yaml`](config.yaml) (models `mistral`, `qwen2.5`,
`remote-gpt` are defined there):

```bash
./bin/llm-bench \
  --config config.yaml \
  --prompt-file prompts/prompt.txt \
  --models mistral,qwen2.5 \
  --project mistral-vs-qwen \
  --format table
```

Run without a config file (models are treated as Ollama model IDs):

```bash
./bin/llm-bench \
  --config /nonexistent.yaml \
  --prompt-file prompts/prompt.txt \
  --models llama3.2:3b,phi3:mini \
  --ollama-url http://localhost:11434 \
  --format table
```

CPU-only run with explicit guardrails and no colors (good for CI logs):

```bash
./bin/llm-bench \
  --prompt-file prompts/prompt.txt \
  --models mistral,qwen2.5 \
  --device cpu \
  --cpu-threads 8 \
  --max-tokens 256 \
  --num-ctx 2048 \
  --timeout 90s \
  --concurrency 2 \
  --no-color
```

GPU run offloading all layers:

```bash
./bin/llm-bench \
  --prompt-file prompts/prompt.txt \
  --models mistral,qwen2.5 \
  --device gpu \
  --gpu-layers -1 \
  --concurrency 1
```

JSON output piped to a file (artifacts are still written to `logs/`):

```bash
./bin/llm-bench \
  --prompt-file prompts/prompt.txt \
  --models mistral,qwen2.5 \
  --format json \
  --no-color \
  --project json-run \
  --log-dir logs \
  > report.json
```

Markdown report:

```bash
./bin/llm-bench \
  --prompt-file prompts/prompt.txt \
  --models mistral,qwen2.5 \
  --format markdown \
  --no-color \
  > report.md
```

Remote OpenAI-compatible endpoint (key read from the env var named in
`api_key_env`):

```bash
export OPENAI_API_KEY=sk-...
./bin/llm-bench \
  --config config.yaml \
  --prompt-file prompts/prompt.txt \
  --models remote-gpt \
  --format table
```

Local `exec` engine (prompt is passed on stdin, stdout is the response). Add an
entry to `config.yaml` first:

```yaml
models:
  - name: local-cli
    engine: exec
    command: "my-llm-cli --model ./models/llama.gguf"
```

```bash
./bin/llm-bench \
  --config config.yaml \
  --prompt-file prompts/prompt.txt \
  --models local-cli \
  --timeout 300s
```

Scan a models directory for local weights/configs:

```bash
./bin/llm-bench \
  --prompt-file prompts/prompt.txt \
  --models mistral \
  --models-dir ./models
```

Show help and all flags:

```bash
./bin/llm-bench --help
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

The compose stack bundles its own Ollama service, so nothing has to be installed
on the host. `llm-bench` talks to `http://ollama:11434` over the compose network,
and Ollama stores downloaded weights in the local `./models/ollama` directory
(bind mount), so you can inspect, back up, or move them like ordinary files.

```bash
docker compose up -d ollama          # start Ollama
docker compose run --rm ollama-pull  # download mistral:7b + qwen2.5:7b
docker compose run --rm llm-bench    # run the benchmark
```

The compose file mounts `config.yaml`, `prompts/`, `models/`, and `logs/`, and
still maps `host.docker.internal` in case you prefer a host-installed Ollama.

### Complete Docker examples

Start the bundled Ollama and pull the models once:

```bash
docker compose up -d ollama
docker compose run --rm ollama-pull
```

Verify the models are available inside the stack:

```bash
docker compose exec ollama ollama list
```

Build the image and run the default compose command:

```bash
docker compose build
docker compose run --rm llm-bench
```

Pull additional models into `./models/ollama`:

```bash
docker compose run --rm ollama-pull ollama pull llama3.2:3b
```

Run against the bundled Ollama with custom flags:

```bash
docker compose run --rm llm-bench \
  --config /app/config.yaml \
  --prompt-file /app/prompts/prompt.txt \
  --models mistral,qwen2.5 \
  --ollama-url http://ollama:11434 \
  --project docker-run \
  --format json \
  --no-color
```

Use a host-installed Ollama instead of the bundled one (skip the `ollama`
service and point at the host gateway):

```bash
docker compose run --rm llm-bench \
  --config /app/config.yaml \
  --prompt-file /app/prompts/prompt.txt \
  --models mistral,qwen2.5 \
  --ollama-url http://host.docker.internal:11434 \
  --project host-ollama
```

Stop the stack (weights stay in `./models/ollama`):

```bash
docker compose down
```

Remove the downloaded weights:

```bash
rm -rf ./models/ollama
```

Inspect what Ollama stored on disk:

```bash
du -sh ./models/ollama
ls ./models/ollama/models/blobs | head
```

Override the compose command with your own flags (arguments after the service
name replace the default `command`):

```bash
docker compose run --rm llm-bench \
  --config /app/config.yaml \
  --prompt-file /app/prompts/prompt.txt \
  --models mistral,qwen2.5 \
  --project docker-run \
  --format json \
  --no-color
```

Build and run the image directly with `docker run`:

```bash
docker build -t llm-bench:latest .

docker run --rm \
  --add-host=host.docker.internal:host-gateway \
  -v "$PWD/config.yaml:/app/config.yaml:ro" \
  -v "$PWD/prompts:/app/prompts:ro" \
  -v "$PWD/logs:/app/logs" \
  llm-bench:latest \
  --config /app/config.yaml \
  --prompt-file /app/prompts/prompt.txt \
  --models mistral,qwen2.5 \
  --ollama-url http://host.docker.internal:11434 \
  --project docker-run
```

Show the container help:

```bash
docker run --rm llm-bench:latest --help
```

## Artifacts

With `--project` set, each run writes `logs/<project>_<timestamp>.json`
containing run metadata (device, limits, concurrency, timings) and per-model
results (tokens, TTFT, throughput, status, raw response).
