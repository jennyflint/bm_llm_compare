FROM golang:1.24-alpine AS builder

WORKDIR /src

RUN apk add --no-cache git ca-certificates

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/llm-bench .

FROM alpine:3.20

RUN apk add --no-cache ca-certificates tzdata && adduser -D -u 10001 bench

WORKDIR /app

COPY --from=builder /out/llm-bench /usr/local/bin/llm-bench

RUN mkdir -p /app/logs /app/prompts /app/models && chown -R bench:bench /app

USER bench

ENTRYPOINT ["llm-bench"]
CMD ["--help"]
