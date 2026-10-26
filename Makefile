BINARY      := llm-bench
WINDOWS_BIN := llm-bench.exe
PKG         := ./...
LDFLAGS     := -s -w

.PHONY: all build build-linux build-windows test test-race vet fmt tidy clean docker docker-run

all: build

build:
	CGO_ENABLED=0 go build -trimpath -ldflags="$(LDFLAGS)" -o bin/$(BINARY) .

build-linux:
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags="$(LDFLAGS)" -o bin/$(BINARY)-linux-amd64 .

build-windows:
	CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -trimpath -ldflags="$(LDFLAGS)" -o bin/$(WINDOWS_BIN) .

test:
	go test -v $(PKG)

test-race:
	go test -v -race $(PKG)

vet:
	go vet $(PKG)

fmt:
	gofmt -l -w .

tidy:
	go mod tidy

docker:
	docker build -t llm-bench:latest .

docker-run:
	docker compose run --rm llm-bench

clean:
	rm -rf bin
