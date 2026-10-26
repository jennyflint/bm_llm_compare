package runner

import (
	"context"
	"os"
	"os/signal"
	"syscall"
)

// WithSignalContext returns a context cancelled on os.Interrupt or SIGTERM.
func WithSignalContext(parent context.Context) (context.Context, context.CancelFunc) {
	return signal.NotifyContext(parent, os.Interrupt, syscall.SIGTERM)
}
