package main

import (
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/bryanC29/infra-play/simulation-engine/api"
)

func main() {
    // Configuration (could be made dynamic via env vars or flags)
    addr := ":3080"
    server := &http.Server{
        Addr:         addr,
        Handler:      api.NewRouter(),
        ReadTimeout:  15 * time.Second,
        WriteTimeout: 15 * time.Second,
        IdleTimeout:  60 * time.Second,
    }

    // Channel to catch shutdown signals
    stop := make(chan os.Signal, 1)
    signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

    go func() {
        log.Printf("Simulator Engine is running at %s\n", addr)
        if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
            log.Fatalf("Failed to start server: %v", err)
        }
    }()

    <-stop // Wait for shutdown signal

    log.Println("Shutting down engine")
    shutdownCtx, cancel := time.WithTimeout(time.Background(), 10*time.Second)
    defer cancel()

    if err := server.Shutdown(shutdownCtx); err != nil {
        log.Fatalf("Shutdown failed: %v", err)
    }

    log.Println("Engine stopped cleanly")
}
