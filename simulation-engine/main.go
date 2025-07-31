package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"simengine/api"
	"syscall"
	"time"
)

func main() {

	port := ":3030"
	server := &http.Server{
		Addr: port,
		Handler: api.NewRouter(),
		ReadTimeout: 15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout: 60 * time.Second,
	}

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

	go func() {
		log.Printf("Engine running at %s\n", port)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Engine failed to start %v", err)
		}
	}()

	<-stop

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 15 * time.Second)
	defer cancel()

	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Fatalf("Enging stopped %v", err)
	}

	log.Printf("Shutdown complete")
}
