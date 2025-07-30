package api

import (
	"net/http"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
)

// NewRouter initializes and returns a configured HTTP router.
func NewRouter() http.Handler {
    r := chi.NewRouter()

    // === Global Middleware ===
    r.Use(middleware.RequestID)       // Adds request ID to context
    r.Use(middleware.RealIP)          // Gets the real IP from proxy headers
    r.Use(middleware.Logger)          // Logs all requests
    r.Use(middleware.Recoverer)       // Catches panics and returns 500
    r.Use(middleware.Timeout(10 * 1e9)) // Optional: 10s timeout per request

    // === Routes ===
    r.Post("/simulate", handleSimulation)

    // Health check
    r.Get("/healthz", func(w http.ResponseWriter, r *http.Request) {
        RespondWithJSON(w, http.StatusOK, map[string]string{"status": "ok"})
    })

    return r
}
