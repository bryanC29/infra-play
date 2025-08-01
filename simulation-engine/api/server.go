package api

import (
	"net/http"

	"github.com/go-chi/chi"
)

func NewRouter() http.Handler {
	r := chi.NewRouter()

	r.Get("/", func(w http.ResponseWriter, r *http.Request) {
		RespondWithJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})

	r.Post("/simulate", handleSimulation)

	return r
}