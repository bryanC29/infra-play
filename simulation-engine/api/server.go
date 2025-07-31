package api

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/go-chi/chi"
)

func NewRouter() http.Handler {
	r := chi.NewRouter()

	r.Get("/test", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		payload := map[string]string{"status": "ok"}

		if err := json.NewEncoder(w).Encode(payload); err != nil {
			http.Error(w, "Failed", http.StatusInternalServerError)
		}
	})
	fmt.Printf("Hello")

	return r
}