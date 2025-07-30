package api

import (
	"encoding/json"
	"net/http"
)

// errorResponse defines the structure of an error payload.
type errorResponse struct {
    Error string `json:"error"`
}

// RespondWithJSON sends a JSON response with a specified status code.
func RespondWithJSON(w http.ResponseWriter, statusCode int, payload interface{}) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(statusCode)

    if payload != nil {
        if err := json.NewEncoder(w).Encode(payload); err != nil {
            http.Error(w, "Failed to encode response", http.StatusInternalServerError)
        }
    }
}

// RespondWithError sends a JSON error response with a message and HTTP status code.
func RespondWithError(w http.ResponseWriter, statusCode int, message string) {
    RespondWithJSON(w, statusCode, errorResponse{Error: message})
}
