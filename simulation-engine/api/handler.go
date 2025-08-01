package api

import (
	"encoding/json"
	"net/http"
	"simengine/engine"
	"simengine/models"
)

type SimulationRequest struct {
	Problem models.Problem `json:"problem"`
	Design models.Design `json:"design"`
}

type SimulationResponse struct {
	Result models.Result `json:"result"`
}

func handleSimulation(w http.ResponseWriter, r *http.Request) {
	var req SimulationRequest

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		RespondWithError(w, http.StatusBadRequest, "Invalid JSON")
		return
	}

	if req.Problem.Id == "" || len(req.Design.Nodes) == 0 || len(req.Design.Edges) == 0 {
		RespondWithError(w, http.StatusBadRequest, "Missing required fields")
		return
	}

	result, err := engine.Run(req.Problem, req.Design)
	if err != nil {
		RespondWithError(w, http.StatusInternalServerError, err.Error())
		return
	}

	response := SimulationResponse{Result: result}

	RespondWithJSON(w, http.StatusOK, response)
}
