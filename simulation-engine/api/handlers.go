package api

import (
	"encoding/json"
	"net/http"

	"github.com/bryanC29/infra-play/simulation-engine/engine"
	"github.com/bryanC29/infra-play/simulation-engine/models"
	"github.com/bryanC29/infra-play/simulation-engine/utils"
)

// SimulationRequest defines the expected structure of the incoming POST request.
type SimulationRequest struct {
    Problem models.Problem `json:"problem"`
    Design  models.Design  `json:"design"`
}

// SimulationResponse defines the structure of the returned result.
type SimulationResponse struct {
    Result models.Result `json:"result"`
}

// handleSimulation handles POST /simulate
func handleSimulation(w http.ResponseWriter, r *http.Request) {
    var req SimulationRequest

    // Decode JSON payload
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        utils.RespondWithError(w, http.StatusBadRequest, "Invalid JSON payload")
        return
    }

    // Basic validation
    if req.Problem.ID == "" || len(req.Design.Nodes) == 0 {
        utils.RespondWithError(w, http.StatusBadRequest, "Missing required fields in problem or design")
        return
    }

    // Run the simulation engine
    result, err := engine.Run(req.Problem, req.Design)
    if err != nil {
        utils.RespondWithError(w, http.StatusInternalServerError, err.Error())
        return
    }

    // Return result
    response := SimulationResponse{Result: result}
    utils.RespondWithJSON(w, http.StatusOK, response)
}
