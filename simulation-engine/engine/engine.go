package engine

import (
	"fmt"

	"github.com/bryanC29/infra-play/simulation-engine/models"
	"github.com/bryanC29/infra-play/simulation-engine/utils"
)

// Run executes the full lifecycle of a system design simulation.
// It returns a Result object containing availability, latency, cost, and pass/fail status.
func Run(problem models.Problem, design models.Design) (models.Result, error) {
    // Step 1: Validate design structure
    if err := ValidateDesign(design); err != nil {
        return models.Result{}, fmt.Errorf("validation failed: %w", err)
    }

    // Step 2: Build graph representation
    graph, err := utils.BuildGraph(design)
    if err != nil {
        return models.Result{}, fmt.Errorf("graph build failed: %w", err)
    }

    // Step 3: Load simulation config
    config := LoadDefaultConfig()

    // Step 4: Simulate base traffic (normal load)
    baselineMetrics := SimulateTraffic(graph, problem.TargetQPS, config, false)

    // Step 5: Inject failures & simulate impact
    for _, failure := range problem.FailureScenarios {
        ApplyFailure(graph, failure)
        SimulateTraffic(graph, problem.TargetQPS, config, true)
        ResetFailures(graph) // Reset system state for next scenario
    }

    // Step 6: Apply surge pattern if defined
    for _, qps := range problem.SurgePattern {
        SimulateTraffic(graph, qps, config, false)
    }

    // Step 7: Collect metrics and score
    aggregate := CollectMetrics()
    score := ScoreResult(aggregate, config.SLA)

    return score, nil
}
