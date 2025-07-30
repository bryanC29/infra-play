package engine

import (
	"github.com/bryanC29/infra-play/simulation-engine/models"
)

// ScoreResult evaluates simulation metrics against SLA and returns a Result.
func ScoreResult(metrics Metrics, sla SLAConfig) models.Result {
	availability := Availability()
	latency := AverageLatency()
	cost := metrics.SimulatedCostUSD

	// Determine fault tolerance as a ratio of tolerated vs total simulated failures
	var faultToleranceScore float64
	if metrics.FailureScenariosRun > 0 {
		faultToleranceScore = float64(metrics.FailuresTolerated) / float64(metrics.FailureScenariosRun)
	} else {
		faultToleranceScore = 1.0 // no failures injected, assume perfect
	}

	// Define SLA pass/fail conditions
	pass := availability >= sla.MinAvailabilityRate &&
		latency <= float64(sla.MaxAvgLatencyMs)

	return models.Result{
		Availability:        availability,
		AvgLatencyMs:        latency,
		CostEstimateUSD:     cost,
		FaultToleranceScore: faultToleranceScore,
		Pass:                pass,
	}
}
