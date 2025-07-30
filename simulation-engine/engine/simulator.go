package engine

import (
	"math/rand"
	"time"

	"github.com/bryanC29/infra-play/simulation-engine/models"
	"github.com/bryanC29/infra-play/simulation-engine/utils"
)

// SimulateTraffic routes requests through the graph under a given QPS.
// It updates internal metrics with availability, latency, and cost.
func SimulateTraffic(graph *models.Graph, qps int, config SystemConfig, duringFailure bool) Metrics {
	ResetMetrics()

	// Total simulated duration
	simStart := time.Now()

	// Distribute requests into batches per second
	for i := 0; i < int(SimulationTimeWindow.Seconds()); i++ {
		simulateSecond(graph, qps, config)
	}

	// Estimate total cost after sim
	estimateTotalCost(graph, config)

	// If running failure scenario, evaluate fault tolerance
	if duringFailure {
		tolerated := Availability() >= config.SLA.MinAvailabilityRate
		RecordFailureScenario(tolerated)
	}

	// Return snapshot of current metrics
	return CollectMetrics()
}

// simulateSecond simulates one second of QPS worth of traffic.
func simulateSecond(graph *models.Graph, qps int, config SystemConfig) {
	for i := 0; i < qps; i++ {
		success, latency := routeRequest(graph, config)
		RecordRequest(success, latency)
	}
}

// routeRequest attempts to find a valid request path from source to sink.
// It simulates latency and success/failure at each step.
func routeRequest(graph *models.Graph, config SystemConfig) (bool, int) {
	path, err := utils.FindPath(graph)
	if err != nil || len(path) == 0 {
		return false, 0
	}

	totalLatency := 0

	for _, node := range path {
		// Skip failed node or edge
		if node.Failed {
			return false, totalLatency
		}

		profile, ok := config.NodeDefaults[node.Type]
		if !ok {
			return false, totalLatency
		}

		// Estimate latency per node type with random jitter
		nodeLatency := simulateLatency(profile)
		totalLatency += nodeLatency
	}

	// Simulate probabilistic failure (e.g. 1% transient fail)
	if rand.Float64() < 0.01 {
		return false, totalLatency
	}

	return true, totalLatency
}

// simulateLatency generates a randomized latency for a node.
func simulateLatency(profile NodeProfile) int {
	base := 5 + rand.Intn(10) // Base latency 5–15 ms
	// Add load-dependent latency multiplier if needed
	return base
}

// estimateTotalCost computes the overall infrastructure cost of the design.
func estimateTotalCost(graph *models.Graph, config SystemConfig) {
	durationHours := SimulationTimeWindow.Hours()

	for _, node := range graph.Nodes {
		if node.Failed {
			continue
		}

		profile, ok := config.NodeDefaults[node.Type]
		if !ok {
			continue
		}

		nodeCost := profile.CostPerHour * float64(node.Config.Replicas) * durationHours
		AddCost(nodeCost)
	}
}
