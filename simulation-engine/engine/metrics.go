package engine

import (
	"sync"
)

// Metrics holds aggregated data collected during simulations.
type Metrics struct {
	mu sync.Mutex

	TotalRequests       int
	SuccessfulRequests  int
	FailedRequests      int
	TotalLatencyMs      int64
	LatencySamples      []int
	SimulatedCostUSD    float64
	FailuresTolerated   int
	FailureScenariosRun int
}

// singleton instance (global for this simulation run)
var currentMetrics *Metrics

func init() {
	currentMetrics = &Metrics{
		LatencySamples: make([]int, 0, 1000),
	}
}

// Reset clears the metrics for a fresh simulation run.
func ResetMetrics() {
	currentMetrics.mu.Lock()
	defer currentMetrics.mu.Unlock()

	currentMetrics = &Metrics{
		LatencySamples: make([]int, 0, 1000),
	}
}

// RecordRequest adds a request result to the metrics.
func RecordRequest(success bool, latencyMs int) {
	currentMetrics.mu.Lock()
	defer currentMetrics.mu.Unlock()

	currentMetrics.TotalRequests++
	if success {
		currentMetrics.SuccessfulRequests++
		currentMetrics.TotalLatencyMs += int64(latencyMs)
		currentMetrics.LatencySamples = append(currentMetrics.LatencySamples, latencyMs)
	} else {
		currentMetrics.FailedRequests++
	}
}

// RecordFailureScenario increments the number of simulated failures and tolerances.
func RecordFailureScenario(tolerated bool) {
	currentMetrics.mu.Lock()
	defer currentMetrics.mu.Unlock()

	currentMetrics.FailureScenariosRun++
	if tolerated {
		currentMetrics.FailuresTolerated++
	}
}

// AddCost accumulates cost over simulation time.
func AddCost(costUSD float64) {
	currentMetrics.mu.Lock()
	defer currentMetrics.mu.Unlock()

	currentMetrics.SimulatedCostUSD += costUSD
}

// CollectMetrics returns a snapshot of all collected metrics.
func CollectMetrics() Metrics {
	currentMetrics.mu.Lock()
	defer currentMetrics.mu.Unlock()

	return *currentMetrics
}

// AverageLatency returns the mean latency in milliseconds.
func AverageLatency() float64 {
	currentMetrics.mu.Lock()
	defer currentMetrics.mu.Unlock()

	if currentMetrics.SuccessfulRequests == 0 {
		return 0
	}
	return float64(currentMetrics.TotalLatencyMs) / float64(currentMetrics.SuccessfulRequests)
}

// Availability returns the availability ratio of successful to total requests.
func Availability() float64 {
	currentMetrics.mu.Lock()
	defer currentMetrics.mu.Unlock()

	if currentMetrics.TotalRequests == 0 {
		return 0
	}
	return float64(currentMetrics.SuccessfulRequests) / float64(currentMetrics.TotalRequests)
}
