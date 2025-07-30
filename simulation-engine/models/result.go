package models

// Result is the output returned after simulating a user-submitted design.
type Result struct {
	Passed             bool              `json:"passed"`             // Whether design met SLA and constraints
	TotalScore         float64           `json:"totalScore"`         // Composite score (0-100)
	Metrics            PerformanceMetrics `json:"metrics"`           // Latency, availability, cost
	Errors             []string          `json:"errors,omitempty"`   // Validation or simulation errors
	FailureTolerance   FailureTolerance   `json:"failureTolerance"`  // Fault tolerance results
	Breakdown          ScoreBreakdown     `json:"breakdown"`         // Sub-scores per criterion
}

// PerformanceMetrics captures system-level performance during simulation.
type PerformanceMetrics struct {
	AvgLatencyMs    float64 `json:"avgLatencyMs"`
	Availability    float64 `json:"availability"`     // e.g., 0.998
	TotalCostUSD    float64 `json:"totalCostUsd"`     // e.g., $812.50
	TotalRequests   int     `json:"totalRequests"`    // e.g., 1M
	FailedRequests  int     `json:"failedRequests"`   // e.g., 6,000
}

// FailureTolerance reports how the system behaved under stress/failure tests.
type FailureTolerance struct {
	TestedNodes    []string `json:"testedNodes"`     // Node IDs tested for failure
	SurvivedAll    bool     `json:"survivedAll"`     // true if SLA held in all cases
	FailuresPassed int      `json:"failuresPassed"`  // # of successful failover scenarios
	FailuresTotal  int      `json:"failuresTotal"`   // Total failure cases tested
}

// ScoreBreakdown provides transparency on how the total score was computed.
type ScoreBreakdown struct {
	LatencyScore       float64 `json:"latencyScore"`       // 0–100
	AvailabilityScore  float64 `json:"availabilityScore"`  // 0–100
	CostEfficiencyScore float64 `json:"costEfficiencyScore"` // 0–100
	FaultToleranceScore float64 `json:"faultToleranceScore"` // 0–100
}
