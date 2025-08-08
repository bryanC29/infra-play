package models

import (
	"fmt"
	"math"
	"time"
)

// SimulationResult represents the final output structure returned by the simulation engine
type SimulationResult struct {
	// Core performance metrics
	Availability    float64 `json:"availability"`     // % of total requests successfully reaching exit node
	AvgLatencyMS    float64 `json:"avg_latency_ms"`   // Average end-to-end latency across all successful requests
	ThroughputQPS   float64 `json:"throughput_qps"`   // Max sustained QPS before saturation causes drops
	FaultTolerance  float64 `json:"fault_tolerance"`  // 0-1 score indicating resilience under partial failure
	QPSThreshold    float64 `json:"qps_threshold"`    // Maximum QPS sustainable before instability

	// Request tracking metrics
	TotalRequests      int64 `json:"total_requests"`      // Number of requests simulated across all scenarios
	SuccessfulRequests int64 `json:"successful_requests"` // Count of requests that completed successfully
	FailedRequests     int64 `json:"failed_requests"`     // Dropped or failed due to overload or node failure

	// System topology metrics
	NumNodes       int `json:"num_nodes"`        // Number of functional nodes in design (excluding entry/exit)
	NumFailedNodes int `json:"num_failed_nodes"` // Nodes that were forcibly failed during failure stages

	// Stage-specific QPS metrics
	QPSUnder1x        float64 `json:"qps_under_1x"`         // Successful QPS at 1x load
	QPSUnder15x       float64 `json:"qps_under_1_5x"`       // Successful QPS at 1.5x load
	QPSUnder2x        float64 `json:"qps_under_2x"`         // Successful QPS at 2x load
	QPSUnderFailure1x float64 `json:"qps_under_failure_1x"` // QPS under normal load with node failures
	QPSUnderFailure15x float64 `json:"qps_under_failure_1_5x"` // QPS under 1.5x surge + node failures

	// Derived metrics
	RequestDropRate float64 `json:"request_drop_rate"` // failed_requests / total_requests

	// Evaluation results
	Score float64 `json:"score"` // Weighted score [0-100] based on requirements
	Pass  bool    `json:"pass"`  // Boolean: meets latency + availability thresholds

	// Metadata (not included in JSON output)
	Timestamp     time.Time `json:"-"`
	ExecutionTime time.Duration `json:"-"`
	Version       string `json:"-"`
}

// ResultBuilder helps construct SimulationResult with validation
type ResultBuilder struct {
	result *SimulationResult
	errors []error
}

// ResultValidationError represents validation errors in result construction
type ResultValidationError struct {
	Field   string
	Value   interface{}
	Message string
}

func (e *ResultValidationError) Error() string {
	return fmt.Sprintf("validation error for field '%s' (value: %v): %s", e.Field, e.Value, e.Message)
}

// PerformanceCategory categorizes system performance levels
type PerformanceCategory string

const (
	PerformanceExcellent PerformanceCategory = "excellent" // Score >= 90
	PerformanceGood      PerformanceCategory = "good"      // Score >= 70
	PerformanceFair      PerformanceCategory = "fair"      // Score >= 50
	PerformancePoor      PerformanceCategory = "poor"      // Score < 50
)

// QualityMetric represents individual quality measurements
type QualityMetric struct {
	Name        string  `json:"name"`
	Value       float64 `json:"value"`
	Target      float64 `json:"target"`
	Weight      float64 `json:"weight"`
	Score       float64 `json:"score"`       // 0-100
	Importance  string  `json:"importance"`  // "critical", "high", "medium", "low"
	Status      string  `json:"status"`      // "pass", "fail", "warning"
	Description string  `json:"description"`
}

// NewSimulationResult creates a new SimulationResult with default values
func NewSimulationResult() *SimulationResult {
	return &SimulationResult{
		Availability:       0.0,
		AvgLatencyMS:       0.0,
		ThroughputQPS:      0.0,
		FaultTolerance:     0.0,
		QPSThreshold:       0.0,
		TotalRequests:      0,
		SuccessfulRequests: 0,
		FailedRequests:     0,
		NumNodes:           0,
		NumFailedNodes:     0,
		QPSUnder1x:         0.0,
		QPSUnder15x:        0.0,
		QPSUnder2x:         0.0,
		QPSUnderFailure1x:  0.0,
		QPSUnderFailure15x: 0.0,
		RequestDropRate:    0.0,
		Score:              0.0,
		Pass:               false,
		Timestamp:          time.Now(),
		Version:            "1.0.0",
	}
}

// NewResultBuilder creates a new result builder
func NewResultBuilder() *ResultBuilder {
	return &ResultBuilder{
		result: NewSimulationResult(),
		errors: make([]error, 0),
	}
}

// SetAvailability sets the availability metric with validation
func (rb *ResultBuilder) SetAvailability(availability float64) *ResultBuilder {
	if availability < 0.0 || availability > 1.0 {
		rb.errors = append(rb.errors, &ResultValidationError{
			Field:   "availability",
			Value:   availability,
			Message: "must be between 0.0 and 1.0",
		})
	}
	rb.result.Availability = availability
	return rb
}

// SetLatency sets the average latency metric with validation
func (rb *ResultBuilder) SetLatency(latencyMS float64) *ResultBuilder {
	if latencyMS < 0.0 {
		rb.errors = append(rb.errors, &ResultValidationError{
			Field:   "avg_latency_ms",
			Value:   latencyMS,
			Message: "must be non-negative",
		})
	}
	if latencyMS > 60000.0 { // 1 minute seems unreasonable
		rb.errors = append(rb.errors, &ResultValidationError{
			Field:   "avg_latency_ms",
			Value:   latencyMS,
			Message: "exceeds reasonable maximum (60000ms)",
		})
	}
	rb.result.AvgLatencyMS = latencyMS
	return rb
}

// SetThroughput sets the throughput metric with validation
func (rb *ResultBuilder) SetThroughput(throughputQPS float64) *ResultBuilder {
	if throughputQPS < 0.0 {
		rb.errors = append(rb.errors, &ResultValidationError{
			Field:   "throughput_qps",
			Value:   throughputQPS,
			Message: "must be non-negative",
		})
	}
	rb.result.ThroughputQPS = throughputQPS
	return rb
}

// SetFaultTolerance sets the fault tolerance metric with validation
func (rb *ResultBuilder) SetFaultTolerance(faultTolerance float64) *ResultBuilder {
	if faultTolerance < 0.0 || faultTolerance > 1.0 {
		rb.errors = append(rb.errors, &ResultValidationError{
			Field:   "fault_tolerance",
			Value:   faultTolerance,
			Message: "must be between 0.0 and 1.0",
		})
	}
	rb.result.FaultTolerance = faultTolerance
	return rb
}

// SetRequestMetrics sets request counting metrics with validation
func (rb *ResultBuilder) SetRequestMetrics(total, successful, failed int64) *ResultBuilder {
	if total < 0 {
		rb.errors = append(rb.errors, &ResultValidationError{
			Field:   "total_requests",
			Value:   total,
			Message: "must be non-negative",
		})
	}
	if successful < 0 {
		rb.errors = append(rb.errors, &ResultValidationError{
			Field:   "successful_requests",
			Value:   successful,
			Message: "must be non-negative",
		})
	}
	if failed < 0 {
		rb.errors = append(rb.errors, &ResultValidationError{
			Field:   "failed_requests",
			Value:   failed,
			Message: "must be non-negative",
		})
	}
	if successful+failed != total {
		rb.errors = append(rb.errors, &ResultValidationError{
			Field:   "request_metrics",
			Value:   fmt.Sprintf("total=%d, successful=%d, failed=%d", total, successful, failed),
			Message: "successful + failed must equal total requests",
		})
	}

	rb.result.TotalRequests = total
	rb.result.SuccessfulRequests = successful
	rb.result.FailedRequests = failed

	// Calculate drop rate
	if total > 0 {
		rb.result.RequestDropRate = float64(failed) / float64(total)
	}

	return rb
}

// SetNodeMetrics sets node counting metrics with validation
func (rb *ResultBuilder) SetNodeMetrics(totalNodes, failedNodes int) *ResultBuilder {
	if totalNodes < 0 {
		rb.errors = append(rb.errors, &ResultValidationError{
			Field:   "num_nodes",
			Value:   totalNodes,
			Message: "must be non-negative",
		})
	}
	if failedNodes < 0 {
		rb.errors = append(rb.errors, &ResultValidationError{
			Field:   "num_failed_nodes",
			Value:   failedNodes,
			Message: "must be non-negative",
		})
	}
	if failedNodes > totalNodes {
		rb.errors = append(rb.errors, &ResultValidationError{
			Field:   "node_metrics",
			Value:   fmt.Sprintf("total=%d, failed=%d", totalNodes, failedNodes),
			Message: "failed nodes cannot exceed total nodes",
		})
	}

	rb.result.NumNodes = totalNodes
	rb.result.NumFailedNodes = failedNodes
	return rb
}

// SetQPSMetrics sets all QPS stage metrics with validation
func (rb *ResultBuilder) SetQPSMetrics(qps1x, qps15x, qps2x, qpsFailure1x, qpsFailure15x, qpsThreshold float64) *ResultBuilder {
	qpsValues := map[string]float64{
		"qps_under_1x":          qps1x,
		"qps_under_1_5x":        qps15x,
		"qps_under_2x":          qps2x,
		"qps_under_failure_1x":  qpsFailure1x,
		"qps_under_failure_1_5x": qpsFailure15x,
		"qps_threshold":         qpsThreshold,
	}

	for field, value := range qpsValues {
		if value < 0.0 {
			rb.errors = append(rb.errors, &ResultValidationError{
				Field:   field,
				Value:   value,
				Message: "must be non-negative",
			})
		}
	}

	// Logical validations
	if qps15x > qps1x*2.0 {
		rb.errors = append(rb.errors, &ResultValidationError{
			Field:   "qps_progression",
			Value:   fmt.Sprintf("1x=%.1f, 1.5x=%.1f", qps1x, qps15x),
			Message: "QPS at 1.5x load should not be more than double the 1x QPS",
		})
	}

	if qpsFailure1x > qps1x*1.1 {
		rb.errors = append(rb.errors, &ResultValidationError{
			Field:   "failure_qps",
			Value:   fmt.Sprintf("normal=%.1f, failure=%.1f", qps1x, qpsFailure1x),
			Message: "QPS under failure should not exceed normal QPS significantly",
		})
	}

	rb.result.QPSUnder1x = qps1x
	rb.result.QPSUnder15x = qps15x
	rb.result.QPSUnder2x = qps2x
	rb.result.QPSUnderFailure1x = qpsFailure1x
	rb.result.QPSUnderFailure15x = qpsFailure15x
	rb.result.QPSThreshold = qpsThreshold

	return rb
}

// SetScore sets the final score and pass/fail status
func (rb *ResultBuilder) SetScore(score float64, pass bool) *ResultBuilder {
	if score < 0.0 || score > 100.0 {
		rb.errors = append(rb.errors, &ResultValidationError{
			Field:   "score",
			Value:   score,
			Message: "must be between 0.0 and 100.0",
		})
	}

	rb.result.Score = score
	rb.result.Pass = pass
	return rb
}

// SetMetadata sets optional metadata fields
func (rb *ResultBuilder) SetMetadata(timestamp time.Time, executionTime time.Duration, version string) *ResultBuilder {
	rb.result.Timestamp = timestamp
	rb.result.ExecutionTime = executionTime
	if version != "" {
		rb.result.Version = version
	}
	return rb
}

// Build constructs the final result with validation
func (rb *ResultBuilder) Build() (*SimulationResult, error) {
	if len(rb.errors) > 0 {
		return nil, fmt.Errorf("result validation failed with %d errors: %v", len(rb.errors), rb.errors[0])
	}

	// Perform final consistency checks
	if err := rb.result.Validate(); err != nil {
		return nil, fmt.Errorf("final result validation failed: %w", err)
	}

	return rb.result, nil
}

// Methods for SimulationResult

// Validate performs comprehensive validation of the result
func (sr *SimulationResult) Validate() error {
	// Check for NaN or infinite values
	if math.IsNaN(sr.Availability) || math.IsInf(sr.Availability, 0) {
		return fmt.Errorf("availability contains invalid value: %f", sr.Availability)
	}
	if math.IsNaN(sr.AvgLatencyMS) || math.IsInf(sr.AvgLatencyMS, 0) {
		return fmt.Errorf("average latency contains invalid value: %f", sr.AvgLatencyMS)
	}
	if math.IsNaN(sr.Score) || math.IsInf(sr.Score, 0) {
		return fmt.Errorf("score contains invalid value: %f", sr.Score)
	}

	// Range validations
	if sr.Availability < 0.0 || sr.Availability > 1.0 {
		return fmt.Errorf("availability must be between 0 and 1, got: %f", sr.Availability)
	}
	if sr.FaultTolerance < 0.0 || sr.FaultTolerance > 1.0 {
		return fmt.Errorf("fault tolerance must be between 0 and 1, got: %f", sr.FaultTolerance)
	}
	if sr.Score < 0.0 || sr.Score > 100.0 {
		return fmt.Errorf("score must be between 0 and 100, got: %f", sr.Score)
	}

	// Consistency checks
	if sr.TotalRequests < 0 {
		return fmt.Errorf("total requests cannot be negative: %d", sr.TotalRequests)
	}
	if sr.SuccessfulRequests+sr.FailedRequests != sr.TotalRequests {
		return fmt.Errorf("request counts inconsistent: %d + %d != %d", 
			sr.SuccessfulRequests, sr.FailedRequests, sr.TotalRequests)
	}

	return nil
}

// GetPerformanceCategory returns the performance category based on score
func (sr *SimulationResult) GetPerformanceCategory() PerformanceCategory {
	switch {
	case sr.Score >= 90.0:
		return PerformanceExcellent
	case sr.Score >= 70.0:
		return PerformanceGood
	case sr.Score >= 50.0:
		return PerformanceFair
	default:
		return PerformancePoor
	}
}

// GetSuccessRate returns the success rate as a percentage (0-100)
func (sr *SimulationResult) GetSuccessRate() float64 {
	if sr.TotalRequests == 0 {
		return 0.0
	}
	return (float64(sr.SuccessfulRequests) / float64(sr.TotalRequests)) * 100.0
}

// GetFailureRate returns the failure rate as a percentage (0-100)
func (sr *SimulationResult) GetFailureRate() float64 {
	return sr.RequestDropRate * 100.0
}

// GetAvailabilityPercentage returns availability as a percentage (0-100)
func (sr *SimulationResult) GetAvailabilityPercentage() float64 {
	return sr.Availability * 100.0
}

// GetFaultTolerancePercentage returns fault tolerance as a percentage (0-100)
func (sr *SimulationResult) GetFaultTolerancePercentage() float64 {
	return sr.FaultTolerance * 100.0
}

// IsHighPerforming returns true if the system meets high performance criteria
func (sr *SimulationResult) IsHighPerforming() bool {
	return sr.Score >= 80.0 && sr.Pass && sr.Availability >= 0.99 && sr.FaultTolerance >= 0.8
}

// GetBottleneckIndicators returns potential performance bottleneck indicators
func (sr *SimulationResult) GetBottleneckIndicators() []string {
	var indicators []string

	if sr.Availability < 0.95 {
		indicators = append(indicators, "Low availability indicates request failures")
	}
	if sr.AvgLatencyMS > 500.0 {
		indicators = append(indicators, "High average latency suggests processing bottlenecks")
	}
	if sr.FaultTolerance < 0.5 {
		indicators = append(indicators, "Low fault tolerance indicates insufficient redundancy")
	}
	if sr.QPSUnder2x < sr.QPSUnder15x*0.8 {
		indicators = append(indicators, "Performance degradation under high load")
	}
	if sr.QPSUnderFailure1x < sr.QPSUnder1x*0.7 {
		indicators = append(indicators, "Significant performance loss under failure conditions")
	}
	if float64(sr.NumFailedNodes)/float64(sr.NumNodes) > 0.2 {
		indicators = append(indicators, "High percentage of failed nodes")
	}

	return indicators
}

// GetQualityMetrics returns detailed quality metrics for analysis
func (sr *SimulationResult) GetQualityMetrics(requirements *Question) []*QualityMetric {
	metrics := make([]*QualityMetric, 0)

	// Availability metric
	availabilityScore := math.Min(100.0, (sr.Availability/requirements.RequiredAvailability)*100.0)
	availabilityStatus := "pass"
	if sr.Availability < requirements.RequiredAvailability {
		availabilityStatus = "fail"
	} else if sr.Availability < requirements.RequiredAvailability*1.05 {
		availabilityStatus = "warning"
	}

	metrics = append(metrics, &QualityMetric{
		Name:        "Availability",
		Value:       sr.Availability,
		Target:      requirements.RequiredAvailability,
		Weight:      30.0,
		Score:       availabilityScore,
		Importance:  "critical",
		Status:      availabilityStatus,
		Description: "Percentage of requests successfully processed",
	})

	// Latency metric
	latencyScore := math.Max(0.0, 100.0*(1.0-math.Max(0.0, sr.AvgLatencyMS-float64(requirements.RequiredLatencyMS))/float64(requirements.RequiredLatencyMS)))
	latencyStatus := "pass"
	if sr.AvgLatencyMS > float64(requirements.RequiredLatencyMS) {
		latencyStatus = "fail"
	} else if sr.AvgLatencyMS > float64(requirements.RequiredLatencyMS)*0.9 {
		latencyStatus = "warning"
	}

	metrics = append(metrics, &QualityMetric{
		Name:        "Latency",
		Value:       sr.AvgLatencyMS,
		Target:      float64(requirements.RequiredLatencyMS),
		Weight:      25.0,
		Score:       latencyScore,
		Importance:  "critical",
		Status:      latencyStatus,
		Description: "Average response time in milliseconds",
	})

	// Throughput metric
	expectedThroughput := float64(requirements.BaseQPS) * 2.0 // Expect to handle 2x base load
	throughputScore := math.Min(100.0, (sr.ThroughputQPS/expectedThroughput)*100.0)
	throughputStatus := "pass"
	if sr.ThroughputQPS < float64(requirements.BaseQPS) {
		throughputStatus = "fail"
	} else if sr.ThroughputQPS < float64(requirements.BaseQPS)*1.5 {
		throughputStatus = "warning"
	}

	metrics = append(metrics, &QualityMetric{
		Name:        "Throughput",
		Value:       sr.ThroughputQPS,
		Target:      expectedThroughput,
		Weight:      20.0,
		Score:       throughputScore,
		Importance:  "high",
		Status:      throughputStatus,
		Description: "Maximum sustainable queries per second",
	})

	// Fault tolerance metric
	faultToleranceScore := sr.FaultTolerance * 100.0
	faultToleranceStatus := "pass"
	if sr.FaultTolerance < 0.5 {
		faultToleranceStatus = "fail"
	} else if sr.FaultTolerance < 0.7 {
		faultToleranceStatus = "warning"
	}

	metrics = append(metrics, &QualityMetric{
		Name:        "Fault Tolerance",
		Value:       sr.FaultTolerance,
		Target:      0.8,
		Weight:      15.0,
		Score:       faultToleranceScore,
		Importance:  "high",
		Status:      faultToleranceStatus,
		Description: "Resilience to node failures",
	})

	// Scalability metric (based on QPS progression)
	scalabilityScore := 100.0
	if sr.QPSUnder1x > 0 {
		efficiency15x := sr.QPSUnder15x / (sr.QPSUnder1x * 1.5)
		efficiency2x := sr.QPSUnder2x / (sr.QPSUnder1x * 2.0)
		scalabilityScore = math.Min(100.0, (efficiency15x+efficiency2x)/2.0*100.0)
	}

	scalabilityStatus := "pass"
	if scalabilityScore < 60.0 {
		scalabilityStatus = "fail"
	} else if scalabilityScore < 80.0 {
		scalabilityStatus = "warning"
	}

	metrics = append(metrics, &QualityMetric{
		Name:        "Scalability",
		Value:       scalabilityScore / 100.0,
		Target:      0.8,
		Weight:      10.0,
		Score:       scalabilityScore,
		Importance:  "medium",
		Status:      scalabilityStatus,
		Description: "Ability to handle increased load efficiently",
	})

	return metrics
}

// Clone creates a deep copy of the SimulationResult
func (sr *SimulationResult) Clone() *SimulationResult {
	return &SimulationResult{
		Availability:       sr.Availability,
		AvgLatencyMS:       sr.AvgLatencyMS,
		ThroughputQPS:      sr.ThroughputQPS,
		FaultTolerance:     sr.FaultTolerance,
		QPSThreshold:       sr.QPSThreshold,
		TotalRequests:      sr.TotalRequests,
		SuccessfulRequests: sr.SuccessfulRequests,
		FailedRequests:     sr.FailedRequests,
		NumNodes:           sr.NumNodes,
		NumFailedNodes:     sr.NumFailedNodes,
		QPSUnder1x:         sr.QPSUnder1x,
		QPSUnder15x:        sr.QPSUnder15x,
		QPSUnder2x:         sr.QPSUnder2x,
		QPSUnderFailure1x:  sr.QPSUnderFailure1x,
		QPSUnderFailure15x: sr.QPSUnderFailure15x,
		RequestDropRate:    sr.RequestDropRate,
		Score:              sr.Score,
		Pass:               sr.Pass,
		Timestamp:          sr.Timestamp,
		ExecutionTime:      sr.ExecutionTime,
		Version:            sr.Version,
	}
}

// String returns a human-readable representation of the result
func (sr *SimulationResult) String() string {
	status := "FAIL"
	if sr.Pass {
		status = "PASS"
	}

	return fmt.Sprintf("SimulationResult{Score: %.1f, Status: %s, Availability: %.2f%%, Latency: %.1fms, Throughput: %.1f QPS, Fault Tolerance: %.1f%%}",
		sr.Score, status, sr.GetAvailabilityPercentage(), sr.AvgLatencyMS, sr.ThroughputQPS, sr.GetFaultTolerancePercentage())
}

// Summary returns a concise summary of the simulation results
func (sr *SimulationResult) Summary() string {
	category := sr.GetPerformanceCategory()
	return fmt.Sprintf("Performance: %s (%.1f/100) | Availability: %.2f%% | Latency: %.1fms | Status: %v", 
		string(category), sr.Score, sr.GetAvailabilityPercentage(), sr.AvgLatencyMS, sr.Pass)
}

// ToMap converts the result to a map for flexible serialization
func (sr *SimulationResult) ToMap() map[string]interface{} {
	return map[string]interface{}{
		"availability":           sr.Availability,
		"avg_latency_ms":         sr.AvgLatencyMS,
		"throughput_qps":         sr.ThroughputQPS,
		"fault_tolerance":        sr.FaultTolerance,
		"qps_threshold":          sr.QPSThreshold,
		"total_requests":         sr.TotalRequests,
		"successful_requests":    sr.SuccessfulRequests,
		"failed_requests":        sr.FailedRequests,
		"num_nodes":              sr.NumNodes,
		"num_failed_nodes":       sr.NumFailedNodes,
		"qps_under_1x":           sr.QPSUnder1x,
		"qps_under_1_5x":         sr.QPSUnder15x,
		"qps_under_2x":           sr.QPSUnder2x,
		"qps_under_failure_1x":   sr.QPSUnderFailure1x,
		"qps_under_failure_1_5x": sr.QPSUnderFailure15x,
		"request_drop_rate":      sr.RequestDropRate,
		"score":                  sr.Score,
		"pass":                   sr.Pass,
	}
}