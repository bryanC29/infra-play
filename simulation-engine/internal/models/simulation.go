package models

import (
	"fmt"
	"math"
	"sort"
	"time"
)

// SimulationStage represents different load scenarios during simulation
type SimulationStage string

const (
	StageNormal1x      SimulationStage = "normal_1x"
	StageSurge15x      SimulationStage = "surge_1_5x"
	StageSurge2x       SimulationStage = "surge_2x"
	StageFailureNormal SimulationStage = "failure_normal_1x"
	StageFailureSurge  SimulationStage = "failure_surge_1_5x"
)

// SimulationPhase represents the current phase of simulation execution
type SimulationPhase string

const (
	PhaseInitialization SimulationPhase = "initialization"
	PhaseExecution      SimulationPhase = "execution"
	PhaseAggregation    SimulationPhase = "aggregation"
	PhaseScoring        SimulationPhase = "scoring"
	PhaseCompleted      SimulationPhase = "completed"
	PhaseFailed         SimulationPhase = "failed"
)

// SimulationContext holds the complete context for a simulation run
type SimulationContext struct {
	Design           *SystemDesign           `json:"design"`
	Question         *Question               `json:"question"`
	SessionID        string                  `json:"session_id"`
	StartTime        time.Time               `json:"start_time"`
	CurrentStage     SimulationStage         `json:"current_stage"`
	CurrentPhase     SimulationPhase         `json:"current_phase"`
	StageResults     map[SimulationStage]*StageResult `json:"stage_results"`
	NodeStates       map[string]*NodeState   `json:"node_states"`
	RequestTracking  *RequestTracker         `json:"request_tracking"`
	SimulationConfig *SimulationConfig       `json:"simulation_config"`
	Errors           []string                `json:"errors,omitempty"`
}

// SimulationConfig holds configuration parameters for simulation execution
type SimulationConfig struct {
	StageDuration     time.Duration `json:"stage_duration"`
	RequestInterval   time.Duration `json:"request_interval"`
	NodeFailureRate   float64       `json:"node_failure_rate"`
	LoadDecayRate     float64       `json:"load_decay_rate"`
	MaxLatencyMS      float64       `json:"max_latency_ms"`
	SaturationThreshold float64     `json:"saturation_threshold"`
	RandomSeed        int64         `json:"random_seed"`
}

// StageResult holds comprehensive results from a single simulation stage
type StageResult struct {
	Stage              SimulationStage         `json:"stage"`
	QPS                int                     `json:"qps"`
	Duration           time.Duration           `json:"duration"`
	StartTime          time.Time               `json:"start_time"`
	EndTime            time.Time               `json:"end_time"`
	TotalRequests      int64                   `json:"total_requests"`
	SuccessfulRequests int64                   `json:"successful_requests"`
	FailedRequests     int64                   `json:"failed_requests"`
	DroppedRequests    int64                   `json:"dropped_requests"`
	TimeoutRequests    int64                   `json:"timeout_requests"`
	
	// Latency metrics
	AvgLatencyMS       float64                 `json:"avg_latency_ms"`
	MinLatencyMS       float64                 `json:"min_latency_ms"`
	MaxLatencyMS       float64                 `json:"max_latency_ms"`
	P50LatencyMS       float64                 `json:"p50_latency_ms"`
	P90LatencyMS       float64                 `json:"p90_latency_ms"`
	P95LatencyMS       float64                 `json:"p95_latency_ms"`
	P99LatencyMS       float64                 `json:"p99_latency_ms"`
	LatencyStdDev      float64                 `json:"latency_std_dev"`
	
	// Performance metrics
	Availability       float64                 `json:"availability"`
	ThroughputQPS      float64                 `json:"throughput_qps"`
	ErrorRate          float64                 `json:"error_rate"`
	SaturationRate     float64                 `json:"saturation_rate"`
	
	// Node-specific results
	SaturatedNodes     []string                `json:"saturated_nodes"`
	FailedNodes        []string                `json:"failed_nodes"`
	OverloadedNodes    []string                `json:"overloaded_nodes"`
	NodeMetrics        map[string]*NodeMetrics `json:"node_metrics"`
	
	// Path-specific results
	PathMetrics        map[string]*PathMetrics `json:"path_metrics"`
	BottleneckPaths    []string                `json:"bottleneck_paths"`
	
	// Resource utilization
	AvgCPUUtilization  float64                 `json:"avg_cpu_utilization"`
	AvgMemoryUtilization float64               `json:"avg_memory_utilization"`
	PeakCPUUtilization float64                 `json:"peak_cpu_utilization"`
	PeakMemoryUtilization float64              `json:"peak_memory_utilization"`
}

// NodeState tracks the runtime state of a node during simulation
type NodeState struct {
	NodeID              string                  `json:"node_id"`
	NodeType            string                  `json:"node_type"`
	MaxCapacity         float64                 `json:"max_capacity"`
	CurrentLoad         float64                 `json:"current_load"`
	CPUUtilization      float64                 `json:"cpu_utilization"`
	MemoryUtilization   float64                 `json:"memory_utilization"`
	IsFailed            bool                    `json:"is_failed"`
	IsSaturated         bool                    `json:"is_saturated"`
	IsOverloaded        bool                    `json:"is_overloaded"`
	FailureStartTime    *time.Time              `json:"failure_start_time,omitempty"`
	LastRequestTime     time.Time               `json:"last_request_time"`
	ProcessedRequests   int64                   `json:"processed_requests"`
	DroppedRequests     int64                   `json:"dropped_requests"`
	SuccessfulRequests  int64                   `json:"successful_requests"`
	FailedRequests      int64                   `json:"failed_requests"`
	TotalLatencyMS      float64                 `json:"total_latency_ms"`
	RequestQueue        []RequestInfo           `json:"request_queue"`
	LoadHistory         []LoadSnapshot          `json:"load_history"`
}

// NodeMetrics holds detailed metrics for a specific node during a simulation stage
type NodeMetrics struct {
	NodeID              string    `json:"node_id"`
	NodeType            string    `json:"node_type"`
	TotalRequests       int64     `json:"total_requests"`
	SuccessfulRequests  int64     `json:"successful_requests"`
	FailedRequests      int64     `json:"failed_requests"`
	DroppedRequests     int64     `json:"dropped_requests"`
	AvgLatencyMS        float64   `json:"avg_latency_ms"`
	MaxLatencyMS        float64   `json:"max_latency_ms"`
	MinLatencyMS        float64   `json:"min_latency_ms"`
	AvgCPUUtilization   float64   `json:"avg_cpu_utilization"`
	PeakCPUUtilization  float64   `json:"peak_cpu_utilization"`
	AvgMemoryUtilization float64  `json:"avg_memory_utilization"`
	PeakMemoryUtilization float64 `json:"peak_memory_utilization"`
	AvgLoad             float64   `json:"avg_load"`
	PeakLoad            float64   `json:"peak_load"`
	SaturationEvents    int       `json:"saturation_events"`
	SaturationDuration  time.Duration `json:"saturation_duration"`
	FailureDuration     time.Duration `json:"failure_duration"`
	Availability        float64   `json:"availability"`
	Reliability         float64   `json:"reliability"`
	BottleneckScore     float64   `json:"bottleneck_score"`
}

// PathMetrics holds metrics for individual request paths through the system
type PathMetrics struct {
	PathID             string    `json:"path_id"`
	Nodes              []string  `json:"nodes"`
	TotalRequests      int64     `json:"total_requests"`
	SuccessfulRequests int64     `json:"successful_requests"`
	FailedRequests     int64     `json:"failed_requests"`
	AvgLatencyMS       float64   `json:"avg_latency_ms"`
	MinLatencyMS       float64   `json:"min_latency_ms"`
	MaxLatencyMS       float64   `json:"max_latency_ms"`
	P95LatencyMS       float64   `json:"p95_latency_ms"`
	P99LatencyMS       float64   `json:"p99_latency_ms"`
	Availability       float64   `json:"availability"`
	ThroughputQPS      float64   `json:"throughput_qps"`
	BottleneckNode     string    `json:"bottleneck_node"`
	BottleneckScore    float64   `json:"bottleneck_score"`
	CriticalPath       bool      `json:"critical_path"`
	PathLength         int       `json:"path_length"`
	LatencyBreakdown   map[string]float64 `json:"latency_breakdown"`
}

// RequestInfo represents information about a single request
type RequestInfo struct {
	RequestID   string    `json:"request_id"`
	StartTime   time.Time `json:"start_time"`
	EndTime     time.Time `json:"end_time"`
	PathID      string    `json:"path_id"`
	LatencyMS   float64   `json:"latency_ms"`
	Success     bool      `json:"success"`
	FailureReason string  `json:"failure_reason,omitempty"`
	NodeLatencies map[string]float64 `json:"node_latencies"`
}

// LoadSnapshot captures load state at a specific point in time
type LoadSnapshot struct {
	Timestamp         time.Time `json:"timestamp"`
	CPUUtilization    float64   `json:"cpu_utilization"`
	MemoryUtilization float64   `json:"memory_utilization"`
	CurrentLoad       float64   `json:"current_load"`
	QueueLength       int       `json:"queue_length"`
	ActiveRequests    int       `json:"active_requests"`
}

// RequestTracker manages request flow and statistics across the simulation
type RequestTracker struct {
	TotalRequests      int64                    `json:"total_requests"`
	SuccessfulRequests int64                    `json:"successful_requests"`
	FailedRequests     int64                    `json:"failed_requests"`
	DroppedRequests    int64                    `json:"dropped_requests"`
	TimeoutRequests    int64                    `json:"timeout_requests"`
	RequestsByStage    map[SimulationStage]int64 `json:"requests_by_stage"`
	RequestsByNode     map[string]int64         `json:"requests_by_node"`
	RequestsByPath     map[string]int64         `json:"requests_by_path"`
	LatencyHistogram   []LatencyBucket          `json:"latency_histogram"`
	RequestSamples     []RequestInfo            `json:"request_samples"`
	MaxSampleSize      int                      `json:"max_sample_size"`
}

// LatencyBucket represents a bucket in the latency histogram
type LatencyBucket struct {
	LowerBoundMS float64 `json:"lower_bound_ms"`
	UpperBoundMS float64 `json:"upper_bound_ms"`
	Count        int64   `json:"count"`
	Percentage   float64 `json:"percentage"`
}

// FailureScenario defines different types of failures to inject
type FailureScenario struct {
	Type           FailureType `json:"type"`
	AffectedNodes  []string    `json:"affected_nodes"`
	FailureRate    float64     `json:"failure_rate"`
	Duration       time.Duration `json:"duration"`
	StartTime      time.Time   `json:"start_time"`
	RecoveryTime   time.Duration `json:"recovery_time"`
}

// FailureType represents different kinds of node failures
type FailureType string

const (
	FailureTypeComplete    FailureType = "complete"    // Node completely unavailable
	FailureTypePartial     FailureType = "partial"     // Node has reduced capacity
	FailureTypeIntermittent FailureType = "intermittent" // Node fails intermittently
	FailureTypeLatency     FailureType = "latency"     // Node has increased latency
	FailureTypeCascading   FailureType = "cascading"   // Failure spreads to connected nodes
)

// NewSimulationContext creates a new simulation context
func NewSimulationContext(design *SystemDesign, question *Question) *SimulationContext {
	sessionID := fmt.Sprintf("sim_%d_%s", time.Now().Unix(), question.SubmissionID)
	
	return &SimulationContext{
		Design:           design.Clone(),
		Question:         question.Clone(),
		SessionID:        sessionID,
		StartTime:        time.Now(),
		CurrentPhase:     PhaseInitialization,
		StageResults:     make(map[SimulationStage]*StageResult),
		NodeStates:       make(map[string]*NodeState),
		RequestTracking:  NewRequestTracker(),
		SimulationConfig: NewDefaultSimulationConfig(),
		Errors:           make([]string, 0),
	}
}

// NewDefaultSimulationConfig creates a default simulation configuration
func NewDefaultSimulationConfig() *SimulationConfig {
	return &SimulationConfig{
		StageDuration:       30 * time.Second,
		RequestInterval:     time.Millisecond,
		NodeFailureRate:     0.01, // 1% failure rate
		LoadDecayRate:       0.1,
		MaxLatencyMS:        5000.0,
		SaturationThreshold: 0.8,
		RandomSeed:          time.Now().UnixNano(),
	}
}

// NewRequestTracker creates a new request tracker
func NewRequestTracker() *RequestTracker {
	return &RequestTracker{
		RequestsByStage:  make(map[SimulationStage]int64),
		RequestsByNode:   make(map[string]int64),
		RequestsByPath:   make(map[string]int64),
		LatencyHistogram: make([]LatencyBucket, 0),
		RequestSamples:   make([]RequestInfo, 0),
		MaxSampleSize:    1000,
	}
}

// NewNodeState creates a new node state for the given node
func NewNodeState(node *Node, maxCapacity float64) *NodeState {
	return &NodeState{
		NodeID:            node.ID,
		NodeType:          node.Type,
		MaxCapacity:       maxCapacity,
		CurrentLoad:       0.0,
		CPUUtilization:    0.0,
		MemoryUtilization: 0.0,
		IsFailed:          false,
		IsSaturated:       false,
		IsOverloaded:      false,
		LastRequestTime:   time.Now(),
		RequestQueue:      make([]RequestInfo, 0),
		LoadHistory:       make([]LoadSnapshot, 0),
	}
}

// NewStageResult creates a new stage result
func NewStageResult(stage SimulationStage, qps int, duration time.Duration) *StageResult {
	return &StageResult{
		Stage:           stage,
		QPS:             qps,
		Duration:        duration,
		StartTime:       time.Now(),
		NodeMetrics:     make(map[string]*NodeMetrics),
		PathMetrics:     make(map[string]*PathMetrics),
		SaturatedNodes:  make([]string, 0),
		FailedNodes:     make([]string, 0),
		OverloadedNodes: make([]string, 0),
		BottleneckPaths: make([]string, 0),
		MinLatencyMS:    math.Inf(1),
		MaxLatencyMS:    0.0,
	}
}

// Methods for SimulationContext

// AddError adds an error to the simulation context
func (sc *SimulationContext) AddError(err error) {
	sc.Errors = append(sc.Errors, err.Error())
}

// HasErrors returns true if there are any errors
func (sc *SimulationContext) HasErrors() bool {
	return len(sc.Errors) > 0
}

// SetPhase updates the current simulation phase
func (sc *SimulationContext) SetPhase(phase SimulationPhase) {
	sc.CurrentPhase = phase
}

// SetStage updates the current simulation stage
func (sc *SimulationContext) SetStage(stage SimulationStage) {
	sc.CurrentStage = stage
}

// GetElapsedTime returns the time elapsed since simulation start
func (sc *SimulationContext) GetElapsedTime() time.Duration {
	return time.Since(sc.StartTime)
}

// IsStageCompleted checks if a specific stage has been completed
func (sc *SimulationContext) IsStageCompleted(stage SimulationStage) bool {
	_, exists := sc.StageResults[stage]
	return exists
}

// GetCompletedStages returns a list of completed stages
func (sc *SimulationContext) GetCompletedStages() []SimulationStage {
	stages := make([]SimulationStage, 0, len(sc.StageResults))
	for stage := range sc.StageResults {
		stages = append(stages, stage)
	}
	return stages
}

// Methods for NodeState

// UpdateLoad updates the current load and utilization metrics
func (ns *NodeState) UpdateLoad(additionalLoad float64) {
	ns.CurrentLoad += additionalLoad
	ns.CPUUtilization = math.Min(1.0, ns.CurrentLoad/ns.MaxCapacity)
	ns.MemoryUtilization = math.Min(1.0, ns.CurrentLoad/(ns.MaxCapacity*1.2)) // Memory has 20% buffer
	
	// Update saturation status
	ns.IsSaturated = ns.CurrentLoad >= ns.MaxCapacity*0.8
	ns.IsOverloaded = ns.CurrentLoad >= ns.MaxCapacity
	
	ns.LastRequestTime = time.Now()
}

// DecayLoad applies exponential decay to the current load
func (ns *NodeState) DecayLoad(decayRate float64) {
	ns.CurrentLoad *= (1.0 - decayRate)
	if ns.CurrentLoad < 0.001 {
		ns.CurrentLoad = 0.0
	}
	ns.UpdateLoad(0) // Refresh utilization metrics
}

// CanProcess returns true if the node can process additional requests
func (ns *NodeState) CanProcess() bool {
	return !ns.IsFailed && !ns.IsOverloaded
}

// GetLoadRatio returns the current load as a ratio of maximum capacity
func (ns *NodeState) GetLoadRatio() float64 {
	if ns.MaxCapacity <= 0 {
		return 0.0
	}
	return ns.CurrentLoad / ns.MaxCapacity
}

// AddLoadSnapshot adds a snapshot of current load to history
func (ns *NodeState) AddLoadSnapshot() {
	snapshot := LoadSnapshot{
		Timestamp:         time.Now(),
		CPUUtilization:    ns.CPUUtilization,
		MemoryUtilization: ns.MemoryUtilization,
		CurrentLoad:       ns.CurrentLoad,
		QueueLength:       len(ns.RequestQueue),
		ActiveRequests:    int(ns.ProcessedRequests - ns.SuccessfulRequests - ns.FailedRequests),
	}
	
	ns.LoadHistory = append(ns.LoadHistory, snapshot)
	
	// Keep only last 100 snapshots to prevent memory bloat
	if len(ns.LoadHistory) > 100 {
		ns.LoadHistory = ns.LoadHistory[len(ns.LoadHistory)-100:]
	}
}

// Methods for StageResult

// UpdateLatencyMetrics updates latency statistics with a new measurement
func (sr *StageResult) UpdateLatencyMetrics(latency float64) {
	if latency < sr.MinLatencyMS {
		sr.MinLatencyMS = latency
	}
	if latency > sr.MaxLatencyMS {
		sr.MaxLatencyMS = latency
	}
	
	// Update running average
	if sr.SuccessfulRequests == 0 {
		sr.AvgLatencyMS = latency
	} else {
		oldAvg := sr.AvgLatencyMS
		sr.AvgLatencyMS = oldAvg + (latency-oldAvg)/float64(sr.SuccessfulRequests+1)
	}
}

// CalculatePercentiles calculates latency percentiles from collected samples
func (sr *StageResult) CalculatePercentiles(latencies []float64) {
	if len(latencies) == 0 {
		return
	}
	
	sorted := make([]float64, len(latencies))
	copy(sorted, latencies)
	sort.Float64s(sorted)
	
	sr.P50LatencyMS = percentile(sorted, 0.50)
	sr.P90LatencyMS = percentile(sorted, 0.90)
	sr.P95LatencyMS = percentile(sorted, 0.95)
	sr.P99LatencyMS = percentile(sorted, 0.99)
	
	// Calculate standard deviation
	mean := sr.AvgLatencyMS
	sumSquares := 0.0
	for _, latency := range sorted {
		diff := latency - mean
		sumSquares += diff * diff
	}
	sr.LatencyStdDev = math.Sqrt(sumSquares / float64(len(sorted)))
}

// Finalize completes the stage result with final calculations
func (sr *StageResult) Finalize() {
	sr.EndTime = time.Now()
	
	// Calculate availability
	if sr.TotalRequests > 0 {
		sr.Availability = float64(sr.SuccessfulRequests) / float64(sr.TotalRequests)
		sr.ErrorRate = float64(sr.FailedRequests) / float64(sr.TotalRequests)
	}
	
	// Calculate throughput
	if sr.Duration.Seconds() > 0 {
		sr.ThroughputQPS = float64(sr.SuccessfulRequests) / sr.Duration.Seconds()
	}
	
	// Calculate saturation rate
	totalNodes := len(sr.NodeMetrics)
	saturatedNodes := len(sr.SaturatedNodes)
	if totalNodes > 0 {
		sr.SaturationRate = float64(saturatedNodes) / float64(totalNodes)
	}
	
	// Identify bottleneck paths
	sr.identifyBottleneckPaths()
}

// identifyBottleneckPaths identifies paths that are performance bottlenecks
func (sr *StageResult) identifyBottleneckPaths() {
	var pathScores []struct {
		pathID string
		score  float64
	}
	
	for pathID, metrics := range sr.PathMetrics {
		// Bottleneck score based on latency and failure rate
		latencyScore := metrics.AvgLatencyMS / sr.AvgLatencyMS
		availabilityScore := 1.0 - metrics.Availability
		score := latencyScore + availabilityScore
		
		pathScores = append(pathScores, struct {
			pathID string
			score  float64
		}{pathID, score})
	}
	
	// Sort by score (higher is worse)
	sort.Slice(pathScores, func(i, j int) bool {
		return pathScores[i].score > pathScores[j].score
	})
	
	// Mark top 20% as bottlenecks
	bottleneckCount := int(math.Ceil(float64(len(pathScores)) * 0.2))
	sr.BottleneckPaths = make([]string, 0, bottleneckCount)
	
	for i := 0; i < bottleneckCount && i < len(pathScores); i++ {
		sr.BottleneckPaths = append(sr.BottleneckPaths, pathScores[i].pathID)
		if metrics, exists := sr.PathMetrics[pathScores[i].pathID]; exists {
			metrics.BottleneckScore = pathScores[i].score
		}
	}
}

// Methods for RequestTracker

// TrackRequest records a completed request
func (rt *RequestTracker) TrackRequest(req RequestInfo, stage SimulationStage) {
	rt.TotalRequests++
	rt.RequestsByStage[stage]++
	
	for nodeID := range req.NodeLatencies {
		rt.RequestsByNode[nodeID]++
	}
	
	if req.PathID != "" {
		rt.RequestsByPath[req.PathID]++
	}
	
	if req.Success {
		rt.SuccessfulRequests++
	} else {
		rt.FailedRequests++
		if req.FailureReason == "timeout" {
			rt.TimeoutRequests++
		} else if req.FailureReason == "dropped" {
			rt.DroppedRequests++
		}
	}
	
	// Add to histogram
	rt.addToHistogram(req.LatencyMS)
	
	// Sample requests for detailed analysis
	if len(rt.RequestSamples) < rt.MaxSampleSize {
		rt.RequestSamples = append(rt.RequestSamples, req)
	}
}

// addToHistogram adds a latency measurement to the histogram
func (rt *RequestTracker) addToHistogram(latency float64) {
	// Define histogram buckets (in milliseconds)
	buckets := []float64{1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000}
	
	// Initialize histogram if empty
	if len(rt.LatencyHistogram) == 0 {
		rt.LatencyHistogram = make([]LatencyBucket, len(buckets))
		for i, bound := range buckets {
			rt.LatencyHistogram[i] = LatencyBucket{
				LowerBoundMS: 0,
				UpperBoundMS: bound,
			}
			if i > 0 {
				rt.LatencyHistogram[i].LowerBoundMS = buckets[i-1]
			}
		}
	}
	
	// Find appropriate bucket and increment
	for i := range rt.LatencyHistogram {
		if latency <= rt.LatencyHistogram[i].UpperBoundMS {
			rt.LatencyHistogram[i].Count++
			break
		}
	}
	
	// Update percentages
	total := rt.TotalRequests
	if total > 0 {
		for i := range rt.LatencyHistogram {
			rt.LatencyHistogram[i].Percentage = float64(rt.LatencyHistogram[i].Count) / float64(total) * 100.0
		}
	}
}

// GetAverageLatency returns the overall average latency
func (rt *RequestTracker) GetAverageLatency() float64 {
	if len(rt.RequestSamples) == 0 {
		return 0.0
	}
	
	total := 0.0
	count := 0
	for _, req := range rt.RequestSamples {
		if req.Success {
			total += req.LatencyMS
			count++
		}
	}
	
	if count == 0 {
		return 0.0
	}
	
	return total / float64(count)
}

// Helper function to calculate percentiles
func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0.0
	}
	
	if p <= 0 {
		return sorted[0]
	}
	if p >= 1 {
		return sorted[len(sorted)-1]
	}
	
	index := p * float64(len(sorted)-1)
	lower := int(math.Floor(index))
	upper := int(math.Ceil(index))
	
	if lower == upper {
		return sorted[lower]
	}
	
	// Linear interpolation
	weight := index - float64(lower)
	return sorted[lower]*(1-weight) + sorted[upper]*weight
}

// String methods for debugging

func (sc *SimulationContext) String() string {
	return fmt.Sprintf("SimulationContext{SessionID: %s, Phase: %s, Stage: %s, Elapsed: %v, Errors: %d}",
		sc.SessionID, sc.CurrentPhase, sc.CurrentStage, sc.GetElapsedTime(), len(sc.Errors))
}

func (ns *NodeState) String() string {
	return fmt.Sprintf("NodeState{ID: %s, Load: %.2f/%.2f (%.1f%%), Failed: %v, Saturated: %v}",
		ns.NodeID, ns.CurrentLoad, ns.MaxCapacity, ns.GetLoadRatio()*100, ns.IsFailed, ns.IsSaturated)
}

func (sr *StageResult) String() string {
	return fmt.Sprintf("StageResult{Stage: %s, Requests: %d/%d, Availability: %.2f%%, Latency: %.1fms, Throughput: %.1f QPS}",
		sr.Stage, sr.SuccessfulRequests, sr.TotalRequests, sr.Availability*100, sr.AvgLatencyMS, sr.ThroughputQPS)
}