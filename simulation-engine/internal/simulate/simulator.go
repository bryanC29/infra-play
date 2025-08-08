package simulate

import (
	"fmt"
	"math/rand"
	"simengine/internal/config"
	"simengine/internal/graph"
	"simengine/internal/metrics"
	"simengine/internal/models"
	"sync"
	"time"
)

// Simulator orchestrates all simulation stages and aggregates results
type Simulator struct {
	design      *models.SystemDesign
	question    *models.Question
	graph       *graph.DAG
	constants   *config.SimulationConstants
	metrics     *metrics.Aggregator
	rng         *rand.Rand
	mu          sync.RWMutex
}

// SimulationStage represents different load scenarios
type SimulationStage string

const (
	StageNormal1x       SimulationStage = "normal_1x"
	StageSurge15x       SimulationStage = "surge_1_5x"
	StageSurge2x        SimulationStage = "surge_2x"
	StageFailureNormal  SimulationStage = "failure_normal_1x"
	StageFailureSurge   SimulationStage = "failure_surge_1_5x"
)

// StageConfig holds configuration for each simulation stage
type StageConfig struct {
	Stage          SimulationStage
	QPS            int
	Duration       time.Duration
	FailureEnabled bool
	FailureRate    float64
	Description    string
}

// SimulationResult holds the complete results from all simulation stages
type SimulationResult struct {
	Design        *models.SystemDesign       `json:"design"`
	Question      *models.Question           `json:"question"`
	Stages        map[SimulationStage]*StageResult `json:"stages"`
	Aggregated    *metrics.AggregatedMetrics `json:"aggregated"`
	ExecutionTime time.Duration              `json:"execution_time"`
	Timestamp     time.Time                  `json:"timestamp"`
}

// StageResult holds results from a single simulation stage
type StageResult struct {
	Stage             SimulationStage        `json:"stage"`
	QPS               int                    `json:"qps"`
	Duration          time.Duration          `json:"duration"`
	TotalRequests     int64                  `json:"total_requests"`
	SuccessfulRequests int64                 `json:"successful_requests"`
	FailedRequests    int64                  `json:"failed_requests"`
	AvgLatencyMS      float64                `json:"avg_latency_ms"`
	MaxLatencyMS      float64                `json:"max_latency_ms"`
	P95LatencyMS      float64                `json:"p95_latency_ms"`
	P99LatencyMS      float64                `json:"p99_latency_ms"`
	Availability      float64                `json:"availability"`
	ThroughputQPS     float64                `json:"throughput_qps"`
	DroppedRequests   int64                  `json:"dropped_requests"`
	SaturatedNodes    []string               `json:"saturated_nodes"`
	FailedNodes       []string               `json:"failed_nodes"`
	PathMetrics       map[string]*PathMetrics `json:"path_metrics"`
}

// PathMetrics holds metrics for individual request paths through the system
type PathMetrics struct {
	PathID            string    `json:"path_id"`
	Nodes             []string  `json:"nodes"`
	TotalRequests     int64     `json:"total_requests"`
	SuccessfulRequests int64    `json:"successful_requests"`
	AvgLatencyMS      float64   `json:"avg_latency_ms"`
	MaxLatencyMS      float64   `json:"max_latency_ms"`
	Availability      float64   `json:"availability"`
	BottleneckNode    string    `json:"bottleneck_node"`
}

// NewSimulator creates a new simulator instance
func NewSimulator(design *models.SystemDesign, question *models.Question) (*Simulator, error) {
	if design == nil {
		return nil, fmt.Errorf("system design cannot be nil")
	}
	if question == nil {
		return nil, fmt.Errorf("question cannot be nil")
	}

	// Build the DAG from the system design
	dag, err := graph.BuildDAG(design)
	if err != nil {
		return nil, fmt.Errorf("failed to build DAG: %w", err)
	}

	// Create metrics aggregator
	metricsAgg := metrics.NewAggregator()

	// Create deterministic random source for consistent results
	seed := generateSeed(design, question)
	rng := rand.New(rand.NewSource(seed))

	return &Simulator{
		design:    design,
		question:  question,
		graph:     dag,
		constants: config.GetSimulationConstants(),
		metrics:   metricsAgg,
		rng:       rng,
	}, nil
}

// Run executes all simulation stages and returns aggregated results
func (s *Simulator) Run() (*SimulationResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	startTime := time.Now()

	// Initialize result structure
	result := &SimulationResult{
		Design:    s.design.Clone(),
		Question:  s.question.Clone(),
		Stages:    make(map[SimulationStage]*StageResult),
		Timestamp: startTime,
	}

	// Define simulation stages
	stages := s.getSimulationStages()

	// Execute each stage
	for _, stageConfig := range stages {
		stageResult, err := s.runStage(stageConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to run stage %s: %w", stageConfig.Stage, err)
		}
		result.Stages[stageConfig.Stage] = stageResult
	}

	// Aggregate metrics across all stages
	aggregated, err := s.aggregateResults(result.Stages)
	if err != nil {
		return nil, fmt.Errorf("failed to aggregate results: %w", err)
	}
	result.Aggregated = aggregated

	result.ExecutionTime = time.Since(startTime)

	return result, nil
}

// getSimulationStages returns the configuration for all simulation stages
func (s *Simulator) getSimulationStages() []StageConfig {
	baseQPS := s.question.GetBaseQPS()
	duration := s.constants.StageDuration

	return []StageConfig{
		{
			Stage:          StageNormal1x,
			QPS:            baseQPS,
			Duration:       duration,
			FailureEnabled: false,
			FailureRate:    0.0,
			Description:    "Normal load (1x base QPS)",
		},
		{
			Stage:          StageSurge15x,
			QPS:            s.question.GetSurge15QPS(),
			Duration:       duration,
			FailureEnabled: false,
			FailureRate:    0.0,
			Description:    "Surge load (1.5x base QPS)",
		},
		{
			Stage:          StageSurge2x,
			QPS:            s.question.GetSurge2QPS(),
			Duration:       duration,
			FailureEnabled: false,
			FailureRate:    0.0,
			Description:    "High surge load (2x base QPS)",
		},
		{
			Stage:          StageFailureNormal,
			QPS:            baseQPS,
			Duration:       duration,
			FailureEnabled: true,
			FailureRate:    s.constants.NodeFailureRate,
			Description:    "Normal load with node failures",
		},
		{
			Stage:          StageFailureSurge,
			QPS:            s.question.GetSurge15QPS(),
			Duration:       duration,
			FailureEnabled: true,
			FailureRate:    s.constants.NodeFailureRate,
			Description:    "Surge load with node failures",
		},
	}
}

// runStage executes a single simulation stage
func (s *Simulator) runStage(config StageConfig) (*StageResult, error) {
	// startTime := time.Now()

	// Initialize stage result
	result := &StageResult{
		Stage:       config.Stage,
		QPS:         config.QPS,
		Duration:    config.Duration,
		PathMetrics: make(map[string]*PathMetrics),
	}

	// Get all valid paths through the system
	paths, err := s.graph.GetAllPaths()
	if err != nil {
		return nil, fmt.Errorf("failed to get system paths: %w", err)
	}

	if len(paths) == 0 {
		return nil, fmt.Errorf("no valid paths found in system design")
	}

	// Initialize node states
	nodeStates := s.initializeNodeStates(config)

	// Simulate requests for the stage duration
	totalRequests := int64(config.QPS) * int64(config.Duration.Seconds())
	requestInterval := config.Duration / time.Duration(totalRequests)

	var allLatencies []float64
	var pathResults = make(map[string]*PathMetrics)

	// Initialize path metrics
	for i, path := range paths {
		pathID := fmt.Sprintf("path_%d", i)
		pathResults[pathID] = &PathMetrics{
			PathID: pathID,
			Nodes:  path.NodeIDs(),
		}
	}

	// Process requests
	for i := int64(0); i < totalRequests; i++ {
		// Select a random path for this request
		pathIndex := s.rng.Intn(len(paths))
		selectedPath := paths[pathIndex]
		pathID := fmt.Sprintf("path_%d", pathIndex)

		// Process the request through the selected path
		latency, success := s.processRequest(selectedPath, nodeStates, config)

		// Update metrics
		result.TotalRequests++
		pathResults[pathID].TotalRequests++

		if success {
			result.SuccessfulRequests++
			pathResults[pathID].SuccessfulRequests++
			allLatencies = append(allLatencies, latency)

			// Update path latency metrics
			pathMetrics := pathResults[pathID]
			if pathMetrics.TotalRequests == 1 {
				pathMetrics.AvgLatencyMS = latency
				pathMetrics.MaxLatencyMS = latency
			} else {
				// Update running average
				oldAvg := pathMetrics.AvgLatencyMS
				pathMetrics.AvgLatencyMS = oldAvg + (latency-oldAvg)/float64(pathMetrics.SuccessfulRequests)
				if latency > pathMetrics.MaxLatencyMS {
					pathMetrics.MaxLatencyMS = latency
				}
			}
		} else {
			result.FailedRequests++
		}

		// Update node utilization (simplified)
		s.updateNodeUtilization(selectedPath, nodeStates, requestInterval)
	}

	// Calculate final metrics
	s.calculateStageMetrics(result, allLatencies, pathResults, nodeStates)

	result.PathMetrics = pathResults

	return result, nil
}

// processRequest simulates processing a single request through a path
func (s *Simulator) processRequest(path *graph.Path, nodeStates map[string]*NodeState, config StageConfig) (latency float64, success bool) {
	totalLatency := 0.0
	success = true

	// Process through each node in the path
	for _, nodeID := range path.NodeIDs() {
		node, exists := s.design.GetNodeByID(nodeID)
		if !exists {
			success = false
			break
		}

		// Skip entry and exit points
		if node.IsSpecialNode() {
			continue
		}

		nodeState := nodeStates[nodeID]

		// Check if node is failed
		if nodeState.IsFailed {
			success = false
			break
		}

		// Check if node is saturated
		if nodeState.CurrentLoad >= nodeState.MaxCapacity {
			success = false
			nodeState.DroppedRequests++
			break
		}

		// Calculate latency for this node
		nodeLatency := s.calculateNodeLatency(node, nodeState)
		totalLatency += nodeLatency

		// Update node load
		nodeState.CurrentLoad += 1.0 / float64(node.GetReplicas())
	}

	return totalLatency, success
}

// calculateNodeLatency calculates the latency contribution of a single node
func (s *Simulator) calculateNodeLatency(node *models.Node, state *NodeState) float64 {
	baseLatency := s.constants.GetBaseLatencyForNodeType(node.Type)

	// Apply load-based latency increase
	loadFactor := state.CurrentLoad / state.MaxCapacity
	if loadFactor > 0.8 {
		// Exponential increase when approaching capacity
		loadMultiplier := 1.0 + (loadFactor-0.8)*5.0
		baseLatency *= loadMultiplier
	}

	// Add resource-based adjustments
	cpuFactor := s.constants.CPULatencyFactor / node.GetCPU()
	memoryFactor := 1.0
	if node.GetMemoryMB() > 0 {
		memoryFactor = s.constants.MemoryLatencyFactor / float64(node.GetMemoryMB())
	}

	adjustedLatency := baseLatency * cpuFactor * memoryFactor

	// Add some randomness for realism (±10%)
	randomFactor := 0.9 + s.rng.Float64()*0.2
	return adjustedLatency * randomFactor
}

// initializeNodeStates creates initial state for all nodes
func (s *Simulator) initializeNodeStates(config StageConfig) map[string]*NodeState {
	states := make(map[string]*NodeState)

	for _, node := range s.design.Nodes {
		if node.IsSpecialNode() {
			continue
		}

		state := &NodeState{
			NodeID:      node.ID,
			MaxCapacity: s.calculateNodeCapacity(&node),
			CurrentLoad: 0.0,
			IsFailed:    false,
		}

		// Apply failure if enabled for this stage
		if config.FailureEnabled {
			if s.rng.Float64() < config.FailureRate {
				state.IsFailed = true
			}
		}

		states[node.ID] = state
	}

	return states
}

// calculateNodeCapacity determines the maximum capacity of a node
func (s *Simulator) calculateNodeCapacity(node *models.Node) float64 {
	baseCPUCapacity := node.GetCPU() * s.constants.CPUToQPSRatio
	replicaMultiplier := float64(node.GetReplicas())

	// Memory constraints
	memoryMB := node.GetMemoryMB()
	memoryCapacity := float64(memoryMB) * s.constants.MemoryToQPSRatio

	// Take the minimum of CPU and memory constraints
	nodeCapacity := baseCPUCapacity
	if memoryMB > 0 && memoryCapacity < baseCPUCapacity {
		nodeCapacity = memoryCapacity
	}

	return nodeCapacity * replicaMultiplier
}

// updateNodeUtilization updates node load based on request processing
func (s *Simulator) updateNodeUtilization(path *graph.Path, nodeStates map[string]*NodeState, interval time.Duration) {
	// Decay load over time (simple exponential decay)
	decayFactor := 1.0 - interval.Seconds()*s.constants.LoadDecayRate

	for _, state := range nodeStates {
		if !state.IsFailed {
			state.CurrentLoad *= decayFactor
			if state.CurrentLoad < 0 {
				state.CurrentLoad = 0
			}
		}
	}
}

// calculateStageMetrics computes final metrics for a simulation stage
func (s *Simulator) calculateStageMetrics(result *StageResult, latencies []float64, pathResults map[string]*PathMetrics, nodeStates map[string]*NodeState) {
	// Calculate availability
	if result.TotalRequests > 0 {
		result.Availability = float64(result.SuccessfulRequests) / float64(result.TotalRequests)
	}

	// Calculate latency metrics
	if len(latencies) > 0 {
		result.AvgLatencyMS = calculateMean(latencies)
		result.MaxLatencyMS = calculateMax(latencies)
		result.P95LatencyMS = calculatePercentile(latencies, 0.95)
		result.P99LatencyMS = calculatePercentile(latencies, 0.99)
	}

	// Calculate throughput
	if result.Duration.Seconds() > 0 {
		result.ThroughputQPS = float64(result.SuccessfulRequests) / result.Duration.Seconds()
	}

	// Identify saturated and failed nodes
	for nodeID, state := range nodeStates {
		if state.IsFailed {
			result.FailedNodes = append(result.FailedNodes, nodeID)
		} else if state.CurrentLoad >= state.MaxCapacity*0.9 {
			result.SaturatedNodes = append(result.SaturatedNodes, nodeID)
		}
	}

	// Update path availability metrics
	for _, pathMetric := range pathResults {
		if pathMetric.TotalRequests > 0 {
			pathMetric.Availability = float64(pathMetric.SuccessfulRequests) / float64(pathMetric.TotalRequests)
		}
	}

	result.DroppedRequests = result.FailedRequests
}

// aggregateResults combines metrics from all simulation stages
func (s *Simulator) aggregateResults(stages map[SimulationStage]*StageResult) (*metrics.AggregatedMetrics, error) {
	aggregated := &metrics.AggregatedMetrics{}

	// var allLatencies []float64
	var totalRequests, totalSuccessful, totalFailed int64
	var qpsMetrics = make(map[string]float64)

	// Aggregate across all stages
	for stage, result := range stages {
		totalRequests += result.TotalRequests
		totalSuccessful += result.SuccessfulRequests
		totalFailed += result.FailedRequests

		// Store QPS metrics for specific stages
		switch stage {
		case StageNormal1x:
			qpsMetrics["qps_under_1x"] = result.ThroughputQPS
		case StageSurge15x:
			qpsMetrics["qps_under_1_5x"] = result.ThroughputQPS
		case StageSurge2x:
			qpsMetrics["qps_under_2x"] = result.ThroughputQPS
		case StageFailureNormal:
			qpsMetrics["qps_under_failure_1x"] = result.ThroughputQPS
		case StageFailureSurge:
			qpsMetrics["qps_under_failure_1_5x"] = result.ThroughputQPS
		}
	}

	// Calculate overall metrics
	aggregated.TotalRequests = totalRequests
	aggregated.SuccessfulRequests = totalSuccessful
	aggregated.FailedRequests = totalFailed

	if totalRequests > 0 {
		aggregated.Availability = float64(totalSuccessful) / float64(totalRequests)
		aggregated.RequestDropRate = float64(totalFailed) / float64(totalRequests)
	}

	// Find maximum throughput achieved
	maxThroughput := 0.0
	for _, qps := range qpsMetrics {
		if qps > maxThroughput {
			maxThroughput = qps
		}
	}
	aggregated.ThroughputQPS = maxThroughput

	// Calculate weighted average latency
	totalLatencyWeighted := 0.0
	totalWeightedRequests := int64(0)
	for _, result := range stages {
		if result.SuccessfulRequests > 0 {
			totalLatencyWeighted += result.AvgLatencyMS * float64(result.SuccessfulRequests)
			totalWeightedRequests += result.SuccessfulRequests
		}
	}
	if totalWeightedRequests > 0 {
		aggregated.AvgLatencyMS = totalLatencyWeighted / float64(totalWeightedRequests)
	}

	// Set node counts
	aggregated.NumNodes = s.design.FunctionalNodeCount()

	// Count failed nodes (from failure stages)
	failedNodesSet := make(map[string]bool)
	if failureStage, exists := stages[StageFailureNormal]; exists {
		for _, nodeID := range failureStage.FailedNodes {
			failedNodesSet[nodeID] = true
		}
	}
	if failureStage, exists := stages[StageFailureSurge]; exists {
		for _, nodeID := range failureStage.FailedNodes {
			failedNodesSet[nodeID] = true
		}
	}
	aggregated.NumFailedNodes = len(failedNodesSet)

	// Set QPS metrics
	aggregated.QPSUnder1x = qpsMetrics["qps_under_1x"]
	aggregated.QPSUnder15x = qpsMetrics["qps_under_1_5x"]
	aggregated.QPSUnder2x = qpsMetrics["qps_under_2x"]
	aggregated.QPSUnderFailure1x = qpsMetrics["qps_under_failure_1x"]
	aggregated.QPSUnderFailure15x = qpsMetrics["qps_under_failure_1_5x"]

	// Calculate fault tolerance score
	aggregated.FaultTolerance = s.calculateFaultTolerance(stages)

	// Estimate QPS threshold (point where performance degrades significantly)
	aggregated.QPSThreshold = s.estimateQPSThreshold(qpsMetrics)

	return aggregated, nil
}

// calculateFaultTolerance computes a fault tolerance score (0-1)
func (s *Simulator) calculateFaultTolerance(stages map[SimulationStage]*StageResult) float64 {
	normalStage := stages[StageNormal1x]
	failureStage := stages[StageFailureNormal]

	if normalStage == nil || failureStage == nil {
		return 0.0
	}

	if normalStage.Availability == 0 {
		return 0.0
	}

	// Fault tolerance is the ratio of availability under failure vs normal conditions
	return failureStage.Availability / normalStage.Availability
}

// estimateQPSThreshold estimates the maximum sustainable QPS
func (s *Simulator) estimateQPSThreshold(qpsMetrics map[string]float64) float64 {
	qps1x := qpsMetrics["qps_under_1x"]
	qps15x := qpsMetrics["qps_under_1_5x"]
	qps2x := qpsMetrics["qps_under_2x"]

	// Find the point where QPS growth plateaus or decreases
	if qps2x >= qps15x*0.95 {
		return qps2x * 1.2 // System can handle more
	} else if qps15x >= qps1x*0.95 {
		return qps15x * 1.1 // Plateau at 1.5x
	} else {
		return qps1x * 1.05 // Saturated at 1x
	}
}

// generateSeed creates a deterministic seed from design and question
func generateSeed(design *models.SystemDesign, question *models.Question) int64 {
	// Create a simple hash from key properties to ensure deterministic results
	seed := int64(0)
	seed += int64(len(design.Nodes)) * 1000
	seed += int64(len(design.Connections)) * 100
	seed += int64(question.BaseQPS)
	seed += int64(question.RequiredLatencyMS) * 10
	seed += int64(question.RequiredAvailability * 10000)

	// Add hash of node types and IDs
	for _, node := range design.Nodes {
		for _, char := range node.ID + node.Type {
			seed += int64(char)
		}
	}

	if seed == 0 {
		seed = 12345 // Fallback seed
	}

	return seed
}

// calculateMean returns the mean of a slice of float64 values
func calculateMean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

// calculateMax returns the max of a slice of float64 values
func calculateMax(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum
}

// calculatePercentile returns the p-th percentile (0.0-1.0) of a slice of float64 values
func calculatePercentile(values []float64, percentile float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := make([]float64, len(values))
	copy(sorted, values)
	// Simple sort (could use sort.Float64s for large slices)
	for i := 0; i < len(sorted)-1; i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}
	index := int(percentile * float64(len(sorted)-1))
	return sorted[index]
}

// NodeState tracks the runtime state of a node during simulation
type NodeState struct {
	NodeID          string
	MaxCapacity     float64
	CurrentLoad     float64
	IsFailed        bool
	DroppedRequests int64
}