package metrics

import (
	"fmt"
	"math"
	"sort"
	"sync"
	"time"
)

// Aggregator collects and aggregates metrics from simulation stages
type Aggregator struct {
	mu            sync.RWMutex
	stageMetrics  map[string]*StageMetrics
	nodeMetrics   map[string]*NodeMetrics
	pathMetrics   map[string]*PathMetrics
	systemMetrics *SystemMetrics
}

// StageMetrics contains metrics for a single simulation stage
type StageMetrics struct {
	StageName          string        `json:"stage_name"`
	TotalRequests      int64         `json:"total_requests"`
	SuccessfulRequests int64         `json:"successful_requests"`
	FailedRequests     int64         `json:"failed_requests"`
	DroppedRequests    int64         `json:"dropped_requests"`
	TotalLatency       float64       `json:"total_latency"`
	MinLatency         float64       `json:"min_latency"`
	MaxLatency         float64       `json:"max_latency"`
	LatencySum         float64       `json:"latency_sum"`
	LatencySumSquares  float64       `json:"latency_sum_squares"`
	Latencies          []float64     `json:"latencies"`
	Availability       float64       `json:"availability"`
	ThroughputQPS      float64       `json:"throughput_qps"`
	Duration           time.Duration `json:"duration"`
	StartTime          time.Time     `json:"start_time"`
	EndTime            time.Time     `json:"end_time"`
	SaturatedNodes     []string      `json:"saturated_nodes"`
	FailedNodes        []string      `json:"failed_nodes"`
}

// NodeMetrics tracks per-node performance metrics
type NodeMetrics struct {
	NodeID             string  `json:"node_id"`
	TotalRequests      int64   `json:"total_requests"`
	SuccessfulRequests int64   `json:"successful_requests"`
	FailedRequests     int64   `json:"failed_requests"`
	DroppedRequests    int64   `json:"dropped_requests"`
	TotalLatency       float64 `json:"total_latency"`
	MinLatency         float64 `json:"min_latency"`
	MaxLatency         float64 `json:"max_latency"`
	CurrentLoad        float64 `json:"current_load"`
	MaxCapacity        float64 `json:"max_capacity"`
	UtilizationRate    float64 `json:"utilization_rate"`
	IsSaturated        bool    `json:"is_saturated"`
	IsFailed           bool    `json:"is_failed"`
	FailureCount       int     `json:"failure_count"`
	LastFailureTime    int64   `json:"last_failure_time"`
}

// PathMetrics tracks per-path performance metrics
type PathMetrics struct {
	PathID             string    `json:"path_id"`
	Nodes              []string  `json:"nodes"`
	TotalRequests      int64     `json:"total_requests"`
	SuccessfulRequests int64     `json:"successful_requests"`
	FailedRequests     int64     `json:"failed_requests"`
	TotalLatency       float64   `json:"total_latency"`
	MinLatency         float64   `json:"min_latency"`
	MaxLatency         float64   `json:"max_latency"`
	Availability       float64   `json:"availability"`
	BottleneckNode     string    `json:"bottleneck_node"`
	PathWeight         float64   `json:"path_weight"`
}

// SystemMetrics contains overall system performance metrics
type SystemMetrics struct {
	TotalRequests      int64   `json:"total_requests"`
	SuccessfulRequests int64   `json:"successful_requests"`
	FailedRequests     int64   `json:"failed_requests"`
	DroppedRequests    int64   `json:"dropped_requests"`
	OverallAvailability float64 `json:"overall_availability"`
	AverageLatency     float64 `json:"average_latency"`
	MedianLatency      float64 `json:"median_latency"`
	P95Latency         float64 `json:"p95_latency"`
	P99Latency         float64 `json:"p99_latency"`
	MaxThroughputQPS   float64 `json:"max_throughput_qps"`
	RequestDropRate    float64 `json:"request_drop_rate"`
	NodeCount          int     `json:"node_count"`
	FailedNodeCount    int     `json:"failed_node_count"`
	SaturatedNodeCount int     `json:"saturated_node_count"`
}

// AggregatedMetrics represents the final aggregated metrics across all stages
type AggregatedMetrics struct {
	// Primary performance metrics
	Availability       float64 `json:"availability"`
	AvgLatencyMS      float64 `json:"avg_latency_ms"`
	ThroughputQPS     float64 `json:"throughput_qps"`
	FaultTolerance    float64 `json:"fault_tolerance"`
	QPSThreshold      float64 `json:"qps_threshold"`
	
	// Request counts
	TotalRequests     int64 `json:"total_requests"`
	SuccessfulRequests int64 `json:"successful_requests"`
	FailedRequests    int64 `json:"failed_requests"`
	RequestDropRate   float64 `json:"request_drop_rate"`
	
	// Node counts
	NumNodes        int `json:"num_nodes"`
	NumFailedNodes  int `json:"num_failed_nodes"`
	
	// QPS metrics per scenario
	QPSUnder1x         float64 `json:"qps_under_1x"`
	QPSUnder15x        float64 `json:"qps_under_1_5x"`
	QPSUnder2x         float64 `json:"qps_under_2x"`
	QPSUnderFailure1x  float64 `json:"qps_under_failure_1x"`
	QPSUnderFailure15x float64 `json:"qps_under_failure_1_5x"`
}

// NewAggregator creates a new metrics aggregator
func NewAggregator() *Aggregator {
	return &Aggregator{
		stageMetrics: make(map[string]*StageMetrics),
		nodeMetrics:  make(map[string]*NodeMetrics),
		pathMetrics:  make(map[string]*PathMetrics),
		systemMetrics: &SystemMetrics{},
	}
}

// RecordStageStart marks the beginning of a simulation stage
func (a *Aggregator) RecordStageStart(stageName string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	
	a.stageMetrics[stageName] = &StageMetrics{
		StageName:     stageName,
		StartTime:     time.Now(),
		MinLatency:    math.MaxFloat64,
		MaxLatency:    0.0,
		Latencies:     make([]float64, 0),
		SaturatedNodes: make([]string, 0),
		FailedNodes:   make([]string, 0),
	}
}

// RecordStageEnd marks the completion of a simulation stage
func (a *Aggregator) RecordStageEnd(stageName string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	
	stage, exists := a.stageMetrics[stageName]
	if !exists {
		return
	}
	
	stage.EndTime = time.Now()
	stage.Duration = stage.EndTime.Sub(stage.StartTime)
	
	// Calculate availability
	if stage.TotalRequests > 0 {
		stage.Availability = float64(stage.SuccessfulRequests) / float64(stage.TotalRequests)
	}
	
	// Calculate throughput QPS
	if stage.Duration.Seconds() > 0 {
		stage.ThroughputQPS = float64(stage.SuccessfulRequests) / stage.Duration.Seconds()
	}
}

// RecordRequest records metrics for a single request
func (a *Aggregator) RecordRequest(stageName, nodeID string, latency float64, success bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	
	// Update stage metrics
	stage, exists := a.stageMetrics[stageName]
	if !exists {
		stage = &StageMetrics{
			StageName: stageName,
			MinLatency: math.MaxFloat64,
			Latencies: make([]float64, 0),
			SaturatedNodes: make([]string, 0),
			FailedNodes: make([]string, 0),
		}
		a.stageMetrics[stageName] = stage
	}
	
	stage.TotalRequests++
	
	if success {
		stage.SuccessfulRequests++
		stage.TotalLatency += latency
		stage.LatencySum += latency
		stage.LatencySumSquares += latency * latency
		stage.Latencies = append(stage.Latencies, latency)
		
		if latency < stage.MinLatency {
			stage.MinLatency = latency
		}
		if latency > stage.MaxLatency {
			stage.MaxLatency = latency
		}
	} else {
		stage.FailedRequests++
	}
	
	// Update node metrics
	node, exists := a.nodeMetrics[nodeID]
	if !exists {
		node = &NodeMetrics{
			NodeID:     nodeID,
			MinLatency: math.MaxFloat64,
		}
		a.nodeMetrics[nodeID] = node
	}
	
	node.TotalRequests++
	
	if success {
		node.SuccessfulRequests++
		node.TotalLatency += latency
		
		if latency < node.MinLatency {
			node.MinLatency = latency
		}
		if latency > node.MaxLatency {
			node.MaxLatency = latency
		}
	} else {
		node.FailedRequests++
	}
}

// RecordRequestDrop records a dropped request (due to overload)
func (a *Aggregator) RecordRequestDrop(stageName, nodeID string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	
	// Update stage metrics
	if stage, exists := a.stageMetrics[stageName]; exists {
		stage.DroppedRequests++
		stage.FailedRequests++
		stage.TotalRequests++
	}
	
	// Update node metrics
	if node, exists := a.nodeMetrics[nodeID]; exists {
		node.DroppedRequests++
		node.FailedRequests++
		node.TotalRequests++
	}
}

// RecordPathRequest records metrics for a request traversing a specific path
func (a *Aggregator) RecordPathRequest(pathID string, nodes []string, latency float64, success bool, bottleneckNode string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	
	path, exists := a.pathMetrics[pathID]
	if !exists {
		path = &PathMetrics{
			PathID:     pathID,
			Nodes:      nodes,
			MinLatency: math.MaxFloat64,
		}
		a.pathMetrics[pathID] = path
	}
	
	path.TotalRequests++
	
	if success {
		path.SuccessfulRequests++
		path.TotalLatency += latency
		
		if latency < path.MinLatency {
			path.MinLatency = latency
		}
		if latency > path.MaxLatency {
			path.MaxLatency = latency
		}
	} else {
		path.FailedRequests++
	}
	
	if bottleneckNode != "" {
		path.BottleneckNode = bottleneckNode
	}
	
	// Calculate path availability
	if path.TotalRequests > 0 {
		path.Availability = float64(path.SuccessfulRequests) / float64(path.TotalRequests)
	}
}

// UpdateNodeState updates the current state of a node
func (a *Aggregator) UpdateNodeState(nodeID string, currentLoad, maxCapacity float64, isSaturated, isFailed bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	
	node, exists := a.nodeMetrics[nodeID]
	if !exists {
		node = &NodeMetrics{
			NodeID:     nodeID,
			MinLatency: math.MaxFloat64,
		}
		a.nodeMetrics[nodeID] = node
	}
	
	node.CurrentLoad = currentLoad
	node.MaxCapacity = maxCapacity
	
	if maxCapacity > 0 {
		node.UtilizationRate = currentLoad / maxCapacity
	}
	
	node.IsSaturated = isSaturated
	
	if isFailed && !node.IsFailed {
		node.FailureCount++
		node.LastFailureTime = time.Now().Unix()
	}
	node.IsFailed = isFailed
}

// RecordSaturatedNode records a node that has reached saturation
func (a *Aggregator) RecordSaturatedNode(stageName, nodeID string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	
	if stage, exists := a.stageMetrics[stageName]; exists {
		// Check if node is already recorded as saturated
		for _, id := range stage.SaturatedNodes {
			if id == nodeID {
				return
			}
		}
		stage.SaturatedNodes = append(stage.SaturatedNodes, nodeID)
	}
}

// RecordFailedNode records a node that has failed
func (a *Aggregator) RecordFailedNode(stageName, nodeID string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	
	if stage, exists := a.stageMetrics[stageName]; exists {
		// Check if node is already recorded as failed
		for _, id := range stage.FailedNodes {
			if id == nodeID {
				return
			}
		}
		stage.FailedNodes = append(stage.FailedNodes, nodeID)
	}
}

// GetStageMetrics returns metrics for a specific stage
func (a *Aggregator) GetStageMetrics(stageName string) (*StageMetrics, bool) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	
	metrics, exists := a.stageMetrics[stageName]
	if !exists {
		return nil, false
	}
	
	// Return a copy to prevent concurrent modifications
	return a.copyStageMetrics(metrics), true
}

// GetAllStageMetrics returns metrics for all stages
func (a *Aggregator) GetAllStageMetrics() map[string]*StageMetrics {
	a.mu.RLock()
	defer a.mu.RUnlock()
	
	result := make(map[string]*StageMetrics)
	for name, metrics := range a.stageMetrics {
		result[name] = a.copyStageMetrics(metrics)
	}
	
	return result
}

// GetNodeMetrics returns metrics for a specific node
func (a *Aggregator) GetNodeMetrics(nodeID string) (*NodeMetrics, bool) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	
	metrics, exists := a.nodeMetrics[nodeID]
	if !exists {
		return nil, false
	}
	
	// Return a copy
	return a.copyNodeMetrics(metrics), true
}

// GetAllNodeMetrics returns metrics for all nodes
func (a *Aggregator) GetAllNodeMetrics() map[string]*NodeMetrics {
	a.mu.RLock()
	defer a.mu.RUnlock()
	
	result := make(map[string]*NodeMetrics)
	for nodeID, metrics := range a.nodeMetrics {
		result[nodeID] = a.copyNodeMetrics(metrics)
	}
	
	return result
}

// GetPathMetrics returns metrics for a specific path
func (a *Aggregator) GetPathMetrics(pathID string) (*PathMetrics, bool) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	
	metrics, exists := a.pathMetrics[pathID]
	if !exists {
		return nil, false
	}
	
	// Return a copy
	return a.copyPathMetrics(metrics), true
}

// GetAllPathMetrics returns metrics for all paths
func (a *Aggregator) GetAllPathMetrics() map[string]*PathMetrics {
	a.mu.RLock()
	defer a.mu.RUnlock()
	
	result := make(map[string]*PathMetrics)
	for pathID, metrics := range a.pathMetrics {
		result[pathID] = a.copyPathMetrics(metrics)
	}
	
	return result
}

// AggregateAll computes final aggregated metrics across all stages
func (a *Aggregator) AggregateAll() (*AggregatedMetrics, error) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	
	if len(a.stageMetrics) == 0 {
		return nil, fmt.Errorf("no stage metrics available for aggregation")
	}
	
	aggregated := &AggregatedMetrics{}
	
	var allLatencies []float64
	var totalRequests, successfulRequests, failedRequests int64
	var qpsMetrics = make(map[string]float64)
	var maxThroughput float64
	
	// Process each stage
	for stageName, stage := range a.stageMetrics {
		totalRequests += stage.TotalRequests
		successfulRequests += stage.SuccessfulRequests
		failedRequests += stage.FailedRequests
		
		// Collect latencies
		allLatencies = append(allLatencies, stage.Latencies...)
		
		// Track max throughput
		if stage.ThroughputQPS > maxThroughput {
			maxThroughput = stage.ThroughputQPS
		}
		
		// Store QPS metrics by stage
		switch stageName {
		case "normal_1x":
			qpsMetrics["qps_under_1x"] = stage.ThroughputQPS
		case "surge_1_5x":
			qpsMetrics["qps_under_1_5x"] = stage.ThroughputQPS
		case "surge_2x":
			qpsMetrics["qps_under_2x"] = stage.ThroughputQPS
		case "failure_normal_1x":
			qpsMetrics["qps_under_failure_1x"] = stage.ThroughputQPS
		case "failure_surge_1_5x":
			qpsMetrics["qps_under_failure_1_5x"] = stage.ThroughputQPS
		}
	}
	
	// Set basic counts
	aggregated.TotalRequests = totalRequests
	aggregated.SuccessfulRequests = successfulRequests
	aggregated.FailedRequests = failedRequests
	
	// Calculate overall availability
	if totalRequests > 0 {
		aggregated.Availability = float64(successfulRequests) / float64(totalRequests)
		aggregated.RequestDropRate = float64(failedRequests) / float64(totalRequests)
	}
	
	// Calculate average latency (weighted by successful requests)
	if len(allLatencies) > 0 {
		latencySum := 0.0
		for _, latency := range allLatencies {
			latencySum += latency
		}
		aggregated.AvgLatencyMS = latencySum / float64(len(allLatencies))
	}
	
	// Set throughput metrics
	aggregated.ThroughputQPS = maxThroughput
	aggregated.QPSUnder1x = qpsMetrics["qps_under_1x"]
	aggregated.QPSUnder15x = qpsMetrics["qps_under_1_5x"]
	aggregated.QPSUnder2x = qpsMetrics["qps_under_2x"]
	aggregated.QPSUnderFailure1x = qpsMetrics["qps_under_failure_1x"]
	aggregated.QPSUnderFailure15x = qpsMetrics["qps_under_failure_1_5x"]
	
	// Calculate fault tolerance
	aggregated.FaultTolerance = a.calculateFaultTolerance(qpsMetrics)
	
	// Estimate QPS threshold
	aggregated.QPSThreshold = a.estimateQPSThreshold(qpsMetrics)
	
	// Count nodes and failures
	aggregated.NumNodes = len(a.nodeMetrics)
	
	failedNodes := make(map[string]bool)
	for _, nodeMetrics := range a.nodeMetrics {
		if nodeMetrics.IsFailed || nodeMetrics.FailureCount > 0 {
			failedNodes[nodeMetrics.NodeID] = true
		}
	}
	aggregated.NumFailedNodes = len(failedNodes)
	
	return aggregated, nil
}

// GetSystemMetrics returns aggregated system-wide metrics
func (a *Aggregator) GetSystemMetrics() *SystemMetrics {
	a.mu.RLock()
	defer a.mu.RUnlock()
	
	system := &SystemMetrics{}
	
	var allLatencies []float64
	
	// Aggregate across all stages
	for _, stage := range a.stageMetrics {
		system.TotalRequests += stage.TotalRequests
		system.SuccessfulRequests += stage.SuccessfulRequests
		system.FailedRequests += stage.FailedRequests
		system.DroppedRequests += stage.DroppedRequests
		
		allLatencies = append(allLatencies, stage.Latencies...)
		
		if stage.ThroughputQPS > system.MaxThroughputQPS {
			system.MaxThroughputQPS = stage.ThroughputQPS
		}
	}
	
	// Calculate overall availability
	if system.TotalRequests > 0 {
		system.OverallAvailability = float64(system.SuccessfulRequests) / float64(system.TotalRequests)
		system.RequestDropRate = float64(system.FailedRequests) / float64(system.TotalRequests)
	}
	
	// Calculate latency statistics
	if len(allLatencies) > 0 {
		sort.Float64s(allLatencies)
		
		// Average
		sum := 0.0
		for _, latency := range allLatencies {
			sum += latency
		}
		system.AverageLatency = sum / float64(len(allLatencies))
		
		// Median
		mid := len(allLatencies) / 2
		if len(allLatencies)%2 == 0 {
			system.MedianLatency = (allLatencies[mid-1] + allLatencies[mid]) / 2.0
		} else {
			system.MedianLatency = allLatencies[mid]
		}
		
		// Percentiles
		system.P95Latency = a.calculatePercentile(allLatencies, 0.95)
		system.P99Latency = a.calculatePercentile(allLatencies, 0.99)
	}
	
	// Count nodes by status
	system.NodeCount = len(a.nodeMetrics)
	for _, node := range a.nodeMetrics {
		if node.IsFailed {
			system.FailedNodeCount++
		}
		if node.IsSaturated {
			system.SaturatedNodeCount++
		}
	}
	
	return system
}

// Reset clears all collected metrics
func (a *Aggregator) Reset() {
	a.mu.Lock()
	defer a.mu.Unlock()
	
	a.stageMetrics = make(map[string]*StageMetrics)
	a.nodeMetrics = make(map[string]*NodeMetrics)
	a.pathMetrics = make(map[string]*PathMetrics)
	a.systemMetrics = &SystemMetrics{}
}

// Helper methods

func (a *Aggregator) calculateFaultTolerance(qpsMetrics map[string]float64) float64 {
	normalQPS := qpsMetrics["qps_under_1x"]
	failureQPS := qpsMetrics["qps_under_failure_1x"]
	
	if normalQPS == 0 {
		return 0.0
	}
	
	tolerance := failureQPS / normalQPS
	if tolerance > 1.0 {
		tolerance = 1.0
	}
	
	return tolerance
}

func (a *Aggregator) estimateQPSThreshold(qpsMetrics map[string]float64) float64 {
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

func (a *Aggregator) calculatePercentile(sortedValues []float64, percentile float64) float64 {
	if len(sortedValues) == 0 {
		return 0.0
	}
	
	if percentile <= 0 {
		return sortedValues[0]
	}
	if percentile >= 1 {
		return sortedValues[len(sortedValues)-1]
	}
	
	index := percentile * float64(len(sortedValues)-1)
	lower := int(index)
	upper := lower + 1
	
	if upper >= len(sortedValues) {
		return sortedValues[lower]
	}
	
	weight := index - float64(lower)
	return sortedValues[lower]*(1-weight) + sortedValues[upper]*weight
}

// Copy methods to prevent concurrent access issues

func (a *Aggregator) copyStageMetrics(original *StageMetrics) *StageMetrics {
	cloned := *original
	
	// Deep copy slices
	cloned.Latencies = make([]float64, len(original.Latencies))
	copySlice := cloned.Latencies
	_ = copySlice
	copy(cloned.Latencies, original.Latencies)
	
	cloned.SaturatedNodes = make([]string, len(original.SaturatedNodes))
	copy(cloned.SaturatedNodes, original.SaturatedNodes)
	
	cloned.FailedNodes = make([]string, len(original.FailedNodes))
	copy(cloned.FailedNodes, original.FailedNodes)
	
	return &cloned
}

func (a *Aggregator) copyNodeMetrics(original *NodeMetrics) *NodeMetrics {
	copy := *original
	return &copy
}

func (a *Aggregator) copyPathMetrics(original *PathMetrics) *PathMetrics {
	cloned := *original
	
	// Deep copy nodes slice
	cloned.Nodes = make([]string, len(original.Nodes))
	copy(cloned.Nodes, original.Nodes)
	
	return &cloned
}

// String returns a string representation of the aggregated metrics
func (am *AggregatedMetrics) String() string {
	return fmt.Sprintf("AggregatedMetrics{Availability: %.3f, AvgLatency: %.1fms, Throughput: %.1f QPS, FaultTolerance: %.3f}",
		am.Availability, am.AvgLatencyMS, am.ThroughputQPS, am.FaultTolerance)
}