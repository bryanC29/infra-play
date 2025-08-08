package metrics

import (
	"math"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// RequestTracker provides thread-safe tracking of requests across nodes and paths
type RequestTracker struct {
	mu                sync.RWMutex
	nodeTrackers      map[string]*NodeTracker
	pathTrackers      map[string]*PathTracker
	systemTracker     *SystemTracker
	stageName         string
	startTime         time.Time
	totalRequestCount int64
	activeRequests    int64
}

// NodeTracker tracks request metrics for a single node
type NodeTracker struct {
	mu                 sync.RWMutex
	nodeID             string
	totalRequests      int64
	successfulRequests int64
	failedRequests     int64
	droppedRequests    int64
	totalLatency       float64
	minLatency         float64
	maxLatency         float64
	latencySum         float64
	latencySumSquares  float64
	currentLoad        float64
	maxCapacity        float64
	lastRequestTime    time.Time
	requestTimes       []time.Time
	latencies          []float64
	isActive           bool
	isSaturated        bool
	isFailed           bool
	failureCount       int64
	lastFailureTime    time.Time
}

// PathTracker tracks request metrics for a specific path through the system
type PathTracker struct {
	mu                 sync.RWMutex
	pathID             string
	nodes              []string
	totalRequests      int64
	successfulRequests int64
	failedRequests     int64
	totalLatency       float64
	minLatency         float64
	maxLatency         float64
	latencySum         float64
	bottleneckNode     string
	pathWeight         float64
	reliability        float64
	lastRequestTime    time.Time
	latencies          []float64
}

// SystemTracker tracks overall system-wide metrics
type SystemTracker struct {
	mu                 sync.RWMutex
	totalRequests      int64
	successfulRequests int64
	failedRequests     int64
	droppedRequests    int64
	totalLatency       float64
	minLatency         float64
	maxLatency         float64
	requestRate        float64  // Current requests per second
	peakRequestRate    float64  // Peak requests per second seen
	averageRequestRate float64  // Running average
	requestRateHistory []float64
	startTime          time.Time
	lastUpdateTime     time.Time
	windowSize         time.Duration
}

// RequestEvent represents a single request event
type RequestEvent struct {
	RequestID   string
	NodeID      string
	PathID      string
	Timestamp   time.Time
	Latency     float64
	Success     bool
	Dropped     bool
	EventType   RequestEventType
	Metadata    map[string]interface{}
}

// RequestEventType represents different types of request events
type RequestEventType string

const (
	EventTypeStart    RequestEventType = "start"
	EventTypeComplete RequestEventType = "complete"
	EventTypeFail     RequestEventType = "fail"
	EventTypeDrop     RequestEventType = "drop"
	EventTypeTimeout  RequestEventType = "timeout"
)

// RequestStats provides summary statistics for requests
type RequestStats struct {
	Count             int64   `json:"count"`
	SuccessCount      int64   `json:"success_count"`
	FailureCount      int64   `json:"failure_count"`
	DropCount         int64   `json:"drop_count"`
	SuccessRate       float64 `json:"success_rate"`
	FailureRate       float64 `json:"failure_rate"`
	DropRate          float64 `json:"drop_rate"`
	AverageLatency    float64 `json:"average_latency"`
	MinLatency        float64 `json:"min_latency"`
	MaxLatency        float64 `json:"max_latency"`
	MedianLatency     float64 `json:"median_latency"`
	P95Latency        float64 `json:"p95_latency"`
	P99Latency        float64 `json:"p99_latency"`
	StandardDeviation float64 `json:"standard_deviation"`
	RequestRate       float64 `json:"request_rate"`
}

// NewRequestTracker creates a new request tracker for a simulation stage
func NewRequestTracker(stageName string) *RequestTracker {
	return &RequestTracker{
		nodeTrackers:  make(map[string]*NodeTracker),
		pathTrackers:  make(map[string]*PathTracker),
		stageName:     stageName,
		startTime:     time.Now(),
		systemTracker: &SystemTracker{
			startTime:     time.Now(),
			windowSize:    time.Second * 10, // 10-second sliding window
			minLatency:    float64(^uint(0) >> 1), // Max float64
		},
	}
}

// TrackRequest records a request event for a specific node and path
func (rt *RequestTracker) TrackRequest(event *RequestEvent) {
	atomic.AddInt64(&rt.totalRequestCount, 1)
	
	if event.EventType == EventTypeStart {
		atomic.AddInt64(&rt.activeRequests, 1)
	} else if event.EventType == EventTypeComplete || event.EventType == EventTypeFail || event.EventType == EventTypeDrop {
		atomic.AddInt64(&rt.activeRequests, -1)
	}

	// Track at node level
	rt.trackNodeRequest(event)
	
	// Track at path level
	if event.PathID != "" {
		rt.trackPathRequest(event)
	}
	
	// Track at system level
	rt.trackSystemRequest(event)
}

// trackNodeRequest updates metrics for a specific node
func (rt *RequestTracker) trackNodeRequest(event *RequestEvent) {
	rt.mu.Lock()
	tracker, exists := rt.nodeTrackers[event.NodeID]
	if !exists {
		tracker = &NodeTracker{
			nodeID:     event.NodeID,
			minLatency: float64(^uint(0) >> 1), // Max float64
			isActive:   true,
		}
		rt.nodeTrackers[event.NodeID] = tracker
	}
	rt.mu.Unlock()

	tracker.mu.Lock()
	defer tracker.mu.Unlock()

	tracker.lastRequestTime = event.Timestamp
	tracker.totalRequests++

	switch event.EventType {
	case EventTypeComplete:
		tracker.successfulRequests++
		tracker.totalLatency += event.Latency
		tracker.latencySum += event.Latency
		tracker.latencySumSquares += event.Latency * event.Latency
		tracker.latencies = append(tracker.latencies, event.Latency)

		if event.Latency < tracker.minLatency {
			tracker.minLatency = event.Latency
		}
		if event.Latency > tracker.maxLatency {
			tracker.maxLatency = event.Latency
		}

	case EventTypeFail:
		tracker.failedRequests++
		if !tracker.isFailed {
			tracker.failureCount++
			tracker.lastFailureTime = event.Timestamp
			tracker.isFailed = true
		}

	case EventTypeDrop:
		tracker.droppedRequests++
		tracker.failedRequests++
		tracker.isSaturated = true

	case EventTypeTimeout:
		tracker.failedRequests++
	}

	// Update request times for rate calculation
	tracker.requestTimes = append(tracker.requestTimes, event.Timestamp)
	
	// Trim old request times (keep last 100 for rate calculation)
	if len(tracker.requestTimes) > 100 {
		tracker.requestTimes = tracker.requestTimes[len(tracker.requestTimes)-100:]
	}

	// Update load information if provided in metadata
	if load, exists := event.Metadata["current_load"]; exists {
		if loadFloat, ok := load.(float64); ok {
			tracker.currentLoad = loadFloat
		}
	}
	if capacity, exists := event.Metadata["max_capacity"]; exists {
		if capacityFloat, ok := capacity.(float64); ok {
			tracker.maxCapacity = capacityFloat
		}
	}
}

// trackPathRequest updates metrics for a specific path
func (rt *RequestTracker) trackPathRequest(event *RequestEvent) {
	rt.mu.Lock()
	tracker, exists := rt.pathTrackers[event.PathID]
	if !exists {
		tracker = &PathTracker{
			pathID:     event.PathID,
			minLatency: float64(^uint(0) >> 1), // Max float64
		}
		
		// Extract nodes from metadata if provided
		if nodes, exists := event.Metadata["path_nodes"]; exists {
			if nodeSlice, ok := nodes.([]string); ok {
				tracker.nodes = nodeSlice
			}
		}
		
		rt.pathTrackers[event.PathID] = tracker
	}
	rt.mu.Unlock()

	tracker.mu.Lock()
	defer tracker.mu.Unlock()

	tracker.lastRequestTime = event.Timestamp
	tracker.totalRequests++

	switch event.EventType {
	case EventTypeComplete:
		tracker.successfulRequests++
		tracker.totalLatency += event.Latency
		tracker.latencySum += event.Latency
		tracker.latencies = append(tracker.latencies, event.Latency)

		if event.Latency < tracker.minLatency {
			tracker.minLatency = event.Latency
		}
		if event.Latency > tracker.maxLatency {
			tracker.maxLatency = event.Latency
		}

	case EventTypeFail, EventTypeDrop, EventTypeTimeout:
		tracker.failedRequests++
	}

	// Update bottleneck information if provided
	if bottleneck, exists := event.Metadata["bottleneck_node"]; exists {
		if bottleneckStr, ok := bottleneck.(string); ok {
			tracker.bottleneckNode = bottleneckStr
		}
	}

	// Update path weight if provided
	if weight, exists := event.Metadata["path_weight"]; exists {
		if weightFloat, ok := weight.(float64); ok {
			tracker.pathWeight = weightFloat
		}
	}
}

// trackSystemRequest updates system-wide metrics
func (rt *RequestTracker) trackSystemRequest(event *RequestEvent) {
	rt.systemTracker.mu.Lock()
	defer rt.systemTracker.mu.Unlock()

	rt.systemTracker.totalRequests++
	rt.systemTracker.lastUpdateTime = event.Timestamp

	switch event.EventType {
	case EventTypeComplete:
		rt.systemTracker.successfulRequests++
		rt.systemTracker.totalLatency += event.Latency

		if event.Latency < rt.systemTracker.minLatency {
			rt.systemTracker.minLatency = event.Latency
		}
		if event.Latency > rt.systemTracker.maxLatency {
			rt.systemTracker.maxLatency = event.Latency
		}

	case EventTypeFail, EventTypeTimeout:
		rt.systemTracker.failedRequests++

	case EventTypeDrop:
		rt.systemTracker.droppedRequests++
		rt.systemTracker.failedRequests++
	}

	// Update request rate (requests per second in the current window)
	rt.updateSystemRequestRate(event.Timestamp)
}

// updateSystemRequestRate calculates the current request rate
func (rt *RequestTracker) updateSystemRequestRate(timestamp time.Time) {
	// windowStart := timestamp.Add(-rt.systemTracker.windowSize)
	
	// Count requests in the current window
	// requestsInWindow := int64(0)
	duration := time.Since(rt.startTime)
	
	if duration > 0 {
		// Simple rate calculation: total requests / elapsed time
		rt.systemTracker.requestRate = float64(rt.systemTracker.totalRequests) / duration.Seconds()
		
		if rt.systemTracker.requestRate > rt.systemTracker.peakRequestRate {
			rt.systemTracker.peakRequestRate = rt.systemTracker.requestRate
		}
		
		// Update running average
		rt.systemTracker.requestRateHistory = append(rt.systemTracker.requestRateHistory, rt.systemTracker.requestRate)
		if len(rt.systemTracker.requestRateHistory) > 100 {
			rt.systemTracker.requestRateHistory = rt.systemTracker.requestRateHistory[1:]
		}
		
		// Calculate average request rate
		if len(rt.systemTracker.requestRateHistory) > 0 {
			sum := 0.0
			for _, rate := range rt.systemTracker.requestRateHistory {
				sum += rate
			}
			rt.systemTracker.averageRequestRate = sum / float64(len(rt.systemTracker.requestRateHistory))
		}
	}
}

// UpdateNodeCapacity updates the capacity information for a node
func (rt *RequestTracker) UpdateNodeCapacity(nodeID string, currentLoad, maxCapacity float64) {
	rt.mu.RLock()
	tracker, exists := rt.nodeTrackers[nodeID]
	rt.mu.RUnlock()

	if !exists {
		rt.mu.Lock()
		tracker = &NodeTracker{
			nodeID:     nodeID,
			minLatency: float64(^uint(0) >> 1),
			isActive:   true,
		}
		rt.nodeTrackers[nodeID] = tracker
		rt.mu.Unlock()
	}

	tracker.mu.Lock()
	defer tracker.mu.Unlock()

	tracker.currentLoad = currentLoad
	tracker.maxCapacity = maxCapacity
	
	// Update saturation status
	if maxCapacity > 0 {
		utilizationRate := currentLoad / maxCapacity
		tracker.isSaturated = utilizationRate >= 0.9 // 90% threshold for saturation
	}
}

// MarkNodeFailed marks a node as failed
func (rt *RequestTracker) MarkNodeFailed(nodeID string, failed bool) {
	rt.mu.RLock()
	tracker, exists := rt.nodeTrackers[nodeID]
	rt.mu.RUnlock()

	if !exists {
		rt.mu.Lock()
		tracker = &NodeTracker{
			nodeID:     nodeID,
			minLatency: float64(^uint(0) >> 1),
			isActive:   true,
		}
		rt.nodeTrackers[nodeID] = tracker
		rt.mu.Unlock()
	}

	tracker.mu.Lock()
	defer tracker.mu.Unlock()

	if failed && !tracker.isFailed {
		tracker.failureCount++
		tracker.lastFailureTime = time.Now()
	}
	tracker.isFailed = failed
}

// GetNodeStats returns statistics for a specific node
func (rt *RequestTracker) GetNodeStats(nodeID string) (*RequestStats, bool) {
	rt.mu.RLock()
	tracker, exists := rt.nodeTrackers[nodeID]
	rt.mu.RUnlock()

	if !exists {
		return nil, false
	}

	tracker.mu.RLock()
	defer tracker.mu.RUnlock()

	stats := &RequestStats{
		Count:        tracker.totalRequests,
		SuccessCount: tracker.successfulRequests,
		FailureCount: tracker.failedRequests,
		DropCount:    tracker.droppedRequests,
		MinLatency:   tracker.minLatency,
		MaxLatency:   tracker.maxLatency,
	}

	if tracker.totalRequests > 0 {
		stats.SuccessRate = float64(tracker.successfulRequests) / float64(tracker.totalRequests)
		stats.FailureRate = float64(tracker.failedRequests) / float64(tracker.totalRequests)
		stats.DropRate = float64(tracker.droppedRequests) / float64(tracker.totalRequests)
	}

	if tracker.successfulRequests > 0 {
		stats.AverageLatency = tracker.latencySum / float64(tracker.successfulRequests)
		
		// Calculate standard deviation
		if len(tracker.latencies) > 1 {
			variance := (tracker.latencySumSquares - (tracker.latencySum*tracker.latencySum)/float64(len(tracker.latencies))) / float64(len(tracker.latencies)-1)
			if variance > 0 {
				stats.StandardDeviation = math.Sqrt(variance)
			}
		}
		
		// Calculate percentiles
		if len(tracker.latencies) > 0 {
			sortedLatencies := make([]float64, len(tracker.latencies))
			copy(sortedLatencies, tracker.latencies)
			sort.Float64s(sortedLatencies)
			
			stats.MedianLatency = calculatePercentile(sortedLatencies, 0.50)
			stats.P95Latency = calculatePercentile(sortedLatencies, 0.95)
			stats.P99Latency = calculatePercentile(sortedLatencies, 0.99)
		}
	}

	// Calculate request rate
	if len(tracker.requestTimes) > 1 {
		duration := tracker.requestTimes[len(tracker.requestTimes)-1].Sub(tracker.requestTimes[0])
		if duration.Seconds() > 0 {
			stats.RequestRate = float64(len(tracker.requestTimes)) / duration.Seconds()
		}
	}

	return stats, true
}

// GetPathStats returns statistics for a specific path
func (rt *RequestTracker) GetPathStats(pathID string) (*RequestStats, bool) {
	rt.mu.RLock()
	tracker, exists := rt.pathTrackers[pathID]
	rt.mu.RUnlock()

	if !exists {
		return nil, false
	}

	tracker.mu.RLock()
	defer tracker.mu.RUnlock()

	stats := &RequestStats{
		Count:        tracker.totalRequests,
		SuccessCount: tracker.successfulRequests,
		FailureCount: tracker.failedRequests,
		MinLatency:   tracker.minLatency,
		MaxLatency:   tracker.maxLatency,
	}

	if tracker.totalRequests > 0 {
		stats.SuccessRate = float64(tracker.successfulRequests) / float64(tracker.totalRequests)
		stats.FailureRate = float64(tracker.failedRequests) / float64(tracker.totalRequests)
	}

	if tracker.successfulRequests > 0 {
		stats.AverageLatency = tracker.latencySum / float64(tracker.successfulRequests)
		
		// Calculate percentiles
		if len(tracker.latencies) > 0 {
			sortedLatencies := make([]float64, len(tracker.latencies))
			copy(sortedLatencies, tracker.latencies)
			sort.Float64s(sortedLatencies)
			
			stats.MedianLatency = calculatePercentile(sortedLatencies, 0.50)
			stats.P95Latency = calculatePercentile(sortedLatencies, 0.95)
			stats.P99Latency = calculatePercentile(sortedLatencies, 0.99)
		}
	}

	return stats, true
}

// GetSystemStats returns overall system statistics
func (rt *RequestTracker) GetSystemStats() *RequestStats {
	rt.systemTracker.mu.RLock()
	defer rt.systemTracker.mu.RUnlock()

	stats := &RequestStats{
		Count:        rt.systemTracker.totalRequests,
		SuccessCount: rt.systemTracker.successfulRequests,
		FailureCount: rt.systemTracker.failedRequests,
		DropCount:    rt.systemTracker.droppedRequests,
		MinLatency:   rt.systemTracker.minLatency,
		MaxLatency:   rt.systemTracker.maxLatency,
		RequestRate:  rt.systemTracker.requestRate,
	}

	if rt.systemTracker.totalRequests > 0 {
		stats.SuccessRate = float64(rt.systemTracker.successfulRequests) / float64(rt.systemTracker.totalRequests)
		stats.FailureRate = float64(rt.systemTracker.failedRequests) / float64(rt.systemTracker.totalRequests)
		stats.DropRate = float64(rt.systemTracker.droppedRequests) / float64(rt.systemTracker.totalRequests)
	}

	if rt.systemTracker.successfulRequests > 0 {
		stats.AverageLatency = rt.systemTracker.totalLatency / float64(rt.systemTracker.successfulRequests)
	}

	return stats
}

// GetAllNodeStats returns statistics for all tracked nodes
func (rt *RequestTracker) GetAllNodeStats() map[string]*RequestStats {
	rt.mu.RLock()
	defer rt.mu.RUnlock()

	result := make(map[string]*RequestStats)
	for nodeID := range rt.nodeTrackers {
		if stats, exists := rt.GetNodeStats(nodeID); exists {
			result[nodeID] = stats
		}
	}

	return result
}

// GetAllPathStats returns statistics for all tracked paths
func (rt *RequestTracker) GetAllPathStats() map[string]*RequestStats {
	rt.mu.RLock()
	defer rt.mu.RUnlock()

	result := make(map[string]*RequestStats)
	for pathID := range rt.pathTrackers {
		if stats, exists := rt.GetPathStats(pathID); exists {
			result[pathID] = stats
		}
	}

	return result
}

// GetActiveRequestCount returns the current number of active requests
func (rt *RequestTracker) GetActiveRequestCount() int64 {
	return atomic.LoadInt64(&rt.activeRequests)
}

// GetTotalRequestCount returns the total number of requests tracked
func (rt *RequestTracker) GetTotalRequestCount() int64 {
	return atomic.LoadInt64(&rt.totalRequestCount)
}

// GetSaturatedNodes returns a list of nodes that are currently saturated
func (rt *RequestTracker) GetSaturatedNodes() []string {
	rt.mu.RLock()
	defer rt.mu.RUnlock()

	var saturated []string
	for nodeID, tracker := range rt.nodeTrackers {
		tracker.mu.RLock()
		if tracker.isSaturated {
			saturated = append(saturated, nodeID)
		}
		tracker.mu.RUnlock()
	}

	return saturated
}

// GetFailedNodes returns a list of nodes that have failed
func (rt *RequestTracker) GetFailedNodes() []string {
	rt.mu.RLock()
	defer rt.mu.RUnlock()

	var failed []string
	for nodeID, tracker := range rt.nodeTrackers {
		tracker.mu.RLock()
		if tracker.isFailed {
			failed = append(failed, nodeID)
		}
		tracker.mu.RUnlock()
	}

	return failed
}

// Reset clears all tracking data
func (rt *RequestTracker) Reset() {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	rt.nodeTrackers = make(map[string]*NodeTracker)
	rt.pathTrackers = make(map[string]*PathTracker)
	rt.systemTracker = &SystemTracker{
		startTime:  time.Now(),
		windowSize: time.Second * 10,
		minLatency: float64(^uint(0) >> 1),
	}
	
	atomic.StoreInt64(&rt.totalRequestCount, 0)
	atomic.StoreInt64(&rt.activeRequests, 0)
	rt.startTime = time.Now()
}

// GetDuration returns the total duration since tracking started
func (rt *RequestTracker) GetDuration() time.Duration {
	return time.Since(rt.startTime)
}

// GetStageName returns the name of the simulation stage being tracked
func (rt *RequestTracker) GetStageName() string {
	return rt.stageName
}

// Helper function to calculate percentiles from sorted data
func calculatePercentile(sortedValues []float64, percentile float64) float64 {
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

// Clone creates a deep copy of the request tracker
func (rt *RequestTracker) Clone() *RequestTracker {
	rt.mu.RLock()
	defer rt.mu.RUnlock()

	clone := &RequestTracker{
		nodeTrackers:      make(map[string]*NodeTracker),
		pathTrackers:      make(map[string]*PathTracker),
		stageName:         rt.stageName,
		startTime:         rt.startTime,
		totalRequestCount: atomic.LoadInt64(&rt.totalRequestCount),
		activeRequests:    atomic.LoadInt64(&rt.activeRequests),
	}

	// Clone system tracker
	rt.systemTracker.mu.RLock()
	clone.systemTracker = &SystemTracker{
		totalRequests:      rt.systemTracker.totalRequests,
		successfulRequests: rt.systemTracker.successfulRequests,
		failedRequests:     rt.systemTracker.failedRequests,
		droppedRequests:    rt.systemTracker.droppedRequests,
		totalLatency:       rt.systemTracker.totalLatency,
		minLatency:         rt.systemTracker.minLatency,
		maxLatency:         rt.systemTracker.maxLatency,
		requestRate:        rt.systemTracker.requestRate,
		peakRequestRate:    rt.systemTracker.peakRequestRate,
		averageRequestRate: rt.systemTracker.averageRequestRate,
		startTime:          rt.systemTracker.startTime,
		lastUpdateTime:     rt.systemTracker.lastUpdateTime,
		windowSize:         rt.systemTracker.windowSize,
	}
	
	// Deep copy request rate history
	if len(rt.systemTracker.requestRateHistory) > 0 {
		clone.systemTracker.requestRateHistory = make([]float64, len(rt.systemTracker.requestRateHistory))
		copy(clone.systemTracker.requestRateHistory, rt.systemTracker.requestRateHistory)
	}
	rt.systemTracker.mu.RUnlock()

	// Clone node trackers
	for nodeID, tracker := range rt.nodeTrackers {
		tracker.mu.RLock()
		clonedTracker := &NodeTracker{
			nodeID:             tracker.nodeID,
			totalRequests:      tracker.totalRequests,
			successfulRequests: tracker.successfulRequests,
			failedRequests:     tracker.failedRequests,
			droppedRequests:    tracker.droppedRequests,
			totalLatency:       tracker.totalLatency,
			minLatency:         tracker.minLatency,
			maxLatency:         tracker.maxLatency,
			latencySum:         tracker.latencySum,
			latencySumSquares:  tracker.latencySumSquares,
			currentLoad:        tracker.currentLoad,
			maxCapacity:        tracker.maxCapacity,
			lastRequestTime:    tracker.lastRequestTime,
			isActive:           tracker.isActive,
			isSaturated:        tracker.isSaturated,
			isFailed:           tracker.isFailed,
			failureCount:       tracker.failureCount,
			lastFailureTime:    tracker.lastFailureTime,
		}
		
		// Deep copy slices
		if len(tracker.requestTimes) > 0 {
			clonedTracker.requestTimes = make([]time.Time, len(tracker.requestTimes))
			copy(clonedTracker.requestTimes, tracker.requestTimes)
		}
		if len(tracker.latencies) > 0 {
			clonedTracker.latencies = make([]float64, len(tracker.latencies))
			copy(clonedTracker.latencies, tracker.latencies)
		}
		
		clone.nodeTrackers[nodeID] = clonedTracker
		tracker.mu.RUnlock()
	}

	// Clone path trackers
	for pathID, tracker := range rt.pathTrackers {
		tracker.mu.RLock()
		clonedTracker := &PathTracker{
			pathID:             tracker.pathID,
			totalRequests:      tracker.totalRequests,
			successfulRequests: tracker.successfulRequests,
			failedRequests:     tracker.failedRequests,
			totalLatency:       tracker.totalLatency,
			minLatency:         tracker.minLatency,
			maxLatency:         tracker.maxLatency,
			latencySum:         tracker.latencySum,
			bottleneckNode:     tracker.bottleneckNode,
			pathWeight:         tracker.pathWeight,
			reliability:        tracker.reliability,
			lastRequestTime:    tracker.lastRequestTime,
		}
		
		// Deep copy slices
		if len(tracker.nodes) > 0 {
			clonedTracker.nodes = make([]string, len(tracker.nodes))
			copy(clonedTracker.nodes, tracker.nodes)
		}
		if len(tracker.latencies) > 0 {
			clonedTracker.latencies = make([]float64, len(tracker.latencies))
			copy(clonedTracker.latencies, tracker.latencies)
		}
		
		clone.pathTrackers[pathID] = clonedTracker
		tracker.mu.RUnlock()
	}

	return clone
}