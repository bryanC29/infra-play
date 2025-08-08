package config

import (
	"fmt"
	"time"
)

// SimulationConstants holds all hardcoded constants for simulation behavior
type SimulationConstants struct {
	// Resource-to-performance mappings
	CPUToQPSRatio          float64 `json:"cpu_to_qps_ratio"`           // QPS capacity per CPU core
	MemoryToQPSRatio       float64 `json:"memory_to_qps_ratio"`        // QPS capacity per MB of memory
	CPULatencyFactor       float64 `json:"cpu_latency_factor"`         // Latency factor for CPU (lower CPU = higher latency)
	MemoryLatencyFactor    float64 `json:"memory_latency_factor"`      // Latency factor for memory
	
	// Base latencies by node type (milliseconds)
	BaseLatencies          map[string]float64 `json:"base_latencies"`
	
	// Load and saturation behavior
	LoadDecayRate          float64 `json:"load_decay_rate"`            // Rate at which node load decays over time
	SaturationThreshold    float64 `json:"saturation_threshold"`       // Load percentage at which performance degrades
	OverloadPenalty        float64 `json:"overload_penalty"`           // Latency multiplier when overloaded
	QueueingDelay          float64 `json:"queueing_delay"`             // Additional delay per queued request (ms)
	
	// Failure injection
	NodeFailureRate        float64 `json:"node_failure_rate"`          // Percentage of nodes to fail during failure stages
	FailureDuration        time.Duration `json:"failure_duration"`     // How long failures last
	CascadeFailureRate     float64 `json:"cascade_failure_rate"`       // Probability of cascading failures
	PartialFailureRate     float64 `json:"partial_failure_rate"`       // Rate of partial performance degradation
	
	// Simulation timing
	StageDuration          time.Duration `json:"stage_duration"`        // Duration of each simulation stage
	RequestBatchSize       int           `json:"request_batch_size"`    // Number of requests to process in each batch
	SimulationPrecision    time.Duration `json:"simulation_precision"`  // Time granularity for simulation steps
	
	// Network and communication
	NetworkLatencyBase     float64 `json:"network_latency_base"`       // Base network latency between nodes (ms)
	NetworkLatencyPerHop   float64 `json:"network_latency_per_hop"`    // Additional latency per network hop (ms)
	BandwidthLimitMbps     float64 `json:"bandwidth_limit_mbps"`       // Default bandwidth limit
	
	// Replica and load balancing
	ReplicaLoadVariance    float64 `json:"replica_load_variance"`      // Variance in load distribution across replicas
	LoadBalancingOverhead  float64 `json:"load_balancing_overhead"`    // Additional latency for load balancing (ms)
	HealthCheckLatency     float64 `json:"health_check_latency"`       // Latency for health checks (ms)
	
	// Request processing
	RequestSize            float64 `json:"request_size"`               // Average request size in KB
	ResponseSize           float64 `json:"response_size"`              // Average response size in KB
	SerializationOverhead  float64 `json:"serialization_overhead"`    // Overhead for request/response serialization (ms)
	
	// Performance degradation curves
	LatencyDegradationCurve map[float64]float64 `json:"latency_degradation_curve"` // Load percentage -> latency multiplier
	ThroughputDegradationCurve map[float64]float64 `json:"throughput_degradation_curve"` // Load percentage -> throughput multiplier
	
	// Error rates and timeouts
	BaseErrorRate          float64 `json:"base_error_rate"`            // Base error rate under normal conditions
	TimeoutThreshold       float64 `json:"timeout_threshold"`          // Request timeout threshold (ms)
	RetryAttempts          int     `json:"retry_attempts"`             // Number of retry attempts for failed requests
	RetryBackoffMs         float64 `json:"retry_backoff_ms"`           // Backoff time between retries (ms)
	
	// Resource limits and constraints
	MaxConnections         int     `json:"max_connections"`            // Maximum connections per node
	ConnectionPoolSize     int     `json:"connection_pool_size"`       // Size of connection pools
	ThreadPoolSize         int     `json:"thread_pool_size"`           // Number of worker threads per node
	BufferSize             int     `json:"buffer_size"`                // Buffer size for request queues
	
	// Monitoring and observability
	MetricsCollectionOverhead float64 `json:"metrics_collection_overhead"` // Overhead for metrics collection (ms)
	LoggingOverhead          float64 `json:"logging_overhead"`             // Overhead for logging (ms)
	TracingOverhead          float64 `json:"tracing_overhead"`             // Overhead for distributed tracing (ms)
}

// Node type constants for base latencies and capacities
const (
	NodeTypeEntryPoint    = "EntryPoint"
	NodeTypeExitPoint     = "ExitPoint"
	NodeTypeAPIGateway    = "APIGateway"
	NodeTypeLoadBalancer  = "LoadBalancer"
	NodeTypeService       = "Service"
	NodeTypeDatabase      = "Database"
	NodeTypeCache         = "Cache"
	NodeTypeMessageQueue  = "MessageQueue"
	NodeTypeCDN          = "CDN"
	NodeTypeProxy        = "Proxy"
	NodeTypeFirewall     = "Firewall"
	NodeTypeMonitoring   = "Monitoring"
)

// Performance tier constants
const (
	PerformanceTierBasic      = "basic"
	PerformanceTierStandard   = "standard"
	PerformanceTierPremium    = "premium"
	PerformanceTierEnterprise = "enterprise"
)

// Simulation stage constants
const (
	StageNormal1x      = "normal_1x"
	StageSurge15x      = "surge_1_5x"
	StageSurge2x       = "surge_2x"
	StageFailureNormal = "failure_normal_1x"
	StageFailureSurge  = "failure_surge_1_5x"
)

// Default simulation constants
var defaultSimulationConstants = &SimulationConstants{
	// Resource-to-performance mappings
	CPUToQPSRatio:       1000.0, // 1000 QPS per CPU core
	MemoryToQPSRatio:    2.0,    // 2 QPS per MB of memory
	CPULatencyFactor:    50.0,   // Base latency factor for CPU calculations
	MemoryLatencyFactor: 0.01,   // Memory latency factor (minimal impact)
	
	// Base latencies by node type (milliseconds)
	BaseLatencies: map[string]float64{
		NodeTypeEntryPoint:    0.0,   // Entry and exit points have no latency
		NodeTypeExitPoint:     0.0,
		NodeTypeAPIGateway:    5.0,   // API Gateway processing
		NodeTypeLoadBalancer:  2.0,   // Load balancer routing
		NodeTypeService:       10.0,  // Application service processing
		NodeTypeDatabase:      25.0,  // Database query processing
		NodeTypeCache:         1.0,   // Cache lookup
		NodeTypeMessageQueue:  8.0,   // Message queue operations
		NodeTypeCDN:          3.0,   // CDN edge processing
		NodeTypeProxy:        1.5,   // Proxy forwarding
		NodeTypeFirewall:     2.5,   // Firewall inspection
		NodeTypeMonitoring:   1.0,   // Monitoring overhead
	},
	
	// Load and saturation behavior
	LoadDecayRate:       0.05,  // 5% decay per simulation step
	SaturationThreshold: 0.80,  // Performance degrades at 80% capacity
	OverloadPenalty:     3.0,   // 3x latency penalty when overloaded
	QueueingDelay:       0.5,   // 0.5ms additional delay per queued request
	
	// Failure injection
	NodeFailureRate:    0.01,             // 1% of nodes fail during failure stages
	FailureDuration:    30 * time.Second, // Failures last 30 seconds
	CascadeFailureRate: 0.005,            // 0.5% chance of cascading failures
	PartialFailureRate: 0.02,             // 2% chance of partial performance degradation
	
	// Simulation timing
	StageDuration:       30 * time.Second,  // Each stage runs for 30 seconds
	RequestBatchSize:    100,               // Process 100 requests per batch
	SimulationPrecision: 10 * time.Millisecond, // 10ms simulation steps
	
	// Network and communication
	NetworkLatencyBase:   1.0,  // 1ms base network latency
	NetworkLatencyPerHop: 0.5,  // 0.5ms per network hop
	BandwidthLimitMbps:   1000, // 1 Gbps default bandwidth
	
	// Replica and load balancing
	ReplicaLoadVariance:   0.1, // 10% variance in load distribution
	LoadBalancingOverhead: 0.5, // 0.5ms load balancing overhead
	HealthCheckLatency:    1.0, // 1ms health check latency
	
	// Request processing
	RequestSize:           4.0,  // 4KB average request
	ResponseSize:          8.0,  // 8KB average response
	SerializationOverhead: 0.2,  // 0.2ms serialization overhead
	
	// Performance degradation curves
	LatencyDegradationCurve: map[float64]float64{
		0.0:  1.0,  // No load = no degradation
		0.5:  1.1,  // 50% load = 10% latency increase
		0.7:  1.3,  // 70% load = 30% latency increase
		0.8:  1.8,  // 80% load = 80% latency increase (saturation point)
		0.9:  3.0,  // 90% load = 200% latency increase
		0.95: 5.0,  // 95% load = 400% latency increase
		1.0:  10.0, // 100% load = 900% latency increase
	},
	
	ThroughputDegradationCurve: map[float64]float64{
		0.0:  1.0,  // No load = full throughput
		0.5:  1.0,  // 50% load = full throughput
		0.7:  1.0,  // 70% load = full throughput
		0.8:  0.95, // 80% load = 5% throughput reduction
		0.9:  0.85, // 90% load = 15% throughput reduction
		0.95: 0.70, // 95% load = 30% throughput reduction
		1.0:  0.50, // 100% load = 50% throughput reduction (significant queuing)
	},
	
	// Error rates and timeouts
	BaseErrorRate:    0.001, // 0.1% base error rate
	TimeoutThreshold: 5000,  // 5 second timeout
	RetryAttempts:    3,     // 3 retry attempts
	RetryBackoffMs:   100,   // 100ms backoff between retries
	
	// Resource limits and constraints
	MaxConnections:     1000, // 1000 max connections per node
	ConnectionPoolSize: 50,   // 50 connections in pool
	ThreadPoolSize:     20,   // 20 worker threads per node
	BufferSize:         1000, // 1000 request buffer size
	
	// Monitoring and observability
	MetricsCollectionOverhead: 0.1, // 0.1ms metrics collection overhead
	LoggingOverhead:          0.05, // 0.05ms logging overhead
	TracingOverhead:          0.2,  // 0.2ms tracing overhead
}

// GetSimulationConstants returns the default simulation constants
func GetSimulationConstants() *SimulationConstants {
	return defaultSimulationConstants.Clone()
}

// GetBaseLatencyForNodeType returns the base latency for a specific node type
func (sc *SimulationConstants) GetBaseLatencyForNodeType(nodeType string) float64 {
	if latency, exists := sc.BaseLatencies[nodeType]; exists {
		return latency
	}
	// Return default service latency if node type not found
	return sc.BaseLatencies[NodeTypeService]
}

// GetLatencyMultiplierForLoad returns the latency multiplier based on current load
func (sc *SimulationConstants) GetLatencyMultiplierForLoad(loadPercent float64) float64 {
	if loadPercent <= 0 {
		return 1.0
	}
	if loadPercent >= 1.0 {
		return sc.LatencyDegradationCurve[1.0]
	}
	
	// Find the closest load points and interpolate
	var lowerLoad, upperLoad float64
	var lowerMultiplier, upperMultiplier float64
	
	for load, multiplier := range sc.LatencyDegradationCurve {
		if load <= loadPercent {
			if load > lowerLoad {
				lowerLoad = load
				lowerMultiplier = multiplier
			}
		}
		if load >= loadPercent {
			if upperLoad == 0 || load < upperLoad {
				upperLoad = load
				upperMultiplier = multiplier
			}
		}
	}
	
	// Linear interpolation
	if upperLoad == lowerLoad {
		return lowerMultiplier
	}
	
	ratio := (loadPercent - lowerLoad) / (upperLoad - lowerLoad)
	return lowerMultiplier + ratio*(upperMultiplier-lowerMultiplier)
}

// GetThroughputMultiplierForLoad returns the throughput multiplier based on current load
func (sc *SimulationConstants) GetThroughputMultiplierForLoad(loadPercent float64) float64 {
	if loadPercent <= 0 {
		return 1.0
	}
	if loadPercent >= 1.0 {
		return sc.ThroughputDegradationCurve[1.0]
	}
	
	// Find the closest load points and interpolate
	var lowerLoad, upperLoad float64
	var lowerMultiplier, upperMultiplier float64
	
	for load, multiplier := range sc.ThroughputDegradationCurve {
		if load <= loadPercent {
			if load > lowerLoad {
				lowerLoad = load
				lowerMultiplier = multiplier
			}
		}
		if load >= loadPercent {
			if upperLoad == 0 || load < upperLoad {
				upperLoad = load
				upperMultiplier = multiplier
			}
		}
	}
	
	// Linear interpolation
	if upperLoad == lowerLoad {
		return lowerMultiplier
	}
	
	ratio := (loadPercent - lowerLoad) / (upperLoad - lowerLoad)
	return lowerMultiplier + ratio*(upperMultiplier-lowerMultiplier)
}

// CalculateNodeCapacity calculates the theoretical maximum capacity of a node
func (sc *SimulationConstants) CalculateNodeCapacity(cpu float64, memoryMB int, replicas int) float64 {
	cpuCapacity := cpu * sc.CPUToQPSRatio
	memoryCapacity := float64(memoryMB) * sc.MemoryToQPSRatio
	
	// Take the minimum of CPU and memory constraints
	nodeCapacity := cpuCapacity
	if memoryMB > 0 && memoryCapacity < cpuCapacity {
		nodeCapacity = memoryCapacity
	}
	
	// Apply replica multiplier
	return nodeCapacity * float64(replicas)
}

// CalculateNetworkLatency calculates network latency between nodes
func (sc *SimulationConstants) CalculateNetworkLatency(hops int) float64 {
	return sc.NetworkLatencyBase + (float64(hops) * sc.NetworkLatencyPerHop)
}

// CalculateProcessingLatency calculates the processing latency for a node
func (sc *SimulationConstants) CalculateProcessingLatency(nodeType string, cpu float64, memoryMB int, currentLoad float64, maxCapacity float64) float64 {
	baseLatency := sc.GetBaseLatencyForNodeType(nodeType)
	
	// Apply resource-based factors
	cpuFactor := sc.CPULatencyFactor / cpu
	memoryFactor := 1.0
	if memoryMB > 0 {
		memoryFactor = sc.MemoryLatencyFactor / float64(memoryMB)
	}
	
	// Apply load-based multiplier
	loadPercent := currentLoad / maxCapacity
	loadMultiplier := sc.GetLatencyMultiplierForLoad(loadPercent)
	
	// Calculate final latency
	latency := baseLatency * cpuFactor * memoryFactor * loadMultiplier
	
	// Add serialization overhead
	latency += sc.SerializationOverhead
	
	// Add monitoring overhead
	latency += sc.MetricsCollectionOverhead + sc.LoggingOverhead + sc.TracingOverhead
	
	return latency
}

// CalculateErrorRate calculates the error rate for a node based on load
func (sc *SimulationConstants) CalculateErrorRate(currentLoad float64, maxCapacity float64) float64 {
	baseErrorRate := sc.BaseErrorRate
	loadPercent := currentLoad / maxCapacity
	
	if loadPercent > sc.SaturationThreshold {
		// Exponentially increase error rate as load approaches capacity
		overloadFactor := (loadPercent - sc.SaturationThreshold) / (1.0 - sc.SaturationThreshold)
		return baseErrorRate * (1.0 + overloadFactor*10.0)
	}
	
	return baseErrorRate
}

// IsNodeSaturated checks if a node is saturated based on current load
func (sc *SimulationConstants) IsNodeSaturated(currentLoad float64, maxCapacity float64) bool {
	return currentLoad >= maxCapacity*sc.SaturationThreshold
}

// IsNodeOverloaded checks if a node is overloaded
func (sc *SimulationConstants) IsNodeOverloaded(currentLoad float64, maxCapacity float64) bool {
	return currentLoad >= maxCapacity*0.95 // 95% is considered overloaded
}

// GetQPSMultiplierForStage returns the QPS multiplier for a specific simulation stage
func (sc *SimulationConstants) GetQPSMultiplierForStage(stage string) float64 {
	switch stage {
	case StageNormal1x, StageFailureNormal:
		return 1.0
	case StageSurge15x, StageFailureSurge:
		return 1.5
	case StageSurge2x:
		return 2.0
	default:
		return 1.0
	}
}

// Clone creates a deep copy of the simulation constants
func (sc *SimulationConstants) Clone() *SimulationConstants {
	clone := &SimulationConstants{
		CPUToQPSRatio:          sc.CPUToQPSRatio,
		MemoryToQPSRatio:       sc.MemoryToQPSRatio,
		CPULatencyFactor:       sc.CPULatencyFactor,
		MemoryLatencyFactor:    sc.MemoryLatencyFactor,
		LoadDecayRate:          sc.LoadDecayRate,
		SaturationThreshold:    sc.SaturationThreshold,
		OverloadPenalty:        sc.OverloadPenalty,
		QueueingDelay:          sc.QueueingDelay,
		NodeFailureRate:        sc.NodeFailureRate,
		FailureDuration:        sc.FailureDuration,
		CascadeFailureRate:     sc.CascadeFailureRate,
		PartialFailureRate:     sc.PartialFailureRate,
		StageDuration:          sc.StageDuration,
		RequestBatchSize:       sc.RequestBatchSize,
		SimulationPrecision:    sc.SimulationPrecision,
		NetworkLatencyBase:     sc.NetworkLatencyBase,
		NetworkLatencyPerHop:   sc.NetworkLatencyPerHop,
		BandwidthLimitMbps:     sc.BandwidthLimitMbps,
		ReplicaLoadVariance:    sc.ReplicaLoadVariance,
		LoadBalancingOverhead:  sc.LoadBalancingOverhead,
		HealthCheckLatency:     sc.HealthCheckLatency,
		RequestSize:            sc.RequestSize,
		ResponseSize:           sc.ResponseSize,
		SerializationOverhead:  sc.SerializationOverhead,
		BaseErrorRate:          sc.BaseErrorRate,
		TimeoutThreshold:       sc.TimeoutThreshold,
		RetryAttempts:          sc.RetryAttempts,
		RetryBackoffMs:         sc.RetryBackoffMs,
		MaxConnections:         sc.MaxConnections,
		ConnectionPoolSize:     sc.ConnectionPoolSize,
		ThreadPoolSize:         sc.ThreadPoolSize,
		BufferSize:             sc.BufferSize,
		MetricsCollectionOverhead: sc.MetricsCollectionOverhead,
		LoggingOverhead:          sc.LoggingOverhead,
		TracingOverhead:          sc.TracingOverhead,
		BaseLatencies:            make(map[string]float64),
		LatencyDegradationCurve:  make(map[float64]float64),
		ThroughputDegradationCurve: make(map[float64]float64),
	}
	
	// Deep copy maps
	for k, v := range sc.BaseLatencies {
		clone.BaseLatencies[k] = v
	}
	for k, v := range sc.LatencyDegradationCurve {
		clone.LatencyDegradationCurve[k] = v
	}
	for k, v := range sc.ThroughputDegradationCurve {
		clone.ThroughputDegradationCurve[k] = v
	}
	
	return clone
}

// Validate checks if the simulation constants are valid
func (sc *SimulationConstants) Validate() error {
	if sc.CPUToQPSRatio <= 0 {
		return fmt.Errorf("CPUToQPSRatio must be positive, got %.2f", sc.CPUToQPSRatio)
	}
	if sc.MemoryToQPSRatio <= 0 {
		return fmt.Errorf("MemoryToQPSRatio must be positive, got %.2f", sc.MemoryToQPSRatio)
	}
	if sc.StageDuration <= 0 {
		return fmt.Errorf("StageDuration must be positive, got %v", sc.StageDuration)
	}
	if sc.NodeFailureRate < 0 || sc.NodeFailureRate > 1 {
		return fmt.Errorf("NodeFailureRate must be between 0 and 1, got %.2f", sc.NodeFailureRate)
	}
	if sc.SaturationThreshold <= 0 || sc.SaturationThreshold > 1 {
		return fmt.Errorf("SaturationThreshold must be between 0 and 1, got %.2f", sc.SaturationThreshold)
	}
	
	// Validate that all required node types have base latencies
	requiredNodeTypes := []string{
		NodeTypeAPIGateway, NodeTypeLoadBalancer, NodeTypeService,
		NodeTypeDatabase, NodeTypeCache, NodeTypeMessageQueue,
	}
	
	for _, nodeType := range requiredNodeTypes {
		if _, exists := sc.BaseLatencies[nodeType]; !exists {
			return fmt.Errorf("missing base latency for node type: %s", nodeType)
		}
	}
	
	return nil
}

// Performance tier configurations
var (
	// BasicTierConstants provides minimal performance settings for basic tier
	BasicTierConstants = &SimulationConstants{
		CPUToQPSRatio:       500.0,  // Lower capacity per CPU
		MemoryToQPSRatio:    1.0,    // Lower memory efficiency
		CPULatencyFactor:    100.0,  // Higher latency factor
		SaturationThreshold: 0.70,   // Saturate earlier
		OverloadPenalty:     5.0,    // Higher penalty
		NodeFailureRate:     0.02,   // Higher failure rate
	}
	
	// PremiumTierConstants provides enhanced performance settings for premium tier
	PremiumTierConstants = &SimulationConstants{
		CPUToQPSRatio:       2000.0, // Higher capacity per CPU
		MemoryToQPSRatio:    4.0,    // Better memory efficiency
		CPULatencyFactor:    25.0,   // Lower latency factor
		SaturationThreshold: 0.90,   // Saturate later
		OverloadPenalty:     2.0,    // Lower penalty
		NodeFailureRate:     0.005,  // Lower failure rate
	}
)

// GetConstantsForTier returns simulation constants for a specific performance tier
func GetConstantsForTier(tier string) *SimulationConstants {
	switch tier {
	case PerformanceTierBasic:
		return mergeConstants(defaultSimulationConstants, BasicTierConstants)
	case PerformanceTierPremium, PerformanceTierEnterprise:
		return mergeConstants(defaultSimulationConstants, PremiumTierConstants)
	default:
		return GetSimulationConstants()
	}
}

// mergeConstants merges tier-specific constants with defaults
func mergeConstants(base, override *SimulationConstants) *SimulationConstants {
	result := base.Clone()
	
	if override.CPUToQPSRatio > 0 {
		result.CPUToQPSRatio = override.CPUToQPSRatio
	}
	if override.MemoryToQPSRatio > 0 {
		result.MemoryToQPSRatio = override.MemoryToQPSRatio
	}
	if override.CPULatencyFactor > 0 {
		result.CPULatencyFactor = override.CPULatencyFactor
	}
	if override.SaturationThreshold > 0 {
		result.SaturationThreshold = override.SaturationThreshold
	}
	if override.OverloadPenalty > 0 {
		result.OverloadPenalty = override.OverloadPenalty
	}
	if override.NodeFailureRate >= 0 {
		result.NodeFailureRate = override.NodeFailureRate
	}
	
	return result
}

// Helper functions for common calculations

// calculateMean calculates the mean of a slice of float64 values
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

// calculateMax finds the maximum value in a slice of float64 values
func calculateMax(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	
	max := values[0]
	for _, v := range values[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

// calculatePercentile calculates the specified percentile of a slice of float64 values
func calculatePercentile(values []float64, percentile float64) float64 {
	if len(values) == 0 {
		return 0
	}
	if percentile <= 0 {
		return calculateMin(values)
	}
	if percentile >= 1 {
		return calculateMax(values)
	}
	
	// Sort values first (simple bubble sort for small datasets)
	sorted := make([]float64, len(values))
	copy(sorted, values)
	
	for i := 0; i < len(sorted)-1; i++ {
		for j := 0; j < len(sorted)-i-1; j++ {
			if sorted[j] > sorted[j+1] {
				sorted[j], sorted[j+1] = sorted[j+1], sorted[j]
			}
		}
	}
	
	index := percentile * float64(len(sorted)-1)
	lower := int(index)
	upper := lower + 1
	
	if upper >= len(sorted) {
		return sorted[len(sorted)-1]
	}
	
	weight := index - float64(lower)
	return sorted[lower]*(1-weight) + sorted[upper]*weight
}

// calculateMin finds the minimum value in a slice of float64 values
func calculateMin(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	
	min := values[0]
	for _, v := range values[1:] {
		if v < min {
			min = v
		}
	}
	return min
}