package config

import (
	"fmt"
	"time"
)

// SimulationDefaults holds all default configuration values for the simulation engine
type SimulationDefaults struct {
	// Simulation timing
	StageDuration    time.Duration `json:"stage_duration"`
	RequestInterval  time.Duration `json:"request_interval"`
	WarmupDuration   time.Duration `json:"warmup_duration"`
	CooldownDuration time.Duration `json:"cooldown_duration"`
	
	// Load and capacity defaults
	LoadDecayRate         float64 `json:"load_decay_rate"`
	SaturationThreshold   float64 `json:"saturation_threshold"`
	OverloadThreshold     float64 `json:"overload_threshold"`
	MaxQueueSize          int     `json:"max_queue_size"`
	
	// Failure simulation
	NodeFailureRate       float64       `json:"node_failure_rate"`
	FailureRecoveryTime   time.Duration `json:"failure_recovery_time"`
	CascadeFailureRate    float64       `json:"cascade_failure_rate"`
	
	// Performance limits
	MaxLatencyMS          float64 `json:"max_latency_ms"`
	TimeoutThresholdMS    float64 `json:"timeout_threshold_ms"`
	MaxConcurrentRequests int     `json:"max_concurrent_requests"`
	
	// Sampling and tracking
	MaxRequestSamples     int     `json:"max_request_samples"`
	LoadHistorySize       int     `json:"load_history_size"`
	MetricsPrecision      int     `json:"metrics_precision"`
	
	// Scoring weights
	AvailabilityWeight    float64 `json:"availability_weight"`
	LatencyWeight         float64 `json:"latency_weight"`
	ThroughputWeight      float64 `json:"throughput_weight"`
	FaultToleranceWeight  float64 `json:"fault_tolerance_weight"`
	ScalabilityWeight     float64 `json:"scalability_weight"`
	
	// Pass/fail thresholds
	MinAvailabilityScore  float64 `json:"min_availability_score"`
	MinLatencyScore       float64 `json:"min_latency_score"`
	MinOverallScore       float64 `json:"min_overall_score"`
}

// ScoringWeights defines the default weights for different scoring components
type ScoringWeights struct {
	Availability   float64 `json:"availability"`   // Weight for availability score (0-100)
	Latency        float64 `json:"latency"`        // Weight for latency score (0-100)
	Throughput     float64 `json:"throughput"`     // Weight for throughput score (0-100)
	FaultTolerance float64 `json:"fault_tolerance"` // Weight for fault tolerance score (0-100)
	Scalability    float64 `json:"scalability"`    // Weight for scalability score (0-100)
	Efficiency     float64 `json:"efficiency"`     // Weight for resource efficiency score (0-100)
}

// PerformanceThresholds defines performance evaluation thresholds
type PerformanceThresholds struct {
	// Availability thresholds (as percentage)
	ExcellentAvailability float64 `json:"excellent_availability"` // >= 99.9%
	GoodAvailability      float64 `json:"good_availability"`      // >= 99.0%
	AcceptableAvailability float64 `json:"acceptable_availability"` // >= 95.0%
	
	// Latency multipliers for scoring
	ExcellentLatencyMultiplier float64 `json:"excellent_latency_multiplier"` // <= 0.5x required
	GoodLatencyMultiplier      float64 `json:"good_latency_multiplier"`      // <= 0.8x required
	AcceptableLatencyMultiplier float64 `json:"acceptable_latency_multiplier"` // <= 1.0x required
	
	// Throughput multipliers for scoring
	ExcellentThroughputMultiplier float64 `json:"excellent_throughput_multiplier"` // >= 2.0x base QPS
	GoodThroughputMultiplier      float64 `json:"good_throughput_multiplier"`      // >= 1.5x base QPS
	AcceptableThroughputMultiplier float64 `json:"acceptable_throughput_multiplier"` // >= 1.0x base QPS
	
	// Fault tolerance thresholds
	ExcellentFaultTolerance float64 `json:"excellent_fault_tolerance"` // >= 0.9
	GoodFaultTolerance      float64 `json:"good_fault_tolerance"`      // >= 0.7
	AcceptableFaultTolerance float64 `json:"acceptable_fault_tolerance"` // >= 0.5
}

// ResourceLimits defines default resource constraints and limits
type ResourceLimits struct {
	// Memory limits
	MaxMemoryPerNodeMB    int `json:"max_memory_per_node_mb"`
	DefaultMemoryMB       int `json:"default_memory_mb"`
	MinMemoryMB           int `json:"min_memory_mb"`
	
	// CPU limits
	MaxCPUPerNode         float64 `json:"max_cpu_per_node"`
	DefaultCPU            float64 `json:"default_cpu"`
	MinCPU                float64 `json:"min_cpu"`
	
	// Replica limits
	MaxReplicasPerNode    int `json:"max_replicas_per_node"`
	DefaultReplicas       int `json:"default_replicas"`
	MinReplicas           int `json:"min_replicas"`
	
	// Storage limits
	MaxStoragePerNodeGB   int `json:"max_storage_per_node_gb"`
	DefaultStorageGB      int `json:"default_storage_gb"`
	
	// Network limits
	MaxBandwidthMbps      int `json:"max_bandwidth_mbps"`
	DefaultBandwidthMbps  int `json:"default_bandwidth_mbps"`
}

// SimulationLimits defines limits for simulation execution
type SimulationLimits struct {
	MaxSimulationTime     time.Duration `json:"max_simulation_time"`
	MaxTotalRequests      int64         `json:"max_total_requests"`
	MaxNodesInDesign      int           `json:"max_nodes_in_design"`
	MaxConnectionsInDesign int          `json:"max_connections_in_design"`
	MaxPathsToAnalyze     int           `json:"max_paths_to_analyze"`
	MaxQPS                int           `json:"max_qps"`
	MinQPS                int           `json:"min_qps"`
}

// Default configuration instances
var (
	// DefaultSimulationConfig provides the standard simulation configuration
	DefaultSimulationConfig = &SimulationDefaults{
		// Timing configuration
		StageDuration:         30 * time.Second,
		RequestInterval:       time.Millisecond,
		WarmupDuration:        2 * time.Second,
		CooldownDuration:      1 * time.Second,
		
		// Load and capacity
		LoadDecayRate:         0.05,  // 5% decay per interval
		SaturationThreshold:   0.80,  // 80% of capacity
		OverloadThreshold:     0.95,  // 95% of capacity
		MaxQueueSize:          1000,
		
		// Failure simulation
		NodeFailureRate:       0.01,  // 1% failure rate
		FailureRecoveryTime:   5 * time.Second,
		CascadeFailureRate:    0.005, // 0.5% cascade rate
		
		// Performance limits
		MaxLatencyMS:          10000.0, // 10 seconds max
		TimeoutThresholdMS:    5000.0,  // 5 seconds timeout
		MaxConcurrentRequests: 10000,
		
		// Sampling and tracking
		MaxRequestSamples:     1000,
		LoadHistorySize:       100,
		MetricsPrecision:      2, // 2 decimal places
		
		// Scoring weights (must sum to 100.0)
		AvailabilityWeight:    35.0,
		LatencyWeight:         30.0,
		ThroughputWeight:      20.0,
		FaultToleranceWeight:  10.0,
		ScalabilityWeight:     5.0,
		
		// Pass/fail thresholds
		MinAvailabilityScore:  70.0, // Must score at least 70% on availability
		MinLatencyScore:       70.0, // Must score at least 70% on latency
		MinOverallScore:       60.0, // Must score at least 60% overall
	}
	
	// DefaultScoringWeights provides the standard scoring weight distribution
	DefaultScoringWeights = &ScoringWeights{
		Availability:   35.0, // Most critical - system must be available
		Latency:        30.0, // Very important - user experience
		Throughput:     20.0, // Important - system capacity
		FaultTolerance: 10.0, // Important - system resilience
		Scalability:    4.0,  // Nice to have - future growth
		Efficiency:     1.0,  // Nice to have - resource utilization
	}
	
	// DefaultPerformanceThresholds defines standard performance evaluation criteria
	DefaultPerformanceThresholds = &PerformanceThresholds{
		// Availability thresholds
		ExcellentAvailability:  0.999, // 99.9%
		GoodAvailability:       0.990, // 99.0%
		AcceptableAvailability: 0.950, // 95.0%
		
		// Latency multipliers (relative to required latency)
		ExcellentLatencyMultiplier:  0.5, // Response time <= 50% of requirement
		GoodLatencyMultiplier:       0.8, // Response time <= 80% of requirement
		AcceptableLatencyMultiplier: 1.0, // Response time <= 100% of requirement
		
		// Throughput multipliers (relative to base QPS)
		ExcellentThroughputMultiplier:  2.0, // Can handle 2x base load
		GoodThroughputMultiplier:       1.5, // Can handle 1.5x base load
		AcceptableThroughputMultiplier: 1.0, // Can handle 1x base load
		
		// Fault tolerance thresholds
		ExcellentFaultTolerance:  0.90, // 90% performance under failure
		GoodFaultTolerance:       0.70, // 70% performance under failure
		AcceptableFaultTolerance: 0.50, // 50% performance under failure
	}
	
	// DefaultResourceLimits defines standard resource constraints
	DefaultResourceLimits = &ResourceLimits{
		// Memory limits (in MB)
		MaxMemoryPerNodeMB: 32768, // 32 GB max per node
		DefaultMemoryMB:    512,   // 512 MB default
		MinMemoryMB:        128,   // 128 MB minimum
		
		// CPU limits (in cores)
		MaxCPUPerNode:      16.0,  // 16 cores max per node
		DefaultCPU:         1.0,   // 1 core default
		MinCPU:             0.1,   // 0.1 core minimum
		
		// Replica limits
		MaxReplicasPerNode: 10,    // 10 replicas max per node
		DefaultReplicas:    1,     // 1 replica default
		MinReplicas:        1,     // 1 replica minimum
		
		// Storage limits (in GB)
		MaxStoragePerNodeGB: 1000, // 1 TB max per node
		DefaultStorageGB:    10,   // 10 GB default
		
		// Network limits (in Mbps)
		MaxBandwidthMbps:     10000, // 10 Gbps max
		DefaultBandwidthMbps: 1000,  // 1 Gbps default
	}
	
	// DefaultSimulationLimits defines bounds for simulation execution
	DefaultSimulationLimits = &SimulationLimits{
		MaxSimulationTime:      5 * time.Minute,  // 5 minutes max total time
		MaxTotalRequests:       1000000,          // 1M requests max
		MaxNodesInDesign:       50,               // 50 nodes max in design
		MaxConnectionsInDesign: 200,              // 200 connections max
		MaxPathsToAnalyze:      100,              // 100 paths max to analyze
		MaxQPS:                 100000,           // 100K QPS max
		MinQPS:                 1,                // 1 QPS minimum
	}
)

// GetStageDurations returns the duration for each simulation stage
func GetStageDurations() map[string]time.Duration {
	baseDuration := DefaultSimulationConfig.StageDuration
	
	return map[string]time.Duration{
		"normal_1x":           baseDuration,
		"surge_1_5x":          baseDuration,
		"surge_2x":            baseDuration,
		"failure_normal_1x":   baseDuration,
		"failure_surge_1_5x":  baseDuration,
	}
}

// GetFailureRates returns the failure rates for different failure scenarios
func GetFailureRates() map[string]float64 {
	return map[string]float64{
		"node_failure":        DefaultSimulationConfig.NodeFailureRate,
		"cascade_failure":     DefaultSimulationConfig.CascadeFailureRate,
		"intermittent_failure": DefaultSimulationConfig.NodeFailureRate * 0.5,
		"latency_degradation": DefaultSimulationConfig.NodeFailureRate * 2.0,
	}
}

// GetScoringConfig returns the complete scoring configuration
func GetScoringConfig() ScoringConfig {
	return ScoringConfig{
		Weights:     *DefaultScoringWeights,
		Thresholds:  *DefaultPerformanceThresholds,
		MinScores: MinimumScores{
			Availability:   DefaultSimulationConfig.MinAvailabilityScore,
			Latency:        DefaultSimulationConfig.MinLatencyScore,
			Overall:        DefaultSimulationConfig.MinOverallScore,
			FaultTolerance: 40.0, // 40% minimum for fault tolerance
			Throughput:     50.0, // 50% minimum for throughput
		},
	}
}

// ScoringConfig combines all scoring-related configuration
type ScoringConfig struct {
	Weights     ScoringWeights         `json:"weights"`
	Thresholds  PerformanceThresholds  `json:"thresholds"`
	MinScores   MinimumScores          `json:"min_scores"`
}

// MinimumScores defines minimum acceptable scores for each component
type MinimumScores struct {
	Availability   float64 `json:"availability"`
	Latency        float64 `json:"latency"`
	Throughput     float64 `json:"throughput"`
	FaultTolerance float64 `json:"fault_tolerance"`
	Overall        float64 `json:"overall"`
}

// GetQPSMultipliers returns the QPS multipliers for different stages
func GetQPSMultipliers() map[string]float64 {
	return map[string]float64{
		"normal_1x":           1.0,
		"surge_1_5x":          1.5,
		"surge_2x":            2.0,
		"failure_normal_1x":   1.0,
		"failure_surge_1_5x":  1.5,
	}
}

// GetLatencyBuckets returns the default latency histogram buckets (in milliseconds)
func GetLatencyBuckets() []float64 {
	return []float64{
		1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000,
	}
}

// GetNodeTypeDefaults returns default resource allocations by node type
func GetNodeTypeDefaults() map[string]NodeTypeDefault {
	return map[string]NodeTypeDefault{
		"APIGateway": {
			CPU:        1.0,
			MemoryMB:   512,
			Replicas:   2,
			StorageGB:  1,
			BandwidthMbps: 1000,
		},
		"LoadBalancer": {
			CPU:        2.0,
			MemoryMB:   1024,
			Replicas:   2,
			StorageGB:  1,
			BandwidthMbps: 2000,
		},
		"Service": {
			CPU:        1.0,
			MemoryMB:   512,
			Replicas:   3,
			StorageGB:  5,
			BandwidthMbps: 500,
		},
		"Database": {
			CPU:        4.0,
			MemoryMB:   4096,
			Replicas:   1,
			StorageGB:  100,
			BandwidthMbps: 1000,
		},
		"Cache": {
			CPU:        2.0,
			MemoryMB:   2048,
			Replicas:   2,
			StorageGB:  5,
			BandwidthMbps: 1000,
		},
		"MessageQueue": {
			CPU:        1.0,
			MemoryMB:   1024,
			Replicas:   3,
			StorageGB:  20,
			BandwidthMbps: 500,
		},
		"CDN": {
			CPU:        1.0,
			MemoryMB:   512,
			Replicas:   1,
			StorageGB:  50,
			BandwidthMbps: 5000,
		},
		"Proxy": {
			CPU:        0.5,
			MemoryMB:   256,
			Replicas:   2,
			StorageGB:  1,
			BandwidthMbps: 2000,
		},
		"Firewall": {
			CPU:        1.0,
			MemoryMB:   512,
			Replicas:   2,
			StorageGB:  1,
			BandwidthMbps: 5000,
		},
		"Monitoring": {
			CPU:        1.0,
			MemoryMB:   1024,
			Replicas:   1,
			StorageGB:  50,
			BandwidthMbps: 100,
		},
	}
}

// NodeTypeDefault defines default resource allocation for a node type
type NodeTypeDefault struct {
	CPU           float64 `json:"cpu"`
	MemoryMB      int     `json:"memory_mb"`
	Replicas      int     `json:"replicas"`
	StorageGB     int     `json:"storage_gb"`
	BandwidthMbps int     `json:"bandwidth_mbps"`
}

// Validation functions

// ValidateScoringWeights ensures scoring weights sum to 100%
func ValidateScoringWeights(weights ScoringWeights) error {
	total := weights.Availability + weights.Latency + weights.Throughput + 
			 weights.FaultTolerance + weights.Scalability + weights.Efficiency
	
	if total < 99.9 || total > 100.1 { // Allow small floating point tolerance
		return fmt.Errorf("scoring weights must sum to 100.0, got %.2f", total)
	}
	
	return nil
}

// ValidateResourceLimits ensures resource limits are reasonable
func ValidateResourceLimits(limits ResourceLimits) error {
	if limits.MinCPU <= 0 {
		return fmt.Errorf("minimum CPU must be positive, got %.2f", limits.MinCPU)
	}
	if limits.MaxCPUPerNode < limits.MinCPU {
		return fmt.Errorf("maximum CPU (%.2f) must be >= minimum CPU (%.2f)", 
			limits.MaxCPUPerNode, limits.MinCPU)
	}
	if limits.MinMemoryMB <= 0 {
		return fmt.Errorf("minimum memory must be positive, got %d", limits.MinMemoryMB)
	}
	if limits.MaxMemoryPerNodeMB < limits.MinMemoryMB {
		return fmt.Errorf("maximum memory (%d) must be >= minimum memory (%d)", 
			limits.MaxMemoryPerNodeMB, limits.MinMemoryMB)
	}
	if limits.MinReplicas <= 0 {
		return fmt.Errorf("minimum replicas must be positive, got %d", limits.MinReplicas)
	}
	
	return nil
}

// GetEffectiveDefaults returns a complete default configuration with all values filled
func GetEffectiveDefaults() *EffectiveDefaults {
	return &EffectiveDefaults{
		Simulation:   *DefaultSimulationConfig,
		Scoring:      GetScoringConfig(),
		Resources:    *DefaultResourceLimits,
		Limits:       *DefaultSimulationLimits,
		NodeTypes:    GetNodeTypeDefaults(),
		QPSMultipliers: GetQPSMultipliers(),
		LatencyBuckets: GetLatencyBuckets(),
		FailureRates:   GetFailureRates(),
		StageDurations: GetStageDurations(),
	}
}

// EffectiveDefaults combines all default configurations into a single structure
type EffectiveDefaults struct {
	Simulation     SimulationDefaults           `json:"simulation"`
	Scoring        ScoringConfig                `json:"scoring"`
	Resources      ResourceLimits               `json:"resources"`
	Limits         SimulationLimits             `json:"limits"`
	NodeTypes      map[string]NodeTypeDefault   `json:"node_types"`
	QPSMultipliers map[string]float64           `json:"qps_multipliers"`
	LatencyBuckets []float64                    `json:"latency_buckets"`
	FailureRates   map[string]float64           `json:"failure_rates"`
	StageDurations map[string]time.Duration     `json:"stage_durations"`
}

// Clone creates a deep copy of the effective defaults
func (ed *EffectiveDefaults) Clone() *EffectiveDefaults {
	clone := &EffectiveDefaults{
		Simulation: ed.Simulation,
		Scoring:    ed.Scoring,
		Resources:  ed.Resources,
		Limits:     ed.Limits,
		NodeTypes:  make(map[string]NodeTypeDefault),
		QPSMultipliers: make(map[string]float64),
		LatencyBuckets: make([]float64, len(ed.LatencyBuckets)),
		FailureRates:   make(map[string]float64),
		StageDurations: make(map[string]time.Duration),
	}
	
	// Copy maps
	for k, v := range ed.NodeTypes {
		clone.NodeTypes[k] = v
	}
	for k, v := range ed.QPSMultipliers {
		clone.QPSMultipliers[k] = v
	}
	for k, v := range ed.FailureRates {
		clone.FailureRates[k] = v
	}
	for k, v := range ed.StageDurations {
		clone.StageDurations[k] = v
	}
	
	// Copy slice
	copy(clone.LatencyBuckets, ed.LatencyBuckets)
	
	return clone
}