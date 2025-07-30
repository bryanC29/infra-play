package engine

import "time"

// NodeProfile defines default capabilities for each node type.
type NodeProfile struct {
	MaxQPSPerCPU   int     // Max QPS that 1 vCPU can handle
	MemoryPerQPSMB float64 // Memory (MB) per QPS
	CostPerHour    float64 // Cost in $ per replica per hour
}

// SLAConfig holds global SLA thresholds for simulations.
type SLAConfig struct {
	MaxAvgLatencyMs     int     // Acceptable average latency
	MinAvailabilityRate float64 // e.g. 0.999 for 99.9%
}

// SystemConfig aggregates all static configuration.
type SystemConfig struct {
	NodeDefaults map[string]NodeProfile // key: node type (e.g. "api", "db")
	SLA          SLAConfig
}

// LoadDefaultConfig initializes all config values for the simulation engine.
func LoadDefaultConfig() SystemConfig {
	return SystemConfig{
		NodeDefaults: map[string]NodeProfile{
			"api": {
				MaxQPSPerCPU:   500,
				MemoryPerQPSMB: 0.2,
				CostPerHour:    0.05,
			},
			"db": {
				MaxQPSPerCPU:   1000,
				MemoryPerQPSMB: 1.0,
				CostPerHour:    0.20,
			},
			"cache": {
				MaxQPSPerCPU:   5000,
				MemoryPerQPSMB: 0.1,
				CostPerHour:    0.10,
			},
			"lb": {
				MaxQPSPerCPU:   10000,
				MemoryPerQPSMB: 0.05,
				CostPerHour:    0.03,
			},
			"queue": {
				MaxQPSPerCPU:   2000,
				MemoryPerQPSMB: 0.15,
				CostPerHour:    0.06,
			},
		},
		SLA: SLAConfig{
			MaxAvgLatencyMs:     200,
			MinAvailabilityRate: 0.999,
		},
	}
}

// SimulationDefaults
const (
	RequestTimeout       = 2 * time.Second
	SimulationTimeWindow = 60 * time.Second // Total duration for simulation
)
