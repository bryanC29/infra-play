package models

// Problem defines a system design challenge with constraints and objectives.
type Problem struct {
	ID                string   `json:"id"`                  // Unique problem ID
	Title             string   `json:"title"`               // Human-readable title
	Description       string   `json:"description"`         // Markdown or plain-text
	TargetQPS         int      `json:"targetQps"`           // Simulated traffic
	EntryPoint        string   `json:"entryPoint"`          // Root node (e.g., load balancer ID)
	RequiredServices  []string `json:"requiredServices"`    // Nodes that must be reachable
	MaxCPU            int      `json:"maxCpu"`              // Max total vCPUs allowed
	MaxMemoryMB       int      `json:"maxMemoryMb"`         // Max total RAM in MB
	SLA               SLA      `json:"sla"`                 // Latency & availability targets
	FailureScenarios  []string `json:"failureScenarios"`    // Optional node IDs to simulate as down
}

// SLA defines service level expectations.
type SLA struct {
	MaxAvgLatencyMs   int     `json:"maxAvgLatencyMs"`   // e.g. 200ms
	MinAvailabilityRate float64 `json:"minAvailabilityRate"` // e.g. 0.999
}
