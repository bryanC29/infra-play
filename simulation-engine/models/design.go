package models

// Design represents a complete user-submitted system design.
type Design struct {
	Nodes []Node `json:"nodes"`
	Edges []Edge `json:"edges"`
}

// Node represents a component in the system (e.g., API, DB, Cache).
type Node struct {
	ID     string     `json:"id"`     // Unique identifier
	Type   string     `json:"type"`   // e.g., "api", "db", "cache", etc.
	Config NodeConfig `json:"config"` // Resource & replica config
}

// NodeConfig describes the deployment configuration for a node.
type NodeConfig struct {
	VCPU     int `json:"vcpu"`     // CPU per replica
	MemoryMB int `json:"memoryMb"` // Memory per replica in MB
	Replicas int `json:"replicas"` // Number of instances
}

// Edge represents a connection between two nodes.
type Edge struct {
	From string `json:"from"` // Source node ID
	To   string `json:"to"`   // Destination node ID
}

// Graph is an internal representation of the design with fast lookup.
type Graph struct {
	Nodes map[string]*Node   // NodeID → Node pointer
	Edges map[string][]*Node // NodeID → List of downstream nodes
}
