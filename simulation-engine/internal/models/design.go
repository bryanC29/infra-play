package models

import (
	"fmt"
	"reflect"
)

// SystemDesign represents the complete infrastructure design submitted by users
type SystemDesign struct {
	Nodes       []Node       `json:"nodes" validate:"required,min=2"`
	Connections []Connection `json:"connections" validate:"required,min=1"`
}

// Node represents a single infrastructure component in the system design
type Node struct {
	ID        string                 `json:"id" validate:"required"`
	Type      string                 `json:"type" validate:"required"`
	Resources map[string]interface{} `json:"resources"`
}

// Connection represents a directed edge between two nodes in the system design
type Connection struct {
	From string `json:"from" validate:"required"`
	To   string `json:"to" validate:"required"`
}

// NodeType constants for different infrastructure components
const (
	NodeTypeEntryPoint     = "EntryPoint"
	NodeTypeExitPoint      = "ExitPoint"
	NodeTypeAPIGateway     = "APIGateway"
	NodeTypeLoadBalancer   = "LoadBalancer"
	NodeTypeDatabase       = "Database"
	NodeTypeCache          = "Cache"
	NodeTypeMessageQueue   = "MessageQueue"
	NodeTypeService        = "Service"
	NodeTypeCDN            = "CDN"
	NodeTypeProxy          = "Proxy"
	NodeTypeFirewall       = "Firewall"
	NodeTypeMonitoring     = "Monitoring"
)

// IsSpecialNode returns true if the node is EntryPoint or ExitPoint
// These nodes are dummy nodes that don't contribute to simulation metrics
func (n *Node) IsSpecialNode() bool {
	return n.Type == NodeTypeEntryPoint || n.Type == NodeTypeExitPoint
}

// IsEntryPoint returns true if this is an entry point node
func (n *Node) IsEntryPoint() bool {
	return n.Type == NodeTypeEntryPoint
}

// IsExitPoint returns true if this is an exit point node
func (n *Node) IsExitPoint() bool {
	return n.Type == NodeTypeExitPoint
}

// GetCPU extracts CPU resource allocation from the node's resources
// Returns 0 if not specified or invalid
func (n *Node) GetCPU() float64 {
	if n.Resources == nil {
		return 0
	}
	
	cpu, exists := n.Resources["cpu"]
	if !exists {
		return 0
	}
	
	switch v := cpu.(type) {
	case float64:
		return v
	case int:
		return float64(v)
	case string:
		// Handle string numbers if needed
		return 0
	default:
		return 0
	}
}

// GetMemoryMB extracts memory allocation in MB from the node's resources
// Returns 0 if not specified or invalid
func (n *Node) GetMemoryMB() int {
	if n.Resources == nil {
		return 0
	}
	
	memory, exists := n.Resources["memoryMB"]
	if !exists {
		return 0
	}
	
	switch v := memory.(type) {
	case float64:
		return int(v)
	case int:
		return v
	default:
		return 0
	}
}

// GetReplicas extracts the number of replicas from the node's resources
// Returns 1 if not specified (default single instance)
func (n *Node) GetReplicas() int {
	if n.Resources == nil {
		return 1
	}
	
	replicas, exists := n.Resources["replicas"]
	if !exists {
		return 1
	}
	
	switch v := replicas.(type) {
	case float64:
		if v <= 0 {
			return 1
		}
		return int(v)
	case int:
		if v <= 0 {
			return 1
		}
		return v
	default:
		return 1
	}
}

// GetStorageGB extracts storage allocation in GB from the node's resources
// Returns 0 if not specified or invalid
func (n *Node) GetStorageGB() int {
	if n.Resources == nil {
		return 0
	}
	
	storage, exists := n.Resources["storageGB"]
	if !exists {
		return 0
	}
	
	switch v := storage.(type) {
	case float64:
		return int(v)
	case int:
		return v
	default:
		return 0
	}
}

// GetBandwidthMbps extracts bandwidth allocation in Mbps from the node's resources
// Returns 0 if not specified or invalid
func (n *Node) GetBandwidthMbps() int {
	if n.Resources == nil {
		return 0
	}
	
	bandwidth, exists := n.Resources["bandwidthMbps"]
	if !exists {
		return 0
	}
	
	switch v := bandwidth.(type) {
	case float64:
		return int(v)
	case int:
		return v
	default:
		return 0
	}
}

// HasResources returns true if the node has any resource specifications
func (n *Node) HasResources() bool {
	return n.Resources != nil && len(n.Resources) > 0
}

// GetResourceValue safely extracts a resource value with type assertion
func (n *Node) GetResourceValue(key string) (interface{}, bool) {
	if n.Resources == nil {
		return nil, false
	}
	
	value, exists := n.Resources[key]
	return value, exists
}

// SetResource safely sets a resource value
func (n *Node) SetResource(key string, value interface{}) {
	if n.Resources == nil {
		n.Resources = make(map[string]interface{})
	}
	n.Resources[key] = value
}

// ValidateResources performs basic validation on node resources
func (n *Node) ValidateResources() error {
	if n.IsSpecialNode() {
		// Entry and exit points should not have resources
		if n.HasResources() {
			return fmt.Errorf("node %s of type %s should not have resources", n.ID, n.Type)
		}
		return nil
	}
	
	// Non-special nodes should have at least CPU specification
	cpu := n.GetCPU()
	if cpu <= 0 {
		return fmt.Errorf("node %s must have positive CPU allocation", n.ID)
	}
	
	// Memory should be positive if specified
	memory := n.GetMemoryMB()
	if memory < 0 {
		return fmt.Errorf("node %s cannot have negative memory allocation", n.ID)
	}
	
	// Replicas should be positive
	replicas := n.GetReplicas()
	if replicas <= 0 {
		return fmt.Errorf("node %s must have at least 1 replica", n.ID)
	}
	
	return nil
}

// Clone creates a deep copy of the node
func (n *Node) Clone() *Node {
	clone := &Node{
		ID:   n.ID,
		Type: n.Type,
	}
	
	if n.Resources != nil {
		clone.Resources = make(map[string]interface{})
		for k, v := range n.Resources {
			// Deep copy the value
			clone.Resources[k] = cloneValue(v)
		}
	}
	
	return clone
}

// cloneValue performs deep copy of interface{} values
func cloneValue(v interface{}) interface{} {
	if v == nil {
		return nil
	}
	
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Map:
		m := make(map[string]interface{})
		for _, key := range rv.MapKeys() {
			m[key.String()] = cloneValue(rv.MapIndex(key).Interface())
		}
		return m
	case reflect.Slice:
		s := make([]interface{}, rv.Len())
		for i := 0; i < rv.Len(); i++ {
			s[i] = cloneValue(rv.Index(i).Interface())
		}
		return s
	default:
		return v
	}
}

// String returns a string representation of the node
func (n *Node) String() string {
	return fmt.Sprintf("Node{ID: %s, Type: %s, CPU: %.1f, Memory: %dMB, Replicas: %d}", 
		n.ID, n.Type, n.GetCPU(), n.GetMemoryMB(), n.GetReplicas())
}

// Equal checks if two nodes are equivalent
func (n *Node) Equal(other *Node) bool {
	if other == nil {
		return false
	}
	
	if n.ID != other.ID || n.Type != other.Type {
		return false
	}
	
	return reflect.DeepEqual(n.Resources, other.Resources)
}

// String returns a string representation of the connection
func (c *Connection) String() string {
	return fmt.Sprintf("Connection{From: %s, To: %s}", c.From, c.To)
}

// Equal checks if two connections are equivalent
func (c *Connection) Equal(other *Connection) bool {
	if other == nil {
		return false
	}
	return c.From == other.From && c.To == other.To
}

// GetNodeByID finds a node by its ID in the system design
func (sd *SystemDesign) GetNodeByID(id string) (*Node, bool) {
	for i := range sd.Nodes {
		if sd.Nodes[i].ID == id {
			return &sd.Nodes[i], true
		}
	}
	return nil, false
}

// GetEntryPoint finds the entry point node in the system design
func (sd *SystemDesign) GetEntryPoint() (*Node, error) {
	for i := range sd.Nodes {
		if sd.Nodes[i].IsEntryPoint() {
			return &sd.Nodes[i], nil
		}
	}
	return nil, fmt.Errorf("no entry point found in system design")
}

// GetExitPoint finds the exit point node in the system design
func (sd *SystemDesign) GetExitPoint() (*Node, error) {
	for i := range sd.Nodes {
		if sd.Nodes[i].IsExitPoint() {
			return &sd.Nodes[i], nil
		}
	}
	return nil, fmt.Errorf("no exit point found in system design")
}

// GetFunctionalNodes returns all nodes except entry and exit points
func (sd *SystemDesign) GetFunctionalNodes() []Node {
	var functional []Node
	for _, node := range sd.Nodes {
		if !node.IsSpecialNode() {
			functional = append(functional, node)
		}
	}
	return functional
}

// GetConnectionsFrom returns all connections originating from the specified node
func (sd *SystemDesign) GetConnectionsFrom(nodeID string) []Connection {
	var connections []Connection
	for _, conn := range sd.Connections {
		if conn.From == nodeID {
			connections = append(connections, conn)
		}
	}
	return connections
}

// GetConnectionsTo returns all connections terminating at the specified node
func (sd *SystemDesign) GetConnectionsTo(nodeID string) []Connection {
	var connections []Connection
	for _, conn := range sd.Connections {
		if conn.To == nodeID {
			connections = append(connections, conn)
		}
	}
	return connections
}

// HasConnection checks if a direct connection exists between two nodes
func (sd *SystemDesign) HasConnection(from, to string) bool {
	for _, conn := range sd.Connections {
		if conn.From == from && conn.To == to {
			return true
		}
	}
	return false
}

// GetAllNodeIDs returns a slice of all node IDs in the system design
func (sd *SystemDesign) GetAllNodeIDs() []string {
	ids := make([]string, len(sd.Nodes))
	for i, node := range sd.Nodes {
		ids[i] = node.ID
	}
	return ids
}

// NodeCount returns the total number of nodes in the system design
func (sd *SystemDesign) NodeCount() int {
	return len(sd.Nodes)
}

// FunctionalNodeCount returns the number of functional nodes (excluding entry/exit)
func (sd *SystemDesign) FunctionalNodeCount() int {
	return len(sd.GetFunctionalNodes())
}

// ConnectionCount returns the total number of connections in the system design
func (sd *SystemDesign) ConnectionCount() int {
	return len(sd.Connections)
}

// Clone creates a deep copy of the system design
func (sd *SystemDesign) Clone() *SystemDesign {
	clone := &SystemDesign{
		Nodes:       make([]Node, len(sd.Nodes)),
		Connections: make([]Connection, len(sd.Connections)),
	}
	
	// Deep copy nodes
	for i, node := range sd.Nodes {
		clone.Nodes[i] = *node.Clone()
	}
	
	// Copy connections (shallow copy is sufficient)
	copy(clone.Connections, sd.Connections)
	
	return clone
}

// String returns a string representation of the system design
func (sd *SystemDesign) String() string {
	return fmt.Sprintf("SystemDesign{Nodes: %d, Connections: %d, Functional: %d}", 
		sd.NodeCount(), sd.ConnectionCount(), sd.FunctionalNodeCount())
}