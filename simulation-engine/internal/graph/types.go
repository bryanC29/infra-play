package graph

import (
	"fmt"
	"simengine/internal/models"
	"strings"
)

// DAG represents a directed acyclic graph built from a system design
type DAG struct {
	Nodes     []*DAGNode            `json:"nodes"`
	Edges     []*DAGEdge            `json:"edges"`
	NodeMap   map[string]*DAGNode   `json:"node_map"`
	EdgeMap   map[string][]*DAGEdge `json:"edge_map"`
	EntryNode *DAGNode              `json:"entry_node"`
	ExitNode  *DAGNode              `json:"exit_node"`
}

// DAGNode represents a node in the DAG with computed properties
type DAGNode struct {
	ID          string        `json:"id"`
	Type        string        `json:"type"`
	Design      *models.Node  `json:"design"`      // Reference to original design node
	Index       int           `json:"index"`       // Position in original design
	InDegree    int           `json:"in_degree"`   // Number of incoming edges
	Incoming    []*DAGEdge    `json:"incoming"`    // Incoming edges
	Outgoing    []*DAGEdge    `json:"outgoing"`    // Outgoing edges
	IsEntry     bool          `json:"is_entry"`    // True if this is the entry point
	IsExit      bool          `json:"is_exit"`     // True if this is the exit point
	IsSpecial   bool          `json:"is_special"`  // True for entry/exit nodes
	Capacity    float64       `json:"capacity"`    // Maximum QPS this node can handle
	BaseLatency float64       `json:"base_latency"` // Base latency in milliseconds
	Depth       int           `json:"depth"`       // Distance from entry node
}

// DAGEdge represents a directed edge between two nodes
type DAGEdge struct {
	ID     string   `json:"id"`
	From   *DAGNode `json:"from"`
	To     *DAGNode `json:"to"`
	Weight float64  `json:"weight"` // Edge weight/cost
}

// Path represents a sequence of nodes from entry to exit
type Path struct {
	Nodes        []*DAGNode `json:"nodes"`
	TotalWeight  float64    `json:"total_weight"`
	TotalLatency float64    `json:"total_latency"`
	MinCapacity  float64    `json:"min_capacity"`  // Bottleneck capacity
	Reliability  float64    `json:"reliability"`   // Path reliability (0-1)
	Length       int        `json:"length"`        // Number of nodes in path
}

// GraphMetrics contains computed metrics about the DAG structure
type GraphMetrics struct {
	TotalNodes        int     `json:"total_nodes"`
	FunctionalNodes   int     `json:"functional_nodes"`
	TotalEdges        int     `json:"total_edges"`
	MaxDepth          int     `json:"max_depth"`
	MinDepth          int     `json:"min_depth"`
	AvgDepth          float64 `json:"avg_depth"`
	Complexity        float64 `json:"complexity"`        // Graph complexity score
	Connectivity      float64 `json:"connectivity"`      // How well connected the graph is
	Redundancy        float64 `json:"redundancy"`        // Number of alternative paths
	CriticalNodes     int     `json:"critical_nodes"`    // Nodes that are single points of failure
	ParallelismFactor float64 `json:"parallelism_factor"` // Potential for parallel processing
}

// NodeType constants for different infrastructure components
const (
	NodeTypeEntryPoint   = "EntryPoint"
	NodeTypeExitPoint    = "ExitPoint"
	NodeTypeAPIGateway   = "APIGateway"
	NodeTypeLoadBalancer = "LoadBalancer"
	NodeTypeDatabase     = "Database"
	NodeTypeCache        = "Cache"
	NodeTypeMessageQueue = "MessageQueue"
	NodeTypeService      = "Service"
	NodeTypeCDN          = "CDN"
	NodeTypeProxy        = "Proxy"
	NodeTypeFirewall     = "Firewall"
	NodeTypeMonitoring   = "Monitoring"
)

// EdgeType constants for different connection types
const (
	EdgeTypeHTTP      = "HTTP"
	EdgeTypeDatabase  = "Database"
	EdgeTypeCache     = "Cache"
	EdgeTypeMessage   = "Message"
	EdgeTypeInternal  = "Internal"
	EdgeTypeExternal  = "External"
)

// PathType constants for categorizing paths
type PathType string

const (
	PathTypeCritical  = "critical"   // Longest path (determines system latency)
	PathTypeShortest  = "shortest"   // Shortest path by weight
	PathTypeOptimal   = "optimal"    // Best balance of latency and capacity
	PathTypeBackup    = "backup"     // Alternative/redundant path
)

// NodeCategory groups node types by their architectural role
type NodeCategory string

const (
	CategoryGateway    NodeCategory = "gateway"    // Entry points and load balancers
	CategoryCompute    NodeCategory = "compute"    // Services and processing nodes
	CategoryStorage    NodeCategory = "storage"    // Databases and caches
	CategoryNetwork    NodeCategory = "network"    // Proxies, CDNs, firewalls
	CategoryObservability NodeCategory = "observability" // Monitoring and logging
	CategoryMessaging  NodeCategory = "messaging"  // Message queues and brokers
	CategorySpecial    NodeCategory = "special"    // Entry and exit points
)

// NodeState represents the runtime state of a node during simulation
type NodeState struct {
	NodeID            string  `json:"node_id"`
	IsHealthy         bool    `json:"is_healthy"`
	CurrentLoad       float64 `json:"current_load"`       // Current QPS load
	MaxCapacity       float64 `json:"max_capacity"`       // Maximum sustainable QPS
	UtilizationRate   float64 `json:"utilization_rate"`   // currentLoad / maxCapacity
	AverageLatency    float64 `json:"average_latency"`    // Current average latency
	RequestsProcessed int64   `json:"requests_processed"` // Total requests processed
	RequestsDropped   int64   `json:"requests_dropped"`   // Requests dropped due to overload
	IsSaturated       bool    `json:"is_saturated"`       // True if at capacity
	FailureCount      int     `json:"failure_count"`      // Number of failures
	LastFailureTime   int64   `json:"last_failure_time"`  // Timestamp of last failure
}

// Methods for DAG

// GetNodeByID returns a node by its ID
func (d *DAG) GetNodeByID(id string) (*DAGNode, bool) {
	node, exists := d.NodeMap[id]
	return node, exists
}

// GetEdgesFromNode returns all outgoing edges from a node
func (d *DAG) GetEdgesFromNode(nodeID string) []*DAGEdge {
	edges, exists := d.EdgeMap[nodeID]
	if !exists {
		return make([]*DAGEdge, 0)
	}
	return edges
}

// GetAllPaths returns all valid paths from entry to exit
func (d *DAG) GetAllPaths() ([]*Path, error) {
	traversal := NewTraversal(d)
	return traversal.GetAllPaths()
}

// GetCriticalPath returns the longest path (critical path)
func (d *DAG) GetCriticalPath() (*Path, error) {
	traversal := NewTraversal(d)
	return traversal.GetCriticalPath()
}

// GetShortestPath returns the shortest path between two nodes
func (d *DAG) GetShortestPath(fromID, toID string) (*Path, error) {
	traversal := NewTraversal(d)
	return traversal.GetShortestPath(fromID, toID)
}

// GetTopologicalSort returns nodes in topological order
func (d *DAG) GetTopologicalSort() ([]*DAGNode, error) {
	traversal := NewTraversal(d)
	return traversal.GetTopologicalSort()
}

// GetMetrics computes structural metrics for the DAG
func (d *DAG) GetMetrics() *GraphMetrics {
	metrics := &GraphMetrics{}
	
	// Basic counts
	metrics.TotalNodes = len(d.Nodes)
	metrics.TotalEdges = len(d.Edges)
	
	// Count functional nodes
	functionalCount := 0
	depthSum := 0
	minDepth := ^int(0) // Max int
	maxDepth := 0
	
	for _, node := range d.Nodes {
		if !node.IsSpecial {
			functionalCount++
		}
		
		// Update depth statistics
		if node.Depth < minDepth {
			minDepth = node.Depth
		}
		if node.Depth > maxDepth {
			maxDepth = node.Depth
		}
		depthSum += node.Depth
	}
	
	metrics.FunctionalNodes = functionalCount
	metrics.MaxDepth = maxDepth
	metrics.MinDepth = minDepth
	
	if len(d.Nodes) > 0 {
		metrics.AvgDepth = float64(depthSum) / float64(len(d.Nodes))
	}
	
	// Calculate complexity (based on nodes, edges, and depth variance)
	metrics.Complexity = d.calculateComplexity()
	
	// Calculate connectivity (how well connected the graph is)
	metrics.Connectivity = d.calculateConnectivity()
	
	// Calculate redundancy (alternative path availability)
	metrics.Redundancy = d.calculateRedundancy()
	
	// Count critical nodes (single points of failure)
	metrics.CriticalNodes = d.countCriticalNodes()
	
	// Calculate parallelism factor
	metrics.ParallelismFactor = d.calculateParallelismFactor()
	
	return metrics
}

// calculateComplexity computes a complexity score for the graph
func (d *DAG) calculateComplexity() float64 {
	if len(d.Nodes) <= 1 {
		return 0.0
	}
	
	// Factors contributing to complexity:
	// 1. Number of nodes and edges
	// 2. Depth variance
	// 3. Branching factor
	
	nodeCount := float64(len(d.Nodes))
	edgeCount := float64(len(d.Edges))
	
	// Edge density (0-1, where 1 is fully connected)
	maxPossibleEdges := nodeCount * (nodeCount - 1)
	edgeDensity := edgeCount / maxPossibleEdges
	
	// Branching factor variance
	branchingVariance := d.calculateBranchingVariance()
	
	// Combine factors (normalized to 0-10 scale)
	complexity := 0.0
	complexity += nodeCount * 0.1                    // Node contribution
	complexity += edgeCount * 0.05                   // Edge contribution
	complexity += edgeDensity * 2.0                  // Density contribution
	complexity += branchingVariance * 3.0            // Branching variance
	
	// Cap at 10.0
	if complexity > 10.0 {
		complexity = 10.0
	}
	
	return complexity
}

// calculateConnectivity measures how well connected the graph is
func (d *DAG) calculateConnectivity() float64 {
	if len(d.Nodes) <= 1 {
		return 1.0
	}
	
	totalNodes := float64(len(d.Nodes))
	totalEdges := float64(len(d.Edges))
	
	// Maximum possible edges in a DAG
	maxEdges := totalNodes * (totalNodes - 1) / 2
	
	if maxEdges == 0 {
		return 0.0
	}
	
	return totalEdges / maxEdges
}

// calculateRedundancy measures the availability of alternative paths
func (d *DAG) calculateRedundancy() float64 {
	paths, err := d.GetAllPaths()
	if err != nil || len(paths) == 0 {
		return 0.0
	}
	
	// Redundancy based on number of paths
	pathCount := float64(len(paths))
	
	// More paths = higher redundancy, but with diminishing returns
	redundancy := 1.0 - (1.0 / pathCount)
	
	// Cap at 0.95 (95% redundancy)
	if redundancy > 0.95 {
		redundancy = 0.95
	}
	
	return redundancy
}

// countCriticalNodes identifies nodes that are single points of failure
func (d *DAG) countCriticalNodes() int {
	criticalCount := 0
	
	for _, node := range d.Nodes {
		if node.IsSpecial {
			continue
		}
		
		// A node is critical if removing it would disconnect entry from exit
		if d.isNodeCritical(node) {
			criticalCount++
		}
	}
	
	return criticalCount
}

// isNodeCritical checks if a node is a single point of failure
func (d *DAG) isNodeCritical(node *DAGNode) bool {
	// Simple heuristic: if all paths go through this node, it's critical
	paths, err := d.GetAllPaths()
	if err != nil || len(paths) == 0 {
		return false
	}
	
	// Check if the node appears in all paths
	for _, path := range paths {
		foundInPath := false
		for _, pathNode := range path.Nodes {
			if pathNode.ID == node.ID {
				foundInPath = true
				break
			}
		}
		if !foundInPath {
			return false // Found a path that doesn't include this node
		}
	}
	
	return true // Node appears in all paths
}

// calculateParallelismFactor estimates the potential for parallel processing
func (d *DAG) calculateParallelismFactor() float64 {
	if len(d.Nodes) <= 2 {
		return 1.0
	}
	
	// Calculate the maximum width (parallelism) at any depth level
	depthCounts := make(map[int]int)
	
	for _, node := range d.Nodes {
		if !node.IsSpecial {
			depthCounts[node.Depth]++
		}
	}
	
	maxWidth := 0
	for _, count := range depthCounts {
		if count > maxWidth {
			maxWidth = count
		}
	}
	
	// Parallelism factor is the ratio of max width to total functional nodes
	functionalNodes := 0
	for _, node := range d.Nodes {
		if !node.IsSpecial {
			functionalNodes++
		}
	}
	
	if functionalNodes == 0 {
		return 1.0
	}
	
	return float64(maxWidth) / float64(functionalNodes)
}

// calculateBranchingVariance computes the variance in node branching factors
func (d *DAG) calculateBranchingVariance() float64 {
	if len(d.Nodes) == 0 {
		return 0.0
	}
	
	// Calculate branching factor for each node
	branchingFactors := make([]float64, 0, len(d.Nodes))
	totalBranching := 0.0
	
	for _, node := range d.Nodes {
		if !node.IsSpecial {
			branching := float64(len(node.Outgoing))
			branchingFactors = append(branchingFactors, branching)
			totalBranching += branching
		}
	}
	
	if len(branchingFactors) == 0 {
		return 0.0
	}
	
	// Calculate mean
	mean := totalBranching / float64(len(branchingFactors))
	
	// Calculate variance
	variance := 0.0
	for _, branching := range branchingFactors {
		diff := branching - mean
		variance += diff * diff
	}
	variance /= float64(len(branchingFactors))
	
	return variance
}

// Methods for DAGNode

// IsEntryPoint returns true if this node is the entry point
func (n *DAGNode) IsEntryPoint() bool {
	return n.IsEntry
}

// IsExitPoint returns true if this node is the exit point
func (n *DAGNode) IsExitPoint() bool {
	return n.IsExit
}

// IsSpecialNode returns true if this node is entry or exit
func (n *DAGNode) IsSpecialNode() bool {
	return n.IsSpecial
}

// GetCategory returns the architectural category of this node
func (n *DAGNode) GetCategory() NodeCategory {
	switch n.Type {
	case NodeTypeEntryPoint, NodeTypeExitPoint:
		return CategorySpecial
	case NodeTypeAPIGateway, NodeTypeLoadBalancer:
		return CategoryGateway
	case NodeTypeService:
		return CategoryCompute
	case NodeTypeDatabase, NodeTypeCache:
		return CategoryStorage
	case NodeTypeProxy, NodeTypeCDN, NodeTypeFirewall:
		return CategoryNetwork
	case NodeTypeMonitoring:
		return CategoryObservability
	case NodeTypeMessageQueue:
		return CategoryMessaging
	default:
		return CategoryCompute // Default fallback
	}
}

// GetIncomingNodeIDs returns IDs of all nodes that have edges to this node
func (n *DAGNode) GetIncomingNodeIDs() []string {
	nodeIDs := make([]string, 0, len(n.Incoming))
	for _, edge := range n.Incoming {
		nodeIDs = append(nodeIDs, edge.From.ID)
	}
	return nodeIDs
}

// GetOutgoingNodeIDs returns IDs of all nodes that this node has edges to
func (n *DAGNode) GetOutgoingNodeIDs() []string {
	nodeIDs := make([]string, 0, len(n.Outgoing))
	for _, edge := range n.Outgoing {
		nodeIDs = append(nodeIDs, edge.To.ID)
	}
	return nodeIDs
}

// Clone creates a deep copy of the DAGNode
func (n *DAGNode) Clone() *DAGNode {
	clone := &DAGNode{
		ID:          n.ID,
		Type:        n.Type,
		Index:       n.Index,
		InDegree:    n.InDegree,
		IsEntry:     n.IsEntry,
		IsExit:      n.IsExit,
		IsSpecial:   n.IsSpecial,
		Capacity:    n.Capacity,
		BaseLatency: n.BaseLatency,
		Depth:       n.Depth,
	}
	
	if n.Design != nil {
		clone.Design = n.Design.Clone()
	}
	
	// Note: Incoming and Outgoing edges are not cloned to avoid circular references
	// They should be reconstructed when cloning the entire DAG
	
	return clone
}

// String returns a string representation of the node
func (n *DAGNode) String() string {
	return fmt.Sprintf("DAGNode{ID: %s, Type: %s, Capacity: %.1f, Latency: %.2fms, InDegree: %d, OutDegree: %d}",
		n.ID, n.Type, n.Capacity, n.BaseLatency, n.InDegree, len(n.Outgoing))
}

// Methods for DAGEdge

// GetOppositeNode returns the node on the other end of this edge
func (e *DAGEdge) GetOppositeNode(node *DAGNode) *DAGNode {
	if e.From.ID == node.ID {
		return e.To
	}
	if e.To.ID == node.ID {
		return e.From
	}
	return nil
}

// String returns a string representation of the edge
func (e *DAGEdge) String() string {
	return fmt.Sprintf("DAGEdge{%s -> %s, Weight: %.2f}", e.From.ID, e.To.ID, e.Weight)
}

// Methods for Path

// NodeIDs returns the IDs of all nodes in the path
func (p *Path) NodeIDs() []string {
	ids := make([]string, 0, len(p.Nodes))
	for _, node := range p.Nodes {
		ids = append(ids, node.ID)
	}
	return ids
}

// ContainsNode returns true if the path contains the specified node
func (p *Path) ContainsNode(nodeID string) bool {
	for _, node := range p.Nodes {
		if node.ID == nodeID {
			return true
		}
	}
	return false
}

// GetBottleneckNode returns the node with the lowest capacity in the path
func (p *Path) GetBottleneckNode() *DAGNode {
	if len(p.Nodes) == 0 {
		return nil
	}
	
	var bottleneck *DAGNode
	minCapacity := float64(^uint(0) >> 1) // Max float64
	
	for _, node := range p.Nodes {
		if !node.IsSpecial && node.Capacity < minCapacity {
			minCapacity = node.Capacity
			bottleneck = node
		}
	}
	
	return bottleneck
}

// GetPathType categorizes the path based on its characteristics
func (p *Path) GetPathType() PathType {
	// This is a simplified categorization - in practice, this would be
	// determined by comparing with other paths in the system
	
	if p.Length <= 3 {
		return PathTypeShortest
	}
	
	if p.TotalLatency > 100.0 {
		return PathTypeCritical
	}
	
	if p.MinCapacity > 1000.0 {
		return PathTypeOptimal
	}
	
	return PathTypeBackup
}

// String returns a string representation of the path
func (p *Path) String() string {
	nodeIDs := p.NodeIDs()
	return fmt.Sprintf("Path{%s, Length: %d, Weight: %.2f, Latency: %.2fms, MinCapacity: %.1f}",
		strings.Join(nodeIDs, " -> "), p.Length, p.TotalWeight, p.TotalLatency, p.MinCapacity)
}

// Clone creates a deep copy of the path
func (p *Path) Clone() *Path {
	clone := &Path{
		TotalWeight:  p.TotalWeight,
		TotalLatency: p.TotalLatency,
		MinCapacity:  p.MinCapacity,
		Reliability:  p.Reliability,
		Length:       p.Length,
	}
	
	// Deep copy nodes array
	clone.Nodes = make([]*DAGNode, len(p.Nodes))
	for i, node := range p.Nodes {
		clone.Nodes[i] = node.Clone()
	}
	
	return clone
}

// Methods for NodeState

// GetUtilizationPercentage returns utilization as a percentage (0-100)
func (ns *NodeState) GetUtilizationPercentage() float64 {
	return ns.UtilizationRate * 100.0
}

// IsOverloaded returns true if the node is operating above safe capacity
func (ns *NodeState) IsOverloaded() bool {
	return ns.UtilizationRate > 0.8 // 80% threshold
}

// GetSuccessRate returns the success rate (0-1) based on processed vs dropped requests
func (ns *NodeState) GetSuccessRate() float64 {
	total := ns.RequestsProcessed + ns.RequestsDropped
	if total == 0 {
		return 1.0
	}
	return float64(ns.RequestsProcessed) / float64(total)
}

// String returns a string representation of the node state
func (ns *NodeState) String() string {
	return fmt.Sprintf("NodeState{ID: %s, Healthy: %t, Load: %.1f/%.1f (%.1f%%), Requests: %d/%d}",
		ns.NodeID, ns.IsHealthy, ns.CurrentLoad, ns.MaxCapacity, 
		ns.GetUtilizationPercentage(), ns.RequestsProcessed, ns.RequestsDropped)
}