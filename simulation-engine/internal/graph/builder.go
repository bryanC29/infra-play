package graph

import (
	"fmt"
	"simengine/internal/models"
	"sort"
	"strings"
)

// Builder constructs a DAG from a system design
type Builder struct {
	design *models.SystemDesign
	nodes  map[string]*DAGNode
	edges  map[string][]*DAGEdge
}

// BuildResult contains the result of building a DAG
type BuildResult struct {
	DAG    *DAG
	Errors []error
	Stats  *BuildStats
}

// BuildStats provides statistics about the DAG construction process
type BuildStats struct {
	TotalNodes       int
	FunctionalNodes  int
	TotalEdges       int
	EntryPoints      int
	ExitPoints       int
	MaxDepth         int
	CyclicEdges      []string
	OrphanedNodes    []string
	DeadEndNodes     []string
	ValidationIssues []string
}

// NewBuilder creates a new DAG builder
func NewBuilder(design *models.SystemDesign) *Builder {
	return &Builder{
		design: design,
		nodes:  make(map[string]*DAGNode),
		edges:  make(map[string][]*DAGEdge),
	}
}

// BuildDAG constructs a DAG from the system design with comprehensive validation
func BuildDAG(design *models.SystemDesign) (*DAG, error) {
	if design == nil {
		return nil, fmt.Errorf("system design cannot be nil")
	}

	builder := NewBuilder(design)
	result := builder.Build()

	if len(result.Errors) > 0 {
		return nil, fmt.Errorf("DAG construction failed: %v", result.Errors[0])
	}

	return result.DAG, nil
}

// Build performs the complete DAG construction process
func (b *Builder) Build() *BuildResult {
	result := &BuildResult{
		Stats: &BuildStats{},
	}

	// Step 1: Validate basic structure
	if err := b.validateBasicStructure(); err != nil {
		result.Errors = append(result.Errors, err)
		return result
	}

	// Step 2: Create nodes
	if err := b.createNodes(); err != nil {
		result.Errors = append(result.Errors, err)
		return result
	}

	// Step 3: Create edges
	if err := b.createEdges(); err != nil {
		result.Errors = append(result.Errors, err)
		return result
	}

	// Step 4: Validate topology
	if err := b.validateTopology(); err != nil {
		result.Errors = append(result.Errors, err)
		return result
	}

	// Step 5: Detect cycles
	cycles := b.detectCycles()
	if len(cycles) > 0 {
		result.Errors = append(result.Errors, fmt.Errorf("cycles detected in graph: %v", cycles))
		return result
	}

	// Step 6: Validate reachability
	if err := b.validateReachability(); err != nil {
		result.Errors = append(result.Errors, err)
		return result
	}

	// Step 7: Build final DAG
	dag, err := b.constructDAG()
	if err != nil {
		result.Errors = append(result.Errors, err)
		return result
	}

	// Step 8: Generate statistics
	result.Stats = b.generateStats(cycles)
	result.DAG = dag

	return result
}

// validateBasicStructure performs initial validation of the system design
func (b *Builder) validateBasicStructure() error {
	if len(b.design.Nodes) < 2 {
		return fmt.Errorf("system design must have at least 2 nodes (entry and exit)")
	}

	if len(b.design.Connections) == 0 {
		return fmt.Errorf("system design must have at least 1 connection")
	}

	// Check for required entry and exit points
	hasEntry, hasExit := false, false
	for _, node := range b.design.Nodes {
		if node.IsEntryPoint() {
			hasEntry = true
		}
		if node.IsExitPoint() {
			hasExit = true
		}
	}

	if !hasEntry {
		return fmt.Errorf("system design must have exactly one entry point")
	}
	if !hasExit {
		return fmt.Errorf("system design must have exactly one exit point")
	}

	// Validate node IDs are unique
	nodeIDs := make(map[string]bool)
	for _, node := range b.design.Nodes {
		if node.ID == "" {
			return fmt.Errorf("node ID cannot be empty")
		}
		if nodeIDs[node.ID] {
			return fmt.Errorf("duplicate node ID: %s", node.ID)
		}
		nodeIDs[node.ID] = true
	}

	return nil
}

// createNodes creates DAG nodes from the system design nodes
func (b *Builder) createNodes() error {
	for i, designNode := range b.design.Nodes {
		// Validate node
		if err := designNode.ValidateResources(); err != nil {
			return fmt.Errorf("invalid node %s: %w", designNode.ID, err)
		}

		// Create DAG node
		dagNode := &DAGNode{
			ID:       designNode.ID,
			Type:     designNode.Type,
			Design:   designNode.Clone(),
			Index:    i,
			InDegree: 0,
			Incoming: make([]*DAGEdge, 0),
			Outgoing: make([]*DAGEdge, 0),
		}

		// Set node properties
		dagNode.IsEntry = designNode.IsEntryPoint()
		dagNode.IsExit = designNode.IsExitPoint()
		dagNode.IsSpecial = designNode.IsSpecialNode()

		// Calculate node capacity and performance characteristics
		if !dagNode.IsSpecial {
			dagNode.Capacity = b.calculateNodeCapacity(&designNode)
			dagNode.BaseLatency = b.calculateBaseLatency(&designNode)
		}

		b.nodes[designNode.ID] = dagNode
	}

	return nil
}

// createEdges creates DAG edges from the system design connections
func (b *Builder) createEdges() error {
	for i, conn := range b.design.Connections {
		// Validate connection references
		fromNode, fromExists := b.nodes[conn.From]
		toNode, toExists := b.nodes[conn.To]

		if !fromExists {
			return fmt.Errorf("connection references non-existent source node: %s", conn.From)
		}
		if !toExists {
			return fmt.Errorf("connection references non-existent target node: %s", conn.To)
		}

		// Prevent self-loops
		if conn.From == conn.To {
			return fmt.Errorf("self-loops are not allowed: %s -> %s", conn.From, conn.To)
		}

		// Create edge
		edge := &DAGEdge{
			ID:     fmt.Sprintf("edge_%d", i),
			From:   fromNode,
			To:     toNode,
			Weight: b.calculateEdgeWeight(fromNode, toNode),
		}

		// Add edge to nodes
		fromNode.Outgoing = append(fromNode.Outgoing, edge)
		toNode.Incoming = append(toNode.Incoming, edge)
		toNode.InDegree++

		// Store edge for lookup
		if b.edges[fromNode.ID] == nil {
			b.edges[fromNode.ID] = make([]*DAGEdge, 0)
		}
		b.edges[fromNode.ID] = append(b.edges[fromNode.ID], edge)
	}

	return nil
}

// validateTopology validates the overall topology of the graph
func (b *Builder) validateTopology() error {
	entryNode, err := b.getEntryNode()
	if err != nil {
		return err
	}

	exitNode, err := b.getExitNode()
	if err != nil {
		return err
	}

	// Entry node should have no incoming edges
	if len(entryNode.Incoming) > 0 {
		return fmt.Errorf("entry node %s should not have incoming connections", entryNode.ID)
	}

	// Exit node should have no outgoing edges
	if len(exitNode.Outgoing) > 0 {
		return fmt.Errorf("exit node %s should not have outgoing connections", exitNode.ID)
	}

	// Entry node must have at least one outgoing edge
	if len(entryNode.Outgoing) == 0 {
		return fmt.Errorf("entry node %s must have at least one outgoing connection", entryNode.ID)
	}

	// Exit node must have at least one incoming edge
	if len(exitNode.Incoming) == 0 {
		return fmt.Errorf("exit node %s must have at least one incoming connection", exitNode.ID)
	}

	// Check for orphaned nodes (nodes with no connections)
	for _, node := range b.nodes {
		if !node.IsSpecial && len(node.Incoming) == 0 && len(node.Outgoing) == 0 {
			return fmt.Errorf("orphaned node detected: %s has no connections", node.ID)
		}
	}

	return nil
}

// detectCycles uses DFS to detect cycles in the graph
func (b *Builder) detectCycles() []string {
	white := make(map[string]bool) // Unvisited
	gray := make(map[string]bool)  // Currently being processed
	black := make(map[string]bool) // Completely processed
	var cycles []string

	// Initialize all nodes as white
	for nodeID := range b.nodes {
		white[nodeID] = true
	}

	// DFS from each unvisited node
	for nodeID := range white {
		if white[nodeID] {
			path := make([]string, 0)
			if b.dfsDetectCycle(nodeID, white, gray, black, path, &cycles) {
				break // Found at least one cycle
			}
		}
	}

	return cycles
}

// dfsDetectCycle performs DFS cycle detection
func (b *Builder) dfsDetectCycle(nodeID string, white, gray, black map[string]bool, path []string, cycles *[]string) bool {
	// Move from white to gray
	white[nodeID] = false
	gray[nodeID] = true
	path = append(path, nodeID)

	// Visit all adjacent nodes
	for _, edge := range b.edges[nodeID] {
		adjNodeID := edge.To.ID

		if gray[adjNodeID] {
			// Back edge found - cycle detected
			cycleStart := -1
			for i, id := range path {
				if id == adjNodeID {
					cycleStart = i
					break
				}
			}
			if cycleStart >= 0 {
				cycle := append(path[cycleStart:], adjNodeID)
				*cycles = append(*cycles, strings.Join(cycle, " -> "))
				return true
			}
		} else if white[adjNodeID] {
			if b.dfsDetectCycle(adjNodeID, white, gray, black, path, cycles) {
				return true
			}
		}
	}

	// Move from gray to black
	gray[nodeID] = false
	black[nodeID] = true
	return false
}

// validateReachability ensures all nodes are reachable from entry and can reach exit
func (b *Builder) validateReachability() error {
	entryNode, err := b.getEntryNode()
	if err != nil {
		return err
	}

	exitNode, err := b.getExitNode()
	if err != nil {
		return err
	}

	// Check forward reachability (from entry)
	forwardReachable := b.getReachableNodes(entryNode.ID, true)
	
	// Check backward reachability (to exit)
	backwardReachable := b.getReachableNodes(exitNode.ID, false)

	// Find unreachable nodes
	var unreachableFromEntry []string
	var unreachableToExit []string

	for nodeID, node := range b.nodes {
		if !node.IsSpecial {
			if !forwardReachable[nodeID] {
				unreachableFromEntry = append(unreachableFromEntry, nodeID)
			}
			if !backwardReachable[nodeID] {
				unreachableToExit = append(unreachableToExit, nodeID)
			}
		}
	}

	if len(unreachableFromEntry) > 0 {
		return fmt.Errorf("nodes unreachable from entry: %v", unreachableFromEntry)
	}

	if len(unreachableToExit) > 0 {
		return fmt.Errorf("nodes that cannot reach exit: %v", unreachableToExit)
	}

	return nil
}

// getReachableNodes performs BFS to find all reachable nodes
func (b *Builder) getReachableNodes(startNodeID string, forward bool) map[string]bool {
	visited := make(map[string]bool)
	queue := []string{startNodeID}
	visited[startNodeID] = true

	for len(queue) > 0 {
		currentID := queue[0]
		queue = queue[1:]

		var edges []*DAGEdge
		if forward {
			// Forward traversal - follow outgoing edges
			edges = b.edges[currentID]
		} else {
			// Backward traversal - follow incoming edges
			if node := b.nodes[currentID]; node != nil {
				edges = node.Incoming
			}
		}

		for _, edge := range edges {
			var nextID string
			if forward {
				nextID = edge.To.ID
			} else {
				nextID = edge.From.ID
			}

			if !visited[nextID] {
				visited[nextID] = true
				queue = append(queue, nextID)
			}
		}
	}

	return visited
}

// constructDAG builds the final DAG structure
func (b *Builder) constructDAG() (*DAG, error) {
	entryNode, err := b.getEntryNode()
	if err != nil {
		return nil, err
	}

	exitNode, err := b.getExitNode()
	if err != nil {
		return nil, err
	}

	// Convert maps to slices for the DAG
	nodes := make([]*DAGNode, 0, len(b.nodes))
	for _, node := range b.nodes {
		nodes = append(nodes, node)
	}

	// Sort nodes by index for consistent ordering
	sort.Slice(nodes, func(i, j int) bool {
		return nodes[i].Index < nodes[j].Index
	})

	edges := make([]*DAGEdge, 0)
	for _, edgeList := range b.edges {
		edges = append(edges, edgeList...)
	}

	dag := &DAG{
		Nodes:     nodes,
		Edges:     edges,
		NodeMap:   b.nodes,
		EdgeMap:   b.edges,
		EntryNode: entryNode,
		ExitNode:  exitNode,
	}

	return dag, nil
}

// generateStats creates comprehensive statistics about the built DAG
func (b *Builder) generateStats(cycles []string) *BuildStats {
	stats := &BuildStats{
		TotalNodes:      len(b.nodes),
		TotalEdges:      0,
		CyclicEdges:     cycles,
		OrphanedNodes:   make([]string, 0),
		DeadEndNodes:    make([]string, 0),
		ValidationIssues: make([]string, 0),
	}

	// Count edges and analyze nodes
	for nodeID, node := range b.nodes {
		stats.TotalEdges += len(b.edges[nodeID])

		if node.IsEntryPoint() {
			stats.EntryPoints++
		}
		if node.IsExitPoint() {
			stats.ExitPoints++
		}
		if !node.IsSpecial {
			stats.FunctionalNodes++
		}

		// Check for orphaned nodes
		if len(node.Incoming) == 0 && len(node.Outgoing) == 0 && !node.IsSpecial {
			stats.OrphanedNodes = append(stats.OrphanedNodes, nodeID)
		}

		// Check for dead-end nodes (except exit)
		if len(node.Outgoing) == 0 && !node.IsExit {
			stats.DeadEndNodes = append(stats.DeadEndNodes, nodeID)
		}
	}

	// Calculate maximum depth
	if stats.EntryPoints > 0 {
		entryNode, _ := b.getEntryNode()
		stats.MaxDepth = b.calculateMaxDepth(entryNode.ID, make(map[string]bool))
	}

	return stats
}

// calculateMaxDepth computes the maximum depth of the DAG
func (b *Builder) calculateMaxDepth(nodeID string, visited map[string]bool) int {
	if visited[nodeID] {
		return 0 // Avoid infinite recursion in case of cycles
	}

	visited[nodeID] = true
	maxDepth := 0

	for _, edge := range b.edges[nodeID] {
		depth := b.calculateMaxDepth(edge.To.ID, visited)
		if depth > maxDepth {
			maxDepth = depth
		}
	}

	delete(visited, nodeID) // Allow revisiting from different paths
	return maxDepth + 1
}

// Helper methods

// getEntryNode finds and returns the entry node
func (b *Builder) getEntryNode() (*DAGNode, error) {
	for _, node := range b.nodes {
		if node.IsEntry {
			return node, nil
		}
	}
	return nil, fmt.Errorf("entry node not found")
}

// getExitNode finds and returns the exit node
func (b *Builder) getExitNode() (*DAGNode, error) {
	for _, node := range b.nodes {
		if node.IsExit {
			return node, nil
		}
	}
	return nil, fmt.Errorf("exit node not found")
}

// calculateNodeCapacity computes the processing capacity of a node
func (b *Builder) calculateNodeCapacity(node *models.Node) float64 {
	// Base capacity from CPU
	cpuCapacity := node.GetCPU() * 100.0 // Base: 100 QPS per CPU core

	// Memory-based capacity (if specified)
	memoryMB := node.GetMemoryMB()
	memoryCapacity := float64(memoryMB) * 0.1 // Base: 0.1 QPS per MB

	// Use the more restrictive constraint
	baseCapacity := cpuCapacity
	if memoryMB > 0 && memoryCapacity < cpuCapacity {
		baseCapacity = memoryCapacity
	}

	// Scale by replicas
	replicas := float64(node.GetReplicas())
	return baseCapacity * replicas
}

// calculateBaseLatency computes the base latency for a node type
func (b *Builder) calculateBaseLatency(node *models.Node) float64 {
	// Base latencies by node type (in milliseconds)
	baseLatencies := map[string]float64{
		models.NodeTypeAPIGateway:   10.0,
		models.NodeTypeLoadBalancer: 2.0,
		models.NodeTypeService:      15.0,
		models.NodeTypeDatabase:     25.0,
		models.NodeTypeCache:        1.0,
		models.NodeTypeMessageQueue: 5.0,
		models.NodeTypeCDN:          50.0,
		models.NodeTypeProxy:        3.0,
		models.NodeTypeFirewall:     1.0,
		models.NodeTypeMonitoring:   2.0,
	}

	baseLatency, exists := baseLatencies[node.Type]
	if !exists {
		baseLatency = 10.0 // Default latency
	}

	// Adjust for resources
	cpuFactor := 2.0 / node.GetCPU() // Inversely proportional to CPU
	if cpuFactor > 5.0 {
		cpuFactor = 5.0 // Cap the factor
	}

	return baseLatency * cpuFactor
}

// calculateEdgeWeight computes the weight/cost of an edge
func (b *Builder) calculateEdgeWeight(from, to *DAGNode) float64 {
	// Simple weight calculation based on node types
	weight := 1.0

	// Add weight for network latency between different node types
	if from.Type != to.Type {
		weight += 0.5
	}

	// Add weight for special transitions
	if from.IsEntry || to.IsExit {
		weight = 0.1 // Entry/exit transitions are nearly free
	}

	return weight
}

// ValidateDAGIntegrity performs comprehensive validation of a constructed DAG
func ValidateDAGIntegrity(dag *DAG) error {
	if dag == nil {
		return fmt.Errorf("DAG cannot be nil")
	}

	// Check basic structure
	if len(dag.Nodes) == 0 {
		return fmt.Errorf("DAG must have at least one node")
	}

	if dag.EntryNode == nil {
		return fmt.Errorf("DAG must have an entry node")
	}

	if dag.ExitNode == nil {
		return fmt.Errorf("DAG must have an exit node")
	}

	// Validate node consistency
	for _, node := range dag.Nodes {
		if node == nil {
			return fmt.Errorf("DAG contains nil node")
		}

		// Check if node exists in node map
		if dag.NodeMap[node.ID] != node {
			return fmt.Errorf("node map inconsistency for node %s", node.ID)
		}

		// Validate edges
		for _, edge := range node.Outgoing {
			if edge.From != node {
				return fmt.Errorf("outgoing edge from %s has incorrect From reference", node.ID)
			}
		}

		for _, edge := range node.Incoming {
			if edge.To != node {
				return fmt.Errorf("incoming edge to %s has incorrect To reference", node.ID)
			}
		}
	}

	return nil
}