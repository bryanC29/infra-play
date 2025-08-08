package graph

import (
	"fmt"
	"sort"
	"strings"
)

// Traversal provides algorithms for DAG traversal and path analysis
type Traversal struct {
	dag   *DAG
	cache *TraversalCache
}

// TraversalCache caches computed results for performance optimization
type TraversalCache struct {
	allPaths        []*Path
	pathsComputed   bool
	topologicalSort []*DAGNode
	sortComputed    bool
	criticalPath    *Path
	criticalComputed bool
	shortestPaths   map[string]map[string]*Path
}

// TraversalResult contains results from path analysis
type TraversalResult struct {
	AllPaths        []*Path
	CriticalPath    *Path
	ShortestPath    *Path
	LongestPath     *Path
	TotalPaths      int
	MaxPathLength   int
	MinPathLength   int
	AvgPathLength   float64
	PathComplexity  float64
	Bottlenecks     []*BottleneckInfo
}

// BottleneckInfo identifies potential bottlenecks in the system
type BottleneckInfo struct {
	NodeID          string  `json:"node_id"`
	PathsThrough    int     `json:"paths_through"`
	PathPercentage  float64 `json:"path_percentage"`
	TotalCapacity   float64 `json:"total_capacity"`
	ExpectedLoad    float64 `json:"expected_load"`
	UtilizationRate float64 `json:"utilization_rate"`
	BottleneckScore float64 `json:"bottleneck_score"`
	Severity        string  `json:"severity"`
}

// PathFinder provides advanced path finding capabilities
type PathFinder struct {
	dag    *DAG
	config *PathFinderConfig
}

// PathFinderConfig configures path finding behavior
type PathFinderConfig struct {
	MaxPaths           int     // Maximum number of paths to return
	MaxPathLength      int     // Maximum allowed path length
	IncludeRedundant   bool    // Include redundant paths
	WeightThreshold    float64 // Minimum weight threshold for paths
	OptimizeFor        string  // "latency", "throughput", "reliability"
}

// NewTraversal creates a new traversal instance
func NewTraversal(dag *DAG) *Traversal {
	return &Traversal{
		dag: dag,
		cache: &TraversalCache{
			shortestPaths: make(map[string]map[string]*Path),
		},
	}
}

// GetAllPaths returns all valid paths from entry to exit
func (t *Traversal) GetAllPaths() ([]*Path, error) {
	if t.cache.pathsComputed {
		return t.cache.allPaths, nil
	}

	if t.dag.EntryNode == nil || t.dag.ExitNode == nil {
		return nil, fmt.Errorf("DAG must have both entry and exit nodes")
	}

	paths := make([]*Path, 0)
	visited := make(map[string]bool)
	currentPath := make([]*DAGNode, 0)

	// Start DFS from entry node
	t.dfsAllPaths(t.dag.EntryNode, t.dag.ExitNode, visited, currentPath, &paths)

	if len(paths) == 0 {
		return nil, fmt.Errorf("no valid paths found from entry to exit")
	}

	// Sort paths by weight for consistent ordering
	sort.Slice(paths, func(i, j int) bool {
		return paths[i].TotalWeight < paths[j].TotalWeight
	})

	t.cache.allPaths = paths
	t.cache.pathsComputed = true

	return paths, nil
}

// dfsAllPaths performs DFS to find all paths from start to end
func (t *Traversal) dfsAllPaths(current, target *DAGNode, visited map[string]bool, path []*DAGNode, paths *[]*Path) {
	// Add current node to path
	path = append(path, current)
	visited[current.ID] = true

	// If we reached the target, create a path
	if current.ID == target.ID {
		pathCopy := make([]*DAGNode, len(path))
		copy(pathCopy, path)
		
		newPath := &Path{
			Nodes:       pathCopy,
			TotalWeight: t.calculatePathWeight(pathCopy),
			Length:      len(pathCopy),
		}
		newPath.TotalLatency = t.calculatePathLatency(pathCopy)
		newPath.MinCapacity = t.calculatePathMinCapacity(pathCopy)
		newPath.Reliability = t.calculatePathReliability(pathCopy)

		*paths = append(*paths, newPath)
	} else {
		// Continue DFS to adjacent nodes
		for _, edge := range current.Outgoing {
			if !visited[edge.To.ID] {
				t.dfsAllPaths(edge.To, target, visited, path, paths)
			}
		}
	}

	// Backtrack
	visited[current.ID] = false
}

// GetTopologicalSort returns nodes in topological order
func (t *Traversal) GetTopologicalSort() ([]*DAGNode, error) {
	if t.cache.sortComputed {
		return t.cache.topologicalSort, nil
	}

	// Kahn's algorithm for topological sorting
	inDegree := make(map[string]int)
	for _, node := range t.dag.Nodes {
		inDegree[node.ID] = len(node.Incoming)
	}

	// Initialize queue with nodes having zero in-degree
	queue := make([]*DAGNode, 0)
	for _, node := range t.dag.Nodes {
		if inDegree[node.ID] == 0 {
			queue = append(queue, node)
		}
	}

	result := make([]*DAGNode, 0)
	
	for len(queue) > 0 {
		// Remove node from queue
		current := queue[0]
		queue = queue[1:]
		result = append(result, current)

		// Process all outgoing edges
		for _, edge := range current.Outgoing {
			inDegree[edge.To.ID]--
			if inDegree[edge.To.ID] == 0 {
				queue = append(queue, edge.To)
			}
		}
	}

	// Check for cycles
	if len(result) != len(t.dag.Nodes) {
		return nil, fmt.Errorf("cycle detected in DAG - topological sort impossible")
	}

	t.cache.topologicalSort = result
	t.cache.sortComputed = true

	return result, nil
}

// GetCriticalPath finds the longest path (critical path) through the DAG
func (t *Traversal) GetCriticalPath() (*Path, error) {
	if t.cache.criticalComputed {
		return t.cache.criticalPath, nil
	}

	paths, err := t.GetAllPaths()
	if err != nil {
		return nil, fmt.Errorf("failed to get paths for critical path analysis: %w", err)
	}

	if len(paths) == 0 {
		return nil, fmt.Errorf("no paths available for critical path analysis")
	}

	// Find path with maximum latency (critical path)
	var criticalPath *Path
	maxLatency := 0.0

	for _, path := range paths {
		if path.TotalLatency > maxLatency {
			maxLatency = path.TotalLatency
			criticalPath = path
		}
	}

	t.cache.criticalPath = criticalPath
	t.cache.criticalComputed = true

	return criticalPath, nil
}

// GetShortestPath finds the shortest path between two nodes
func (t *Traversal) GetShortestPath(fromID, toID string) (*Path, error) {
	// Check cache first
	if t.cache.shortestPaths[fromID] != nil {
		if path := t.cache.shortestPaths[fromID][toID]; path != nil {
			return path, nil
		}
	}

	fromNode := t.dag.NodeMap[fromID]
	toNode := t.dag.NodeMap[toID]

	if fromNode == nil {
		return nil, fmt.Errorf("source node %s not found", fromID)
	}
	if toNode == nil {
		return nil, fmt.Errorf("target node %s not found", toID)
	}

	// Dijkstra's algorithm
	dist := make(map[string]float64)
	prev := make(map[string]*DAGNode)
	unvisited := make(map[string]bool)

	// Initialize distances
	for _, node := range t.dag.Nodes {
		dist[node.ID] = float64(^uint(0) >> 1) // Max float64
		unvisited[node.ID] = true
	}
	dist[fromID] = 0.0

	for len(unvisited) > 0 {
		// Find unvisited node with minimum distance
		currentID := ""
		minDist := float64(^uint(0) >> 1)
		for nodeID := range unvisited {
			if dist[nodeID] < minDist {
				minDist = dist[nodeID]
				currentID = nodeID
			}
		}

		if currentID == "" || minDist == float64(^uint(0) >> 1) {
			break // No more reachable nodes
		}

		delete(unvisited, currentID)
		current := t.dag.NodeMap[currentID]

		// Check if we reached the target
		if currentID == toID {
			break
		}

		// Update distances to neighbors
		for _, edge := range current.Outgoing {
			neighborID := edge.To.ID
			if !unvisited[neighborID] {
				continue
			}

			alt := dist[currentID] + edge.Weight
			if alt < dist[neighborID] {
				dist[neighborID] = alt
				prev[neighborID] = current
			}
		}
	}

	// Reconstruct path
	if dist[toID] == float64(^uint(0) >> 1) {
		return nil, fmt.Errorf("no path found from %s to %s", fromID, toID)
	}

	path := make([]*DAGNode, 0)
	current := toNode
	for current != nil {
		path = append([]*DAGNode{current}, path...)
		current = prev[current.ID]
	}

	shortestPath := &Path{
		Nodes:       path,
		TotalWeight: dist[toID],
		Length:      len(path),
	}
	shortestPath.TotalLatency = t.calculatePathLatency(path)
	shortestPath.MinCapacity = t.calculatePathMinCapacity(path)
	shortestPath.Reliability = t.calculatePathReliability(path)

	// Cache the result
	if t.cache.shortestPaths[fromID] == nil {
		t.cache.shortestPaths[fromID] = make(map[string]*Path)
	}
	t.cache.shortestPaths[fromID][toID] = shortestPath

	return shortestPath, nil
}

// AnalyzePaths performs comprehensive path analysis
func (t *Traversal) AnalyzePaths() (*TraversalResult, error) {
	paths, err := t.GetAllPaths()
	if err != nil {
		return nil, fmt.Errorf("failed to get paths for analysis: %w", err)
	}

	if len(paths) == 0 {
		return nil, fmt.Errorf("no paths available for analysis")
	}

	result := &TraversalResult{
		AllPaths:   paths,
		TotalPaths: len(paths),
	}

	// Find critical path (longest)
	result.CriticalPath, _ = t.GetCriticalPath()

	// Find shortest and longest paths by weight
	minWeight, maxWeight := paths[0].TotalWeight, paths[0].TotalWeight
	shortestIdx, longestIdx := 0, 0

	totalLength := 0
	for i, path := range paths {
		totalLength += path.Length
		
		if path.TotalWeight < minWeight {
			minWeight = path.TotalWeight
			shortestIdx = i
		}
		if path.TotalWeight > maxWeight {
			maxWeight = path.TotalWeight
			longestIdx = i
		}
		
		if path.Length > result.MaxPathLength {
			result.MaxPathLength = path.Length
		}
		if result.MinPathLength == 0 || path.Length < result.MinPathLength {
			result.MinPathLength = path.Length
		}
	}

	result.ShortestPath = paths[shortestIdx]
	result.LongestPath = paths[longestIdx]
	result.AvgPathLength = float64(totalLength) / float64(len(paths))

	// Calculate path complexity
	result.PathComplexity = t.calculatePathComplexity(paths)

	// Identify bottlenecks
	result.Bottlenecks = t.identifyBottlenecks(paths)

	return result, nil
}

// identifyBottlenecks analyzes paths to identify potential bottlenecks
func (t *Traversal) identifyBottlenecks(paths []*Path) []*BottleneckInfo {
	nodePathCount := make(map[string]int)
	nodeCapacity := make(map[string]float64)

	// Count how many paths go through each node
	for _, path := range paths {
		for _, node := range path.Nodes {
			if !node.IsSpecial {
				nodePathCount[node.ID]++
				nodeCapacity[node.ID] = node.Capacity
			}
		}
	}

	var bottlenecks []*BottleneckInfo
	totalPaths := len(paths)

	for nodeID, pathCount := range nodePathCount {
		if pathCount == 0 {
			continue
		}

		pathPercentage := float64(pathCount) / float64(totalPaths) * 100.0
		capacity := nodeCapacity[nodeID]
		
		// Estimate expected load (simplified)
		expectedLoad := float64(pathCount) * 100.0 // Assuming 100 QPS per path
		utilizationRate := 0.0
		if capacity > 0 {
			utilizationRate = expectedLoad / capacity
		}

		// Calculate bottleneck score
		bottleneckScore := pathPercentage * utilizationRate

		severity := "low"
		if bottleneckScore > 80.0 {
			severity = "critical"
		} else if bottleneckScore > 60.0 {
			severity = "high"
		} else if bottleneckScore > 40.0 {
			severity = "medium"
		}

		bottleneck := &BottleneckInfo{
			NodeID:          nodeID,
			PathsThrough:    pathCount,
			PathPercentage:  pathPercentage,
			TotalCapacity:   capacity,
			ExpectedLoad:    expectedLoad,
			UtilizationRate: utilizationRate,
			BottleneckScore: bottleneckScore,
			Severity:        severity,
		}

		bottlenecks = append(bottlenecks, bottleneck)
	}

	// Sort by bottleneck score (descending)
	sort.Slice(bottlenecks, func(i, j int) bool {
		return bottlenecks[i].BottleneckScore > bottlenecks[j].BottleneckScore
	})

	return bottlenecks
}

// calculatePathComplexity estimates the complexity of the path structure
func (t *Traversal) calculatePathComplexity(paths []*Path) float64 {
	if len(paths) == 0 {
		return 0.0
	}

	// Factors that contribute to complexity:
	// 1. Number of paths
	// 2. Variance in path lengths
	// 3. Number of unique nodes involved
	// 4. Path weight variance

	pathCount := float64(len(paths))
	
	// Calculate length variance
	totalLength := 0
	for _, path := range paths {
		totalLength += path.Length
	}
	avgLength := float64(totalLength) / pathCount
	
	lengthVariance := 0.0
	for _, path := range paths {
		diff := float64(path.Length) - avgLength
		lengthVariance += diff * diff
	}
	lengthVariance /= pathCount

	// Calculate weight variance
	totalWeight := 0.0
	for _, path := range paths {
		totalWeight += path.TotalWeight
	}
	avgWeight := totalWeight / pathCount
	
	weightVariance := 0.0
	for _, path := range paths {
		diff := path.TotalWeight - avgWeight
		weightVariance += diff * diff
	}
	weightVariance /= pathCount

	// Count unique nodes
	uniqueNodes := make(map[string]bool)
	for _, path := range paths {
		for _, node := range path.Nodes {
			uniqueNodes[node.ID] = true
		}
	}

	// Complexity score (normalized to 0-100)
	complexity := 0.0
	complexity += pathCount * 2.0                    // More paths = more complexity
	complexity += lengthVariance * 5.0               // Length variance adds complexity
	complexity += weightVariance * 3.0               // Weight variance adds complexity
	complexity += float64(len(uniqueNodes)) * 1.0    // More nodes = more complexity

	// Normalize to 0-100 scale
	if complexity > 100.0 {
		complexity = 100.0
	}

	return complexity
}

// Helper methods for path calculations

// calculatePathWeight computes the total weight of a path
func (t *Traversal) calculatePathWeight(nodes []*DAGNode) float64 {
	if len(nodes) <= 1 {
		return 0.0
	}

	totalWeight := 0.0
	for i := 0; i < len(nodes)-1; i++ {
		from := nodes[i]
		to := nodes[i+1]
		
		// Find the edge between these nodes
		for _, edge := range from.Outgoing {
			if edge.To.ID == to.ID {
				totalWeight += edge.Weight
				break
			}
		}
	}

	return totalWeight
}

// calculatePathLatency computes the total latency of a path
func (t *Traversal) calculatePathLatency(nodes []*DAGNode) float64 {
	totalLatency := 0.0
	for _, node := range nodes {
		if !node.IsSpecial {
			totalLatency += node.BaseLatency
		}
	}
	return totalLatency
}

// calculatePathMinCapacity finds the minimum capacity along a path (bottleneck)
func (t *Traversal) calculatePathMinCapacity(nodes []*DAGNode) float64 {
	minCapacity := float64(^uint(0) >> 1) // Max float64
	
	for _, node := range nodes {
		if !node.IsSpecial && node.Capacity < minCapacity {
			minCapacity = node.Capacity
		}
	}
	
	if minCapacity == float64(^uint(0) >> 1) {
		return 0.0
	}
	
	return minCapacity
}

// calculatePathReliability estimates the reliability of a path
func (t *Traversal) calculatePathReliability(nodes []*DAGNode) float64 {
	reliability := 1.0
	
	for _, node := range nodes {
		if !node.IsSpecial {
			// Assume 99.9% reliability per node as baseline
			nodeReliability := 0.999
			
			// Adjust based on replicas (higher replicas = higher reliability)
			if node.Design != nil {
				replicas := node.Design.GetReplicas()
				if replicas > 1 {
					// Each additional replica improves reliability
					failureRate := 1.0 - nodeReliability
					replicatedFailureRate := 1.0
					for i := 0; i < replicas; i++ {
						replicatedFailureRate *= failureRate
					}
					nodeReliability = 1.0 - replicatedFailureRate
				}
			}
			
			reliability *= nodeReliability
		}
	}
	
	return reliability
}

// NewPathFinder creates a new path finder with configuration
func NewPathFinder(dag *DAG, config *PathFinderConfig) *PathFinder {
	if config == nil {
		config = &PathFinderConfig{
			MaxPaths:        100,
			MaxPathLength:   20,
			IncludeRedundant: false,
			WeightThreshold: 0.0,
			OptimizeFor:     "latency",
		}
	}
	
	return &PathFinder{
		dag:    dag,
		config: config,
	}
}

// FindOptimalPaths finds paths optimized for the specified criteria
func (pf *PathFinder) FindOptimalPaths() ([]*Path, error) {
	traversal := NewTraversal(pf.dag)
	allPaths, err := traversal.GetAllPaths()
	if err != nil {
		return nil, err
	}

	// Filter and optimize paths based on configuration
	var optimizedPaths []*Path
	
	for _, path := range allPaths {
		// Apply filters
		if pf.config.MaxPathLength > 0 && path.Length > pf.config.MaxPathLength {
			continue
		}
		
		if path.TotalWeight < pf.config.WeightThreshold {
			continue
		}
		
		optimizedPaths = append(optimizedPaths, path)
	}

	// Sort based on optimization criteria
	switch pf.config.OptimizeFor {
	case "latency":
		sort.Slice(optimizedPaths, func(i, j int) bool {
			return optimizedPaths[i].TotalLatency < optimizedPaths[j].TotalLatency
		})
	case "throughput":
		sort.Slice(optimizedPaths, func(i, j int) bool {
			return optimizedPaths[i].MinCapacity > optimizedPaths[j].MinCapacity
		})
	case "reliability":
		sort.Slice(optimizedPaths, func(i, j int) bool {
			return optimizedPaths[i].Reliability > optimizedPaths[j].Reliability
		})
	default:
		sort.Slice(optimizedPaths, func(i, j int) bool {
			return optimizedPaths[i].TotalWeight < optimizedPaths[j].TotalWeight
		})
	}

	// Limit number of paths
	if pf.config.MaxPaths > 0 && len(optimizedPaths) > pf.config.MaxPaths {
		optimizedPaths = optimizedPaths[:pf.config.MaxPaths]
	}

	return optimizedPaths, nil
}

// ClearCache clears the traversal cache
func (t *Traversal) ClearCache() {
	t.cache = &TraversalCache{
		shortestPaths: make(map[string]map[string]*Path),
	}
}

// GetCacheStats returns statistics about the cache
func (t *Traversal) GetCacheStats() map[string]interface{} {
	stats := make(map[string]interface{})
	stats["paths_cached"] = t.cache.pathsComputed
	stats["sort_cached"] = t.cache.sortComputed
	stats["critical_cached"] = t.cache.criticalComputed
	
	if t.cache.pathsComputed {
		stats["total_paths"] = len(t.cache.allPaths)
	}
	
	shortestPathCount := 0
	for _, paths := range t.cache.shortestPaths {
		shortestPathCount += len(paths)
	}
	stats["shortest_paths_cached"] = shortestPathCount
	
	return stats
}

// String returns a string representation of traversal results
func (tr *TraversalResult) String() string {
	var parts []string
	parts = append(parts, fmt.Sprintf("Total Paths: %d", tr.TotalPaths))
	parts = append(parts, fmt.Sprintf("Path Length: %d-%d (avg: %.1f)", tr.MinPathLength, tr.MaxPathLength, tr.AvgPathLength))
	parts = append(parts, fmt.Sprintf("Complexity: %.1f", tr.PathComplexity))
	parts = append(parts, fmt.Sprintf("Bottlenecks: %d", len(tr.Bottlenecks)))
	
	if tr.CriticalPath != nil {
		parts = append(parts, fmt.Sprintf("Critical Path: %.1fms", tr.CriticalPath.TotalLatency))
	}
	
	return fmt.Sprintf("TraversalResult{%s}", strings.Join(parts, ", "))
}