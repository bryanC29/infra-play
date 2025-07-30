package utils

import (
	"errors"
	"fmt"

	"github.com/bryanC29/infra-play/simulation-engine/models"
)

// ValidateNoCycles checks for cycles in the graph using DFS.
func ValidateNoCycles(graph *models.Graph) error {
	visited := make(map[string]bool)
	recStack := make(map[string]bool)

	for nodeID := range graph.Nodes {
		if !visited[nodeID] {
			if detectCycleDFS(nodeID, visited, recStack, graph.Edges) {
				return fmt.Errorf("cycle detected starting from node %s", nodeID)
			}
		}
	}
	return nil
}

// detectCycleDFS is a helper for DFS-based cycle detection.
func detectCycleDFS(nodeID string, visited, recStack map[string]bool, edges map[string][]*models.Node) bool {
	visited[nodeID] = true
	recStack[nodeID] = true

	for _, neighbor := range edges[nodeID] {
		if !visited[neighbor.ID] {
			if detectCycleDFS(neighbor.ID, visited, recStack, edges) {
				return true
			}
		} else if recStack[neighbor.ID] {
			return true
		}
	}

	recStack[nodeID] = false
	return false
}

// ValidateReachability ensures all required services are reachable from the entry point.
func ValidateReachability(graph *models.Graph, entryPoint string, requiredServices []string) error {
	reachable := BFS(graph, entryPoint)

	for _, target := range requiredServices {
		if !reachable[target] {
			return fmt.Errorf("required service %s is not reachable from entry point %s", target, entryPoint)
		}
	}

	return nil
}

// BFS performs a breadth-first search and returns a map of reachable node IDs.
func BFS(graph *models.Graph, start string) map[string]bool {
	visited := make(map[string]bool)
	queue := []string{start}
	visited[start] = true

	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]

		for _, neighbor := range graph.Edges[curr] {
			if !visited[neighbor.ID] {
				visited[neighbor.ID] = true
				queue = append(queue, neighbor.ID)
			}
		}
	}

	return visited
}

// TopoSort returns nodes in topological order. Returns error if cycle exists.
func TopoSort(graph *models.Graph) ([]string, error) {
	inDegree := make(map[string]int)
	for id := range graph.Nodes {
		inDegree[id] = 0
	}
	for _, neighbors := range graph.Edges {
		for _, neighbor := range neighbors {
			inDegree[neighbor.ID]++
		}
	}

	queue := []string{}
	for id, deg := range inDegree {
		if deg == 0 {
			queue = append(queue, id)
		}
	}

	var sorted []string
	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]
		sorted = append(sorted, curr)

		for _, neighbor := range graph.Edges[curr] {
			inDegree[neighbor.ID]--
			if inDegree[neighbor.ID] == 0 {
				queue = append(queue, neighbor.ID)
			}
		}
	}

	if len(sorted) != len(graph.Nodes) {
		return nil, errors.New("cycle detected in graph, cannot topologically sort")
	}

	return sorted, nil
}
