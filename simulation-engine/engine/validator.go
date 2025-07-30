package engine

import (
	"errors"
	"fmt"

	"github.com/bryanC29/infra-play/simulation-engine/models"
	"github.com/bryanC29/infra-play/simulation-engine/utils"
)

// ValidateDesign runs a series of checks on the submitted design.
func ValidateDesign(design models.Design, problem models.Problem) error {
	if len(design.Nodes) == 0 {
		return errors.New("design must include at least one node")
	}

	if len(design.Edges) == 0 {
		return errors.New("design must include at least one edge")
	}

	graph := models.NewGraph(design.Nodes, design.Edges)

	if err := validateUniqueIDs(graph); err != nil {
		return err
	}

	if err := utils.ValidateNoCycles(graph); err != nil {
		return fmt.Errorf("cycle detected in design: %w", err)
	}

	if err := utils.ValidateReachability(graph, problem.EntryPoint, problem.RequiredServices); err != nil {
		return fmt.Errorf("invalid routing paths: %w", err)
	}

	if err := validateResourceLimits(graph, problem); err != nil {
		return fmt.Errorf("resource limit violation: %w", err)
	}

	return nil
}

// validateUniqueIDs ensures no duplicate node IDs exist.
func validateUniqueIDs(graph *models.Graph) error {
	seen := make(map[string]bool)
	for id := range graph.Nodes {
		if seen[id] {
			return fmt.Errorf("duplicate node ID found: %s", id)
		}
		seen[id] = true
	}
	return nil
}

// validateResourceLimits enforces limits from the problem spec.
func validateResourceLimits(graph *models.Graph, problem models.Problem) error {
	totalCPU := 0
	totalMem := 0

	for _, node := range graph.Nodes {
		totalCPU += node.Config.VCPU * node.Config.Replicas
		totalMem += node.Config.MemoryMB * node.Config.Replicas
	}

	if totalCPU > problem.MaxCPU {
		return fmt.Errorf("CPU exceeds limit: %d > %d", totalCPU, problem.MaxCPU)
	}

	if totalMem > problem.MaxMemoryMB {
		return fmt.Errorf("memory exceeds limit: %dMB > %dMB", totalMem, problem.MaxMemoryMB)
	}

	return nil
}
