package engine

import (
	"fmt"
	"strings"

	"github.com/bryanC29/infra-play/simulation-engine/models"
)

// ApplyFailure modifies the system graph or node state to simulate a failure.
// Supports node-level and region-level failures.
func ApplyFailure(graph *models.Graph, failure models.FailureEvent) error {
    switch strings.ToLower(failure.Type) {
    case "node":
        return failNode(graph, failure.Target)

    case "region":
        return failRegion(graph, failure.Target)

    default:
        return fmt.Errorf("unsupported failure type: %s", failure.Type)
    }
}

// ResetFailures restores all nodes and edges to their normal (healthy) state.
// Called between failure scenarios to isolate effects.
func ResetFailures(graph *models.Graph) {
    for _, node := range graph.Nodes {
        node.Failed = false
    }

    for _, edge := range graph.Edges {
        edge.Failed = false
    }
}

// failNode marks a specific node as failed and disables all connected edges.
func failNode(graph *models.Graph, nodeID string) error {
    node, ok := graph.Nodes[nodeID]
    if !ok {
        return fmt.Errorf("node %s not found", nodeID)
    }

    node.Failed = true

    // Disable all edges in/out of this node
    for _, edge := range graph.Edges {
        if edge.Source == nodeID || edge.Target == nodeID {
            edge.Failed = true
        }
    }

    return nil
}

// failRegion marks all nodes in a given region as failed and disables their edges.
func failRegion(graph *models.Graph, region string) error {
    found := false

    for _, node := range graph.Nodes {
        if strings.EqualFold(node.Config.Region, region) {
            node.Failed = true
            found = true

            // Disable edges related to this node
            for _, edge := range graph.Edges {
                if edge.Source == node.ID || edge.Target == node.ID {
                    edge.Failed = true
                }
            }
        }
    }

    if !found {
        return fmt.Errorf("region %s not found in any node", region)
    }

    return nil
}
