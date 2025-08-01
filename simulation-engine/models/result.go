package models

type Result struct {
	Passed     bool               `json:"passed"`
	TotalScore float64            `json:"score"`
	Metrics    PerformanceMetrics `json:"metrics"`
	Errors     []string           `json:"errors,omitempty"`
}

type PerformanceMetrics struct {
	Latency        float64 `json:"latency"`
	Requests       int     `json:"requests"`
	FailedRequests int     `json:"failedRequests"`
	Availability   float64 `json:"availability"`
}