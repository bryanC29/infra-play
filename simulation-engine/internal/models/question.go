package models

import (
	"fmt"
	"strings"
	"time"
)

// Question represents the evaluation context and criteria for a system design submission
type Question struct {
	UserID               string  `json:"user_id" validate:"required"`
	QuestionID           string  `json:"question_id" validate:"required"`
	SubmissionID         string  `json:"submission_id" validate:"required"`
	BaseQPS              int     `json:"base_qps" validate:"required,min=1"`
	RequiredAvailability float64 `json:"required_availability" validate:"required,min=0,max=1"`
	RequiredLatencyMS    int     `json:"required_latency_ms" validate:"required,min=1"`
}

// QuestionMetadata contains additional metadata about the question context
type QuestionMetadata struct {
	CreatedAt     time.Time `json:"created_at"`
	UpdatedAt     time.Time `json:"updated_at"`
	Title         string    `json:"title,omitempty"`
	Description   string    `json:"description,omitempty"`
	Category      string    `json:"category,omitempty"`
	Difficulty    string    `json:"difficulty,omitempty"`
	TimeLimit     int       `json:"time_limit_minutes,omitempty"`
	MaxAttempts   int       `json:"max_attempts,omitempty"`
	CurrentScore  float64   `json:"current_score,omitempty"`
}

// DifficultyLevel constants for question categorization
const (
	DifficultyBeginner     = "beginner"
	DifficultyIntermediate = "intermediate"
	DifficultyAdvanced     = "advanced"
	DifficultyExpert       = "expert"
)

// Category constants for question classification
const (
	CategoryWebServices    = "web_services"
	CategoryMicroservices  = "microservices"
	CategoryDataPipeline   = "data_pipeline"
	CategoryCaching        = "caching"
	CategoryLoadBalancing  = "load_balancing"
	CategoryDatabase       = "database"
	CategoryMessaging      = "messaging"
	CategorySecurity       = "security"
	CategoryMonitoring     = "monitoring"
	CategoryHighAvailability = "high_availability"
)

// ValidationError represents a validation error for question fields
type ValidationError struct {
	Field   string `json:"field"`
	Message string `json:"message"`
	Value   interface{} `json:"value,omitempty"`
}

// Error implements the error interface for ValidationError
func (ve *ValidationError) Error() string {
	if ve.Value != nil {
		return fmt.Sprintf("validation error for field '%s': %s (value: %v)", ve.Field, ve.Message, ve.Value)
	}
	return fmt.Sprintf("validation error for field '%s': %s", ve.Field, ve.Message)
}

// ValidationErrors represents a collection of validation errors
type ValidationErrors []*ValidationError

// Error implements the error interface for ValidationErrors
func (ves ValidationErrors) Error() string {
	if len(ves) == 0 {
		return "no validation errors"
	}
	
	if len(ves) == 1 {
		return ves[0].Error()
	}
	
	var messages []string
	for _, ve := range ves {
		messages = append(messages, ve.Error())
	}
	
	return fmt.Sprintf("multiple validation errors: %s", strings.Join(messages, "; "))
}

// HasErrors returns true if there are any validation errors
func (ves ValidationErrors) HasErrors() bool {
	return len(ves) > 0
}

// AddError adds a new validation error to the collection
func (ves *ValidationErrors) AddError(field, message string, value interface{}) {
	*ves = append(*ves, &ValidationError{
		Field:   field,
		Message: message,
		Value:   value,
	})
}

// GetBaseQPS returns the base QPS for simulation
func (q *Question) GetBaseQPS() int {
	return q.BaseQPS
}

// GetSurge15QPS returns the 1.5x surge QPS
func (q *Question) GetSurge15QPS() int {
	return int(float64(q.BaseQPS) * 1.5)
}

// GetSurge2QPS returns the 2x surge QPS
func (q *Question) GetSurge2QPS() int {
	return q.BaseQPS * 2
}

// GetRequiredAvailability returns the required availability threshold
func (q *Question) GetRequiredAvailability() float64 {
	return q.RequiredAvailability
}

// GetRequiredLatencyMS returns the required latency threshold in milliseconds
func (q *Question) GetRequiredLatencyMS() int {
	return q.RequiredLatencyMS
}

// GetRequiredLatencySeconds returns the required latency threshold in seconds
func (q *Question) GetRequiredLatencySeconds() float64 {
	return float64(q.RequiredLatencyMS) / 1000.0
}

// IsValidUserID checks if the user ID is valid
func (q *Question) IsValidUserID() bool {
	return len(strings.TrimSpace(q.UserID)) > 0
}

// IsValidQuestionID checks if the question ID is valid
func (q *Question) IsValidQuestionID() bool {
	return len(strings.TrimSpace(q.QuestionID)) > 0
}

// IsValidSubmissionID checks if the submission ID is valid
func (q *Question) IsValidSubmissionID() bool {
	return len(strings.TrimSpace(q.SubmissionID)) > 0
}

// Validate performs comprehensive validation of the question
func (q *Question) Validate() error {
	var errors ValidationErrors
	
	// Validate UserID
	if !q.IsValidUserID() {
		errors.AddError("user_id", "user ID cannot be empty", q.UserID)
	}
	
	// Validate QuestionID
	if !q.IsValidQuestionID() {
		errors.AddError("question_id", "question ID cannot be empty", q.QuestionID)
	}
	
	// Validate SubmissionID
	if !q.IsValidSubmissionID() {
		errors.AddError("submission_id", "submission ID cannot be empty", q.SubmissionID)
	}
	
	// Validate BaseQPS
	if q.BaseQPS <= 0 {
		errors.AddError("base_qps", "base QPS must be greater than 0", q.BaseQPS)
	}
	if q.BaseQPS > 1000000 { // Reasonable upper limit
		errors.AddError("base_qps", "base QPS exceeds maximum allowed value (1,000,000)", q.BaseQPS)
	}
	
	// Validate RequiredAvailability
	if q.RequiredAvailability < 0.0 || q.RequiredAvailability > 1.0 {
		errors.AddError("required_availability", "required availability must be between 0.0 and 1.0", q.RequiredAvailability)
	}
	if q.RequiredAvailability < 0.5 {
		errors.AddError("required_availability", "required availability is unreasonably low (< 50%)", q.RequiredAvailability)
	}
	
	// Validate RequiredLatencyMS
	if q.RequiredLatencyMS <= 0 {
		errors.AddError("required_latency_ms", "required latency must be greater than 0", q.RequiredLatencyMS)
	}
	if q.RequiredLatencyMS > 60000 { // 60 seconds max
		errors.AddError("required_latency_ms", "required latency exceeds maximum allowed value (60,000ms)", q.RequiredLatencyMS)
	}
	
	if errors.HasErrors() {
		return errors
	}
	
	return nil
}

// ValidateBusinessRules performs business logic validation
func (q *Question) ValidateBusinessRules() error {
	var errors ValidationErrors
	
	// High availability requirements should have reasonable latency expectations
	if q.RequiredAvailability >= 0.999 && q.RequiredLatencyMS < 50 {
		errors.AddError("requirements", 
			"extremely high availability (>= 99.9%) with very low latency (< 50ms) may be unrealistic", 
			fmt.Sprintf("availability: %.3f, latency: %dms", q.RequiredAvailability, q.RequiredLatencyMS))
	}
	
	// Very high QPS should have reasonable latency expectations
	if q.BaseQPS >= 10000 && q.RequiredLatencyMS < 100 {
		errors.AddError("requirements", 
			"high QPS (>= 10,000) with very low latency (< 100ms) may be challenging", 
			fmt.Sprintf("QPS: %d, latency: %dms", q.BaseQPS, q.RequiredLatencyMS))
	}
	
	// Low QPS with high availability might be over-engineering
	if q.BaseQPS <= 10 && q.RequiredAvailability >= 0.99 {
		errors.AddError("requirements", 
			"low QPS with high availability requirements may indicate over-engineering", 
			fmt.Sprintf("QPS: %d, availability: %.3f", q.BaseQPS, q.RequiredAvailability))
	}
	
	if errors.HasErrors() {
		return errors
	}
	
	return nil
}

// GetDifficultyScore estimates the difficulty based on requirements
func (q *Question) GetDifficultyScore() float64 {
	score := 0.0
	
	// QPS difficulty (0-3 points)
	switch {
	case q.BaseQPS >= 50000:
		score += 3.0
	case q.BaseQPS >= 10000:
		score += 2.5
	case q.BaseQPS >= 1000:
		score += 2.0
	case q.BaseQPS >= 100:
		score += 1.0
	default:
		score += 0.5
	}
	
	// Availability difficulty (0-3 points)
	switch {
	case q.RequiredAvailability >= 0.9999:
		score += 3.0
	case q.RequiredAvailability >= 0.999:
		score += 2.5
	case q.RequiredAvailability >= 0.99:
		score += 2.0
	case q.RequiredAvailability >= 0.95:
		score += 1.0
	default:
		score += 0.5
	}
	
	// Latency difficulty (0-3 points) - lower latency = higher difficulty
	switch {
	case q.RequiredLatencyMS <= 50:
		score += 3.0
	case q.RequiredLatencyMS <= 100:
		score += 2.5
	case q.RequiredLatencyMS <= 200:
		score += 2.0
	case q.RequiredLatencyMS <= 500:
		score += 1.0
	default:
		score += 0.5
	}
	
	// Normalize to 0-10 scale
	return (score / 9.0) * 10.0
}

// GetSuggestedDifficulty returns a suggested difficulty level based on requirements
func (q *Question) GetSuggestedDifficulty() string {
	score := q.GetDifficultyScore()
	
	switch {
	case score >= 8.0:
		return DifficultyExpert
	case score >= 6.0:
		return DifficultyAdvanced
	case score >= 4.0:
		return DifficultyIntermediate
	default:
		return DifficultyBeginner
	}
}

// Clone creates a deep copy of the question
func (q *Question) Clone() *Question {
	return &Question{
		UserID:               q.UserID,
		QuestionID:           q.QuestionID,
		SubmissionID:         q.SubmissionID,
		BaseQPS:              q.BaseQPS,
		RequiredAvailability: q.RequiredAvailability,
		RequiredLatencyMS:    q.RequiredLatencyMS,
	}
}

// Equal checks if two questions are equivalent
func (q *Question) Equal(other *Question) bool {
	if other == nil {
		return false
	}
	
	return q.UserID == other.UserID &&
		q.QuestionID == other.QuestionID &&
		q.SubmissionID == other.SubmissionID &&
		q.BaseQPS == other.BaseQPS &&
		q.RequiredAvailability == other.RequiredAvailability &&
		q.RequiredLatencyMS == other.RequiredLatencyMS
}

// String returns a string representation of the question
func (q *Question) String() string {
	return fmt.Sprintf("Question{User: %s, Question: %s, Submission: %s, QPS: %d, Availability: %.3f, Latency: %dms}", 
		q.UserID, q.QuestionID, q.SubmissionID, q.BaseQPS, q.RequiredAvailability, q.RequiredLatencyMS)
}

// ToMap converts the question to a map for logging or debugging
func (q *Question) ToMap() map[string]interface{} {
	return map[string]interface{}{
		"user_id":                q.UserID,
		"question_id":            q.QuestionID,
		"submission_id":          q.SubmissionID,
		"base_qps":               q.BaseQPS,
		"required_availability":  q.RequiredAvailability,
		"required_latency_ms":    q.RequiredLatencyMS,
		"surge_1_5x_qps":        q.GetSurge15QPS(),
		"surge_2x_qps":          q.GetSurge2QPS(),
		"difficulty_score":       q.GetDifficultyScore(),
		"suggested_difficulty":   q.GetSuggestedDifficulty(),
	}
}

// GetSimulationScenarios returns all QPS scenarios that will be tested
func (q *Question) GetSimulationScenarios() map[string]int {
	return map[string]int{
		"normal_1x":           q.GetBaseQPS(),
		"surge_1_5x":         q.GetSurge15QPS(),
		"surge_2x":           q.GetSurge2QPS(),
		"failure_normal_1x":  q.GetBaseQPS(),
		"failure_surge_1_5x": q.GetSurge15QPS(),
	}
}

// IsHighPerformanceScenario returns true if this question involves high-performance requirements
func (q *Question) IsHighPerformanceScenario() bool {
	return q.BaseQPS >= 5000 || q.RequiredLatencyMS <= 100 || q.RequiredAvailability >= 0.999
}

// IsBasicScenario returns true if this question involves basic requirements
func (q *Question) IsBasicScenario() bool {
	return q.BaseQPS <= 100 && q.RequiredLatencyMS >= 1000 && q.RequiredAvailability <= 0.95
}

// GetRequirementsSummary returns a human-readable summary of the requirements
func (q *Question) GetRequirementsSummary() string {
	return fmt.Sprintf("Handle %d QPS with %.1f%% availability and ≤%dms latency", 
		q.BaseQPS, q.RequiredAvailability*100, q.RequiredLatencyMS)
}