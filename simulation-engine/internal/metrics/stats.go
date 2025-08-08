package metrics

import (
	"fmt"
	"math"
	"sort"
	"sync"
	"time"
)

// StatsCalculator provides statistical calculations for simulation metrics
type StatsCalculator struct {
	mu sync.RWMutex
}

// LatencyStats contains detailed latency statistics
type LatencyStats struct {
	Count             int64   `json:"count"`
	Mean              float64 `json:"mean"`
	Median            float64 `json:"median"`
	Min               float64 `json:"min"`
	Max               float64 `json:"max"`
	StandardDeviation float64 `json:"standard_deviation"`
	Variance          float64 `json:"variance"`
	P50               float64 `json:"p50"`
	P75               float64 `json:"p75"`
	P90               float64 `json:"p90"`
	P95               float64 `json:"p95"`
	P99               float64 `json:"p99"`
	P999              float64 `json:"p999"`
}

// AvailabilityStats contains availability and reliability statistics
type AvailabilityStats struct {
	TotalRequests      int64   `json:"total_requests"`
	SuccessfulRequests int64   `json:"successful_requests"`
	FailedRequests     int64   `json:"failed_requests"`
	DroppedRequests    int64   `json:"dropped_requests"`
	TimeoutRequests    int64   `json:"timeout_requests"`
	SuccessRate        float64 `json:"success_rate"`
	FailureRate        float64 `json:"failure_rate"`
	DropRate           float64 `json:"drop_rate"`
	TimeoutRate        float64 `json:"timeout_rate"`
	Availability       float64 `json:"availability"`
	Reliability        float64 `json:"reliability"`
	MTTR               float64 `json:"mttr"` // Mean Time To Recovery (seconds)
	MTBF               float64 `json:"mtbf"` // Mean Time Between Failures (seconds)
}

// ThroughputStats contains throughput and performance statistics
type ThroughputStats struct {
	RequestsPerSecond    float64 `json:"requests_per_second"`
	PeakThroughput       float64 `json:"peak_throughput"`
	AverageThroughput    float64 `json:"average_throughput"`
	MinThroughput        float64 `json:"min_throughput"`
	ThroughputVariance   float64 `json:"throughput_variance"`
	SustainedThroughput  float64 `json:"sustained_throughput"`
	BurstCapacity        float64 `json:"burst_capacity"`
	CapacityUtilization  float64 `json:"capacity_utilization"`
}

// LoadStats contains system load and resource utilization statistics
type LoadStats struct {
	AverageLoad        float64 `json:"average_load"`
	PeakLoad           float64 `json:"peak_load"`
	MinLoad            float64 `json:"min_load"`
	LoadVariance       float64 `json:"load_variance"`
	LoadStandardDev    float64 `json:"load_standard_deviation"`
	UtilizationRate    float64 `json:"utilization_rate"`
	SaturationPoint    float64 `json:"saturation_point"`
	OverloadDuration   float64 `json:"overload_duration"` // Seconds spent overloaded
	LoadDistribution   []LoadBucket `json:"load_distribution"`
}

// LoadBucket represents a load distribution bucket
type LoadBucket struct {
	MinLoad     float64 `json:"min_load"`
	MaxLoad     float64 `json:"max_load"`
	Count       int64   `json:"count"`
	Percentage  float64 `json:"percentage"`
	Duration    float64 `json:"duration"` // Seconds in this load range
}

// ErrorStats contains error and failure statistics
type ErrorStats struct {
	TotalErrors        int64             `json:"total_errors"`
	ErrorRate          float64           `json:"error_rate"`
	ErrorsByType       map[string]int64  `json:"errors_by_type"`
	ErrorRatesByType   map[string]float64 `json:"error_rates_by_type"`
	CriticalErrors     int64             `json:"critical_errors"`
	RecoverableErrors  int64             `json:"recoverable_errors"`
	TransientErrors    int64             `json:"transient_errors"`
	PermanentErrors    int64             `json:"permanent_errors"`
	ErrorBurstCount    int               `json:"error_burst_count"`
	MaxConsecutiveErrors int             `json:"max_consecutive_errors"`
}

// TimeSeries represents a time-series data point
type TimeSeries struct {
	Timestamp time.Time `json:"timestamp"`
	Value     float64   `json:"value"`
	Label     string    `json:"label,omitempty"`
}

// TimeSeriesStats contains time-series analysis
type TimeSeriesStats struct {
	DataPoints      []TimeSeries `json:"data_points"`
	StartTime       time.Time    `json:"start_time"`
	EndTime         time.Time    `json:"end_time"`
	Duration        float64      `json:"duration"` // Seconds
	SampleCount     int          `json:"sample_count"`
	SamplingRate    float64      `json:"sampling_rate"` // Samples per second
	Trend           float64      `json:"trend"`         // Linear trend slope
	Volatility      float64      `json:"volatility"`    // Standard deviation of changes
	Autocorrelation float64      `json:"autocorrelation"`
}

// NewStatsCalculator creates a new statistics calculator
func NewStatsCalculator() *StatsCalculator {
	return &StatsCalculator{}
}

// CalculateLatencyStats computes comprehensive latency statistics
func (sc *StatsCalculator) CalculateLatencyStats(latencies []float64) *LatencyStats {
	sc.mu.RLock()
	defer sc.mu.RUnlock()

	if len(latencies) == 0 {
		return &LatencyStats{}
	}

	// Sort latencies for percentile calculations
	sortedLatencies := make([]float64, len(latencies))
	copy(sortedLatencies, latencies)
	sort.Float64s(sortedLatencies)

	stats := &LatencyStats{
		Count:  int64(len(latencies)),
		Min:    sortedLatencies[0],
		Max:    sortedLatencies[len(sortedLatencies)-1],
		Median: sc.calculatePercentile(sortedLatencies, 0.50),
		P50:    sc.calculatePercentile(sortedLatencies, 0.50),
		P75:    sc.calculatePercentile(sortedLatencies, 0.75),
		P90:    sc.calculatePercentile(sortedLatencies, 0.90),
		P95:    sc.calculatePercentile(sortedLatencies, 0.95),
		P99:    sc.calculatePercentile(sortedLatencies, 0.99),
		P999:   sc.calculatePercentile(sortedLatencies, 0.999),
	}

	// Calculate mean
	sum := 0.0
	for _, latency := range latencies {
		sum += latency
	}
	stats.Mean = sum / float64(len(latencies))

	// Calculate variance and standard deviation
	sumSquaredDiffs := 0.0
	for _, latency := range latencies {
		diff := latency - stats.Mean
		sumSquaredDiffs += diff * diff
	}
	stats.Variance = sumSquaredDiffs / float64(len(latencies))
	stats.StandardDeviation = math.Sqrt(stats.Variance)

	return stats
}

// CalculateAvailabilityStats computes availability and reliability statistics
func (sc *StatsCalculator) CalculateAvailabilityStats(total, successful, failed, dropped, timeout int64, 
	failureTimes []time.Time, recoveryTimes []time.Time) *AvailabilityStats {
	sc.mu.RLock()
	defer sc.mu.RUnlock()

	stats := &AvailabilityStats{
		TotalRequests:      total,
		SuccessfulRequests: successful,
		FailedRequests:     failed,
		DroppedRequests:    dropped,
		TimeoutRequests:    timeout,
	}

	if total > 0 {
		stats.SuccessRate = float64(successful) / float64(total)
		stats.FailureRate = float64(failed) / float64(total)
		stats.DropRate = float64(dropped) / float64(total)
		stats.TimeoutRate = float64(timeout) / float64(total)
		stats.Availability = stats.SuccessRate
		stats.Reliability = stats.SuccessRate // Simplified - same as success rate
	}

	// Calculate MTTR and MTBF if failure/recovery data is provided
	if len(failureTimes) > 0 && len(recoveryTimes) > 0 {
		stats.MTTR = sc.calculateMTTR(failureTimes, recoveryTimes)
		stats.MTBF = sc.calculateMTBF(failureTimes)
	}

	return stats
}

// CalculateThroughputStats computes throughput and performance statistics
func (sc *StatsCalculator) CalculateThroughputStats(requestCounts []float64, timeWindows []float64, 
	maxCapacity float64, duration time.Duration) *ThroughputStats {
	sc.mu.RLock()
	defer sc.mu.RUnlock()

	if len(requestCounts) == 0 {
		return &ThroughputStats{}
	}

	stats := &ThroughputStats{}
	
	// Calculate throughput values
	throughputs := make([]float64, len(requestCounts))
	totalThroughput := 0.0
	
	for i, count := range requestCounts {
		window := timeWindows[i]
		if window > 0 {
			throughput := count / window
			throughputs[i] = throughput
			totalThroughput += throughput
			
			if throughput > stats.PeakThroughput {
				stats.PeakThroughput = throughput
			}
			if stats.MinThroughput == 0 || throughput < stats.MinThroughput {
				stats.MinThroughput = throughput
			}
		}
	}

	// Calculate average throughput
	if len(throughputs) > 0 {
		stats.AverageThroughput = totalThroughput / float64(len(throughputs))
		stats.RequestsPerSecond = stats.AverageThroughput
	}

	// Calculate total requests over total duration
	if duration.Seconds() > 0 {
		totalRequests := 0.0
		for _, count := range requestCounts {
			totalRequests += count
		}
		stats.SustainedThroughput = totalRequests / duration.Seconds()
	}

	// Calculate throughput variance
	if len(throughputs) > 1 {
		mean := stats.AverageThroughput
		sumSquaredDiffs := 0.0
		for _, throughput := range throughputs {
			diff := throughput - mean
			sumSquaredDiffs += diff * diff
		}
		stats.ThroughputVariance = sumSquaredDiffs / float64(len(throughputs)-1)
	}

	// Calculate capacity metrics
	if maxCapacity > 0 {
		stats.CapacityUtilization = stats.PeakThroughput / maxCapacity
		stats.BurstCapacity = stats.PeakThroughput - stats.AverageThroughput
	}

	return stats
}

// CalculateLoadStats computes load and resource utilization statistics
func (sc *StatsCalculator) CalculateLoadStats(loadSamples []float64, timestamps []time.Time, 
	maxCapacity float64) *LoadStats {
	sc.mu.RLock()
	defer sc.mu.RUnlock()

	if len(loadSamples) == 0 {
		return &LoadStats{}
	}

	stats := &LoadStats{
		MinLoad: math.MaxFloat64,
	}

	// Calculate basic statistics
	totalLoad := 0.0
	overloadTime := 0.0
	saturationThreshold := maxCapacity * 0.8 // 80% of capacity
	
	for i, load := range loadSamples {
		totalLoad += load
		
		if load > stats.PeakLoad {
			stats.PeakLoad = load
		}
		if load < stats.MinLoad {
			stats.MinLoad = load
		}
		
		// Track overload duration
		if load > maxCapacity && i > 0 && i < len(timestamps) {
			timeDiff := timestamps[i].Sub(timestamps[i-1]).Seconds()
			overloadTime += timeDiff
		}
	}

	stats.AverageLoad = totalLoad / float64(len(loadSamples))
	stats.OverloadDuration = overloadTime

	// Calculate variance and standard deviation
	sumSquaredDiffs := 0.0
	for _, load := range loadSamples {
		diff := load - stats.AverageLoad
		sumSquaredDiffs += diff * diff
	}
	stats.LoadVariance = sumSquaredDiffs / float64(len(loadSamples))
	stats.LoadStandardDev = math.Sqrt(stats.LoadVariance)

	// Calculate utilization and saturation
	if maxCapacity > 0 {
		stats.UtilizationRate = stats.AverageLoad / maxCapacity
		stats.SaturationPoint = saturationThreshold
	}

	// Calculate load distribution
	stats.LoadDistribution = sc.calculateLoadDistribution(loadSamples, timestamps, maxCapacity)

	return stats
}

// CalculateErrorStats computes error and failure statistics
func (sc *StatsCalculator) CalculateErrorStats(totalRequests int64, errorEvents []ErrorEvent) *ErrorStats {
	sc.mu.RLock()
	defer sc.mu.RUnlock()

	stats := &ErrorStats{
		ErrorsByType:     make(map[string]int64),
		ErrorRatesByType: make(map[string]float64),
	}

	if len(errorEvents) == 0 {
		return stats
	}

	stats.TotalErrors = int64(len(errorEvents))
	if totalRequests > 0 {
		stats.ErrorRate = float64(stats.TotalErrors) / float64(totalRequests)
	}

	// Categorize errors by type
	consecutiveErrors := 0
	maxConsecutive := 0
	burstCount := 0
	lastErrorTime := time.Time{}

	for _, event := range errorEvents {
		// Count by type
		stats.ErrorsByType[event.ErrorType]++
		
		// Classify error severity
		switch event.Severity {
		case "critical":
			stats.CriticalErrors++
		case "recoverable":
			stats.RecoverableErrors++
		}
		
		// Classify error persistence
		if event.IsTransient {
			stats.TransientErrors++
		} else {
			stats.PermanentErrors++
		}
		
		// Track consecutive errors and bursts
		if !lastErrorTime.IsZero() && event.Timestamp.Sub(lastErrorTime) < time.Second*5 {
			consecutiveErrors++
			if consecutiveErrors > maxConsecutive {
				maxConsecutive = consecutiveErrors
			}
			if consecutiveErrors == 1 { // Start of a new burst
				burstCount++
			}
		} else {
			consecutiveErrors = 1
		}
		
		lastErrorTime = event.Timestamp
	}

	stats.MaxConsecutiveErrors = maxConsecutive
	stats.ErrorBurstCount = burstCount

	// Calculate error rates by type
	for errorType, count := range stats.ErrorsByType {
		if totalRequests > 0 {
			stats.ErrorRatesByType[errorType] = float64(count) / float64(totalRequests)
		}
	}

	return stats
}

// CalculateTimeSeriesStats analyzes time-series data
func (sc *StatsCalculator) CalculateTimeSeriesStats(series []TimeSeries) *TimeSeriesStats {
	sc.mu.RLock()
	defer sc.mu.RUnlock()

	if len(series) == 0 {
		return &TimeSeriesStats{}
	}

	stats := &TimeSeriesStats{
		DataPoints:  make([]TimeSeries, len(series)),
		SampleCount: len(series),
		StartTime:   series[0].Timestamp,
		EndTime:     series[len(series)-1].Timestamp,
	}

	copy(stats.DataPoints, series)
	
	duration := stats.EndTime.Sub(stats.StartTime)
	stats.Duration = duration.Seconds()
	
	if stats.Duration > 0 {
		stats.SamplingRate = float64(stats.SampleCount) / stats.Duration
	}

	// Calculate trend (linear regression slope)
	stats.Trend = sc.calculateLinearTrend(series)

	// Calculate volatility (standard deviation of value changes)
	stats.Volatility = sc.calculateVolatility(series)

	// Calculate autocorrelation (simplified - lag 1)
	stats.Autocorrelation = sc.calculateAutocorrelation(series, 1)

	return stats
}

// Helper methods

func (sc *StatsCalculator) calculatePercentile(sortedValues []float64, percentile float64) float64 {
	if len(sortedValues) == 0 {
		return 0.0
	}
	
	if percentile <= 0 {
		return sortedValues[0]
	}
	if percentile >= 1 {
		return sortedValues[len(sortedValues)-1]
	}
	
	index := percentile * float64(len(sortedValues)-1)
	lower := int(index)
	upper := lower + 1
	
	if upper >= len(sortedValues) {
		return sortedValues[lower]
	}
	
	weight := index - float64(lower)
	return sortedValues[lower]*(1-weight) + sortedValues[upper]*weight
}

func (sc *StatsCalculator) calculateMTTR(failureTimes, recoveryTimes []time.Time) float64 {
	if len(failureTimes) == 0 || len(recoveryTimes) == 0 {
		return 0.0
	}

	totalRecoveryTime := 0.0
	recoveryCount := 0

	// Match failures with recoveries
	for i, failureTime := range failureTimes {
		for _, recoveryTime := range recoveryTimes {
			if recoveryTime.After(failureTime) && (i == len(failureTimes)-1 || recoveryTime.Before(failureTimes[i+1])) {
				totalRecoveryTime += recoveryTime.Sub(failureTime).Seconds()
				recoveryCount++
				break
			}
		}
	}

	if recoveryCount == 0 {
		return 0.0
	}

	return totalRecoveryTime / float64(recoveryCount)
}

func (sc *StatsCalculator) calculateMTBF(failureTimes []time.Time) float64 {
	if len(failureTimes) <= 1 {
		return 0.0
	}

	totalTimeBetweenFailures := 0.0
	for i := 1; i < len(failureTimes); i++ {
		totalTimeBetweenFailures += failureTimes[i].Sub(failureTimes[i-1]).Seconds()
	}

	return totalTimeBetweenFailures / float64(len(failureTimes)-1)
}

func (sc *StatsCalculator) calculateLoadDistribution(loadSamples []float64, timestamps []time.Time, 
	maxCapacity float64) []LoadBucket {
	
	if len(loadSamples) == 0 {
		return []LoadBucket{}
	}

	// Define load buckets (0-20%, 20-40%, 40-60%, 60-80%, 80-100%, 100%+)
	buckets := []LoadBucket{
		{MinLoad: 0, MaxLoad: maxCapacity * 0.2},
		{MinLoad: maxCapacity * 0.2, MaxLoad: maxCapacity * 0.4},
		{MinLoad: maxCapacity * 0.4, MaxLoad: maxCapacity * 0.6},
		{MinLoad: maxCapacity * 0.6, MaxLoad: maxCapacity * 0.8},
		{MinLoad: maxCapacity * 0.8, MaxLoad: maxCapacity},
		{MinLoad: maxCapacity, MaxLoad: math.MaxFloat64},
	}

	totalDuration := 0.0
	if len(timestamps) > 1 {
		totalDuration = timestamps[len(timestamps)-1].Sub(timestamps[0]).Seconds()
		fmt.Printf("Total duration is: %.2f seconds\n", totalDuration)
	}

	// Distribute samples into buckets
	for i, load := range loadSamples {
		for j := range buckets {
			if load >= buckets[j].MinLoad && load < buckets[j].MaxLoad {
				buckets[j].Count++
				
				// Add duration if timestamps are available
				if i > 0 && i < len(timestamps) {
					duration := timestamps[i].Sub(timestamps[i-1]).Seconds()
					buckets[j].Duration += duration
				}
				break
			}
		}
	}

	// Calculate percentages
	totalSamples := int64(len(loadSamples))
	for i := range buckets {
		if totalSamples > 0 {
			buckets[i].Percentage = float64(buckets[i].Count) / float64(totalSamples) * 100.0
		}
	}

	return buckets
}

func (sc *StatsCalculator) calculateLinearTrend(series []TimeSeries) float64 {
	if len(series) <= 1 {
		return 0.0
	}

	n := float64(len(series))
	sumX, sumY, sumXY, sumX2 := 0.0, 0.0, 0.0, 0.0

	for i, point := range series {
		x := float64(i) // Use index as x-coordinate
		y := point.Value
		
		sumX += x
		sumY += y
		sumXY += x * y
		sumX2 += x * x
	}

	// Calculate slope using least squares method
	denominator := n*sumX2 - sumX*sumX
	if denominator == 0 {
		return 0.0
	}

	slope := (n*sumXY - sumX*sumY) / denominator
	return slope
}

func (sc *StatsCalculator) calculateVolatility(series []TimeSeries) float64 {
	if len(series) <= 1 {
		return 0.0
	}

	// Calculate changes between consecutive values
	changes := make([]float64, len(series)-1)
	for i := 1; i < len(series); i++ {
		changes[i-1] = series[i].Value - series[i-1].Value
	}

	// Calculate standard deviation of changes
	if len(changes) == 0 {
		return 0.0
	}

	mean := 0.0
	for _, change := range changes {
		mean += change
	}
	mean /= float64(len(changes))

	variance := 0.0
	for _, change := range changes {
		diff := change - mean
		variance += diff * diff
	}
	variance /= float64(len(changes))

	return math.Sqrt(variance)
}

func (sc *StatsCalculator) calculateAutocorrelation(series []TimeSeries, lag int) float64 {
	if len(series) <= lag {
		return 0.0
	}

	n := len(series) - lag
	if n <= 0 {
		return 0.0
	}

	// Calculate means
	mean1, mean2 := 0.0, 0.0
	for i := 0; i < n; i++ {
		mean1 += series[i].Value
		mean2 += series[i+lag].Value
	}
	mean1 /= float64(n)
	mean2 /= float64(n)

	// Calculate covariance and variances
	covariance := 0.0
	var1, var2 := 0.0, 0.0
	
	for i := 0; i < n; i++ {
		diff1 := series[i].Value - mean1
		diff2 := series[i+lag].Value - mean2
		
		covariance += diff1 * diff2
		var1 += diff1 * diff1
		var2 += diff2 * diff2
	}

	// Calculate correlation coefficient
	denominator := math.Sqrt(var1 * var2)
	if denominator == 0 {
		return 0.0
	}

	return covariance / denominator
}

// ErrorEvent represents an error occurrence
type ErrorEvent struct {
	Timestamp   time.Time `json:"timestamp"`
	ErrorType   string    `json:"error_type"`
	Severity    string    `json:"severity"`
	IsTransient bool      `json:"is_transient"`
	Message     string    `json:"message,omitempty"`
}

// AggregateStats combines multiple statistics into a comprehensive view
type AggregateStats struct {
	Latency     *LatencyStats     `json:"latency"`
	Availability *AvailabilityStats `json:"availability"`
	Throughput  *ThroughputStats  `json:"throughput"`
	Load        *LoadStats        `json:"load"`
	Errors      *ErrorStats       `json:"errors"`
	TimeSeries  *TimeSeriesStats  `json:"time_series,omitempty"`
}

// CalculateAggregateStats computes comprehensive statistics from all available data
func (sc *StatsCalculator) CalculateAggregateStats(
	latencies []float64,
	totalRequests, successful, failed, dropped, timeout int64,
	requestCounts []float64,
	timeWindows []float64,
	loadSamples []float64,
	timestamps []time.Time,
	errorEvents []ErrorEvent,
	maxCapacity float64,
	duration time.Duration,
) *AggregateStats {
	
	return &AggregateStats{
		Latency:     sc.CalculateLatencyStats(latencies),
		Availability: sc.CalculateAvailabilityStats(totalRequests, successful, failed, dropped, timeout, nil, nil),
		Throughput:  sc.CalculateThroughputStats(requestCounts, timeWindows, maxCapacity, duration),
		Load:        sc.CalculateLoadStats(loadSamples, timestamps, maxCapacity),
		Errors:      sc.CalculateErrorStats(totalRequests, errorEvents),
	}
}