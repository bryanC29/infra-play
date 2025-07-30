package utils

import (
	"math"
	"math/rand"
	"time"
)

// ExponentialLatency simulates latency with jitter using exponential distribution.
// mean in milliseconds.
func ExponentialLatency(mean float64) float64 {
	if mean <= 0 {
		return 0
	}
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	return r.ExpFloat64() * mean
}

// GaussianLatency simulates latency with Gaussian distribution.
func GaussianLatency(mean, stddev float64) float64 {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	return r.NormFloat64()*stddev + mean
}

// Clamp clamps a value within min and max.
func Clamp(value, min, max float64) float64 {
	return math.Max(min, math.Min(max, value))
}

// Probability returns true with a given probability (e.g., 0.1 for 10%).
func Probability(p float64) bool {
	if p <= 0 {
		return false
	}
	if p >= 1 {
		return true
	}
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	return r.Float64() < p
}

// SafeDivide returns a / b, or 0 if b == 0 to avoid division by zero.
func SafeDivide(a, b float64) float64 {
	if b == 0 {
		return 0
	}
	return a / b
}

// Round rounds a float64 to the given number of decimal places.
func Round(value float64, places int) float64 {
	pow := math.Pow(10, float64(places))
	return math.Round(value*pow) / pow
}
