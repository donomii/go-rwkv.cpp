package main

import (
	"errors"
	"math"
	"math/rand"
	"sort"
	"time"
)

func softmax(out []float32) []float32 {
	maxVal := out[0]
	for _, val := range out {
		if val > maxVal {
			maxVal = val
		}
	}

	expSum := float32(0.0)
	for i := range out {
		out[i] = float32(math.Exp(float64(out[i] - maxVal)))
		expSum += out[i]
	}

	for i := range out {
		out[i] /= expSum
	}

	return out
}

func sampleLogits(tensor []float32, temperature float32, topP float32, logitBias map[int]float32) (int, error) {
	probs := softmax(tensor)
	return sampleProbs(probs, temperature, topP, logitBias)
}

func sampleProbs(probs []float32, temperature float32, topP float32, logitBias map[int]float32) (int, error) {
	if temperature < 0 {
		return 0, errors.New("temperature must be non-negative")
	}
	if topP < 0 || topP > 1 {
		return 0, errors.New("top_p must be in the range [0, 1]")
	}

	if topP == 0 {
		topP = 1
	}

	if logitBias != nil {
		logits := make([]float32, len(probs))
		copy(logits, probs)
		for i := range logits {
			logits[i] = float32(math.Log(float64(logits[i])))
		}

		for token, bias := range logitBias {
			logits[token] += bias
		}

		expLogitsSum := float32(0.0)
		for i := range logits {
			logits[i] = float32(math.Exp(float64(logits[i])))
			expLogitsSum += logits[i]
		}

		for i := range probs {
			probs[i] = logits[i] / expLogitsSum
		}
	}

	if temperature == 0 {
		return argMax(probs), nil
	}

	if topP < 1 {
		sortedProbs := make([]float32, len(probs))
		copy(sortedProbs, probs)
		sort.Slice(sortedProbs, func(i, j int) bool { return sortedProbs[i] > sortedProbs[j] })

		cumulativeProbs := make([]float32, len(sortedProbs))
		cumulativeProbs[0] = sortedProbs[0]
		for i := 1; i < len(sortedProbs); i++ {
			cumulativeProbs[i] = cumulativeProbs[i-1] + sortedProbs[i]
		}

		cutoff := float32(0.0)
		for i := 0; i < len(cumulativeProbs); i++ {
			if cumulativeProbs[i] > topP {
				cutoff = sortedProbs[i]
				break
			}
		}
		for i, p := range probs {
			if p < cutoff {
				probs[i] = 0
			}
		}
	}

	if temperature != 1 {
		for i := range probs {
			probs[i] = float32(math.Pow(float64(probs[i]), float64(1/temperature)))
		}
	}

	probsSum := float32(0.0)
	for _, p := range probs {
		probsSum += p
	}
	for i := range probs {
		probs[i] /= probsSum
	}

	return randomChoice(len(probs), probs), nil
}

func argMax(slice []float32) int {
	maxIndex := 0
	maxValue := slice[0]
	for i, value := range slice {
		if value > maxValue {
			maxValue = value
			maxIndex = i
		}
	}
	return maxIndex
}

func randomChoice(length int, probabilities []float32) int {
	rand.Seed(time.Now().UnixNano())
	cumulativeProbabilities := make([]float32, length)
	cumulativeProbabilities[0] = probabilities[0]
	for i := 1; i < length; i++ {
		cumulativeProbabilities[i] = cumulativeProbabilities[i-1] + probabilities[i]
	}

	randomValue := rand.Float32()
	for i, cp := range cumulativeProbabilities {
		if randomValue <= cp {
			return i
		}
	}

	return length - 1
}