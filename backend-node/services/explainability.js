// Simple dictionary map for simulated LIME explanations
const LIME_WORD_WEIGHTS = {
  // Positive word contributions
  "good": 0.28, "great": 0.35, "excellent": 0.42, "love": 0.45, "wonderful": 0.38,
  "amazing": 0.40, "happy": 0.32, "best": 0.36, "superb": 0.39, "awesome": 0.41,
  "nice": 0.18, "glad": 0.22, "enjoy": 0.25, "beautiful": 0.20, "perfect": 0.38,
  "fantastic": 0.43, "thrilled": 0.44, "joy": 0.39, "joyful": 0.40,
  
  // Negative word contributions
  "bad": -0.30, "worse": -0.35, "worst": -0.44, "hate": -0.46, "terrible": -0.42,
  "awful": -0.38, "sad": -0.28, "angry": -0.34, "boring": -0.25, "poor": -0.27,
  "pain": -0.22, "dislike": -0.28, "annoyed": -0.30, "fail": -0.35, "worry": -0.18,
  "disaster": -0.45, "horrible": -0.43, "frustrated": -0.36
};

class ExplainabilityService {
  static runLimeText(text, predictionLabel) {
    if (!text) return [];
    
    // Clean text and split into words
    const words = text.toLowerCase().match(/[a-z0-9']+/g) || [];
    const results = [];
    const isPositive = predictionLabel.toLowerCase() === 'positive';
    const isNegative = predictionLabel.toLowerCase() === 'negative';

    for (const word of words) {
      let weight = 0.0;
      if (LIME_WORD_WEIGHTS[word] !== undefined) {
        weight = LIME_WORD_WEIGHTS[word];
        // Invert if predicting the opposite
        if (isNegative && weight > 0) weight = -weight * 0.5;
        if (isPositive && weight < 0) weight = -weight * 0.5;
      } else {
        // Small pseudorandom deterministic variance
        let hash = 0;
        for (let i = 0; i < word.length; i++) hash = ((hash << 5) - hash) + word.charCodeAt(i);
        const randVal = ((Math.abs(hash) % 100) / 100.0) * 0.04 - 0.02;
        weight = randVal;
      }

      results.push({
        token: word,
        weight: Math.round(weight * 10000) / 10000
      });
    }

    return results;
  }
}

module.exports = ExplainabilityService;
