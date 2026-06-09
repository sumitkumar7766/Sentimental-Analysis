import re
import numpy as np
from typing import List, Dict, Any, Tuple

# Simple word weight map for simulated LIME explanations
LIME_WORD_WEIGHTS = {
    # Positive word contributions
    "good": 0.28, "great": 0.35, "excellent": 0.42, "love": 0.45, "wonderful": 0.38,
    "amazing": 0.40, "happy": 0.32, "best": 0.36, "superb": 0.39, "awesome": 0.41,
    "nice": 0.18, "glad": 0.22, "enjoy": 0.25, "beautiful": 0.20, "perfect": 0.38,
    
    # Negative word contributions
    "bad": -0.30, "worse": -0.35, "worst": -0.44, "hate": -0.46, "terrible": -0.42,
    "awful": -0.38, "sad": -0.28, "angry": -0.34, "boring": -0.25, "poor": -0.27,
    "pain": -0.22, "dislike": -0.28, "annoyed": -0.30, "fail": -0.35, "worry": -0.18
}

class ExplainabilityService:
    @staticmethod
    def run_lime_text(text: str, prediction_label: str) -> List[Dict[str, Any]]:
        """
        Computes LIME-like word contribution weights for a given text prediction.
        Returns a list of tokens with their respective alignment scores.
        """
        # Clean text and split into words
        clean_text = re.sub(r"[^\w\s]", "", text.lower())
        words = clean_text.split()
        
        explanations = []
        multiplier = 1.0 if prediction_label == "Positive" else -1.0
        if prediction_label == "Neutral":
            multiplier = 0.1 # Dampen contributions for neutral results
            
        for i, word in enumerate(words):
            # Base contribution is slightly randomized to simulate sampling perturbations
            base_weight = LIME_WORD_WEIGHTS.get(word, 0.0)
            
            if base_weight == 0.0:
                # Add minimal noise to neutral words to simulate non-zero LIME outputs
                weight = float(np.random.normal(0, 0.015))
            else:
                # Align positive weights with target prediction label (positive or negative)
                weight = float(base_weight * multiplier + np.random.normal(0, 0.02))
                
            explanations.append({
                "word": word,
                "index": i,
                "weight": round(weight, 4)
            })
            
        return explanations

    @staticmethod
    def run_shap_multimodal(
        text_scores: Dict[str, float], 
        image_scores: Dict[str, float], 
        audio_scores: Dict[str, float],
        multimodal_label: str
    ) -> Dict[str, Any]:
        """
        Computes SHAP feature importance values mapping modality inputs to the target prediction.
        Computes base value, modal values, and shapley contributions.
        """
        target_key = multimodal_label.lower()
        
        # Base value represents average prediction over dataset (~0.33 per class)
        base_value = 0.333
        
        # Individual modality predictions for target sentiment
        v_t = text_scores.get(target_key, 0.33)
        v_i = image_scores.get(target_key, 0.33)
        v_a = audio_scores.get(target_key, 0.33)
        
        # Calculate marginal contributions (simulated Shapley values)
        # Shapley represents difference from base value distributed across components
        diff_t = v_t - base_value
        diff_i = v_i - base_value
        diff_a = v_a - base_value
        
        total_diff = abs(diff_t) + abs(diff_i) + abs(diff_a)
        if total_diff == 0:
            shap_t = 0.0
            shap_i = 0.0
            shap_a = 0.0
        else:
            # Scale sum of Shapley values to match final output prediction difference
            # Output prediction is average or fusion result
            fusion_output = (v_t + v_i + v_a) / 3.0
            scale = (fusion_output - base_value) / (diff_t + diff_i + diff_a + 1e-6)
            
            shap_t = float(diff_t * scale)
            shap_i = float(diff_i * scale)
            shap_a = float(diff_a * scale)
            
        return {
            "base_value": base_value,
            "shap_values": {
                "text": round(shap_t, 4),
                "image": round(shap_i, 4),
                "audio": round(shap_a, 4)
            },
            "contributions": {
                "text": round(v_t, 4),
                "image": round(v_i, 4),
                "audio": round(v_a, 4)
            }
        }
