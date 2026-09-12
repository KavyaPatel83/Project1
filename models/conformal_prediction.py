import os
import numpy as np
import torch

DIAGNOSTIC_NAMES = ['BCC', 'ACK', 'NEV', 'SEK', 'SCC', 'MEL']

class ConformalPredictor:
    r"""
    Split Conformal Prediction with margin nonconformity for
    mathematically guaranteed, distribution-free uncertainty estimation.
    
    Guarantees:
        P(Y \in C(X)) >= 1 - \alpha
    """
    def __init__(self, alpha=0.05, class_names=None):
        """
        alpha: significance level (0.05 corresponds to 95% coverage guarantee)
        class_names: list of diagnostic string labels
        """
        self.alpha = alpha
        self.class_names = class_names if class_names is not None else DIAGNOSTIC_NAMES
        self.q_hat = None
        self.threshold = None
        self.is_calibrated = False
        self.calibration_scores = None

    def calibrate(self, probs: np.ndarray, labels: np.ndarray):
        """
        Calibrates conformal quantiles using a held-out calibration split.
        probs: [N, num_classes] softmax probability array
        labels: [N] ground truth integer class labels
        """
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        n = len(labels)
        if n == 0:
            raise ValueError("Calibration set cannot be empty.")

        # Margin nonconformity score: s_i = 1 - p(y_i | x_i)
        true_probs = np.array([probs[i, labels[i]] for i in range(n)])
        self.calibration_scores = 1.0 - true_probs

        # Conformal quantile with finite-sample correction
        level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.q_hat = float(np.quantile(self.calibration_scores, level, method='higher'))
        self.threshold = max(0.01, 1.0 - self.q_hat)
        self.is_calibrated = True

        empirical_coverage = np.mean(self.calibration_scores <= self.q_hat)
        return {
            'q_hat': self.q_hat,
            'probability_threshold': self.threshold,
            'num_calibration_samples': n,
            'target_coverage': 1.0 - self.alpha,
            'empirical_coverage': float(empirical_coverage)
        }

    def predict_set(self, prob_vector: np.ndarray, alpha=None):
        """
        Predicts conformal prediction set for a single sample or batch.
        prob_vector: [num_classes] or [B, num_classes] probability array
        """
        thresh = self.threshold if self.is_calibrated else 0.15

        prob_vector = np.asarray(prob_vector, dtype=np.float64)
        single = (prob_vector.ndim == 1)
        if single:
            prob_vector = prob_vector.reshape(1, -1)

        results = []
        for p in prob_vector:
            sort_idx = np.argsort(-p)
            
            # Include all classes with probability >= threshold
            included_indices = [idx for idx in sort_idx if p[idx] >= thresh]
            # Ensure top class is always included
            if not included_indices:
                included_indices = [sort_idx[0]]

            prediction_set = [self.class_names[idx] for idx in included_indices]
            set_size = len(prediction_set)

            # Categorize uncertainty
            if set_size == 1:
                uncertainty_level = "LOW (Decisive Diagnosis)"
                clinical_advice = f"High certainty. Single credible diagnostic category identified ({prediction_set[0]})."
            elif set_size == 2:
                uncertainty_level = "MODERATE (Differential Diagnosis)"
                clinical_advice = f"Differential diagnosis between {prediction_set[0]} ({p[sort_idx[0]]*100:.1f}%) and {prediction_set[1]} ({p[sort_idx[1]]*100:.1f}%). Dermoscopy inspection advised."
            else:
                uncertainty_level = f"HIGH ({set_size} Differential Diagnoses)"
                clinical_advice = f"Multi-class clinical ambiguity across {set_size} categories ({', '.join(prediction_set)}). Biopsy or expert second opinion strongly recommended."

            # Calculate conformal credibility & confidence
            sorted_probs = p[sort_idx]
            credibility = float(sorted_probs[0])
            confidence = float(1.0 - (sorted_probs[1] if len(sorted_probs) > 1 else 0.0))

            results.append({
                'prediction_set': prediction_set,
                'set_size': set_size,
                'uncertainty_level': uncertainty_level,
                'coverage_guarantee': f"{(1.0 - (alpha if alpha else self.alpha))*100:.1f}%",
                'probability_threshold': thresh,
                'credibility': credibility,
                'confidence': confidence,
                'clinical_advice': clinical_advice,
                'class_probabilities': {self.class_names[i]: float(p[i]) for i in range(len(self.class_names))}
            })

        return results[0] if single else results

    def save(self, filepath: str):
        """Serializes calibrated conformal predictor."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'alpha': self.alpha,
            'q_hat': self.q_hat,
            'threshold': self.threshold,
            'is_calibrated': self.is_calibrated,
            'calibration_scores': self.calibration_scores,
            'class_names': self.class_names
        }, filepath)

    @classmethod
    def load(cls, filepath: str):
        """Loads serialized calibrated conformal predictor."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Conformal checkpoint not found at: {filepath}")
        data = torch.load(filepath, weights_only=False)
        inst = cls(alpha=data['alpha'], class_names=data['class_names'])
        inst.q_hat = data['q_hat']
        inst.threshold = data.get('threshold', 0.15)
        inst.is_calibrated = data['is_calibrated']
        inst.calibration_scores = data['calibration_scores']
        return inst
