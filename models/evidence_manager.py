import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon
import cv2

# Diagnosis-adaptive prior utility matrix W_prior [Diagnostic, Modality (Image, Metadata, Text)]
# Order of classes: 0: BCC, 1: ACK, 2: NEV, 3: SEK, 4: SCC, 5: MEL
DIAGNOSTIC_UTILITY_PRIOR = np.array([
    # Image, Metadata, Text
    [0.90, 0.80, 0.75],  # BCC: Strong visual telangiectasia + age/site
    [0.75, 0.85, 0.90],  # ACK: High history & symptom relevance (scaly, sun-damage)
    [0.92, 0.70, 0.70],  # NEV: High visual dermoscopy importance
    [0.85, 0.80, 0.88],  # SEK: Stuck-on appearance + age/location
    [0.88, 0.82, 0.92],  # SCC: High visual induration + clinical history (rapid growth/bleed)
    [0.96, 0.75, 0.90]   # MEL: Crucial dermoscopy ABCDE + patient history
], dtype=np.float32)

class EvidenceManager(nn.Module):
    """
    DERMA-GUARD Evidence Manager:
    Calculates the 5 Core Evidence Dimensions:
    - Q: Quality Score (Image features, PTIS text score, Metadata completeness)
    - R: Reliability Score (Predictive entropy, TTA stability, cross-modal agreement)
    - U: Utility Score (Diagnosis-specific prior utility)
    - C: Cross-Modal Consistency Score (Jensen-Shannon divergence)
    - S: Evidence Sufficiency Score (Clinical decision gating)
    - M: Missingness Mask

    Formulation:
        E_m = Q_m * R_m * U_m * C * S * M_m
        w_m = E_m / sum(E_j)
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim

        # Quality estimation MLP from raw feature descriptors
        self.quality_mlp = nn.Sequential(
            nn.Linear(5, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Reliability estimation MLP
        self.reliability_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    # -------------------------------------------------------------
    # 1. Quality (Q) Computations
    # -------------------------------------------------------------
    @staticmethod
    def compute_image_quality_metrics(image_tensor_or_np) -> np.ndarray:
        """
        Extracts 5 domain quality features:
        - blur (Laplacian variance)
        - brightness (deviation from optimal luminance)
        - contrast (standard deviation of luminance)
        - resolution (scale adequacy)
        - occlusion (border artifact score)
        """
        if isinstance(image_tensor_or_np, torch.Tensor):
            img = image_tensor_or_np.detach().cpu().numpy()
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = np.transpose(img, (1, 2, 0))
            img = np.clip((img * 0.225 + 0.45) * 255, 0, 255).astype(np.uint8)
        else:
            img = np.array(image_tensor_or_np)

        if img.ndim == 2:
            gray = img
        elif img.ndim == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img[:, :, 0]

        # 1. Blur via Laplacian variance
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = float(np.clip(lap_var / 500.0, 0.05, 1.0))

        # 2. Brightness score (optimal around 128)
        mean_lum = float(np.mean(gray))
        brightness_score = float(1.0 - abs(mean_lum - 128.0) / 128.0)
        brightness_score = np.clip(brightness_score, 0.1, 1.0)

        # 3. Contrast via standard deviation
        std_lum = float(np.std(gray))
        contrast_score = float(np.clip(std_lum / 64.0, 0.1, 1.0))

        # 4. Resolution score
        h, w = gray.shape[:2]
        res_score = float(np.clip((h * w) / (224.0 * 224.0), 0.2, 1.0))

        # 5. Occlusion / border darkness check
        border_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
        occlusion_score = float(1.0 - (np.sum(border_pixels < 25) / len(border_pixels)))
        occlusion_score = np.clip(occlusion_score, 0.1, 1.0)

        return np.array([blur_score, brightness_score, contrast_score, res_score, occlusion_score], dtype=np.float32)

    def calculate_q_image(self, img_features_np) -> float:
        feat_t = torch.tensor(img_features_np, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q = float(self.quality_mlp(feat_t).item())
        return float(np.clip(q, 0.1, 1.0))

    @staticmethod
    def calculate_q_metadata(missing_fraction: float) -> float:
        return float(np.clip(1.0 - missing_fraction, 0.05, 1.0))

    @staticmethod
    def calculate_q_text(ptis_score: float) -> float:
        return float(np.clip(ptis_score, 0.1, 1.0))

    # -------------------------------------------------------------
    # 2. Reliability (R) Computations
    # -------------------------------------------------------------
    @staticmethod
    def calculate_predictive_entropy(probs: np.ndarray) -> float:
        """Normalized predictive entropy H in [0, 1]. Lower entropy -> higher certainty."""
        eps = 1e-9
        k = len(probs)
        ent = -np.sum(probs * np.log(probs + eps)) / np.log(k)
        return float(np.clip(ent, 0.0, 1.0))

    def calculate_reliability(self, q: float, entropy: float, tta_variance: float = 0.03, agreement: float = 0.90) -> float:
        stability = 1.0 - float(np.clip(tta_variance, 0.0, 1.0))
        entropy_certainty = 1.0 - float(np.clip(entropy, 0.0, 1.0))
        feat = np.array([q, entropy_certainty, stability, agreement], dtype=np.float32)
        feat_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            r = float(self.reliability_mlp(feat_t).item())
        return float(np.clip(r, 0.1, 1.0))

    # -------------------------------------------------------------
    # 3. Utility (U) Computations
    # -------------------------------------------------------------
    @staticmethod
    def calculate_utility(pred_class_idx: int, modality_name: str) -> float:
        mod_idx_map = {'image': 0, 'metadata': 1, 'text': 2}
        m_idx = mod_idx_map.get(modality_name, 0)
        return float(DIAGNOSTIC_UTILITY_PRIOR[pred_class_idx, m_idx])

    # -------------------------------------------------------------
    # 4. Consistency (C) via Jensen-Shannon Divergence
    # -------------------------------------------------------------
    @staticmethod
    def calculate_consistency(predictions_list: list) -> float:
        """
        Calculates C(x) = 1 - 1/binom(M, 2) * sum JSD(p_i || p_j).
        If only 1 modality is available, returns 1.0.
        """
        num_mods = len(predictions_list)
        if num_mods <= 1:
            return 1.0

        divergences = []
        for i in range(num_mods):
            for j in range(i + 1, num_mods):
                p = np.clip(predictions_list[i], 1e-7, 1.0)
                p = p / np.sum(p)
                q = np.clip(predictions_list[j], 1e-7, 1.0)
                q = q / np.sum(q)
                jsd_val = float(jensenshannon(p, q) ** 2)
                divergences.append(jsd_val)

        if not divergences:
            return 1.0
        c_score = 1.0 - float(np.mean(divergences))
        return float(np.clip(c_score, 0.05, 1.0))

    # -------------------------------------------------------------
    # 5. Sufficiency (S) and Decision Gating
    # -------------------------------------------------------------
    @staticmethod
    def calculate_sufficiency(Q_list: list, R_list: list, C: float, missing_fraction: float) -> tuple:
        """
        Sufficiency formula:
        S = 0.35 * mean(Q) + 0.35 * mean(R) + 0.20 * C + 0.10 * (1 - missing_fraction)
        Decision Gate:
        - S < 0.30 -> INSUFFICIENT_EVIDENCE
        - 0.30 <= S < 0.50 -> ACQUIRE_MORE_EVIDENCE
        - S >= 0.50 -> PROCEED
        """
        mean_q = float(np.mean(Q_list)) if Q_list else 0.0
        mean_r = float(np.mean(R_list)) if R_list else 0.0
        avail = float(1.0 - missing_fraction)

        s_score = 0.35 * mean_q + 0.35 * mean_r + 0.20 * C + 0.10 * avail
        s_score = float(np.clip(s_score, 0.0, 1.0))

        if s_score < 0.30:
            gate = "INSUFFICIENT_EVIDENCE"
            action = "Abstain / Escalate"
        elif s_score < 0.50:
            gate = "ACQUIRE_MORE_EVIDENCE"
            action = "Acquire high-resolution dermoscopy or biopsy"
        else:
            gate = "PROCEED"
            action = "Proceed with clinical prediction"

        return s_score, gate, action

    # -------------------------------------------------------------
    # 6. Final Modality Evidence Score Computation
    # -------------------------------------------------------------
    @staticmethod
    def calculate_evidence_score(Q: float, R: float, U: float, C: float, S: float, M: float = 1.0) -> float:
        """
        Calculates modality evidence score:
        E_m = Q_m * R_m * U_m * C * S * M_m
        """
        evidence = float(Q * R * U * C * S * M)
        return float(np.clip(evidence, 0.0, 1.0))

    @staticmethod
    def normalize_weights(evidence_dict: dict) -> dict:
        """
        Converts modality evidence scores into dynamic fusion weights:
        w_m = E_m / sum(E_j)
        """
        total = sum(evidence_dict.values())
        if total <= 1e-9:
            # Fallback uniform
            n = len(evidence_dict)
            return {k: 1.0 / max(n, 1) for k in evidence_dict}
        return {k: float(v / total) for k, v in evidence_dict.items()}

    @staticmethod
    def get_evidence_level(evidence_score: float) -> tuple:
        """
        Converts evidence score into 5 clinical levels:
        - 0.80 - 1.00: Strong (Proceed)
        - 0.60 - 0.79: Good (Proceed with evidence trace)
        - 0.40 - 0.59: Moderate (Review / consider acquisition)
        - 0.20 - 0.39: Weak (Acquire more evidence)
        - 0.00 - 0.19: Insufficient (Abstain / escalate)
        """
        if evidence_score >= 0.80:
            return "Strong", "Proceed"
        elif evidence_score >= 0.60:
            return "Good", "Proceed with evidence trace"
        elif evidence_score >= 0.40:
            return "Moderate", "Review / consider acquisition"
        elif evidence_score >= 0.20:
            return "Weak", "Acquire more evidence"
        else:
            return "Insufficient", "Abstain / escalate"

    # -------------------------------------------------------------
    # 7. Consensus Level Evaluation
    # -------------------------------------------------------------
    @staticmethod
    def evaluate_consensus(predictions_list: list, C: float, mean_Q: float) -> str:
        """
        Four clinical consensus levels:
        - Level 1: Strong Consensus (>= 3 models agree or top conf >= 0.8 and C >= 0.3)
        - Level 2: Moderate Agreement (conf 0.5-0.8)
        - Level 3: High Uncertainty (conf < 0.5 or mean Q < 0.3)
        - Level 4: Critical Conflict (divergent high-confidence predictions)
        """
        if not predictions_list:
            return "Level 3: High Uncertainty"

        top_preds = [int(np.argmax(p)) for p in predictions_list]
        max_confs = [float(np.max(p)) for p in predictions_list]
        avg_conf = float(np.mean(max_confs))

        unique_preds, counts = np.unique(top_preds, return_counts=True)
        max_count = int(np.max(counts))

        if (max_count >= 3 or (len(predictions_list) >= 2 and max_count == len(predictions_list))) and avg_conf >= 0.8 and C >= 0.3:
            return "Level 1: Strong Consensus"
        elif avg_conf >= 0.5 and C >= 0.25:
            return "Level 2: Moderate Agreement"
        elif avg_conf < 0.5 or mean_Q < 0.3:
            return "Level 3: High Uncertainty (Acquire More Evidence)"
        else:
            return "Level 4: Critical Conflict (Escalate to Dermatologist)"

    @classmethod
    def determine_consensus_level(cls, predictions_list, top_confidence=0.85):
        """Helper to determine consensus level from string predictions or probability vectors."""
        if not predictions_list:
            return "Level 3: High Uncertainty"
        if isinstance(predictions_list[0], (list, np.ndarray, torch.Tensor)):
            return cls.evaluate_consensus(predictions_list, C=0.8, mean_Q=0.8)
        unique_labels = set(predictions_list)
        if len(unique_labels) == 1 and top_confidence >= 0.70:
            return "Level 1: Strong Consensus (All active modalities agree)"
        elif len(unique_labels) <= 2 and top_confidence >= 0.50:
            return "Level 2: Moderate Agreement"
        elif top_confidence < 0.50:
            return "Level 3: High Uncertainty (Acquire More Evidence)"
        else:
            return "Level 4: Critical Conflict (Escalate to Dermatologist)"
