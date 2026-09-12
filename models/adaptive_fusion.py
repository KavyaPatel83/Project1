import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AdaptiveGatedFusion(nn.Module):
    """
    Evidence-Aware Adaptive Gated Fusion with Diagnosis-Adaptive Prior Refinement.
    Combines active modality embeddings dynamically weighted by evidence scores E_m.
    """
    def __init__(self, embed_dim=256, num_classes=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Gating network computing attention logit per modality from (embedding + evidence)
        self.gating_net = nn.Sequential(
            nn.Linear(embed_dim + 1, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Residual fusion projection
        self.fusion_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Final multi-class classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        self.act = nn.GELU()

    def forward(self, modality_embeddings: list, evidence_scores: list):
        """
        modality_embeddings: list of Tensors [B, embed_dim]
        evidence_scores: list of Tensors [B, 1]
        """
        # Apply stochastic modality dropout during training if >1 modality
        if self.training and len(modality_embeddings) > 1 and np.random.rand() < 0.15:
            drop_idx = np.random.randint(0, len(modality_embeddings))
            modality_embeddings = [
                torch.zeros_like(emb) if i == drop_idx else emb
                for i, emb in enumerate(modality_embeddings)
            ]

        gate_logits = []
        for emb, ev in zip(modality_embeddings, evidence_scores):
            emb_n = F.normalize(emb, p=2, dim=-1)
            x = torch.cat([emb_n, ev], dim=1)
            gate_logits.append(self.gating_net(x))

        gate_logits_stacked = torch.cat(gate_logits, dim=1)  # [B, M]
        # Temperature scaling prevents single-modality winner-take-all collapse
        raw_weights = F.softmax(gate_logits_stacked / 2.0, dim=1)  # [B, M]

        # Blend gating weights with input evidence prior
        ev_stacked = torch.cat(evidence_scores, dim=1)  # [B, M]
        ev_prior = ev_stacked / (torch.sum(ev_stacked, dim=1, keepdim=True) + 1e-7)
        dynamic_weights = 0.60 * raw_weights + 0.40 * ev_prior
        dynamic_weights = dynamic_weights / torch.sum(dynamic_weights, dim=1, keepdim=True)

        # Weighted combination
        fused = torch.zeros_like(modality_embeddings[0])
        for i in range(len(modality_embeddings)):
            w = dynamic_weights[:, i:i+1]
            fused = fused + w * modality_embeddings[i]

        proj = self.fusion_projection(fused)
        fused_rep = self.act(fused + proj)
        logits = self.classifier(fused_rep)

        return logits, dynamic_weights, fused_rep

    def refine_weights_with_prior(self, initial_weights: np.ndarray, pred_class_idx: int,
                                   prior_utility: np.ndarray, alpha=0.6, beta=0.4) -> np.ndarray:
        """
        Diagnosis-adaptive iterative refinement:
        w^(k+1) = softmax(alpha * log(w^k + eps) + beta * log(pi(y^k) + eps))
        """
        eps = 1e-7
        log_w = np.log(np.clip(initial_weights, eps, 1.0))
        log_pi = np.log(np.clip(prior_utility, eps, 1.0))
        updated_logits = alpha * log_w + beta * log_pi
        exps = np.exp(updated_logits - np.max(updated_logits))
        refined_w = exps / np.sum(exps)
        return refined_w
