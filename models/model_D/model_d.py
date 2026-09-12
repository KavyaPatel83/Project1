import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoders import ImageEncoder, MetadataEncoder
from models.adaptive_fusion import AdaptiveGatedFusion

class ModelD(nn.Module):
    """
    Model D: Image + Structured Metadata Multimodal Classification Model.
    Fuses ResNet50 visual features with Deep Residual MetaBlock metadata representations
    using DERMA-GUARD Evidence-Aware Adaptive Gated Fusion.
    """
    def __init__(self, meta_input_dim=109, embed_dim=256, num_classes=6, pretrained=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_encoder = ImageEncoder(embed_dim=embed_dim, pretrained=pretrained)
        self.meta_encoder = MetadataEncoder(input_dim=meta_input_dim, embed_dim=embed_dim)
        self.fusion = AdaptiveGatedFusion(embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, img=None, meta=None, img_embed=None, meta_embed=None, ev_img=None, ev_meta=None):
        if img_embed is None and img is not None:
            img_embed = self.image_encoder(img)
        elif img_embed is not None and img_embed.shape[-1] != self.embed_dim:
            img_embed = self.image_encoder.projection(img_embed)

        if meta_embed is None and meta is not None:
            meta_embed = self.meta_encoder(meta)
        elif meta_embed is not None and meta_embed.shape[-1] != self.embed_dim:
            meta_embed = self.meta_encoder(meta_embed)

        batch_size = img_embed.size(0)
        device = img_embed.device
        if ev_img is None:
            ev_img = torch.ones(batch_size, 1, device=device) * 0.85
        if ev_meta is None:
            ev_meta = torch.ones(batch_size, 1, device=device) * 0.80

        logits, dynamic_weights, fused_rep = self.fusion(
            [img_embed, meta_embed],
            [ev_img, ev_meta]
        )
        return logits, dynamic_weights, fused_rep

    def predict_proba(self, img=None, meta=None, img_embed=None, meta_embed=None, ev_img=None, ev_meta=None):
        self.eval()
        with torch.no_grad():
            logits, weights, fused_rep = self.forward(img, meta, img_embed, meta_embed, ev_img, ev_meta)
            probs = F.softmax(logits, dim=-1)
        return probs, weights, fused_rep
