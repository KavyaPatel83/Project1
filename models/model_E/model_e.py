import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoders import ImageEncoder, LanguageEncoder
from models.adaptive_fusion import AdaptiveGatedFusion

class ModelE(nn.Module):
    """
    Model E: Image + Free Text Multimodal Classification Model.
    Fuses ResNet50 visual features with Bio_ClinicalBERT textual representations
    using DERMA-GUARD Evidence-Aware Adaptive Gated Fusion.
    """
    def __init__(self, embed_dim=256, num_classes=6, bert_name='emilyalsentzer/Bio_ClinicalBERT', pretrained=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_encoder = ImageEncoder(embed_dim=embed_dim, pretrained=pretrained)
        self.text_encoder = LanguageEncoder(model_name=bert_name, embed_dim=embed_dim)
        self.fusion = AdaptiveGatedFusion(embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, img=None, input_ids=None, attention_mask=None, img_embed=None, text_embed=None, ev_img=None, ev_text=None):
        if img_embed is None and img is not None:
            img_embed = self.image_encoder(img)
        elif img_embed is not None and img_embed.shape[-1] != self.embed_dim:
            img_embed = self.image_encoder.projection(img_embed)

        if text_embed is None and input_ids is not None:
            text_embed = self.text_encoder(input_ids, attention_mask)
        elif text_embed is not None and text_embed.shape[-1] != self.embed_dim:
            text_embed = self.text_encoder.projection(text_embed)

        batch_size = img_embed.size(0)
        device = img_embed.device
        if ev_img is None:
            ev_img = torch.ones(batch_size, 1, device=device) * 0.85
        if ev_text is None:
            ev_text = torch.ones(batch_size, 1, device=device) * 0.85

        logits, dynamic_weights, fused_rep = self.fusion(
            [img_embed, text_embed],
            [ev_img, ev_text]
        )
        return logits, dynamic_weights, fused_rep

    def predict_proba(self, img=None, input_ids=None, attention_mask=None, img_embed=None, text_embed=None, ev_img=None, ev_text=None):
        self.eval()
        with torch.no_grad():
            logits, weights, fused_rep = self.forward(img, input_ids, attention_mask, img_embed, text_embed, ev_img, ev_text)
            probs = F.softmax(logits, dim=-1)
        return probs, weights, fused_rep
