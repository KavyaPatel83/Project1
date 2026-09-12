import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoders import LanguageEncoder, MetadataEncoder
from models.adaptive_fusion import AdaptiveGatedFusion

class ModelF(nn.Module):
    """
    Model F: Free Text + Metadata Multimodal Classification Model.
    Fuses ClinicalBERT contextual narrative features with Deep Residual MetaBlock
    metadata representations using Evidence-Aware Adaptive Gated Fusion.
    """
    def __init__(self, meta_input_dim=109, embed_dim=256, num_classes=6, bert_name='emilyalsentzer/Bio_ClinicalBERT'):
        super().__init__()
        self.embed_dim = embed_dim
        self.meta_encoder = MetadataEncoder(input_dim=meta_input_dim, embed_dim=embed_dim)
        self.text_encoder = LanguageEncoder(model_name=bert_name, embed_dim=embed_dim)
        self.fusion = AdaptiveGatedFusion(embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, meta=None, input_ids=None, attention_mask=None, meta_embed=None, text_embed=None, ev_meta=None, ev_text=None):
        if meta_embed is None and meta is not None:
            meta_embed = self.meta_encoder(meta)
        elif meta_embed is not None and meta_embed.shape[-1] != self.embed_dim:
            meta_embed = self.meta_encoder(meta_embed)

        if text_embed is None and input_ids is not None:
            text_embed = self.text_encoder(input_ids, attention_mask)
        elif text_embed is not None and text_embed.shape[-1] != self.embed_dim:
            text_embed = self.text_encoder.projection(text_embed)

        batch_size = meta_embed.size(0)
        device = meta_embed.device
        if ev_meta is None:
            ev_meta = torch.ones(batch_size, 1, device=device) * 0.80
        if ev_text is None:
            ev_text = torch.ones(batch_size, 1, device=device) * 0.85

        logits, dynamic_weights, fused_rep = self.fusion(
            [text_embed, meta_embed],
            [ev_text, ev_meta]
        )
        return logits, dynamic_weights, fused_rep

    def predict_proba(self, meta=None, input_ids=None, attention_mask=None, meta_embed=None, text_embed=None, ev_meta=None, ev_text=None):
        self.eval()
        with torch.no_grad():
            logits, weights, fused_rep = self.forward(meta, input_ids, attention_mask, meta_embed, text_embed, ev_meta, ev_text)
            probs = F.softmax(logits, dim=-1)
        return probs, weights, fused_rep
