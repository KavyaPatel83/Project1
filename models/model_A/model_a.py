import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoders import ImageEncoder

class ModelA(nn.Module):
    """
    Model A: Multi-Head Self-Attention Vision Transformer (ViT) Image Classification Model.
    Processes skin lesion images using ResNet50 backbone + Dermatological Channel Attention
    and Multi-Head Self-Attention Vision Transformer Encoder (8 attention heads, CLS token)
    into 6 diagnostic classes: [BCC, ACK, NEV, SEK, SCC, MEL].
    """
    def __init__(self, embed_dim=256, num_classes=6, pretrained=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.image_encoder = ImageEncoder(embed_dim=embed_dim, pretrained=pretrained)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, img=None, img_embed=None):
        if img_embed is None and img is not None:
            img_embed = self.image_encoder(img)
        elif img_embed is not None:
            if img_embed.shape[-1] in (2048, 1792):
                img_embed = self.image_encoder(img_embed)
            elif img_embed.shape[-1] != self.embed_dim:
                img_embed = self.image_encoder.projection(img_embed)

        logits = self.classifier(img_embed)
        return logits, img_embed

    def predict_proba(self, img=None, img_embed=None):
        self.eval()
        with torch.no_grad():
            logits, feats = self.forward(img=img, img_embed=img_embed)
            probs = F.softmax(logits, dim=-1)
        return probs, feats
