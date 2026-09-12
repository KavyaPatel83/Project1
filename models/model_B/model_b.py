import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoders import MetadataEncoder

class ModelB(nn.Module):
    """
    Model B: TabTransformer Clinical Metadata Classification Model.
    Processes heterogeneous patient metadata & clinical risk factors using canonical TabTransformer
    (Column Embeddings, Multi-Head Self-Attention Transformer layers, Continuous feature projections, and MLP)
    into 6 diagnostic categories: [BCC, ACK, NEV, SEK, SCC, MEL].
    """
    def __init__(self, meta_input_dim=116, embed_dim=256, num_classes=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.meta_encoder = MetadataEncoder(input_dim=meta_input_dim, embed_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, meta=None, meta_embed=None):
        if meta_embed is None and meta is not None:
            meta_embed = self.meta_encoder(meta)
        elif meta_embed is not None and meta_embed.shape[-1] != self.embed_dim:
            meta_embed = self.meta_encoder(meta_embed)

        logits = self.classifier(meta_embed)
        return logits, meta_embed

    def predict_proba(self, meta=None, meta_embed=None):
        self.eval()
        with torch.no_grad():
            logits, feats = self.forward(meta=meta, meta_embed=meta_embed)
            probs = F.softmax(logits, dim=-1)
        return probs, feats
