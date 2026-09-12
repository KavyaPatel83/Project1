import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoders import LanguageEncoder

class ModelC(nn.Module):
    """
    Model C: Free Text-Only Classification Baseline.
    Processes clinical patient narratives and symptomatology using ClinicalBERT (Bio_ClinicalBERT).
    Extracts 256-dim clinical textual representations and classifies into 6 diagnostic classes:
    [BCC, ACK, NEV, SEK, SCC, MEL].
    """
    def __init__(self, embed_dim=256, num_classes=6, bert_name='emilyalsentzer/Bio_ClinicalBERT'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.text_encoder = LanguageEncoder(model_name=bert_name, embed_dim=embed_dim)
        self.res_block = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.15)
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids=None, attention_mask=None, text_embed=None):
        if text_embed is None and input_ids is not None:
            text_embed = self.text_encoder(input_ids, attention_mask)
        elif text_embed is not None and text_embed.shape[-1] != self.embed_dim:
            text_embed = self.text_encoder.projection(text_embed)

        h = text_embed + self.res_block(text_embed)
        logits = self.classifier(h)
        return logits, h

    def predict_proba(self, input_ids=None, attention_mask=None, text_embed=None):
        self.eval()
        with torch.no_grad():
            logits, feats = self.forward(input_ids=input_ids, attention_mask=attention_mask, text_embed=text_embed)
            probs = F.softmax(logits, dim=-1)
        return probs, feats
