import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoModel


class DermatologicalChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation Channel Attention with Power Normalization.
    Enhances subtle dermatological lesions (pigment networks, vascular structures, borders)
    while suppressing background skin artifacts.
    """
    def __init__(self, in_features=2048, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features // reduction)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(in_features // reduction, in_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Power normalization (Signed Square Root + L2)
        x_pow = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-7)
        x_norm = F.normalize(x_pow, p=2, dim=-1)
        att = self.sigmoid(self.fc2(self.act(self.fc1(x_norm))))
        return x_norm * (1.0 + att)


class MHSA_TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Multi-Head Self-Attention Transformer Block.
    Applies multi-head self-attention with residual connection followed by feedforward MLP with residual connection.
    """
    def __init__(self, d_model=256, nhead=8, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, _ = self.mha(norm_x, norm_x, norm_x)
        x = x + self.dropout1(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class VisionTransformerImageEncoder(nn.Module):
    """
    Multi-Head Self-Attention Vision Transformer (ViT) Image Encoder.
    Extracts deep 2048-dim visual representations with ResNet50 backbone,
    applies Dermatological Channel Attention, tokenizes into visual patch tokens,
    prepends a learnable [CLS] token, adds 1D positional embeddings, and passes
    through multi-layer Multi-Head Self-Attention Transformer Encoder blocks.
    Outputs a contextualized 256-dim visual embedding.
    """
    def __init__(self, embed_dim=256, num_tokens=8, num_heads=8, num_layers=2, pretrained=True, freeze_backbone=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        try:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
        except Exception as e:
            print(f"[VisionTransformerImageEncoder] Loading fallback ResNet50 ({e})")
            self.backbone = models.resnet50(weights=None)

        self.backbone.fc = nn.Identity()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = 2048
        self.attention = DermatologicalChannelAttention(in_features=in_features, reduction=16)

        # Visual Token Projector: maps 2048-dim vector to sequence of `num_tokens` tokens
        self.token_proj = nn.Sequential(
            nn.Linear(in_features, num_tokens * embed_dim),
            nn.LayerNorm(num_tokens * embed_dim),
            nn.GELU(),
            nn.Dropout(0.10)
        )

        # Learnable CLS token and 1D positional encodings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens + 1, embed_dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            MHSA_TransformerBlock(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2, dropout=0.10)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # Backward compatibility wrapper
        self.projection = nn.Sequential(
            nn.Linear(in_features, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(384, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

    def extract_backbone_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        if x.dim() == 4:
            feats = self.extract_backbone_features(x)
        else:
            feats = x

        b = feats.size(0)
        feats = self.attention(feats)
        tokens = self.token_proj(feats).view(b, self.num_tokens, self.embed_dim)

        cls_tokens = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = tokens + self.pos_embed

        for blk in self.blocks:
            tokens = blk(tokens)

        cls_out = self.norm(tokens[:, 0, :])
        return self.head(cls_out)


# Alias ImageEncoder to VisionTransformerImageEncoder for seamless backward-compatibility
ImageEncoder = VisionTransformerImageEncoder


class TabTransformerEncoder(nn.Module):
    """
    TabTransformer Encoder for Heterogeneous Clinical Metadata (Huang et al.).
    Separates tabular features into categorical risk tokens and continuous clinical metrics.
    Projects risk factors into column embeddings, applies Multi-Head Self-Attention Transformer
    layers, projects continuous variables with LayerNorm, and passes through a residual MLP block
    to generate a 256-dim joint metadata representation.
    """
    def __init__(self, input_dim=116, embed_dim=256, col_dim=64, num_heads=4, num_layers=2, num_col_tokens=12):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_col_tokens = num_col_tokens
        self.col_dim = col_dim

        # Continuous features projection
        self.cont_proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.10)
        )

        # Categorical column embedding generator
        self.col_embedder = nn.Sequential(
            nn.Linear(input_dim, num_col_tokens * col_dim),
            nn.LayerNorm(num_col_tokens * col_dim),
            nn.GELU()
        )

        # Learnable CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, col_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_col_tokens + 1, col_dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            MHSA_TransformerBlock(d_model=col_dim, nhead=num_heads, dim_feedforward=col_dim * 2, dropout=0.10)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(col_dim)

        joint_dim = (num_col_tokens + 1) * col_dim + 128
        self.joint_mlp = nn.Sequential(
            nn.Linear(joint_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(384, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # Backward compatibility wrapper
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(384, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

    def forward(self, x):
        b = x.size(0)
        cont_feats = self.cont_proj(x)

        tokens = self.col_embedder(x).view(b, self.num_col_tokens, self.col_dim)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = tokens + self.pos_embed

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        flat_tokens = tokens.reshape(b, -1)
        joint_input = torch.cat([flat_tokens, cont_feats], dim=-1)
        return self.joint_mlp(joint_input)


class MetaBlockMLPEncoder(nn.Module):
    """
    Deep Residual MetaBlock MLP Encoder for Structured Clinical Metadata (116-dim -> 256-dim).
    Employs LayerNorm, GELU non-linearities, residual skip-connections, and dropout
    for high-precision tabular clinical representation learning.
    """
    def __init__(self, input_dim=116, embed_dim=256, hidden_dim=512, dropout=0.15):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.gelu = nn.GELU()
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

    def forward(self, x):
        h0 = self.input_layer(x)
        h1 = self.gelu(h0 + self.res_block(h0))
        return self.output_layer(h1)


# MetadataEncoder defaults to MetaBlockMLPEncoder as requested by the clinical architecture specification
MetadataEncoder = MetaBlockMLPEncoder


class LanguageEncoder(nn.Module):
    """
    Bio_ClinicalBERT Language Encoder extracting deep contextual representations
    from free-text clinical notes and mapping them into the 256-dim joint embedding.
    """
    def __init__(self, model_name='emilyalsentzer/Bio_ClinicalBERT', embed_dim=256, pretrained=True, freeze_backbone=True):
        super().__init__()
        try:
            self.bert = AutoModel.from_pretrained(model_name)
        except Exception:
            self.bert = AutoModel.from_pretrained(model_name, local_files_only=True)

        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False

        in_features = 768
        self.projection = nn.Sequential(
            nn.Linear(in_features, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(384, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

    def forward(self, input_ids=None, attention_mask=None, text_embed=None):
        if text_embed is not None:
            if text_embed.shape[-1] == 768:
                return self.projection(text_embed)
            return text_embed

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.projection(cls_token)

