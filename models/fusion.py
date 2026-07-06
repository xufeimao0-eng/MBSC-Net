"""
Feature Fusion: Hadamard Bilinear + Cross-Attention + Gating.

Key components:
  1. Hadamard bilinear fusion: captures diagonal multiplicative img*txt interactions
  2. Cross-attention: image attends to text
  3. Modality gating: learnable dynamic weighting per modality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class BilinearFusion(nn.Module):
    """Hadamard bilinear fusion: W @ (img * txt), captures diagonal multiplicative interactions.

    Uses element-wise product + linear transform (512^2+512 ~ 0.26M params) instead
    of full outer-product bilinear (512^3 = 134.2M params).
    """

    def __init__(self, img_dim=768, txt_dim=512, output_dim=None):
        super().__init__()
        output_dim = output_dim or Config.FUSION_DIM
        self.img_proj = nn.Linear(img_dim, output_dim)
        self.txt_proj = nn.Linear(txt_dim, output_dim)
        self.linear = nn.Linear(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, img_feat, txt_feat):
        img = self.img_proj(img_feat)  # (B, hidden)
        txt = self.txt_proj(txt_feat)  # (B, hidden)
        hadamard = img * txt           # (B, hidden) element-wise
        out = self.linear(hadamard)    # (B, hidden)
        return self.norm(out)


class CrossAttentionFusion(nn.Module):
    """Cross-attention: image attends to text."""

    def __init__(self, img_dim=768, txt_dim=512, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or Config.FUSION_DIM
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.txt_proj = nn.Linear(txt_dim, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, img_feat, txt_feat):
        img = self.img_proj(img_feat)
        txt = self.txt_proj(txt_feat)
        q = self.q_proj(img).unsqueeze(1)
        k = self.k_proj(txt).unsqueeze(1)
        v = self.v_proj(txt).unsqueeze(1)
        attn = torch.softmax(q @ k.transpose(-2, -1) / self.scale, dim=-1)
        attended = (attn @ v).squeeze(1)
        out = self.out_proj(attended)
        out = self.norm(out)
        gate = torch.sigmoid(self.gate)
        return gate * out + (1 - gate) * img


class ModalityGate(nn.Module):
    """Dynamic modality weighting: gate = sigma(W.[img, txt])."""

    def __init__(self, dim=None):
        super().__init__()
        dim = dim or Config.FUSION_DIM
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, img_feat, txt_feat):
        gate = self.gate_net(torch.cat([img_feat, txt_feat], dim=-1))  # (B, 1)
        return gate * img_feat + (1 - gate) * txt_feat


class SemanticProjection(nn.Module):
    """Project text features to deep semantic representation."""

    def __init__(self, txt_dim=512, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or Config.FUSION_DIM
        self.net = nn.Sequential(
            nn.Linear(txt_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, txt_feat):
        return self.net(txt_feat)


class SemanticEncoder(nn.Module):
    """Lightweight Transformer for semantic context modeling."""

    def __init__(self, dim=None, num_layers=None, num_heads=None):
        super().__init__()
        dim = dim or Config.X0_DIM
        num_layers = num_layers or Config.SEMANTIC_NUM_LAYERS
        num_heads = num_heads or Config.SEMANTIC_NUM_HEADS
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 4,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.norm(x)


class FusionPipeline(nn.Module):
    """
    Fusion Pipeline:
      CLIP features -> Hadamard Bilinear + CrossAttn + Gating -> X0
      Outputs features for contrastive loss.
    """

    def __init__(self, vision_encoder, text_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

        img_dim = vision_encoder.hidden_size   # 768
        txt_dim = text_encoder.hidden_size     # 512

        self.bilinear = BilinearFusion(img_dim, txt_dim, Config.FUSION_DIM)
        self.cross_attn = CrossAttentionFusion(img_dim, txt_dim, Config.FUSION_DIM)
        self.semantic_proj = SemanticProjection(txt_dim, Config.FUSION_DIM)
        self.gate = ModalityGate(Config.FUSION_DIM)

        # Fuse bilinear + cross_attn + gated -> X0
        fusion_input_dim = Config.FUSION_DIM * 2 + Config.FUSION_DIM  # bilinear + crossattn + gated
        self.final_proj = nn.Sequential(
            nn.Linear(fusion_input_dim, Config.X0_DIM),
            nn.LayerNorm(Config.X0_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(Config.X0_DIM, Config.X0_DIM),
        )
        self.semantic_encoder = SemanticEncoder()
        self.x0_dim = Config.X0_DIM

    def forward(self, images, text_input_ids, text_attention_mask):
        with torch.no_grad():
            img_feat = self.vision_encoder(images)
        txt_feat = self.text_encoder(text_input_ids, text_attention_mask)

        # Multi-branch fusion
        bilinear_out = self.bilinear(img_feat, txt_feat)          # (B, FUSION_DIM)
        cross_out = self.cross_attn(img_feat, txt_feat)           # (B, FUSION_DIM)

        t_sem = self.semantic_proj(txt_feat)                       # (B, FUSION_DIM)
        # Project img_feat to FUSION_DIM for gating
        img_proj = self.bilinear.img_proj(img_feat)                # (B, FUSION_DIM)
        gated = self.gate(img_proj, t_sem)                         # (B, FUSION_DIM)
        gated = self.gate(cross_out, gated)

        # Fuse all branches -> X0
        fused = torch.cat([bilinear_out, cross_out, gated], dim=-1)
        x0 = self.semantic_encoder(self.final_proj(fused))

        return x0, t_sem, img_feat, txt_feat
