"""
Multimodal Fake News Discriminator — MBSC-Net:
  1. Hadamard bilinear fusion + cross-attention + modality gating
  2. Intra-batch supervised contrastive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from models.encoders import CLIPVisionEncoder, TextEncoderWrapper
from models.fusion import FusionPipeline


class FakeNewsDiscriminator(nn.Module):

    def __init__(self, text_encoder_type=None):
        super().__init__()
        text_encoder_type = text_encoder_type or Config.TEXT_ENCODER_TYPE

        self.vision_encoder = CLIPVisionEncoder()
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        self.text_encoder = TextEncoderWrapper(text_encoder_type)
        if not Config.UNFREEZE_TEXT_ENCODER:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.fusion_pipeline = FusionPipeline(
            vision_encoder=self.vision_encoder,
            text_encoder=self.text_encoder.encoder,
        )

        # Main classifier on X0
        self.classifier = nn.Sequential(
            nn.Linear(Config.X0_DIM, Config.DENOISER_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(Config.DENOISER_HIDDEN, Config.DENOISER_HIDDEN // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(Config.DENOISER_HIDDEN // 2, 2),
        )

        # Projection head for intra-batch contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(Config.X0_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, Config.PROJECTION_DIM),
        )

    def forward_pipeline(self, images, text_input_ids, text_attention_mask):
        """Returns X0, condition, img_feat, txt_feat."""
        return self.fusion_pipeline(images, text_input_ids, text_attention_mask)

    def forward_classifier(self, x0):
        return self.classifier(x0)

    def forward_projection(self, x0):
        proj = self.projection_head(x0)
        return F.normalize(proj, dim=-1)

    def get_trainable_parameters(self):
        params = (list(self.fusion_pipeline.parameters())
                  + list(self.classifier.parameters())
                  + list(self.projection_head.parameters()))
        return params
