"""
Standalone ablation experiment — does NOT modify core source files.
Each variant creates a custom FusionPipeline subclass and/or adjusts loss weights.

Usage: python scripts/run_ablation.py --gpu 0 --variant simple_concat --dataset weibo
Variants: full, simple_concat, concat_semantic, ce_only, no_contrastive, frozen_bert
"""
import sys, os, json, argparse, shutil
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data import load_dataset, MultimodalDataset
from models.encoders import CLIPVisionEncoder, TextEncoderWrapper
from models.fusion import BilinearFusion, CrossAttentionFusion, ModalityGate, \
    SemanticProjection, SemanticEncoder
from models.pipeline import FakeNewsDiscriminator
from train import train as train_mbsc

# ============================================================
# Custom Fusion Pipelines for ablation
# ============================================================

class SimpleConcatFusion(nn.Module):
    """Ablation A: simple img⊕txt concat, no semantic encoder."""
    def __init__(self, vision_encoder, text_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        img_dim = vision_encoder.hidden_size
        txt_dim = text_encoder.hidden_size
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, Config.FUSION_DIM),
            nn.LayerNorm(Config.FUSION_DIM),
        )
        self.txt_proj = nn.Sequential(
            nn.Linear(txt_dim, Config.FUSION_DIM),
            nn.LayerNorm(Config.FUSION_DIM),
        )
        self.final_proj = nn.Sequential(
            nn.Linear(Config.FUSION_DIM * 2, Config.X0_DIM),
            nn.LayerNorm(Config.X0_DIM), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(Config.X0_DIM, Config.X0_DIM),
            nn.LayerNorm(Config.X0_DIM),
        )
        self.x0_dim = Config.X0_DIM

    def forward(self, images, text_input_ids, text_attention_mask):
        with torch.no_grad():
            img_feat = self.vision_encoder(images)
        txt_feat = self.text_encoder(text_input_ids, text_attention_mask)
        img_p = self.img_proj(img_feat)
        txt_p = self.txt_proj(txt_feat)
        fused = torch.cat([img_p, txt_p], dim=-1)
        x0 = self.final_proj(fused)
        return x0, txt_p, img_feat, txt_feat


class ConcatSemanticFusion(nn.Module):
    """Ablation B: concat + SemanticEncoder (no bilinear/cross-attn/gate)."""
    def __init__(self, vision_encoder, text_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        img_dim = vision_encoder.hidden_size
        txt_dim = text_encoder.hidden_size
        self.img_proj = nn.Linear(img_dim, Config.FUSION_DIM)
        self.txt_proj = nn.Linear(txt_dim, Config.FUSION_DIM)
        self.final_proj = nn.Sequential(
            nn.Linear(Config.FUSION_DIM * 2, Config.X0_DIM),
            nn.LayerNorm(Config.X0_DIM), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(Config.X0_DIM, Config.X0_DIM),
        )
        self.semantic_encoder = SemanticEncoder()
        self.x0_dim = Config.X0_DIM

    def forward(self, images, text_input_ids, text_attention_mask):
        with torch.no_grad():
            img_feat = self.vision_encoder(images)
        txt_feat = self.text_encoder(text_input_ids, text_attention_mask)
        img_p = self.img_proj(img_feat)
        txt_p = self.txt_proj(txt_feat)
        fused = torch.cat([img_p, txt_p], dim=-1)
        x0 = self.semantic_encoder(self.final_proj(fused))
        return x0, txt_p, img_feat, txt_feat


# ============================================================
# Model builder: patches FakeNewsDiscriminator with custom fusion
# ============================================================

def build_ablated_model(variant, device, text_encoder_type=None):
    """Build FakeNewsDiscriminator with ablated fusion pipeline."""
    text_encoder_type = text_encoder_type or Config.TEXT_ENCODER_TYPE

    # Create encoders
    vision_encoder = CLIPVisionEncoder()
    for p in vision_encoder.parameters():
        p.requires_grad = False

    text_encoder_wrapper = TextEncoderWrapper(text_encoder_type)
    if not Config.UNFREEZE_TEXT_ENCODER:
        for p in text_encoder_wrapper.parameters():
            p.requires_grad = False

    # Build custom fusion
    if variant == "simple_concat":
        fusion = SimpleConcatFusion(vision_encoder, text_encoder_wrapper.encoder)
    elif variant == "concat_semantic":
        fusion = ConcatSemanticFusion(vision_encoder, text_encoder_wrapper.encoder)
    else:
        # full fusion: import the original
        from models.fusion import FusionPipeline
        fusion = FusionPipeline(vision_encoder, text_encoder_wrapper.encoder)

    # Build model and replace fusion_pipeline
    model = FakeNewsDiscriminator.__new__(FakeNewsDiscriminator)
    nn.Module.__init__(model)
    model.vision_encoder = vision_encoder
    model.text_encoder = text_encoder_wrapper
    model.fusion_pipeline = fusion

    # Classifier
    model.classifier = nn.Sequential(
        nn.Linear(Config.X0_DIM, Config.DENOISER_HIDDEN), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(Config.DENOISER_HIDDEN, Config.DENOISER_HIDDEN // 2), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(Config.DENOISER_HIDDEN // 2, 2),
    )
    # Projection head
    model.projection_head = nn.Sequential(
        nn.Linear(Config.X0_DIM, 256), nn.ReLU(),
        nn.Linear(256, Config.PROJECTION_DIM),
    )

    # Bind methods
    model.forward_pipeline = lambda imgs, tids, tmask: model.fusion_pipeline(imgs, tids, tmask)
    model.forward_classifier = lambda x0: model.classifier(x0)
    model.forward_projection = FakeNewsDiscriminator.forward_projection.__get__(model)
    model.get_trainable_parameters = lambda: (
        list(model.fusion_pipeline.parameters())
        + list(model.classifier.parameters())
        + list(model.projection_head.parameters())
    )

    model = model.to(device)
    return model


# ============================================================
# Custom training loop (handles loss ablation)
# ============================================================

import random
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from train import set_seed, supervised_contrastive_loss, validate


def train_ablated(model, train_loader, val_loader, device, variant, ckpt_dir):
    """Train with optional loss ablation."""
    use_align = False  # alignment loss removed
    use_contrastive = variant not in ("no_contrastive", "ce_only")  # both skip contrastive

    # Reduce LR for simple_concat to avoid NaN
    lr = Config.LEARNING_RATE * 0.1 if variant == "simple_concat" else Config.LEARNING_RATE
    print(f"  LR={lr}")

    optimizer = AdamW(model.get_trainable_parameters(),
                      lr=lr, weight_decay=Config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)

    best_val_f1 = 0
    patience = getattr(Config, 'PATIENCE', 50)
    patience_counter = 0
    history = {"train_loss": [], "val_f1": []}
    has_val = val_loader is not None
    n_epochs = Config.NUM_EPOCHS if has_val else 30

    for epoch in range(n_epochs):
        model.train()
        model.vision_encoder.eval()
        total_loss = total_cls = total_cont = 0
        skipped = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for batch in pbar:
            imgs = batch["images"].to(device)
            tids = batch["text_input_ids"].to(device)
            tmask = batch["text_attention_mask"].to(device)
            labels = batch["label"].to(device)

            x0, _, img_feat, txt_feat = model.forward_pipeline(imgs, tids, tmask)

            loss_cls = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)(
                model.forward_classifier(x0), labels)

                          if use_align else torch.tensor(0.0, device=device))

            loss_cont = (supervised_contrastive_loss(model.forward_projection(x0), labels)
                         if use_contrastive else torch.tensor(0.0, device=device))

            loss = Config.CLASSIFIER_LOSS_WEIGHT * loss_cls
            # alignment loss removed
