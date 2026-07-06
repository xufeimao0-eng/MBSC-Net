"""
Baseline comparison — MBSC-Net data pipeline + encoders.
Models: spotfake, eann, safe, hmcan, pivot, bmr
"""
import sys, os, json, argparse, shutil, random
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from data import load_dataset, MultimodalDataset
from models.encoders import CLIPVisionEncoder, BertEncoder
from models.fusion import SemanticEncoder
from models import get_clip_processor, get_tokenizer

# ============================================================
# Gradient Reversal Layer
# ============================================================
class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

# ============================================================
# Model: SpotFake (2019)
# ============================================================
class SpotFakeModel(nn.Module):
    def __init__(self, vision_encoder, text_encoder, txt_dim=768):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        img_dim = vision_encoder.hidden_size
        self.img_proj = nn.Linear(img_dim, Config.FUSION_DIM)
        self.txt_proj = nn.Linear(txt_dim, Config.FUSION_DIM)
        self.classifier = nn.Sequential(
            nn.Linear(Config.FUSION_DIM * 2, Config.FUSION_DIM),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(Config.FUSION_DIM, 2),
        )

    def forward(self, images, text_input_ids, text_attention_mask):
        with torch.no_grad():
            img_feat = self.vision_encoder(images)
        txt_feat = self.text_encoder(text_input_ids, text_attention_mask)
        fused = torch.cat([self.img_proj(img_feat), self.txt_proj(txt_feat)], dim=-1)
        return self.classifier(fused), img_feat, txt_feat

# ============================================================
# Model: EANN (KDD 2018)
# ============================================================
class EANNModel(nn.Module):
    def __init__(self, vision_encoder, text_encoder, num_events, txt_dim=768):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        img_dim = vision_encoder.hidden_size
        self.img_proj = nn.Linear(img_dim, Config.FUSION_DIM)
        self.txt_proj = nn.Linear(txt_dim, Config.FUSION_DIM)
        hidden = Config.FUSION_DIM * 2
        self.fake_classifier = nn.Sequential(
            nn.Linear(hidden, Config.FUSION_DIM),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(Config.FUSION_DIM, 2),
        )
        self.event_classifier = nn.Sequential(
            nn.Linear(hidden, Config.FUSION_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(Config.FUSION_DIM, num_events),
        )
        self.grl_lambda = 1.0

    def forward(self, images, text_input_ids, text_attention_mask, apply_grl=True):
        with torch.no_grad():
            img_feat = self.vision_encoder(images)
        txt_feat = self.text_encoder(text_input_ids, text_attention_mask)
        fused = torch.cat([self.img_proj(img_feat), self.txt_proj(txt_feat)], dim=-1)
        fake_logits = self.fake_classifier(fused)
        if apply_grl:
            reversed_f = GradientReverseLayer.apply(fused, self.grl_lambda)
            event_logits = self.event_classifier(reversed_f)
        else:
            event_logits = self.event_classifier(fused)
        return fake_logits, event_logits, img_feat, txt_feat

# ============================================================
# Model: SAFE (PAKDD 2020)
# ============================================================
class SAFEModel(nn.Module):
    def __init__(self, vision_encoder, text_encoder, txt_dim=768):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        img_dim = vision_encoder.hidden_size
        self.img_proj = nn.Linear(img_dim, Config.FUSION_DIM)
        self.txt_proj = nn.Linear(txt_dim, Config.FUSION_DIM)
        self.sim_fc = nn.Sequential(
            nn.Linear(1, Config.FUSION_DIM), nn.ReLU(),
            nn.Linear(Config.FUSION_DIM, 1), nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(Config.FUSION_DIM * 2, Config.FUSION_DIM),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(Config.FUSION_DIM, 2),
        )

    def forward(self, images, text_input_ids, text_attention_mask):
        with torch.no_grad():
            img_feat = self.vision_encoder(images)
        txt_feat = self.text_encoder(text_input_ids, text_attention_mask)
        img_p = F.normalize(self.img_proj(img_feat), dim=-1)
        txt_p = F.normalize(self.txt_proj(txt_feat), dim=-1)
        cos_sim = (img_p * txt_p).sum(dim=-1, keepdim=True)
        fused = torch.cat([img_p, txt_p], dim=-1)
        logits = self.classifier(fused)
        return logits, cos_sim.squeeze(-1), img_feat, txt_feat

# ============================================================
# Model: HMCAN (SIGIR 2021) — Co-Attention
# ============================================================
class CoAttentionLayer(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.W_t = nn.Linear(dim, dim, bias=False)
        self.W_i = nn.Linear(dim, dim, bias=False)
        self.W_h = nn.Linear(dim, 1, bias=False)

    def forward(self, txt_feat, img_feat):
        t = self.W_t(txt_feat).unsqueeze(1)
        i = self.W_i(img_feat).unsqueeze(1)
        attn_i = torch.sigmoid(self.W_h(torch.tanh(t + i)).squeeze(-1))
        attended_img = attn_i * img_feat
        attn_t = torch.sigmoid(self.W_h(torch.tanh(i + t)).squeeze(-1))
        attended_txt = attn_t * txt_feat
        return attended_txt, attended_img

class HMCANModel(nn.Module):
    def __init__(self, vision_encoder, text_encoder, txt_dim=768):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        img_dim = vision_encoder.hidden_size
        self.img_proj = nn.Linear(img_dim, Config.FUSION_DIM)
        self.txt_proj = nn.Linear(txt_dim, Config.FUSION_DIM)
        self.co_attn = CoAttentionLayer(Config.FUSION_DIM)
        self.classifier = nn.Sequential(
            nn.Linear(Config.FUSION_DIM * 2, Config.FUSION_DIM),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(Config.FUSION_DIM, 2),
        )

    def forward(self, images, text_input_ids, text_attention_mask):
        with torch.no_grad():
            img_feat = self.vision_encoder(images)
        txt_feat = self.text_encoder(text_input_ids, text_attention_mask)
        img_p = self.img_proj(img_feat)
        txt_p = self.txt_proj(txt_feat)
        attended_txt, attended_img = self.co_attn(txt_p, img_p)
        fused = torch.cat([attended_txt, attended_img], dim=-1)
        return self.classifier(fused), img_feat, txt_feat

# ============================================================
# Model: Pivot Transformer (MMDFND-inspired)
# ============================================================
class TransformerLayer(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )
    def forward(self, x):
        x = self.norm1(x + self.attn(x, x, x)[0])
        x = self.norm2(x + self.ffn(x))
        return x

class PivotFusionModel(nn.Module):
    def __init__(self, vision_encoder, text_encoder, txt_dim=768, num_layers=6):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        img_dim = vision_encoder.hidden_size
        dim = Config.FUSION_DIM
        self.num_stars = 4
        self.num_features = 4

        self.img_proj = nn.Linear(img_dim, dim)
        self.txt_proj = nn.Linear(txt_dim, dim)
        self.img_mlps = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
                                        for _ in range(self.num_features)])
        self.txt_mlps = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
                                        for _ in range(self.num_features)])
        self.stars = nn.Parameter(torch.randn(self.num_stars, dim) * 0.02)
        self.transformers = nn.ModuleList([TransformerLayer(dim) for _ in range(num_layers)])
        self.fusion_proj = nn.Sequential(
            nn.Linear(dim * self.num_stars, dim), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(dim, dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(0.3), nn.Linear(dim, 2),
        )

    def forward(self, images, text_input_ids, text_attention_mask):
        B = images.shape[0]
        with torch.no_grad():
            img_feat = self.vision_encoder(images)
        txt_feat = self.text_encoder(text_input_ids, text_attention_mask)
        img_p = self.img_proj(img_feat)
        txt_p = self.txt_proj(txt_feat)
        img_views = torch.stack([mlp(img_p) for mlp in self.img_mlps], dim=1)
        txt_views = torch.stack([mlp(txt_p) for mlp in self.txt_mlps], dim=1)
        stars = self.stars.unsqueeze(0).expand(B, -1, -1)

        for i, transformer in enumerate(self.transformers):
            if i % 2 == 0:
                seq = torch.cat([stars, txt_views], dim=1)
                out = transformer(seq)
                stars = (out[:, :self.num_stars] + stars) / 2
                txt_views = out[:, self.num_stars:] + txt_views
            else:
                seq = torch.cat([stars, img_views], dim=1)
                out = transformer(seq)
                stars = (out[:, :self.num_stars] + stars) / 2
                img_views = out[:, self.num_stars:] + img_views

        fused = self.fusion_proj(stars.reshape(B, -1))
        return self.classifier(fused), img_feat, txt_feat

# ============================================================
# Model: BMR (AAAI 2023) — Bootstrapping Multi-view + iMMoE
# ============================================================
class iMMoE(nn.Module):
    """Improved Mixture of Experts: sigmoid gate (no softmax constraint) + token attention."""
    def __init__(self, input_dim, hidden_dim, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                          nn.Linear(hidden_dim, input_dim))
            for _ in range(num_experts)
        ])
        # Gate: sigmoid instead of softmax (key iMMoE innovation)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, num_experts), nn.Sigmoid(),
        )
        # Token attention
        self.token_attn = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4), nn.GELU(),
            nn.Linear(input_dim // 4, 1),
        )

    def forward(self, x):
        # x: (B, dim)
        gate_val = self.gate(x)  # (B, num_experts)
        out = 0
        for i in range(self.num_experts):
            expert_out = self.experts[i](x)
            out += expert_out * gate_val[:, i].unsqueeze(-1)
        attn = torch.sigmoid(self.token_attn(x))
        return out * attn + x  # residual


class BMRModel(nn.Module):
    """BMR: Bootstrapping Multi-view Representations (simplified from AAAI 2023)."""
    def __init__(self, vision_encoder, text_encoder, txt_dim=768):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        img_dim = vision_encoder.hidden_size
        dim = Config.FUSION_DIM

        # Multi-view projections
        self.img_proj = nn.Linear(img_dim, dim)
        self.txt_proj = nn.Linear(txt_dim, dim)

        # View 1: Image (semantic)
        self.img_view = iMMoE(dim, dim, num_experts=4)
        self.img_coarse = nn.Sequential(nn.Linear(dim, dim//2), nn.ReLU(), nn.Linear(dim//2, 2))

        # View 2: Text
        self.txt_view = iMMoE(dim, dim, num_experts=4)
        self.txt_coarse = nn.Sequential(nn.Linear(dim, dim//2), nn.ReLU(), nn.Linear(dim//2, 2))

        # View 3: Cross-modal fusion
        self.fusion_view = iMMoE(dim * 2, dim * 2, num_experts=4)
        self.fusion_coarse = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, 2))

        # Bootstrap re-weighting MLPs
        self.weight_img = nn.Sequential(nn.Linear(2, dim), nn.ReLU(), nn.Linear(dim, 1), nn.Sigmoid())
        self.weight_txt = nn.Sequential(nn.Linear(2, dim), nn.ReLU(), nn.Linear(dim, 1), nn.Sigmoid())
        self.weight_fusion = nn.Sequential(nn.Linear(2, dim), nn.ReLU(), nn.Linear(dim, 1), nn.Sigmoid())

        # Final classifier (after bootstrapping)
        self.final_classifier = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(dim, dim // 2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(dim // 2, 2),
        )

    def forward(self, images, text_input_ids, text_attention_mask):
        with torch.no_grad():
            img_feat = self.vision_encoder(images)
        txt_feat = self.text_encoder(text_input_ids, text_attention_mask)
        img_p = self.img_proj(img_feat)
        txt_p = self.txt_proj(txt_feat)

        # Multi-view encoding via iMMoE
        img_enc = self.img_view(img_p)            # (B, dim)
        txt_enc = self.txt_view(txt_p)            # (B, dim)
        fusion_input = torch.cat([img_p, txt_p], dim=-1)
        fusion_enc = self.fusion_view(fusion_input)  # (B, 2*dim)

        # Coarse predictions for bootstrapping
        img_coarse = torch.softmax(self.img_coarse(img_enc), dim=-1)      # (B, 2)
        txt_coarse = torch.softmax(self.txt_coarse(txt_enc), dim=-1)      # (B, 2)
        fusion_coarse = torch.softmax(self.fusion_coarse(fusion_enc), dim=-1)  # (B, 2)

        # Bootstrap re-weighting
        w_img = self.weight_img(img_coarse)        # (B, 1)
        w_txt = self.weight_txt(txt_coarse)        # (B, 1)
        w_fusion = self.weight_fusion(fusion_coarse)  # (B, 1)

        # Weighted multi-view fusion
        attended_img = w_img * img_enc
        attended_txt = w_txt * txt_enc
        final_input = torch.cat([attended_img, attended_txt], dim=-1)

        logits = self.final_classifier(final_input)
        return logits, img_feat, txt_feat


# ============================================================
# Model: C3N (IP&M 2025) — Cross-modal Content Correlation Network
# ============================================================
class C3NModel(nn.Module):
    """C3N: Fine-grained cross-modal similarity matrix + 1D-CNN."""
    def __init__(self, vision_encoder, text_encoder, txt_dim=768):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        img_dim = vision_encoder.hidden_size  # 768
        self.txt_proj = nn.Linear(txt_dim, img_dim)

        # 1D-CNN over similarity patterns
        self.conv1d = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveMaxPool1d(64),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveMaxPool1d(32),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, images, text_input_ids, text_attention_mask):
        # Get patch-level image features
        img_seq = self.vision_encoder(images, return_sequence=True)  # (B, L_i, 768)
        # Get token-level text features (handle both BERT and CLIP encoders)
        if hasattr(self.text_encoder, 'bert'):
            txt_seq = self.text_encoder.bert(
                input_ids=text_input_ids, attention_mask=text_attention_mask
            ).last_hidden_state  # (B, L_t, 768)
        else:
            # BertEncoder
            txt_seq = self.text_encoder.clip_model.text_model(
                input_ids=text_input_ids, attention_mask=text_attention_mask
            ).last_hidden_state  # (B, L_t, 512/768)
        txt_seq = self.txt_proj(txt_seq)  # (B, L_t, img_dim)

        # L2 normalize for cosine similarity
        img_norm = F.normalize(img_seq, dim=-1)  # (B, L_i, 768)
        txt_norm = F.normalize(txt_seq, dim=-1)  # (B, L_t, 768)

        # Cross-modal similarity matrix
        sim_matrix = torch.bmm(img_norm, txt_norm.transpose(1, 2))  # (B, L_i, L_t)

        # Aggregate text dimension: mean pooling
        sim_vec = sim_matrix.mean(dim=-1).unsqueeze(1)  # (B, 1, L_i)

        # 1D-CNN over similarity patterns
        conv_out = self.conv1d(sim_vec).squeeze(-1)  # (B, 128)

        logits = self.classifier(conv_out)
        return logits, img_seq.mean(dim=1), txt_seq.mean(dim=1)


# ============================================================
# Model: MDFEND (CIKM 2021) — Mixture of Domain Experts
# ============================================================
class MDFENDModel(nn.Module):
    """MDFEND: Multi-domain Fake News Detection with MoE + Shared Expert."""
    def __init__(self, vision_encoder, text_encoder, num_experts=9, txt_dim=768):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        img_dim = vision_encoder.hidden_size

        self.img_proj = nn.Linear(img_dim, Config.FUSION_DIM)
        self.txt_proj = nn.Linear(txt_dim, Config.FUSION_DIM)
        self.num_experts = num_experts

        # Domain-specific experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(Config.FUSION_DIM * 2, Config.FUSION_DIM),
                nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(Config.FUSION_DIM, Config.FUSION_DIM),
            ) for _ in range(num_experts)
        ])

        # Shared expert (always active)
        self.shared_expert = nn.Sequential(
            nn.Linear(Config.FUSION_DIM * 2, Config.FUSION_DIM),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(Config.FUSION_DIM, Config.FUSION_DIM),
        )

        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(Config.FUSION_DIM * 2, Config.FUSION_DIM // 2),
            nn.ReLU(),
            nn.Linear(Config.FUSION_DIM // 2, num_experts),
            nn.Softmax(dim=-1),
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(Config.FUSION_DIM, Config.FUSION_DIM // 2),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(Config.FUSION_DIM // 2, 2),
        )

    def forward(self, images, text_input_ids, text_attention_mask):
        with torch.no_grad():
            img_feat = self.vision_encoder(images)
        txt_feat = self.text_encoder(text_input_ids, text_attention_mask)
        img_p = self.img_proj(img_feat)
        txt_p = self.txt_proj(txt_feat)
        fused = torch.cat([img_p, txt_p], dim=-1)

        # Gate routing
        gate_w = self.gate(fused)  # (B, num_experts)

        # Expert outputs
        expert_outs = torch.stack([e(fused) for e in self.experts], dim=1)  # (B, K, dim)
        shared_out = self.shared_expert(fused)  # (B, dim)

        # Weighted fusion: shared + gated experts
        weighted = shared_out
        for i in range(self.num_experts):
            weighted = weighted + gate_w[:, i:i+1] * expert_outs[:, i, :]

        return self.classifier(weighted), img_feat, txt_feat


# ============================================================
# Training & Evaluation
# ============================================================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def evaluate_model(model, test_loader, val_loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            img = batch['images'].to(device)
            tids = batch['text_input_ids'].to(device)
            tmask = batch['text_attention_mask'].to(device)
            out = model(img, tids, tmask)
            logits = out[0]
            probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
            labels.append(batch['label'])
    probs = torch.cat(probs).numpy(); labels = torch.cat(labels).numpy()
    best_t = 0.5
    for t in np.linspace(0.05, 0.95, 91):
        preds = (probs > t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        if f1 > best_t: best_t = t

    test_probs, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            img = batch['images'].to(device)
            tids = batch['text_input_ids'].to(device)
            tmask = batch['text_attention_mask'].to(device)
            out = model(img, tids, tmask)
            logits = out[0]
            test_probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
            test_labels.append(batch['label'])
    test_probs = torch.cat(test_probs).numpy(); test_labels = torch.cat(test_labels).numpy()
    test_preds = (test_probs > best_t).astype(int)
    acc = accuracy_score(test_labels, test_preds)
    p, r, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='binary', zero_division=0)
    auc = roc_auc_score(test_labels, test_probs)
    cm = confusion_matrix(test_labels, test_preds)
    return {"threshold": float(best_t), "accuracy": float(acc), "precision": float(p),
            "recall": float(r), "f1": float(f1), "auc": float(auc), "confusion_matrix": cm.tolist()}

def train_model(model, model_name, train_loader, val_loader, device, ckpt_dir, dataset_name, event_ids=None):
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)
    best_val_f1, patience_counter = 0, 0
    n_epochs, patience = Config.NUM_EPOCHS, 50

    for epoch in range(n_epochs):
        model.train()
        model.vision_encoder.eval()
        if model_name == 'eann':
            model.text_encoder.eval()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for batch in pbar:
            img = batch['images'].to(device)
            tids = batch['text_input_ids'].to(device)
            tmask = batch['text_attention_mask'].to(device)
            labels = batch['label'].to(device)

            if model_name == 'eann':
                event_labels = batch.get('event_id', torch.zeros_like(labels))
                if isinstance(event_labels, torch.Tensor):
                    event_labels = event_labels.to(device)
                fake_logits, event_logits, _, _ = model(img, tids, tmask, apply_grl=True)
                loss_cls = nn.CrossEntropyLoss()(fake_logits, labels)
                loss_event = nn.CrossEntropyLoss()(event_logits, event_labels)
                loss = loss_cls + loss_event
            elif model_name == 'safe':
                logits, cos_sim, _, _ = model(img, tids, tmask)
                loss_cls = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)(logits, labels)
                sim_target = 1.0 - labels.float()
                loss_sim = F.mse_loss(torch.sigmoid(cos_sim), sim_target)
                loss = loss_cls + 0.3 * loss_sim
            else:
                logits, _, _ = model(img, tids, tmask)
                loss = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        avg_loss = total_loss / len(train_loader)
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                img = batch['images'].to(device)
                tids = batch['text_input_ids'].to(device)
                tmask = batch['text_attention_mask'].to(device)
                out = model(img, tids, tmask)
                val_preds.append(out[0].argmax(dim=-1).cpu())
                val_labels.append(batch['label'])
        val_preds = torch.cat(val_preds).numpy(); val_labels = torch.cat(val_labels).numpy()
        val_f1 = precision_recall_fscore_support(val_labels, val_preds, average='binary', zero_division=0)[2]
        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} Val F1={val_f1:.4f} Acc={val_acc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1; patience_counter = 0
            torch.save({"epoch": epoch+1, "model_state_dict": model.state_dict(), "val_f1": val_f1},
                       os.path.join(ckpt_dir, "best_model.pt"))
            print(f"  -> Best (F1={val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stop at epoch {epoch+1}"); break
        scheduler.step()

    print(f"Done. Best Val F1={best_val_f1:.4f}")
    return best_val_f1


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--model", type=str, required=True,
                        choices=["spotfake", "eann", "safe", "hmcan", "mdfend", "pivot", "bmr", "c3n"])
    parser.add_argument("--dataset", type=str, default="weibo", choices=["weibo", "weibo21", "pheme"])
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")
    model_name = args.model; dataset_name = args.dataset; seed = 42

    print(f"\n{'='*60}")
    print(f"Baseline: {model_name.upper()} | Dataset: {dataset_name} | GPU: {args.gpu}")
    print(f"{'='*60}")

    set_seed(seed)

    train_samples = load_dataset(dataset_name, "train")
    val_samples = load_dataset(dataset_name, "val")
    test_samples = load_dataset(dataset_name, "test")
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    # Events for EANN
    num_events = None; event_id_map = {}; use_event_loss = True
    if model_name == 'eann':
        all_events = sorted(set(s['event_id'] for s in train_samples if 'event_id' in s))
        if len(all_events) == 0:
            # No event labels (e.g. PHEME) — fall back to dummy 2-class events
            print("Warning: No event labels found, using dummy event classifier (2 pseudo-classes)")
            num_events = 2
            for s in train_samples + val_samples + test_samples:
                s['event_id'] = 0  # all same pseudo-event
        else:
            event_id_map = {eid: i for i, eid in enumerate(all_events)}
            num_events = len(all_events) + 1
            print(f"Events: {len(all_events)}")
            for s in train_samples + val_samples + test_samples:
                if 'event_id' in s:
                    s['event_id'] = event_id_map.get(s['event_id'], num_events - 1)

    # Auto-select text encoder: CLIP for English (pheme), BERT for Chinese (weibo)
    text_enc_type = "clip" if dataset_name == "pheme" else "bert_chinese"
    txt_dim = 512 if text_enc_type == "clip" else 768
    print(f"  Text encoder: {text_enc_type} ({txt_dim}d)")

    tokenizer = get_tokenizer(text_enc_type)
    processor = get_clip_processor()
    train_ds = MultimodalDataset(train_samples, processor, tokenizer, text_encoder_type=text_enc_type)
    val_ds = MultimodalDataset(val_samples, processor, tokenizer, text_encoder_type=text_enc_type)
    test_ds = MultimodalDataset(test_samples, processor, tokenizer, text_encoder_type=text_enc_type)

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)

    # Build model
    vision_encoder = CLIPVisionEncoder().to(device)
    for p in vision_encoder.parameters(): p.requires_grad = False
    text_encoder = BertEncoder().to(device)
    if model_name != 'eann':
        layers = text_encoder.bert.encoder.layer
        for i in range(max(0, len(layers) - Config.TEXT_ENCODER_UNFREEZE_LAYERS), len(layers)):
            for p in layers[i].parameters(): p.requires_grad = True
        for p in text_encoder.bert.pooler.parameters(): p.requires_grad = True

    if model_name == 'spotfake':   model = SpotFakeModel(vision_encoder, text_encoder, txt_dim)
    elif model_name == 'eann':     model = EANNModel(vision_encoder, text_encoder, num_events, txt_dim)
    elif model_name == 'safe':     model = SAFEModel(vision_encoder, text_encoder, txt_dim)
    elif model_name == 'hmcan':    model = HMCANModel(vision_encoder, text_encoder, txt_dim)
    elif model_name == 'mdfend':   model = MDFENDModel(vision_encoder, text_encoder, 9, txt_dim)
    elif model_name == 'pivot':    model = PivotFusionModel(vision_encoder, text_encoder, txt_dim)
    elif model_name == 'bmr':      model = BMRModel(vision_encoder, text_encoder, txt_dim)
    elif model_name == 'c3n':      model = C3NModel(vision_encoder, text_encoder, txt_dim)
    model = model.to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {n_trainable:,}")

    ckpt_dir = os.path.join(Config.CHECKPOINT_DIR, f"{dataset_name}_baselines", model_name, f"seed_{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")

    print("Training...")
    val_f1 = train_model(model, model_name, train_loader, val_loader, device, ckpt_dir, dataset_name)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)

    print("Evaluating...")
    metrics = evaluate_model(model, test_loader, val_loader, device)
    metrics['val_f1'] = float(val_f1)
    metrics['ckpt_epoch'] = ckpt.get('epoch', 0)
    metrics['model'] = model_name; metrics['dataset'] = dataset_name

    result_dir = os.path.join(Config.LOG_DIR, dataset_name)
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"baseline_{model_name}.json")
    with open(result_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"{model_name.upper()} — {dataset_name}")
    print(f"{'='*60}")
    print(f"  Val F1:    {metrics['val_f1']:.4f}")
    print(f"  Test F1:   {metrics['f1']:.4f}")
    print(f"  Test AUC:  {metrics['auc']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"Saved to {result_path}")

    shutil.rmtree(ckpt_dir, ignore_errors=True)
    print("Checkpoint deleted.")

if __name__ == "__main__":
    main()
