"""
MBSC-Net: BERT + Bilinear Fusion + Contrastive Learning.
"""

import os, json, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from config import Config
from data import load_dataset, MultimodalDataset
from models import FakeNewsDiscriminator, get_clip_processor, get_tokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def supervised_contrastive_loss(projections, labels, temperature=0.1):
    device = projections.device
    sim = torch.matmul(projections, projections.T) / temperature
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T).float()
    pos_mask.fill_diagonal_(0)
    neg_mask = (labels != labels.T).float()
    sim_max = sim.max(dim=1, keepdim=True)[0].detach()
    sim = sim - sim_max
    pos_exp = (torch.exp(sim) * pos_mask).sum(dim=1)
    all_exp = (torch.exp(sim) * (pos_mask + neg_mask)).sum(dim=1)
    return -torch.log(pos_exp / (all_exp + 1e-8)).mean()


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(dataloader, desc="Validating"):
        images = batch["images"].to(device)
        tids = batch["text_input_ids"].to(device)
        tmask = batch["text_attention_mask"].to(device)
        x0, _, _, _ = model.forward_pipeline(images, tids, tmask)
        logits = model.forward_classifier(x0)
        all_preds.append(logits.argmax(dim=-1).cpu())
        all_labels.append(batch["label"])
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def train(dataset_name="weibo", text_encoder_type=None, seed=None):
    seed = seed if seed is not None else Config.SEED
    set_seed(seed)
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    text_encoder_type = text_encoder_type or Config.TEXT_ENCODER_TYPE
    vision_type = "clip"

    print(f"\n{'='*60}")
    print(f"MBSC-Net: dataset={dataset_name}")
    print(f"  BERT + Bilinear + Contrastive")
    print(f"  Vision: {vision_type}  |  Seed: {seed}")
    print(f"{'='*60}\n")

    train_all = load_dataset(dataset_name, "train")
    val_samples = load_dataset(dataset_name, "val") if dataset_name in ("weibo", "weibo21", "pheme") else None
    has_val = val_samples is not None
    if not has_val:
        # No built-in val set (e.g. weibo21) — train on all data, skip val
        print(f"Train: {len(train_all)} (no val set — using all for training)")
    else:
        print(f"Train: {len(train_all)}, Val: {len(val_samples)}")

    real_count = sum(1 for s in train_all if s["label"] == 0)
    print(f"Real: {real_count}, Fake: {len(train_all)-real_count}")

    tokenizer = get_tokenizer(text_encoder_type)
    clip_processor = get_clip_processor()

    train_ds = MultimodalDataset(train_all, clip_processor, tokenizer,
                                 text_encoder_type=text_encoder_type)
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True,
                              num_workers=Config.NUM_WORKERS, pin_memory=True)

    if has_val:
        val_ds = MultimodalDataset(val_samples, clip_processor, tokenizer,
                                   text_encoder_type=text_encoder_type)
        val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False,
                                num_workers=Config.NUM_WORKERS, pin_memory=True)

    model = FakeNewsDiscriminator(text_encoder_type=text_encoder_type)
                                  
    model = model.to(device)
    print(f"Trainable: {sum(p.numel() for p in model.get_trainable_parameters()):,}")

    optimizer = AdamW(model.get_trainable_parameters(),
                      lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)

    best_val_f1 = 0
    patience = 50
    patience_counter = 0
    history = {"train_loss": [], "val_f1": []}

    # Without val set, use fixed epochs and save final model
    n_epochs = Config.NUM_EPOCHS if has_val else 30

    model_dir = "default"
    ckpt_dir = os.path.join(Config.CHECKPOINT_DIR, dataset_name, model_dir,
                            f"seed_{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(n_epochs):
        model.train()
        model.vision_encoder.eval()

        total_loss = total_cls = total_cont = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for batch in pbar:
            imgs = batch["images"].to(device)
            tids = batch["text_input_ids"].to(device)
            tmask = batch["text_attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Forward
            x0, _, img_feat, txt_feat = model.forward_pipeline(imgs, tids, tmask)

            # 1. Classification
            loss_cls = nn.CrossEntropyLoss(
                label_smoothing=Config.LABEL_SMOOTHING)(
                model.forward_classifier(x0), labels)

            # 2. Contrastive
            proj = model.forward_projection(x0)
            loss_cont = supervised_contrastive_loss(proj, labels)

            loss = (Config.CLASSIFIER_LOSS_WEIGHT * loss_cls
                    + Config.CONTRASTIVE_LOSS_WEIGHT * loss_cont)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_cls += loss_cls.item()
            total_cont += loss_cont.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}", cls=f"{loss_cls.item():.3f}")

        avg_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_loss)

        if has_val:
            val_preds, val_labels = validate(model, val_loader, device)
            val_f1 = precision_recall_fscore_support(
                val_labels, val_preds, average="binary", zero_division=0)[2]
            val_acc = accuracy_score(val_labels, val_preds)
            history["val_f1"].append(float(val_f1))
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} "
                  f"(cls={total_cls/len(train_loader):.4f} al={.4f}) "
                  f"Val F1={val_f1:.4f} Acc={val_acc:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save({"epoch": epoch+1, "model_state_dict": model.state_dict(),
                            "val_f1": val_f1},
                           os.path.join(ckpt_dir, "best_model.pt"))
                print(f"  -> Best (F1={val_f1:.4f})")
            else:
                patience_counter += 1
                print(f"  -> No improvement ({patience_counter}/{patience})")

            if patience_counter >= patience:
                print(f"Early stop at epoch {epoch+1}")
                break
        else:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} "
                  f"(cls={total_cls/len(train_loader):.4f} al={.4f})")

        scheduler.step()

    if not has_val:
        torch.save({"epoch": n_epochs, "model_state_dict": model.state_dict()},
                   os.path.join(ckpt_dir, "best_model.pt"))
        print(f"\nDone. Final model saved (epoch {n_epochs})")
    else:
        print(f"\nDone. Best Val F1={best_val_f1:.4f}")
    with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="weibo", choices=["weibo", "weibo21", "pheme"])
    parser.add_argument("--text_encoder", type=str, default=None, choices=["clip", "bert_chinese"])
    
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    train(args.dataset, args.text_encoder, args.seed)
