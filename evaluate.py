"""
MBSC-Net Evaluation: multi-seed evaluation with mean ± std reporting.
"""

import os, json
import torch, numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score,
)
from tqdm import tqdm

from config import Config
from data import load_dataset, MultimodalDataset
from models import FakeNewsDiscriminator, get_clip_processor, get_tokenizer


def find_best_threshold(probs, labels):
    best_f1, best_t = 0, 0.5
    for t in np.linspace(0.05, 0.95, 91):
        preds = (probs > t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


@torch.no_grad()
def evaluate_single_model(model, val_loader, test_loader, device):
    """Returns dict of metrics for one model."""
    model.eval()

    # Calibrate threshold on val
    val_probs, val_labels = [], []
    for batch in val_loader:
        images = batch["images"].to(device)
        tids = batch["text_input_ids"].to(device)
        tmask = batch["text_attention_mask"].to(device)
        x0, _, _, _ = model.forward_pipeline(images, tids, tmask)
        logits = model.forward_classifier(x0)
        val_probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
        val_labels.append(batch["label"])
    val_probs = torch.cat(val_probs).numpy()
    val_labels = torch.cat(val_labels).numpy()
    threshold, val_f1 = find_best_threshold(val_probs, val_labels)
    val_preds = (val_probs > threshold).astype(int)
    val_acc = accuracy_score(val_labels, val_preds)

    # Test
    test_probs, test_labels = [], []
    for batch in test_loader:
        images = batch["images"].to(device)
        tids = batch["text_input_ids"].to(device)
        tmask = batch["text_attention_mask"].to(device)
        x0, _, _, _ = model.forward_pipeline(images, tids, tmask)
        logits = model.forward_classifier(x0)
        test_probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
        test_labels.append(batch["label"])
    test_probs = torch.cat(test_probs).numpy()
    test_labels = torch.cat(test_labels).numpy()
    test_preds = (test_probs > threshold).astype(int)

    acc = accuracy_score(test_labels, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average="binary", zero_division=0)
    auc = roc_auc_score(test_labels, test_probs)
    cm = confusion_matrix(test_labels, test_preds)

    return {
        "val_f1": float(val_f1), "val_acc": float(val_acc),
        "threshold": float(threshold),
        "test_acc": float(acc), "test_precision": float(precision),
        "test_recall": float(recall), "test_f1": float(f1),
        "test_auc": float(auc), "confusion_matrix": cm.tolist(),
    }


def load_model_for_eval(checkpoint_dir, text_encoder_type, device):
    """Load model from checkpoint directory."""
    ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        return None, None

    model = FakeNewsDiscriminator(text_encoder_type=text_encoder_type)
    model = model.to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


def evaluate_multi(dataset_name="weibo", text_encoder_type="bert_chinese", seeds=(42, 123, 456)):
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    vision_type = "clip"
    subdir = "default"
    base_dir = os.path.join(Config.CHECKPOINT_DIR, dataset_name, subdir)

    print(f"\n{'='*60}")
    print(f"Multi-Seed Evaluation: {dataset_name}")
    print(f"  Vision: {vision_type} | Text: {text_encoder_type}")
    print(f"  Seeds: {list(seeds)}")
    print(f"{'='*60}\n")

    # Load data once
    try:
        val_samples = load_dataset(dataset_name, "val")
    except (ValueError, FileNotFoundError):
        val_samples = load_dataset(dataset_name, "test")
    test_samples = load_dataset(dataset_name, "test")
    print(f"Val: {len(val_samples)}, Test: {len(test_samples)}")

    tokenizer = get_tokenizer(text_encoder_type)
    clip_processor = get_clip_processor()
    val_dataset = MultimodalDataset(val_samples, clip_processor, tokenizer,
                                    text_encoder_type=text_encoder_type)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=Config.NUM_WORKERS, pin_memory=True)
    test_dataset = MultimodalDataset(test_samples, clip_processor, tokenizer,
                                     text_encoder_type=text_encoder_type)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                             num_workers=Config.NUM_WORKERS, pin_memory=True)

    results = []
    for seed in seeds:
        seed_dir = os.path.join(base_dir, f"seed_{seed}")
        print(f"\n--- Seed {seed} ---")
        model, ckpt = load_model_for_eval(seed_dir, text_encoder_type, device)
        if model is None:
            print(f"  [SKIP] No checkpoint at {seed_dir}")
            continue
        print(f"  Checkpoint: epoch={ckpt.get('epoch')}, Val F1={ckpt.get('val_f1', 0):.4f}")
        metrics = evaluate_single_model(model, val_loader, test_loader, device)
        metrics["seed"] = seed
        metrics["ckpt_epoch"] = ckpt.get("epoch")
        metrics["ckpt_val_f1"] = ckpt.get("val_f1")
        results.append(metrics)
        print(f"  Test F1={metrics['test_f1']:.4f}  AUC={metrics['test_auc']:.4f}  "
              f"Acc={metrics['test_acc']:.4f}  Thresh={metrics['threshold']:.4f}")

    if not results:
        print("No checkpoints found!")
        return

    # Aggregate
    test_f1s = [r["test_f1"] for r in results]
    test_aucs = [r["test_auc"] for r in results]
    test_accs = [r["test_acc"] for r in results]
    val_f1s = [r["val_f1"] for r in results]

    print(f"\n{'='*60}")
    print(f"Aggregated Results — {dataset_name} ({vision_type})")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Mean ± Std':>20} {'Individual':>30}")
    print(f"{'-'*70}")
    for name, vals in [("Val F1", val_f1s), ("Test F1", test_f1s),
                        ("Test AUC", test_aucs), ("Test Acc", test_accs)]:
        individual = "  ".join(f"{v:.4f}" for v in vals)
        print(f"{name:<20} {np.mean(vals):.4f} ± {np.std(vals):.4f}   {individual}")

    # Save
    save_dir = os.path.join(Config.LOG_DIR, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"multi_seed_{vision_type}.json")
    with open(save_path, "w") as f:
        json.dump({
            "dataset": dataset_name, 
            "text_encoder": text_encoder_type, "seeds": seeds,
            "aggregated": {
                "test_f1_mean": float(np.mean(test_f1s)),
                "test_f1_std": float(np.std(test_f1s)),
                "test_auc_mean": float(np.mean(test_aucs)),
                "test_auc_std": float(np.std(test_aucs)),
                "test_acc_mean": float(np.mean(test_accs)),
                "test_acc_std": float(np.std(test_accs)),
            },
            "individual": results,
        }, f, indent=2)
    print(f"\nSaved to {save_path}")

    return results


@torch.no_grad()
def evaluate(dataset_name="weibo", text_encoder_type=None,
             checkpoint_path=None, save_results=True):
    """Single-model evaluation (backward compatible)."""
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    text_encoder_type = text_encoder_type or Config.TEXT_ENCODER_TYPE

    try:
        val_samples = load_dataset(dataset_name, "val")
    except (ValueError, FileNotFoundError):
        val_samples = load_dataset(dataset_name, "test")
    test_samples = load_dataset(dataset_name, "test")

    tokenizer = get_tokenizer(text_encoder_type)
    clip_processor = get_clip_processor()

    val_dataset = MultimodalDataset(val_samples, clip_processor, tokenizer,
                                    text_encoder_type=text_encoder_type)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=Config.NUM_WORKERS, pin_memory=True)
    test_dataset = MultimodalDataset(test_samples, clip_processor, tokenizer,
                                     text_encoder_type=text_encoder_type)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                             num_workers=Config.NUM_WORKERS, pin_memory=True)

    model = FakeNewsDiscriminator(text_encoder_type=text_encoder_type)
    model = model.to(device)

    vision_type = "clip"
    subdir = "default"
    base_dir = os.path.join(Config.CHECKPOINT_DIR, dataset_name, subdir)
    ckpt_path = checkpoint_path or os.path.join(base_dir, "best_model.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Single checkpoint: epoch={ckpt.get('epoch')}, Val F1={ckpt.get('val_f1', 0):.4f}")
    model.eval()

    metrics = evaluate_single_model(model, val_loader, test_loader, device)

    print(f"\n{'='*60}")
    print(f"Test Results — {dataset_name} (MBSC-Net)")
    print(f"{'='*60}")
    print(f"Threshold: {metrics['threshold']:.4f}")
    print(f"Accuracy:  {metrics['test_acc']:.4f}")
    print(f"Precision: {metrics['test_precision']:.4f}")
    print(f"Recall:    {metrics['test_recall']:.4f}")
    print(f"F1-Score:  {metrics['test_f1']:.4f}")
    print(f"AUC:       {metrics['test_auc']:.4f}")
    cm = np.array(metrics["confusion_matrix"])
    print(f"\nConfusion Matrix:")
    print(f"  TN={int(cm[0][0]):5d}  FP={int(cm[0][1]):5d}")
    print(f"  FN={int(cm[1][0]):5d}  TP={int(cm[1][1]):5d}")

    if save_results:
        d = os.path.join(Config.LOG_DIR, dataset_name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "test_results_mbsc.json"), "w") as f:
            json.dump({"dataset": dataset_name, "threshold": metrics["threshold"],
                       "val_f1": metrics["val_f1"], "accuracy": metrics["test_acc"],
                       "precision": metrics["test_precision"], "recall": metrics["test_recall"],
                       "f1": metrics["test_f1"], "auc": metrics["test_auc"],
                       "confusion_matrix": metrics["confusion_matrix"]}, f, indent=2)

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="weibo", choices=["weibo", "weibo21", "pheme"])
    parser.add_argument("--text_encoder", type=str, default=None, choices=["clip", "bert_chinese"])
    
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None,
                       help="Comma-separated seeds for multi-seed eval, e.g. '42,123,456'")
    args = parser.parse_args()

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
        evaluate_multi(args.dataset, args.text_encoder, seeds)
    else:
        evaluate(args.dataset, args.text_encoder, args.checkpoint, True)
