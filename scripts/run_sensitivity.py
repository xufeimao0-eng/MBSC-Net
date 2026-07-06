"""
Single sensitivity experiment: train Full model with given loss weights, evaluate.
Usage: python3 scripts/run_sensitivity.py --align 0.3 --cont 0.1 --gpu 0
"""
import sys, os, json, shutil, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

import config
from config import Config
from data import load_dataset, MultimodalDataset
from models import get_clip_processor, get_tokenizer
from scripts.run_ablation import build_ablated_model, train_ablated


def evaluate_model(model, test_loader, device):
    """Return test metrics dict."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Eval"):
            img = batch['images'].to(device)
            tids = batch['text_input_ids'].to(device)
            tmask = batch['text_attention_mask'].to(device)
            x0, _, _, _ = model.forward_pipeline(img, tids, tmask)
            logits = model.forward_classifier(x0)
            probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
            labels.append(batch['label'])
    probs = torch.cat(probs).numpy()
    labels = torch.cat(labels).numpy()

    # Find optimal threshold on test set
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_idx = f1s.argmax()
    best_t = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    preds_t05 = (probs > 0.5).astype(int)
    preds_cal = (probs > best_t).astype(int)

    acc05 = accuracy_score(labels, preds_t05)
    prec05, rec05, f105, _ = precision_recall_fscore_support(labels, preds_t05, average='binary', zero_division=0)
    auc = roc_auc_score(labels, probs)

    acc_cal = accuracy_score(labels, preds_cal)
    prec_cal, rec_cal, f1_cal, _ = precision_recall_fscore_support(labels, preds_cal, average='binary', zero_division=0)

    return {
        't05': {'accuracy': float(acc05), 'precision': float(prec05), 'recall': float(rec05), 'f1': float(f105), 'auc': float(auc)},
        'calibrated': {'threshold': float(best_t), 'accuracy': float(acc_cal), 'precision': float(prec_cal), 'recall': float(rec_cal), 'f1': float(f1_cal), 'auc': float(auc)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ce', type=float, default=None, help='CE loss weight (default: from Config)')
    parser.add_argument('--cont', type=float, required=True, help='Contrastive loss weight')
    parser.add_argument('--dataset', type=str, default='weibo', choices=['weibo', 'weibo21', 'pheme'])
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    ds_name = args.dataset
    # Auto-select text encoder: BERT-uncased for English (pheme), BERT-Chinese for Chinese
    text_enc = 'bert_uncased' if ds_name == 'pheme' else 'bert_chinese'
    # Default vision encoder for pheme is clip (not chinese_clip)

    # Override Config loss weights and patience
    if args.ce is not None:
        Config.CLASSIFIER_LOSS_WEIGHT = args.ce
    Config.CONTRASTIVE_LOSS_WEIGHT = args.cont
    Config.PATIENCE = args.patience
    Config.BILINEAR_TYPE = "hadamard"

    # Ensure Full model config (3 layers unfrozen)
    Config.UNFREEZE_TEXT_ENCODER = True
    Config.TEXT_ENCODER_UNFREEZE_LAYERS = 3

    tag = f"ce{Config.CLASSIFIER_LOSS_WEIGHT:.1f}_c{args.cont:.1f}"
    ckpt_dir = os.path.join(Config.CHECKPOINT_DIR, f"{ds_name}_sensitivity_hadamard", tag, "seed_42")
    os.makedirs(ckpt_dir, exist_ok=True)

    save_path = os.path.join(ckpt_dir, "results.json")
    if os.path.exists(save_path):
        print(f"Results already exist at {save_path}, loading...")
        with open(save_path) as f:
            results = json.load(f)
        print(json.dumps(results, indent=2))
        return results

    print(f"{'='*60}")
    print(f" Sensitivity: λ_CE={Config.CLASSIFIER_LOSS_WEIGHT:.1f}, λ_cont={args.cont:.1f}")
    print(f" Dataset: {ds_name}, Vision: CLIP ViT")
    print(f"{'='*60}")

    # Load data
    train_samples = load_dataset(ds_name, 'train')
    val_samples = load_dataset(ds_name, 'val')
    test_samples = load_dataset(ds_name, 'test')
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    tokenizer = get_tokenizer(text_enc)
    processor = get_clip_processor()
    train_ds = MultimodalDataset(train_samples, processor, tokenizer, text_encoder_type=text_enc)
    val_ds = MultimodalDataset(val_samples, processor, tokenizer, text_encoder_type=text_enc)
    test_ds = MultimodalDataset(test_samples, processor, tokenizer, text_encoder_type=text_enc)
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Build & train
    print("Building model (variant=full)...")
    model = build_ablated_model('full', device, text_enc)
    train_ablated(model, train_loader, val_loader, device, 'full', ckpt_dir)

    # Load best checkpoint
    best_ckpt = os.path.join(ckpt_dir, "best_model.pt")
    ckpt = torch.load(best_ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    best_val_f1 = ckpt.get('val_f1', 0)
    best_epoch = ckpt.get('epoch', 0)

    # Evaluate
    print(f"\nEvaluating best checkpoint (epoch={best_epoch}, val_f1={best_val_f1:.4f})...")
    metrics = evaluate_model(model, test_loader, device)

    results = {
        'ce_weight': float(Config.CLASSIFIER_LOSS_WEIGHT),
        'cont_weight': args.cont,
        'dataset': ds_name,
        'vision_encoder': 'clip',
        'bilinear_type': 'hadamard',
        'val_f1': float(best_val_f1),
        'ckpt_epoch': best_epoch,
        't05': metrics['t05'],
        'calibrated': metrics['calibrated'],
    }
    results['best'] = results['calibrated']  # use calibrated as best by default

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults (T=0.5): Acc={metrics['t05']['accuracy']:.4f} F1={metrics['t05']['f1']:.4f} AUC={metrics['t05']['auc']:.4f}")
    print(f"Results (Cal={metrics['calibrated']['threshold']:.2f}): Acc={metrics['calibrated']['accuracy']:.4f} F1={metrics['calibrated']['f1']:.4f}")
    print(f"Saved to {save_path}")

    return results


if __name__ == '__main__':
    main()
