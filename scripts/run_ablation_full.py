"""
Ablation experiment pipeline: train → evaluate → delete checkpoint → record.
Run one experiment at a time on specified GPU.
"""
import sys, os, json, re, shutil
import torch, numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data import load_dataset, MultimodalDataset
from models import FakeNewsDiscriminator, get_clip_processor, get_tokenizer
from train import train
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def evaluate_checkpoint(ckpt_path, fusion_mode, loss_mode, freeze_bert):
    """Load checkpoint, evaluate on test set, return metrics dict."""
    device = torch.device('cuda:0')

    # Set ablation config for model creation
    Config.ABLATION_FUSION = fusion_mode
    Config.ABLATION_LOSS = loss_mode
    Config.ABLATION_FREEZE_BERT = freeze_bert

    model = FakeNewsDiscriminator(text_encoder_type='bert_chinese')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    val_samples = load_dataset('weibo', 'val')
    test_samples = load_dataset('weibo', 'test')
    tokenizer = get_tokenizer('bert_chinese')
    processor = get_clip_processor()
    val_ds = MultimodalDataset(val_samples, processor, tokenizer, text_encoder_type='bert_chinese')
    test_ds = MultimodalDataset(test_samples, processor, tokenizer, text_encoder_type='bert_chinese')
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Threshold calibration on val
    val_probs, val_labels = [], []
    for batch in val_loader:
        img = batch['images'].to(device)
        tids = batch['text_input_ids'].to(device)
        tmask = batch['text_attention_mask'].to(device)
        x0, _, _, _ = model.forward_pipeline(img, tids, tmask)
        logits = model.forward_classifier(x0)
        val_probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
        val_labels.append(batch['label'])
    val_probs = torch.cat(val_probs).numpy()
    val_labels = torch.cat(val_labels).numpy()

    best_t = 0.5
    for t in np.linspace(0.05, 0.95, 91):
        preds = (val_probs > t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            val_labels, preds, average='binary', zero_division=0)
        if f1 > best_t:
            best_t = t

    # Test evaluation
    test_probs, test_labels = [], []
    for batch in test_loader:
        img = batch['images'].to(device)
        tids = batch['text_input_ids'].to(device)
        tmask = batch['text_attention_mask'].to(device)
        x0, _, _, _ = model.forward_pipeline(img, tids, tmask)
        logits = model.forward_classifier(x0)
        test_probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
        test_labels.append(batch['label'])
    test_probs = torch.cat(test_probs).numpy()
    test_labels = torch.cat(test_labels).numpy()
    test_preds = (test_probs > best_t).astype(int)

    acc = accuracy_score(test_labels, test_preds)
    _, _, f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average='binary', zero_division=0)
    auc = roc_auc_score(test_labels, test_probs)

    return {
        'val_f1': ckpt.get('val_f1', 0),
        'ckpt_epoch': ckpt.get('epoch', 0),
        'test_f1': float(f1),
        'test_auc': float(auc),
        'test_acc': float(acc),
        'threshold': float(best_t),
    }


def delete_checkpoint(dataset_name, fusion_mode, loss_mode, seed=42):
    """Delete best_model.pt after evaluation to free disk space."""
    # Determine checkpoint dir
    if fusion_mode != "full":
        model_dir = f"ablation/{fusion_mode}"
    elif loss_mode != "full":
        model_dir = f"ablation/{loss_mode}"
    elif Config.ABLATION_FREEZE_BERT:
        model_dir = "ablation/frozen_bert"
    else:
        model_dir = "default"
    ckpt_dir = os.path.join(Config.CHECKPOINT_DIR, dataset_name, model_dir, f"seed_{seed}")
    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f"  Deleted: {ckpt_path}")
    # Also remove history.json to keep dirs clean
    hist_path = os.path.join(ckpt_dir, "history.json")
    if os.path.exists(hist_path):
        os.remove(hist_path)


def run_one(exp_name, gpu, fusion_mode, loss_mode, freeze_bert):
    """Run one full ablation experiment pipeline."""
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {exp_name}")
    print(f"  GPU={gpu}  fusion={fusion_mode}  loss={loss_mode}  freeze_bert={freeze_bert}")
    print(f"{'='*60}")

    # Set config
    Config.ABLATION_FUSION = fusion_mode
    Config.ABLATION_LOSS = loss_mode
    Config.ABLATION_FREEZE_BERT = freeze_bert

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # Train
    print(f"  Training...")
    model = train('weibo', 'bert_chinese', 'clip', 42)
    del model
    torch.cuda.empty_cache()

    # Find checkpoint
    if fusion_mode != "full":
        model_dir = f"ablation/{fusion_mode}"
    elif loss_mode != "full":
        model_dir = f"ablation/{loss_mode}"
    elif freeze_bert:
        model_dir = "ablation/frozen_bert"
    else:
        model_dir = "default"
    ckpt_dir = os.path.join(Config.CHECKPOINT_DIR, 'weibo', model_dir, 'seed_42')
    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")

    if not os.path.exists(ckpt_path):
        print(f"  ERROR: No checkpoint at {ckpt_path}")
        return None

    # Evaluate
    print(f"  Evaluating on test set...")
    # Use GPU from CUDA_VISIBLE_DEVICES
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # eval on GPU 0 if free, else same GPU
    metrics = evaluate_checkpoint(ckpt_path, fusion_mode, loss_mode, freeze_bert)
    metrics['exp_name'] = exp_name
    metrics['fusion'] = fusion_mode
    metrics['loss'] = loss_mode
    metrics['freeze_bert'] = freeze_bert

    print(f"  Val F1={metrics['val_f1']:.4f}  Test F1={metrics['test_f1']:.4f}  "
          f"AUC={metrics['test_auc']:.4f}  Acc={metrics['test_acc']:.4f}")

    # Delete checkpoint to save space
    delete_checkpoint('weibo', fusion_mode, loss_mode)

    # Save metrics incrementally
    results_file = os.path.join(Config.LOG_DIR, 'ablation_results.jsonl')
    with open(results_file, 'a') as f:
        f.write(json.dumps(metrics) + '\n')

    print(f"  ✓ Done. Results saved to {results_file}")
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--exp', type=str, required=True,
                        choices=['A','B','C','D','E','F','all'])
    args = parser.parse_args()

    # Experiment definitions
    experiments = {
        'A': ('simple_concat',   'full',           False),
        'B': ('concat_semantic', 'full',           False),
        'C': ('full',            'full',           False),  # full MBSC-Net
        'D': ('full',            'ce_only',        False),
        'E': ('full',            'no_contrastive', False),
        'F': ('full',            'full',           True),
    }

    os.makedirs(os.path.join(Config.LOG_DIR), exist_ok=True)

    if args.exp == 'all':
        order = ['A', 'B', 'E', 'D', 'F']
        results = {}
        for exp in order:
            fusion, loss, freeze = experiments[exp]
            m = run_one(exp, args.gpu, fusion, loss, freeze)
            if m:
                results[exp] = m
        print(f"\n{'='*60}")
        print("All experiments complete!")
        print(f"{'='*60}")
        for exp, m in results.items():
            print(f"  {exp}: Val={m['val_f1']:.4f} Test F1={m['test_f1']:.4f} AUC={m['test_auc']:.4f}")
    else:
        fusion, loss, freeze = experiments[args.exp]
        run_one(args.exp, args.gpu, fusion, loss, freeze)
