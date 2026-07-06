"""
Statistical significance test: 5 seeds × 2 models on weibo21.
Compares Full (Ours) vs frozen_bert (次优基线) with paired t-test.
Usage: python3 scripts/run_significance.py --model full --seed 42 --gpu 0
"""
import sys, os, json, argparse, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

import config
from config import Config
from data import load_dataset, MultimodalDataset
from models import get_clip_processor, get_tokenizer
from scripts.run_ablation import build_ablated_model, train_ablated


def evaluate_model(model, test_loader, device):
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

    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    auc = roc_auc_score(labels, probs)

    return {'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec),
            'f1': float(f1), 'auc': float(auc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['full', 'frozen_bert'])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--dataset', type=str, default='weibo', choices=['weibo', 'weibo21', 'pheme'])
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    if args.patience is not None:
        Config.PATIENCE = args.patience

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    ds_name = args.dataset
    # Auto-select text encoder: BERT-uncased for English (pheme), BERT-Chinese for Chinese
    text_enc = 'bert_uncased' if ds_name == 'pheme' else 'bert_chinese'
    # Auto-select vision encoder

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Config for Full vs frozen_bert
    if args.model == 'frozen_bert':
        Config.UNFREEZE_TEXT_ENCODER = False
        Config.TEXT_ENCODER_UNFREEZE_LAYERS = 0
    else:
        Config.UNFREEZE_TEXT_ENCODER = True
        Config.TEXT_ENCODER_UNFREEZE_LAYERS = 3

    # Use default loss weights for fair comparison
    Config.CONTRASTIVE_LOSS_WEIGHT = 0.3
    Config.PATIENCE = 15

    tag = f"{args.model}_seed{args.seed}"
    ckpt_dir = os.path.join(Config.CHECKPOINT_DIR, f"{ds_name}_sigtest", tag, "seed_42")
    os.makedirs(ckpt_dir, exist_ok=True)

    save_path = os.path.join(ckpt_dir, "results.json")
    if os.path.exists(save_path):
        print(f"Already done: {save_path}")
        with open(save_path) as f:
            return json.load(f)

    print(f"\n{'='*60}")
    print(f" SigTest: {args.model} | Seed={args.seed} | {ds_name}")
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

    # Use seed for DataLoader shuffle
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True)

    # Build model
    print(f"Building model (variant=full, frozen_bert={args.model=='frozen_bert'})...")
    model = build_ablated_model('full', device, text_enc)
    if args.model == 'frozen_bert':
        for p in model.text_encoder.parameters():
            p.requires_grad = False

    n_trainable = sum(p.numel() for p in model.get_trainable_parameters())
    print(f"Trainable: {n_trainable:,}")

    # Train
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
        'model': args.model, 'seed': args.seed, 'dataset': ds_name,
        'val_f1': float(best_val_f1), 'ckpt_epoch': best_epoch,
        **metrics
    }

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Seed={args.seed} {args.model}: Acc={metrics['accuracy']:.4f} "
          f"F1={metrics['f1']:.4f} AUC={metrics['auc']:.4f}")
    print(f"Saved to {save_path}")

    return results


if __name__ == '__main__':
    main()
