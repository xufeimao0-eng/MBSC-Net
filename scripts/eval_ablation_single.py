"""Evaluate a single ablation checkpoint, then delete it."""
import sys, os, json, numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from data import load_dataset, MultimodalDataset
from models import FakeNewsDiscriminator, get_clip_processor, get_tokenizer

ckpt_path = sys.argv[1]
fusion_mode = sys.argv[2]   # simple_concat | concat_semantic | full
loss_mode = sys.argv[3]     # full | ce_only | no_contrastive
freeze_bert = sys.argv[4] == 'True'
results_file = sys.argv[5]  # append results as JSON line

device = torch.device('cuda:0')

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
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

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
    _, _, f1, _ = precision_recall_fscore_support(val_labels, preds, average='binary', zero_division=0)
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
precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='binary', zero_division=0)
auc = roc_auc_score(test_labels, test_probs)
cm = confusion_matrix(test_labels, test_preds)

metrics = {
    "dataset": "weibo",
    "fusion": fusion_mode,
    "loss": loss_mode,
    "freeze_bert": freeze_bert,
    "threshold": float(best_t),
    "val_f1": float(ckpt.get('val_f1', 0)),
    "ckpt_epoch": ckpt.get('epoch', 0),
    "accuracy": float(acc),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "auc": float(auc),
    "confusion_matrix": cm.tolist(),
}

# Save individual result file (matching test_results_mbsc.json format)
os.makedirs(os.path.dirname(results_file), exist_ok=True)
individual_file = results_file.replace('.jsonl', f'_{fusion_mode}_{loss_mode}.json')
if freeze_bert:
    individual_file = individual_file.replace('.json', '_frozen_bert.json')
with open(individual_file, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"Result saved to {individual_file}")

# Also append to aggregated JSONL
agg_file = results_file.replace('.jsonl', '_aggregated.jsonl').replace(f'_{fusion_mode}_{loss_mode}', '')
with open(agg_file, 'a') as f:
    f.write(json.dumps(metrics) + '\n')

print(f"Val F1={metrics['val_f1']:.4f}  Test F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}  Acc={metrics['accuracy']:.4f}")
print(f"  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  Thresh={metrics['threshold']:.4f}")
print(f"  CM: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
