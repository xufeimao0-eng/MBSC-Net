"""
MBSC-Net Visualization: t-SNE of X₀ features + classifier score distributions.
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data import load_dataset, MultimodalDataset
from models import FakeNewsDiscriminator, get_clip_processor, get_tokenizer


@torch.no_grad()
def extract_features_and_scores(dataset_name, text_encoder_type,
                                checkpoint_path=None, max_samples=None):
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    text_encoder_type = text_encoder_type or Config.TEXT_ENCODER_TYPE

    test_samples = load_dataset(dataset_name, "test")
    if max_samples:
        test_samples = test_samples[:max_samples]

    tokenizer = get_tokenizer(text_encoder_type)
    clip_processor = get_clip_processor()

    dataset = MultimodalDataset(test_samples, clip_processor, tokenizer,
                                text_encoder_type=text_encoder_type)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                        num_workers=Config.NUM_WORKERS, pin_memory=True)

    model = FakeNewsDiscriminator(text_encoder_type=text_encoder_type)
    model = model.to(device)

    if checkpoint_path is None:
        checkpoint_dir = os.path.join(Config.CHECKPOINT_DIR, dataset_name, "default")
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    features_list, probs_list, labels_list = [], [], []

    for batch in tqdm(loader, desc="Extracting"):
        images = batch["images"].to(device)
        text_input_ids = batch["text_input_ids"].to(device)
        text_attention_mask = batch["text_attention_mask"].to(device)
        labels = batch["label"]

        x0, _, _, _ = model.forward_pipeline(images, text_input_ids, text_attention_mask)
        logits = model.forward_classifier(x0)
        probs = torch.softmax(logits, dim=-1)[:, 1]

        features_list.append(x0.cpu().numpy())
        probs_list.append(probs.cpu().numpy())
        labels_list.extend(labels.tolist())

    features = np.concatenate(features_list, axis=0)
    probs = np.concatenate(probs_list, axis=0)
    labels = np.array(labels_list)
    return features, probs, labels


def plot_tsne(features, labels, dataset_name, save_path=None):
    print(f"Running t-SNE on {features.shape[0]} samples ({features.shape[1]}-dim)...")
    tsne = TSNE(n_components=2, random_state=Config.SEED, perplexity=30, n_iter=1000, verbose=1)
    features_2d = tsne.fit_transform(features)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#2ca02c", "#d62728"]
    names = ["Real", "Fake"]

    for i, (name, color) in enumerate(zip(names, colors)):
        mask = labels == i
        axes[0].scatter(features_2d[mask, 0], features_2d[mask, 1],
                        c=color, label=name, alpha=0.6, s=10, edgecolors="none")
    axes[0].set_title(f"t-SNE of X₀ — {dataset_name.upper()} (MBSC-Net)", fontsize=13, fontweight="bold")
    axes[0].legend(markerscale=3)
    axes[0].grid(True, alpha=0.3)

    from scipy.stats import gaussian_kde
    for i, (name, color) in enumerate(zip(names, colors)):
        mask = labels == i
        axes[1].scatter(features_2d[mask, 0], features_2d[mask, 1],
                        c=color, label=name, alpha=0.4, s=8, edgecolors="none")
        if mask.sum() > 2:
            try:
                kde = gaussian_kde(features_2d[mask].T)
                xmin, xmax = features_2d[:, 0].min(), features_2d[:, 0].max()
                ymin, ymax = features_2d[:, 1].min(), features_2d[:, 1].max()
                xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                density = kde(positions).reshape(xx.shape)
                axes[1].contour(xx, yy, density, levels=5, colors=color, alpha=0.7, linewidths=1.5)
            except Exception:
                pass
    axes[1].set_title(f"X₀ Density — {dataset_name.upper()} (MBSC-Net)", fontsize=13, fontweight="bold")
    axes[1].legend(markerscale=3)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is None:
        save_dir = os.path.join(Config.PLOT_DIR, dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "tsne_mbsc.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"t-SNE saved to {save_path}")
    return save_path


def plot_score_distribution(probs, labels, dataset_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Probability histogram
    for label_val, color, name in [(0, "#2ca02c", "Real"), (1, "#d62728", "Fake")]:
        mask = labels == label_val
        axes[0].hist(probs[mask], bins=40, alpha=0.6, color=color,
                     label=f"{name} (μ={probs[mask].mean():.4f})")
    axes[0].set_title("Classifier P(fake) Distribution", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(0.5, color="black", linestyle="--", linewidth=1.5)

    # Box plot
    data = [probs[labels == 0], probs[labels == 1]]
    bp = axes[1].boxplot(data, labels=["Real", "Fake"], patch_artist=True)
    bp["boxes"][0].set_facecolor("#2ca02c")
    bp["boxes"][1].set_facecolor("#d62728")
    axes[1].set_title("P(fake) by Class", fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Score Distribution — {dataset_name.upper()} (MBSC-Net)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    save_dir = os.path.join(Config.PLOT_DIR, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "scores_mbsc.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Score distribution saved to {save_path}")
    return save_path


def run_visualization(dataset_name="weibo", text_encoder_type="bert_chinese",
                      checkpoint_path=None, max_samples=None):
    features, probs, labels = extract_features_and_scores(
        dataset_name, text_encoder_type, checkpoint_path, max_samples
    )
    plot_tsne(features, labels, dataset_name)
    plot_score_distribution(probs, labels, dataset_name)
    return features, probs, labels


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="weibo", choices=["weibo", "weibo21", "pheme"])
    parser.add_argument("--text_encoder", type=str, default="bert_chinese", choices=["bert_chinese", "bert_uncased"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    run_visualization(args.dataset, args.text_encoder,
                      args.checkpoint, args.max_samples)
