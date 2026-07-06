"""
MBSC-Net: Multimodal Bidirectional Semantic Consistency Network for Fake News Detection.

Usage:
    python main.py --mode all --dataset weibo
    python main.py --mode train --dataset weibo
    python main.py --mode visualize --dataset weibo
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from train import train
from evaluate import evaluate
from visualize import run_visualization


def main():
    parser = argparse.ArgumentParser(description="MBSC-Net — Multimodal Fake News Detection")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["train", "eval", "visualize", "all"])
    parser.add_argument("--dataset", type=str, default="weibo",
                        choices=["weibo", "weibo21", "pheme"])
    parser.add_argument("--text_encoder", type=str, default="bert_chinese",
                        choices=["bert_chinese", "bert_uncased"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ensemble", action="store_true")

    args = parser.parse_args()

    Config.print_config()

    if args.mode in ("train", "all"):
        print("\n" + "=" * 60)
        print("STAGE 1: Training")
        print("  BERT-fullFT + FocalLoss + Contrastive")
        print("=" * 60)
        if args.ensemble:
            from train import train_ensemble
            train_ensemble(args.dataset, args.text_encoder)
        else:
            from train import train
            train(args.dataset, args.text_encoder, seed=args.seed)

    if args.mode in ("eval", "all"):
        print("\n" + "=" * 60)
        print("STAGE 2: Evaluation")
        print("=" * 60)
        evaluate(
            dataset_name=args.dataset,
            text_encoder_type=args.text_encoder,
            checkpoint_path=args.checkpoint,
        )

    if args.mode in ("visualize", "all"):
        print("\n" + "=" * 60)
        print("STAGE 3: t-SNE + Score Distribution Visualization")
        print("=" * 60)
        run_visualization(
            dataset_name=args.dataset,
            text_encoder_type=args.text_encoder,
            checkpoint_path=args.checkpoint,
            max_samples=args.max_samples,
        )

    print("\n" + "=" * 60)
    print("All tasks completed!")
    print(f"Results saved to: {Config.RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
