# MBSC-Net: Multimodal Bidirectional Semantic Consistency Network for Fake News Detection

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**MBSC-Net** (Multimodal Bidirectional Semantic Consistency Network) is a deep
learning framework for multimodal fake news detection. It jointly models text
and visual content from social media posts to distinguish real news from rumors,
using:

- **Bidirectional semantic consistency** via bilinear fusion, cross-attention,
  and modality gating
- **Supervised contrastive loss** on fused multimodal features

## Supported Datasets

| Dataset  | Language | Posts  | Modalities     |
|----------|----------|--------|----------------|
| Weibo21  | Chinese  | ~10K   | Text + Image   |
| Weibo    | Chinese  | ~5K    | Text + Image   |
| PHEME    | English  | ~6K    | Text + Image   |

See [DATASETS.md](DATASETS.md) for download and preparation instructions.

## Architecture

```
Images ──→ [CLIP/Chinese-CLIP Vision Encoder] ──→ Image Features ──┐
                                                                     ├──→ Bilinear Fusion
Text ────→ [BERT/CLIP Text Encoder] ────────────→ Text Features ────┤    + Cross-Attention
                                                                     │    + Modality Gating
                                                                     ├──→ X₀ Feature
                                                                     │
                                                                     ├──→ Classifier ──→ Real/Fake
                                                                     └──→ Contrastive Projection
```

## Key Features

- **Multi-branch fusion**: Hadamard bilinear pooling, cross-modal
  attention, and learnable modality gating
- **Flexible encoders**: CLIP ViT-B/32 or Chinese-CLIP ViT-B/16 for vision;
  BERT-base-chinese, BERT-base-uncased, or CLIP for text
- **Supervised contrastive loss**: Intra-batch contrastive learning on fused
  multimodal features
- **Multi-dataset support**: Unified data loading for 3 datasets with automatic
  format detection
- **Multi-seed evaluation**: Robust evaluation with mean ± std reporting across
  random seeds

## Installation

### Requirements

- Python 3.9 or higher
- PyTorch 2.0 or higher
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/xufeimao0-eng/MBSC-Net.git
cd MBSC-Net

# Install dependencies
pip install -r requirements.txt
```

### Model Weights

Pre-trained encoder weights are downloaded automatically from HuggingFace Hub on
first use. To use local checkpoints instead, set the following environment
variables:

```bash
export CLIP_MODEL_PATH=/path/to/clip-vit-base-patch32
export CHINESE_CLIP_MODEL_PATH=/path/to/chinese-clip-vit-base-patch16
export BERT_MODEL_PATH=/path/to/bert-base-chinese
export BERT_UNCASED_MODEL_PATH=/path/to/bert-base-uncased
```

### Results Directory

By default, checkpoints, logs, and plots are saved to `./results/`. Override
with:

```bash
export MBSC_RESULTS_DIR=/path/to/results
```

## Quick Start

### Training

```bash
# Train on Weibo21 with BERT-Chinese text encoder
python main.py --mode train --dataset weibo

# Train on Weibo21 with Chinese-CLIP vision encoder
python main.py --mode train --dataset weibo --text_encoder bert_chinese

# Train with a specific random seed
python main.py --mode train --dataset weibo --seed 42

# Train on PHEME (English) with CLIP encoder
python main.py --mode train --dataset pheme --text_encoder clip

# Train on PHEME (English) with BERT-base-uncased
python main.py --mode train --dataset pheme --text_encoder bert_uncased
```

### Evaluation

```bash
# Single checkpoint evaluation
python main.py --mode eval --dataset weibo --checkpoint results/checkpoints/weibo/default/seed_42/best_model.pt

# Multi-seed evaluation (3 seeds)
python evaluate.py --dataset weibo --seeds 42,123,456
```

### Visualization

```bash
# t-SNE feature visualization + score distributions
python main.py --mode visualize --dataset weibo --max_samples 500
```

### Full Pipeline

```bash
# Train, evaluate, and visualize in one command
python main.py --mode all --dataset weibo
```

## Configuration

All hyperparameters are centralized in `config.py`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 32 | Training batch size |
| `NUM_EPOCHS` | 200 | Maximum training epochs |
| `LEARNING_RATE` | 1e-4 | AdamW learning rate |
| `PATIENCE` | 50 | Early stopping patience |
| `FUSION_DIM` | 512 | Fusion projection dimension |
| `X0_DIM` | 512 | Output feature dimension |
| `BILINEAR_TYPE` | `"hadamard"` | Bilinear fusion type |
| `TEXT_ENCODER_TYPE` | `"bert_chinese"` | Text encoder (`"bert_chinese"`, `"bert_uncased"`, `"clip"`) |
| `VISION_ENCODER_TYPE` | `"clip"` | Vision encoder (`"clip"`, `"chinese_clip"`) |

Override via environment variables:

```bash
export BATCH_SIZE=64
export LEARNING_RATE=5e-5
```

## Project Structure

```
MBSC-Net/
├── main.py                  # Main entry point
├── train.py                 # Training loop with multi-loss optimization
├── evaluate.py              # Single and multi-seed evaluation
├── visualize.py             # t-SNE and score distribution plots
├── config.py                # Central configuration
├── requirements.txt         # Python dependencies
├── data/
│   ├── dataset.py           # PyTorch Dataset classes
│   └── preprocess.py        # Dataset loading and preprocessing
├── models/
│   ├── encoders.py          # Vision and text encoders
│   ├── fusion.py            # Multi-modal fusion pipeline
│   ├── pipeline.py          # Main FakeNewsDiscriminator model
│   └── classifier.py        # Legacy classifier
├── scripts/                 # Experiment scripts
│   ├── run_ablation.py      # Ablation study
│   ├── run_baselines.py     # Baseline model implementations
│   ├── run_sensitivity.py   # Hyperparameter sensitivity analysis
│   ├── run_significance.py  # Statistical significance testing
│   ├── augment_dataset.py   # BERT MLM text augmentation
│   └── *.sh                 # Experiment launcher scripts
└── results/                 # Generated outputs (gitignored)
    ├── checkpoints/
    ├── logs/
    └── plots/
```

## Results

Performance on Weibo21 test set (mean ± std over 3 seeds):

| Method | Accuracy | F1-Score |
|--------|----------|----------|
| MBSC-Net (Hadamard) | -- | -- |

*Fill in your experimental results here.*

## Citation

If you find this work useful, please cite:

```bibtex
@article{mbscnet,
  title={MBSC-Net: Multimodal Bidirectional Semantic Consistency Network
         for Fake News Detection},
  author={},
  journal={},
  year={2025}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgements

This project uses pre-trained models from:
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)

Dataset sources:
- Weibo21 / Weibo: Sina Weibo microblogging platform
- PHEME: Twitter breaking news dataset
