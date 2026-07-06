# Datasets

MBSC-Net supports three multimodal fake news detection datasets. Due to licensing
and size constraints, datasets are **not included** in this repository. Follow
the instructions below to obtain and prepare each dataset.

## Expected Directory Structure

After downloading and preparing the datasets, your project directory should look like:

```
MBSC-Net/
├── weibo21 dataset/          # Weibo21 (Chinese)
│   ├── train_datasets.xlsx
│   ├── val_datasets.xlsx
│   ├── test_datasets.xlsx
│   ├── rumor_images/
│   ├── nonrumor_images/
│   └── processed/            # (auto-generated)
├── weibo_dataset/            # Weibo (Chinese, legacy)
│   ├── tweets/
│   │   ├── train_rumor.txt
│   │   ├── train_nonrumor.txt
│   │   ├── test_rumor.txt
│   │   └── test_nonrumor.txt
│   ├── rumor_images/
│   ├── nonrumor_images/
│   └── processed/            # (auto-generated)
└── pheme dataset/            # PHEME (English)
    ├── pheme.train
    ├── pheme.dev
    ├── pheme.test
    ├── content.csv
    └── pheme_images_jpg/
```

## Dataset Descriptions

### Weibo21 (weibo)

- **Language**: Chinese
- **Source**: Sina Weibo (Chinese microblogging platform)
- **Format**: Excel (.xlsx) files with columns: `title`, `content`, `image`, `label`
- **Labels**: 0 = real (non-rumor), 1 = fake (rumor)
- **Splits**: train / val / test
- **Preprocessing**: Run `python scripts/preprocess_weibo.py` to generate optimized
  JSON files in `weibo21 dataset/processed/`. The data loader will use these
  processed files automatically if present.

### Weibo (weibo)

- **Language**: Chinese
- **Source**: Sina Weibo
- **Format**: Text files with 3-line blocks (meta | image_URL, text)
- **Labels**: 0 = real, 1 = fake
- **Splits**: train / test (no built-in validation split; a 80/20 stratified
  split is created on the fly)

### PHEME (pheme)

- **Language**: English
- **Source**: Twitter (collected during breaking news events)
- **Format**: Tab-separated files (`pheme.train`, `pheme.dev`, `pheme.test`) with
  columns: `tweet_id`, `text`, `label`
- **Image mapping**: `content.csv` maps tweet IDs to image file names
- **Labels**: `non-rumor` → 0, `false` → 1 (rumor)

## Data Loading

The unified `load_dataset()` function in `data/preprocess.py` handles all
datasets. It automatically detects whether processed JSON files exist and falls
back to raw formats:

```python
from data import load_dataset

train_samples = load_dataset("weibo", "train")
val_samples   = load_dataset("weibo", "val")
test_samples  = load_dataset("weibo", "test")
```

Supported dataset keys: `"weibo"`, `"weibo21"`, `"pheme"`.

Supported splits: `"train"`, `"val"`, `"test"`.
