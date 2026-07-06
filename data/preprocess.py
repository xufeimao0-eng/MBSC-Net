import os
import re
import json
import random
import pandas as pd
from PIL import Image
from config import Config

def clean_text(text):
    """Clean text by removing URLs and extra whitespace."""
    if not isinstance(text, str):
        return ""
    if text.strip().lower() == "nan":
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_weibo_data(split="train"):
    """
    Load Weibo21 dataset.
    Returns list of dicts: {id, text, image_paths, label}
    Weibo text = title + content
    Checks processed JSON first (includes event_id from clustering).
    """
    # Check for processed data first
    processed_dir = os.path.join(Config.WEIBO_DIR, "processed")
    json_path = os.path.join(processed_dir, f"{split}.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    file_map = {
        "train": Config.WEIBO_TRAIN,
        "val": Config.WEIBO_VAL,
        "test": Config.WEIBO_TEST,
    }

    if split not in file_map:
        raise ValueError(f"Unknown split: {split}")

    df = pd.read_excel(file_map[split])
    image_dir = Config.WEIBO_DIR
    samples = []

    for idx, row in df.iterrows():
        title_val = row.get("title", "")
        content_val = row.get("content", "")
        title = str(title_val) if pd.notna(title_val) else ""
        content = str(content_val) if pd.notna(content_val) else ""
        text = clean_text(title + " " + content)

        label = int(row["label"])

        image_field = str(row.get("image", ""))
        image_paths = []
        if image_field and image_field != "nan":
            for img_name in image_field.split("|"):
                img_name = img_name.strip()
                full_path = os.path.join(image_dir, img_name)
                if os.path.exists(full_path):
                    image_paths.append(full_path)

        samples.append({
            "id": str(idx),
            "text": text,
            "image_paths": image_paths,
            "label": label,
        })

    return samples


def load_weibo21_data(split="train", val_ratio=0.2, seed=42):
    """
    Load weibo dataset. Uses processed JSON if available, otherwise raw txt.
    split='train': training data
    split='val': validation data
    split='test': test data
    """
    # Check for processed data first
    processed_dir = os.path.join(Config.WEIBO21_DIR, "processed")
    json_path = os.path.join(processed_dir, f"{split}.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Fallback: load from raw txt files with stratified split
    if split == "test":
        rumor_file = os.path.join(Config.WEIBO21_TWEETS, "test_rumor.txt")
        nonrumor_file = os.path.join(Config.WEIBO21_TWEETS, "test_nonrumor.txt")
    else:
        rumor_file = os.path.join(Config.WEIBO21_TWEETS, "train_rumor.txt")
        nonrumor_file = os.path.join(Config.WEIBO21_TWEETS, "train_nonrumor.txt")

    def _load_file(data_file, label_val, img_dir):
        samples = []
        if not os.path.exists(data_file):
            return samples
        with open(data_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for i in range(0, len(lines), 3):
            if i + 2 >= len(lines):
                break
            meta = lines[i].strip()
            img_line = lines[i + 1].strip()
            text = clean_text(lines[i + 2])
            post_id = meta.split("|")[0] if "|" in meta else str(i)
            image_paths = []
            if img_line and img_line != "null":
                for img_url in img_line.split("|"):
                    img_url = img_url.strip()
                    if not img_url or img_url == "null":
                        continue
                    img_name = img_url.split("/")[-1]
                    if img_name:
                        full_path = os.path.join(img_dir, img_name)
                        if os.path.exists(full_path):
                            image_paths.append(full_path)
            if not text:
                continue
            samples.append({
                "id": post_id,
                "text": text,
                "image_paths": image_paths,
                "label": label_val,
            })
        return samples

    # Load both classes
    rumor_samples = _load_file(rumor_file, 1, Config.WEIBO21_RUMOR_IMG)
    nonrumor_samples = _load_file(nonrumor_file, 0, Config.WEIBO21_NONRUMOR_IMG)

    if split == "test":
        return rumor_samples + nonrumor_samples

    # Stratified train/val split
    rng = random.Random(seed)
    rng.shuffle(rumor_samples)
    rng.shuffle(nonrumor_samples)

    rumor_val_n = int(len(rumor_samples) * val_ratio)
    nonrumor_val_n = int(len(nonrumor_samples) * val_ratio)

    if split == "val":
        return rumor_samples[:rumor_val_n] + nonrumor_samples[:nonrumor_val_n]
    else:
        return rumor_samples[rumor_val_n:] + nonrumor_samples[nonrumor_val_n:]


def load_pheme_data(split="train"):
    """
    Load PHEME dataset.
    Returns list of dicts: {id, text, image_paths, label}

    Data format:
      - pheme.train / pheme.dev / pheme.test: tab-separated (tweet_id, text, label)
        label = "non-rumor" or "false"
      - content.csv: maps tweet mid -> imgnum -> image file
    """
    import csv
    base_dir = os.path.join(Config.DATA_DIR, "pheme dataset")

    # Build mid -> imgnum mapping from content.csv
    content_csv = os.path.join(base_dir, "content.csv")
    mid_to_imgnum = {}
    with open(content_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row["mid"].strip()
            imgnum = row["imgnum"].strip()
            mid_to_imgnum[mid] = imgnum

    image_dir = os.path.join(base_dir, "pheme_images_jpg")

    # Load split file
    split_file = os.path.join(base_dir, f"pheme.{'dev' if split == 'val' else split}")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"PHEME split file not found: {split_file}")

    samples = []
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            mid = parts[0]
            text = clean_text(parts[1])
            label_str = parts[2].strip().lower()
            label = 1 if label_str == "false" else 0  # false=rumor→1, non-rumor→0

            imgnum = mid_to_imgnum.get(mid, None)
            image_paths = []
            if imgnum is not None:
                for ext in [".jpg", ".png", ".jpeg", ".gif"]:
                    img_path = os.path.join(image_dir, imgnum + ext)
                    if os.path.exists(img_path):
                        image_paths.append(img_path)
                        break

            samples.append({
                "id": mid,
                "text": text,
                "image_paths": image_paths,
                "label": label,
            })

    return samples


def load_dataset(dataset_name, split="train", **kwargs):
    """Unified data loading interface."""
    if dataset_name == "weibo":
        return load_weibo_data(split)
    elif dataset_name == "weibo21":
        return load_weibo21_data(split)
    elif dataset_name == "pheme":
        return load_pheme_data(split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
