import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from config import Config


class MultimodalDataset(Dataset):
    """Dataset for multimodal fake news detection (text + images).

    Supports BERT MLM text augmentation (v7): pass aug_data (dict mapping
    original_id → list of augmented texts) and aug_prob to enable on-the-fly
    text substitution during training.
    """

    def __init__(self, samples, clip_processor, tokenizer, max_images=None,
                 text_encoder_type="clip", aug_data=None, aug_prob=0.5):
        self.samples = samples
        self.clip_processor = clip_processor
        self.tokenizer = tokenizer
        self.max_images = max_images or Config.MAX_IMAGES_PER_POST
        self.text_encoder_type = text_encoder_type
        self.max_text_length = (
            Config.MAX_TEXT_LENGTH_BERT if text_encoder_type == "bert_chinese"
            else Config.MAX_TEXT_LENGTH_CLIP
        )
        self.aug_data = aug_data or {}
        self.aug_prob = aug_prob

        self.image_transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

    def _load_image(self, path):
        try:
            img = Image.open(path).convert("RGB")
            return self.image_transform(img)
        except Exception:
            return torch.zeros(3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample["label"]
        image_paths = sample["image_paths"]

        # BERT MLM augmentation (v7): randomly substitute text
        text = sample["text"]
        if self.aug_data:
            sample_id = sample.get("id", str(idx))
            if sample_id in self.aug_data:
                augs = self.aug_data[sample_id]
                if augs and random.random() < self.aug_prob:
                    text = random.choice(augs)

        # Load images (up to max_images)
        images = []
        for path in image_paths[:self.max_images]:
            img_tensor = self._load_image(path)
            images.append(img_tensor)

        # Pad to max_images with zeros
        while len(images) < self.max_images:
            images.append(torch.zeros(3, Config.IMAGE_SIZE, Config.IMAGE_SIZE))

        # Image mask: 1 for valid images, 0 for padding
        num_valid = min(len(image_paths), self.max_images)
        image_mask = torch.zeros(self.max_images)
        image_mask[:num_valid] = 1.0

        # Tokenize text
        if self.tokenizer is not None:
            if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
                encoding = self.tokenizer(
                    text,
                    max_length=self.max_text_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = encoding["input_ids"].squeeze(0)
                text_attention_mask = encoding["attention_mask"].squeeze(0)
            else:
                text_input_ids = torch.zeros(self.max_text_length, dtype=torch.long)
                text_attention_mask = torch.zeros(self.max_text_length)
                text_attention_mask[:min(len(text), self.max_text_length)] = 1
        else:
            text_input_ids = torch.zeros(self.max_text_length, dtype=torch.long)
            text_attention_mask = torch.zeros(self.max_text_length)

        result = {
            "images": torch.stack(images),
            "image_mask": image_mask,
            "text": text,
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "label": torch.tensor(label, dtype=torch.long),
            "id": sample["id"],
        }
        if "event_id" in sample:
            result["event_id"] = torch.tensor(sample["event_id"], dtype=torch.long)
        return result


class TextOnlyDataset(Dataset):
    """Text-only dataset (for test sets without images)."""

    def __init__(self, samples, tokenizer, text_encoder_type="clip"):
        self.samples = samples
        self.tokenizer = tokenizer
        self.text_encoder_type = text_encoder_type
        self.max_text_length = (
            Config.MAX_TEXT_LENGTH_BERT if text_encoder_type == "bert_chinese"
            else Config.MAX_TEXT_LENGTH_CLIP
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        label = sample["label"]

        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                max_length=self.max_text_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = encoding["input_ids"].squeeze(0)
            text_attention_mask = encoding["attention_mask"].squeeze(0)
        else:
            text_input_ids = torch.zeros(self.max_text_length, dtype=torch.long)
            text_attention_mask = torch.zeros(self.max_text_length)

        return {
            "text": text,
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "label": torch.tensor(label, dtype=torch.long),
            "id": sample["id"],
        }
