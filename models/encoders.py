"""
Vision and Text Encoders for MBSC-Net.

Vision: CLIP ViT-B/32 (frozen).
Text:   BERT-base-chinese (Chinese datasets) or BERT-base-uncased (English).
"""

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPProcessor, BertModel, BertTokenizer
from config import Config


class CLIPVisionEncoder(nn.Module):
    """CLIP ViT-B/32 vision encoder."""

    def __init__(self, model_path=None):
        super().__init__()
        model_path = model_path or Config.CLIP_MODEL_PATH
        try:
            self.vision_model = CLIPVisionModel.from_pretrained(model_path)
        except Exception:
            self.vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.hidden_size = self.vision_model.config.hidden_size

    def forward(self, pixel_values, return_sequence=False):
        """
        Args:
            pixel_values: (B, 3, 224, 224) or (B, N, 3, 224, 224)
            return_sequence: if True, return all patch tokens
        Returns:
            image_features: (B, hidden_size) or (B, N_patches, hidden_size)
        """
        if pixel_values.dim() == 5:
            B, N, C, H, W = pixel_values.shape
            pixel_values = pixel_values.view(B * N, C, H, W)
            outputs = self.vision_model(pixel_values=pixel_values)
            if return_sequence:
                seq = outputs.last_hidden_state
                seq = seq.view(B, N, seq.shape[1], -1).mean(dim=1)
                return seq
            pooled = outputs.pooler_output
            pooled = pooled.view(B, N, -1)
            pooled = pooled.mean(dim=1)
            return pooled
        else:
            outputs = self.vision_model(pixel_values=pixel_values)
            if return_sequence:
                return outputs.last_hidden_state
            return outputs.pooler_output


class BertEncoder(nn.Module):
    """BERT text encoder. Supports partial fine-tuning of last N layers."""

    def __init__(self, model_path=None, unfreeze_layers=None):
        super().__init__()
        model_path = model_path or Config.BERT_MODEL_PATH
        self.bert = BertModel.from_pretrained(model_path)
        self.hidden_size = self.bert.config.hidden_size  # 768

        # Freeze all params first
        for p in self.bert.parameters():
            p.requires_grad = False

        # Optionally unfreeze last N encoder layers + pooler
        unfreeze_layers = unfreeze_layers or Config.TEXT_ENCODER_UNFREEZE_LAYERS
        if unfreeze_layers > 0:
            layers = self.bert.encoder.layer
            total = len(layers)
            for i in range(max(0, total - unfreeze_layers), total):
                for p in layers[i].parameters():
                    p.requires_grad = True
            for p in self.bert.pooler.parameters():
                p.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output  # (B, 768)


class TextEncoderWrapper(nn.Module):
    """Wrapper that selects between BERT-Chinese and BERT-Uncased."""

    def __init__(self, encoder_type=None, bert_model_path=None):
        super().__init__()
        encoder_type = encoder_type or Config.TEXT_ENCODER_TYPE

        if encoder_type == "bert_uncased":
            self.encoder = BertEncoder(Config.BERT_UNCASED_MODEL_PATH)
            self.encoder_type = "bert_uncased"
        else:
            self.encoder = BertEncoder(bert_model_path)
            self.encoder_type = "bert_chinese"

        self.hidden_size = self.encoder.hidden_size

    def forward(self, input_ids, attention_mask=None):
        return self.encoder(input_ids, attention_mask)


def get_clip_processor():
    """Get CLIP image processor."""
    try:
        return CLIPProcessor.from_pretrained(Config.CLIP_MODEL_PATH)
    except Exception:
        return CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def get_tokenizer(text_encoder_type=None):
    """Get the tokenizer for text encoding."""
    text_encoder_type = text_encoder_type or Config.TEXT_ENCODER_TYPE

    if text_encoder_type == "bert_uncased":
        return BertTokenizer.from_pretrained(Config.BERT_UNCASED_MODEL_PATH)
    else:
        return BertTokenizer.from_pretrained(Config.BERT_MODEL_PATH)
