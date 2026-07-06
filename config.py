import os


class Config:
    # ── Paths (use env vars with sensible defaults) ──
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Model paths: set environment variables to use local checkpoints,
    # otherwise download from HuggingFace Hub automatically.
    CLIP_MODEL_PATH = os.environ.get(
        "CLIP_MODEL_PATH", "openai/clip-vit-base-patch32"
    )
    BERT_MODEL_PATH = os.environ.get(
        "BERT_MODEL_PATH", "google-bert/bert-base-chinese"
    )
    BERT_UNCASED_MODEL_PATH = os.environ.get(
        "BERT_UNCASED_MODEL_PATH", "google-bert/bert-base-uncased"
    )

    # Dataset directories
    DATA_DIR = BASE_DIR
    WEIBO_DIR = os.path.join(BASE_DIR, "weibo21 dataset")
    WEIBO_TRAIN = os.path.join(WEIBO_DIR, "train_datasets.xlsx")
    WEIBO_VAL = os.path.join(WEIBO_DIR, "val_datasets.xlsx")
    WEIBO_TEST = os.path.join(WEIBO_DIR, "test_datasets.xlsx")

    WEIBO21_DIR = os.path.join(BASE_DIR, "weibo_dataset")
    WEIBO21_TWEETS = os.path.join(WEIBO21_DIR, "tweets")
    WEIBO21_RUMOR_IMG = os.path.join(WEIBO21_DIR, "rumor_images")
    WEIBO21_NONRUMOR_IMG = os.path.join(WEIBO21_DIR, "nonrumor_images")

    PHEME_DIR = os.path.join(BASE_DIR, "pheme dataset")

    # Results: project-relative by default, customizable via env var
    RESULTS_DIR = os.environ.get(
        "MBSC_RESULTS_DIR", os.path.join(BASE_DIR, "results")
    )
    CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
    LOG_DIR = os.path.join(RESULTS_DIR, "logs")
    PLOT_DIR = os.path.join(RESULTS_DIR, "plots")

    # ── Device ──
    DEVICE = "cuda"

    # ── Model ──
    TEXT_ENCODER_TYPE = "bert_chinese"  # "bert_chinese" or "bert_uncased"
    UNFREEZE_TEXT_ENCODER = True
    TEXT_ENCODER_UNFREEZE_LAYERS = 3

    CLIP_VISION_HIDDEN = 768
    FUSION_DIM = 512
    X0_DIM = 512
    DENOISER_HIDDEN = 2048
    SEMANTIC_NUM_LAYERS = 2
    SEMANTIC_NUM_HEADS = 8
    BILINEAR_TYPE = "hadamard"

    # ── Loss weights ──
    CLASSIFIER_LOSS_WEIGHT = 1.0
    CONTRASTIVE_LOSS_WEIGHT = 0.3

    # ── Training ──
    BATCH_SIZE = 32
    NUM_EPOCHS = 200
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 4
    SEED = 42
    PATIENCE = 50
    LABEL_SMOOTHING = 0.1
    USE_AMP = False
    CONTRASTIVE_TEMPERATURE = 0.1
    PROJECTION_DIM = 128

    # ── Image ──
    IMAGE_SIZE = 224
    MAX_IMAGES_PER_POST = 3

    # ── Text ──
    MAX_TEXT_LENGTH_BERT = 512

    @classmethod
    def print_config(cls):
        print("=" * 50)
        print("Configuration")
        print("=" * 50)
        for key, value in sorted(cls.__dict__.items()):
            if not key.startswith("_") and key != "print_config":
                print(f"  {key}: {value}")
        print("=" * 50)
