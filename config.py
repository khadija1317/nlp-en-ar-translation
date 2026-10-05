"""All settings in one place. Paths are relative to this file."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODEL_DIR = ROOT / "models" / "nllb_final"

# OPUS-100 en-ar files expected in data/raw/
RAW_FILES = {
    "train": ("opus.ar-en-train.en", "opus.ar-en-train.ar"),
    "dev": ("opus.ar-en-dev.en", "opus.ar-en-dev.ar"),
    "test": ("opus.ar-en-test.en", "opus.ar-en-test.ar"),
}

# cleaning (applied identically to train, dev and test)
MAX_LEN = 100                 # max words per side
MAX_RATIO = 4                 # max(en_len, ar_len) / max(min(en_len, ar_len), 1)
LATIN_RATIO_THRESHOLD = 0.5   # drop Arabic lines whose Latin-word share is >= this
N_TRAIN = 40000
SEED = 42

# counts / sample head of the run that trained the released checkpoint; prepare_data.py checks them
# when it is fed the standard 1,000,000-line OPUS-100 ar-en train file
EXPECTED = {
    "n_input": 1000000,
    "n_after_dedup": 959929,
    "n_after_length_ratio": 949014,
    "n_after_tag_strip": 948984,
    "n_final": 941277,
    "sample_head_en": [
        "Their right to actually have a say in what the post Saddam era would look like",
        "Are you kidding me?",
        "_",
        "Promotional activities.",
        "Cause you only go around once in this life.",
    ],
}

# models
BASE_MODEL = "facebook/nllb-200-distilled-600M"
FINETUNED_MODEL = "khadija-1317/nllb-finetuned"   # Hub repo id or a local path
SRC_LANG, TGT_LANG = "eng_Latn", "arb_Arab"
GENERATION = {"num_beams": 1, "max_length": 128, "batch_size": 16}   # same for every system

# fine-tuning: exactly the run that produced the released checkpoint
TRAINING = dict(
    seed=42, max_length=128, learning_rate=3e-5, lr_scheduler_type="linear", warmup_steps=500,
    per_device_train_batch_size=2, per_device_eval_batch_size=4, gradient_accumulation_steps=8,
    num_train_epochs=3, weight_decay=0.01, eval_steps=250, logging_steps=50, early_stopping_patience=3,
)

IBM1_ITERATIONS = 5
BERTSCORE_LANG = "ar"          # -> bert-base-multilingual-cased
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42
FLORES_DATASET, FLORES_SPLIT = "yash9439/flores200", "devtest"
