"""Fine-tune NLLB-200-distilled-600M (EN -> AR) on data/processed/train_sample.json. Needs a GPU.

Port of the Kaggle run that produced the released checkpoint: same hyperparameters (config.py), 8-bit AdamW,
fp16, gradient checkpointing, early stopping on dev loss, best in-memory weights restored at the end.
Note: that checkpoint used a 1,999-pair dev set (one corrupted pair removed, nothing else); this script uses the
fully cleaned dev set, so a re-run will not reproduce it exactly.
Usage: python finetune.py       (HF_TOKEN env var only needed if the base model download requires auth)
"""
import config
from nllb_utils import hf_token, load_pairs


def main():
    import bitsandbytes as bnb
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, EarlyStoppingCallback,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback)

    if not torch.cuda.is_available():
        raise SystemExit("No GPU found (fp16 and 8-bit AdamW need CUDA).")

    t = config.TRAINING
    train_en, train_ar = load_pairs(config.DATA_PROCESSED / "train_sample.json")
    dev_en, dev_ar = load_pairs(config.DATA_PROCESSED / "dev.json")
    print(f"train: {len(train_en)} pairs | dev: {len(dev_en)} pairs")

    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL, src_lang=config.SRC_LANG, tgt_lang=config.TGT_LANG,
                                        token=hf_token())
    model = AutoModelForSeq2SeqLM.from_pretrained(config.BASE_MODEL, token=hf_token())
    model.config.use_cache = False          # required with gradient checkpointing

    def tokenize(batch):
        return tok(batch["en"], text_target=batch["ar"], max_length=t["max_length"], truncation=True)

    def make(en, ar):
        return Dataset.from_dict({"en": en, "ar": ar}).map(tokenize, batched=True, remove_columns=["en", "ar"])

    class BestInMemory(TrainerCallback):
        """Keeps the weights with the lowest dev loss (no checkpoints written to disk)."""
        def __init__(self):
            self.best_loss, self.best_state = float("inf"), None

        def on_evaluate(self, args, state, control, metrics, model, **kwargs):
            loss = metrics.get("eval_loss")
            if loss is not None and loss < self.best_loss:
                self.best_loss = loss
                self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    best_cb = BestInMemory()
    args = Seq2SeqTrainingArguments(
        output_dir=str(config.MODEL_DIR.parent / "trainer_scratch"),
        learning_rate=t["learning_rate"], lr_scheduler_type=t["lr_scheduler_type"], warmup_steps=t["warmup_steps"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"], gradient_checkpointing=True,
        num_train_epochs=t["num_train_epochs"], weight_decay=t["weight_decay"], fp16=True,
        eval_strategy="steps", eval_steps=t["eval_steps"], save_strategy="no", save_steps=t["eval_steps"],
        save_total_limit=1, load_best_model_at_end=False, save_only_model=True,
        metric_for_best_model="eval_loss", greater_is_better=False, prediction_loss_only=True,
        logging_steps=t["logging_steps"], report_to="none", seed=t["seed"],
    )
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=make(train_en, train_ar), eval_dataset=make(dev_en, dev_ar),
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
        optimizers=(optimizer, None),       # the Trainer builds the scheduler from lr_scheduler_type
        callbacks=[EarlyStoppingCallback(early_stopping_patience=t["early_stopping_patience"]), best_cb],
    )
    trainer.train()
    if best_cb.best_state is not None:
        model.load_state_dict(best_cb.best_state)
        print(f"Restored best weights (dev loss {best_cb.best_loss:.6f})")

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(config.MODEL_DIR))
    tok.save_pretrained(str(config.MODEL_DIR))
    print("Saved model + tokenizer ->", config.MODEL_DIR)


if __name__ == "__main__":
    main()
