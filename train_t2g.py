# train_t2g.py
"""
Text-to-Gloss (T2G) Fine-Tuning on PHOENIX-2014T
===============================================
Fine-tunes a T5 model to translate spoken German text into sign language glosses,
replicating the methodology and baseline evaluation metrics (BLEU-1 to BLEU-4, ROUGE)
from the WSLP 2025 paper.

Usage in Kaggle
---------------
  python train_t2g.py --base_path /kaggle/working/phoenix_annotations --epochs 5 --batch_size 8
"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
from pathlib import Path

# Evaluation metrics
import evaluate
import numpy as np
import torch
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def load_dataset_from_pickle(gz_path: str) -> list[dict]:
    """Load decompressed PHOENIX-2014T annotations."""
    with gzip.open(gz_path, "rb") as f:
        raw_data = pickle.load(f)
    if isinstance(raw_data, dict):
        return list(raw_data.values())
    return raw_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="/kaggle/working/phoenix_annotations")
    parser.add_argument("--model_name", type=str, default="google/t5-v1_1-base")  # Or 't5-base' or 'facebook/mbart-large-50'
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/t2g_model")
    args = parser.parse_args()

    # 1. Load PHOENIX-2014T Splits
    print("[T2G Train] Loading annotations...")
    train_path = Path(args.base_path) / "phoenix14t.pami0.train.annotations_only.gzip"
    test_path  = Path(args.base_path) / "phoenix14t.pami0.test.annotations_only.gzip"
    dev_path   = Path(args.base_path) / "phoenix14t.pami0.dev.annotations_only.gzip"

    train_data = load_dataset_from_pickle(str(train_path))
    dev_data   = load_dataset_from_pickle(str(dev_path))

    # Convert to HuggingFace dataset format
    # Source text: 'text' (spoken German)
    # Target text: 'gloss' (sign language gloss)
    train_ds = HFDataset.from_dict({
        "input_text":  [item["text"] for item in train_data],
        "target_text": [item["gloss"] for item in train_data]
    })
    dev_ds = HFDataset.from_dict({
        "input_text":  [item["text"] for item in dev_data],
        "target_text": [item["gloss"] for item in dev_data]
    })

    print(f"Loaded {len(train_ds)} train examples and {len(dev_ds)} dev examples.")

    # 2. Tokenizer & Model setup
    print(f"[T2G Train] Initializing model & tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model     = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # Prefix required for some T5 models
    prefix = "translate German to Gloss: "

    def preprocess_function(examples):
        inputs  = [prefix + doc for doc in examples["input_text"]]
        targets = [doc for doc in examples["target_text"]]
        
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)
        labels = tokenizer(text_target=targets, max_length=128, truncation=True)
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_ds.map(preprocess_function, batched=True)
    tokenized_dev   = dev_ds.map(preprocess_function, batched=True)

    # 3. Metrics (BLEU and ROUGE)
    sacrebleu = evaluate.load("sacrebleu")
    rouge     = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
            
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in labels as we cannot decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some post-processing
        decoded_preds  = [pred.strip().upper() for pred in decoded_preds]
        decoded_labels = [[label.strip().upper()] for label in decoded_labels]

        # Calculate BLEU-1 to BLEU-4
        bleu_results = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
        
        # Calculate ROUGE
        flat_labels = [lbl[0] for lbl in decoded_labels]
        rouge_results = rouge.compute(predictions=decoded_preds, references=flat_labels)

        return {
            "bleu":       round(bleu_results["score"], 4),
            "rougeL":     round(rouge_results["rougeL"] * 100, 4)
        }

    # 4. Trainer
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none"
    )

    data_collator = DataCollatorForSeq2SeqLM(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("[T2G Train] Starting fine-tuning...")
    trainer.train()

    # Save final model
    print(f"[T2G Train] Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[T2G Train] Completed successfully!")


if __name__ == "__main__":
    main()
