#!/usr/bin/env python3
"""
Optional DistilBERT fine-tuning module for price tier classification.

This module provides a DistilBERT-based text classifier that can be used
as an alternative to TF-IDF + ensemble models. It is NOT run by default in
train.py (to keep training fast), but can be invoked standalone.

Usage:
    python distilbert_classifier.py --data_path data/processed/products.csv \
                                    --output_dir models/distilbert \
                                    --epochs 3

Requirements:
    pip install transformers torch datasets
"""
import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

LABEL_MAP = {"budget": 0, "mid-range": 1, "premium": 2, "luxury": 3}
LABEL_INV = {v: k for k, v in LABEL_MAP.items()}
MODEL_NAME = "distilbert-base-uncased"


def check_dependencies():
    """Verify that optional heavy dependencies are available."""
    missing = []
    try:
        import torch
        log.info(f"PyTorch {torch.__version__} - GPU: {torch.cuda.is_available()}")
    except ImportError:
        missing.append("torch")
    try:
        import transformers
        log.info(f"Transformers {transformers.__version__}")
    except ImportError:
        missing.append("transformers")
    if missing:
        raise ImportError(
            f"Missing dependencies: {missing}. "
            f"Install with: pip install {' '.join(missing)}"
        )


def load_data(data_path: str):
    """Load processed CSV and prepare texts + labels."""
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["price_tier"])
    df["text"] = (df["title"].fillna("") + " " + df["description"].fillna("")).str.lower()
    df["label"] = df["price_tier"].map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    log.info(f"Loaded {len(df)} samples. Label distribution: {df['label'].value_counts().to_dict()}")
    return df[["text", "label"]].reset_index(drop=True)


def train(data_path: str, output_dir: str, epochs: int = 3, batch_size: int = 16,
          max_len: int = 128, lr: float = 2e-5):
    """Fine-tune DistilBERT for 4-class price tier classification."""
    check_dependencies()

    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification,
        AdamW,
        get_linear_schedule_with_warmup,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, accuracy_score

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # -- Data --
    df = load_data(data_path)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    class PriceDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.encodings = tokenizer(
                list(texts), truncation=True, padding="max_length",
                max_length=max_len, return_tensors="pt"
            )
            self.labels = torch.tensor(list(labels), dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids":      self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels":         self.labels[idx],
            }

    train_dataset = PriceDataset(train_df["text"], train_df["label"], tokenizer, max_len)
    val_dataset   = PriceDataset(val_df["text"],   val_df["label"],   tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)

    # -- Model --
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=4, ignore_mismatched_sizes=True
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    best_f1 = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        # -- Train --
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # -- Validate --
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].numpy())

        f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        acc = accuracy_score(all_labels, all_preds)
        log.info(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f} | val F1-macro: {f1:.4f} | acc: {acc:.4f}")
        history.append({"epoch": epoch, "loss": avg_loss, "f1_macro": f1, "accuracy": acc})

        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(str(output_path))
            tokenizer.save_pretrained(str(output_path))
            log.info(f"  Saved best model (F1={f1:.4f}) to {output_path}")

    # Save training history
    with open(output_path / "training_history.json", "w") as fh:
        json.dump({"history": history, "best_f1": best_f1, "label_map": LABEL_MAP}, fh, indent=2)

    log.info(f"DistilBERT training complete. Best F1-macro: {best_f1:.4f}")
    return best_f1


def predict_single(text: str, model_dir: str) -> dict:
    """
    Load a saved DistilBERT model and predict one sample.
    Returns: {"label": str, "probabilities": dict}
    """
    check_dependencies()
    import torch
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    enc = tokenizer(text.lower(), truncation=True, padding="max_length",
                    max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device)
        ).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    return {
        "label": LABEL_INV[pred_idx],
        "probabilities": {LABEL_INV[i]: round(float(p), 4) for i, p in enumerate(probs)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for price tier classification")
    parser.add_argument("--data_path",  default="data/processed/products.csv",
                        help="Path to processed products CSV")
    parser.add_argument("--output_dir", default="models/distilbert",
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len",    type=int, default=128)
    parser.add_argument("--lr",         type=float, default=2e-5)
    args = parser.parse_args()

    log.info("=== DistilBERT Fine-Tuning ===")
    best = train(
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_len=args.max_len,
        lr=args.lr,
    )
    log.info(f"Done. Best F1-macro on validation set: {best:.4f}")
