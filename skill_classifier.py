"""NLBSE 2027: RoBERTa multi-label skill classifier.

Run with --smoke-test before a full experiment. This is a new experimental
classifier, not the competition's archived Random Forest baseline.
"""
import argparse
import json
import random
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "FacebookAI/roberta-base"
TABLE = "nlbse_tool_competition_data_by_issue"
SEED = 42


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(path):
    if not path.is_file():
        raise FileNotFoundError(f"Database not found: {path}")
    with sqlite3.connect(path) as db:
        frame = pd.read_sql_query(f'SELECT * FROM "{TABLE}"', db)
    columns = list(frame.columns)
    if len(columns) != 226 or columns[9] != "Application":
        raise ValueError("Unexpected database layout; inspect before training")
    labels = columns[9:]
    text = (frame["issue text"].fillna("").astype(str).str.strip()
            + " </s> " + frame["issue description"].fillna("").astype(str).str.strip())
    y = frame[labels].apply(pd.to_numeric, errors="coerce").fillna(0).gt(0).to_numpy(dtype=np.float32)
    if len(labels) != 217:
        raise ValueError(f"Expected 217 labels, found {len(labels)}")
    return text.tolist(), y, labels


def split_indices(n):
    # The TEST partition mirrors safe_baseline.py: 20%, seed=42, no stratification.
    trainval, test = train_test_split(np.arange(n), test_size=0.20, random_state=SEED)
    train, val = train_test_split(trainval, test_size=0.125, random_state=SEED)
    return train, val, test  # 70% / 10% / 20% overall


class SkillDataset(Dataset):
    def __init__(self, texts, labels, indices, tokenizer, max_length):
        self.indices = np.asarray(indices)
        self.labels = labels
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = int(self.indices[index])
        result = self.tokenizer(
            self.texts[row], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt"
        )
        return {**{k: v.squeeze(0) for k, v in result.items()},
                "labels": torch.from_numpy(self.labels[row].copy())}


def compute_loss(logits, targets, loss_type, gamma=2.0):
    elementwise_bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if loss_type == "bce":
        return elementwise_bce.mean()
    # Symmetric binary focal loss. gamma=2 downweights well-classified examples.
    pt = torch.exp(-elementwise_bce)
    return ((1.0 - pt) ** gamma * elementwise_bce).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions, targets = [], []
    for batch in loader:
        labels = batch.pop("labels")
        features = {k: v.to(device) for k, v in batch.items()}
        logits = model(**features).logits
        predictions.append((torch.sigmoid(logits).cpu().numpy() >= 0.5).astype(np.uint8))
        targets.append(labels.numpy().astype(np.uint8))
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    return {"precision": float(precision), "recall": float(recall), "micro_f1": float(f1)}


def run(args):
    seed_everything(SEED)
    texts, labels, names = load_dataset(args.db)
    train_idx, val_idx, test_idx = split_indices(len(texts))
    print(f"Rows={len(texts)}, labels={len(names)}, train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}", flush=True)
    print(f"Positive labels per example: {labels.sum(axis=1).mean():.2f} on average", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(names), problem_type="multi_label_classification",
        id2label={i: name for i, name in enumerate(names)},
        label2id={name: i for i, name in enumerate(names)}
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}", flush=True)

    if args.smoke_test:
        # Forward + backward pass on two real training examples; no optimizer update.
        ds = SkillDataset(texts, labels, train_idx[:2], tokenizer, args.max_length)
        batch = next(iter(DataLoader(ds, batch_size=2)))
        y = batch.pop("labels").to(device)
        logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
        loss = compute_loss(logits, y, args.loss)
        loss.backward()
        print(f"SMOKE TEST PASSED: logits shape={tuple(logits.shape)}, {args.loss} loss={loss.item():.4f}", flush=True)
        return

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "split_indices.json").write_text(json.dumps({
        "seed": SEED, "train": train_idx.tolist(), "validation": val_idx.tolist(), "test": test_idx.tolist()
    }), encoding="utf-8")
    train_loader = DataLoader(SkillDataset(texts, labels, train_idx, tokenizer, args.max_length), batch_size=2, shuffle=True)
    val_loader = DataLoader(SkillDataset(texts, labels, val_idx, tokenizer, args.max_length), batch_size=2)
    test_loader = DataLoader(SkillDataset(texts, labels, test_idx, tokenizer, args.max_length), batch_size=2)
    model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_f1 = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            y = batch.pop("labels").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
                loss = compute_loss(logits.float(), y, args.loss)
            running_loss += loss.item()
            scaler.scale(loss / 8).backward()
            if (step + 1) % 8 == 0 or step + 1 == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if (step + 1) % 200 == 0:
                print(f"Epoch {epoch+1}, batch {step+1}/{len(train_loader)}, loss={loss.item():.4f}", flush=True)
        scores = evaluate(model, val_loader, device)
        history.append({"epoch": epoch + 1, "mean_train_loss": running_loss / len(train_loader), **scores})
        print(f"Validation epoch {epoch+1}: {scores}", flush=True)
        if scores["micro_f1"] > best_f1:
            best_f1 = scores["micro_f1"]
            model.save_pretrained(args.output / "best_model")
            tokenizer.save_pretrained(args.output / "best_model")
    best = AutoModelForSequenceClassification.from_pretrained(args.output / "best_model").to(device)
    test_scores = evaluate(best, test_loader, device)  # evaluate test only after best validation epoch selected
    report = {"loss": args.loss, "model": MODEL_ID, "records": len(texts),
              "label_count": len(names), "batch_size": 2, "gradient_accumulation": 8,
              "max_length": args.max_length, "epochs": args.epochs,
              "threshold": 0.5, "seed": SEED, "validation_history": history,
              "test_metrics": test_scores}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"HELD-OUT TEST: {test_scores}", flush=True)
    print(f"Saved model and metrics to: {args.output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLBSE 2027 RoBERTa skill classifier")
    parser.add_argument("--db", type=Path, default=Path("skillscope_data.db"))
    parser.add_argument("--loss", choices=["bce", "focal"], default="bce")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("runs/roberta_bce"))
    args = parser.parse_args()
    if args.epochs < 1 or args.max_length < 16:
        parser.error("epochs must be >=1 and max-length must be >=16")
    run(args)
