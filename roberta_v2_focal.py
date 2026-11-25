"""NLBSE 2027 RoBERTa focal-loss experiment on corrected v2 (validation-only).

Use exactly the same 217 targets and 70/10/20 split as skill_classifier.py.
Leave the test partition unused. Select by validation F1; also test sensitivity
by shuffling validation texts. No API-derived ground-truth values enter inputs.

Usage:
  python roberta_recovery.py --epochs 4
  python roberta_recovery.py --epochs 1 --limit-train 64 --output runs/recovery_smoke

This code relies on the local skill_classifier.py from the same project.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from skill_classifier import (MODEL_ID, SEED, SkillDataset, load_dataset,
                              seed_everything, split_indices)


def score(y_true, probabilities, threshold):
    pred = probabilities >= threshold
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, average='micro', zero_division=0)
    return {'precision': float(p), 'recall': float(r), 'micro_f1': float(f),
            'avg_predicted_labels': float(pred.sum(axis=1).mean())}


def choose_threshold(y_true, probabilities):
    # Search a global threshold on validation only, not the held-out test set.
    results = [(float(t), score(y_true, probabilities, float(t)))
               for t in np.arange(0.10, 0.801, 0.025)]
    threshold, metrics = max(results, key=lambda item: item[1]['micro_f1'])
    return threshold, metrics


@torch.inference_mode()
def probabilities(model, loader, device):
    model.eval()
    all_probs = []
    for batch in loader:
        batch.pop('labels')
        logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
        all_probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def main():
    ap = argparse.ArgumentParser(description='RoBERTa focal-loss v2 training, validation only')
    ap.add_argument('--db', type=Path, default=Path('skillscope_data.db'))
    ap.add_argument('--output', type=Path, default=Path('runs/v2_roberta_focal'))
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--max-length', type=int, default=256)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--grad-accum', type=int, default=8)
    ap.add_argument('--encoder-lr', type=float, default=2e-5)
    ap.add_argument('--head-lr', type=float, default=1e-4)
    ap.add_argument('--gamma', type=float, default=2.0)
    ap.add_argument('--weight-cap', type=float, default=3.0)
    ap.add_argument('--limit-train', type=int, default=0,
                    help='diagnostic only: restrict training records, not a competition experiment')
    args = ap.parse_args()
    if args.gamma < 0:
        ap.error('gamma must be non-negative')
    if min(args.epochs, args.batch_size, args.grad_accum, args.max_length) < 1:
        ap.error('epochs, batch size, grad accumulation and max length must be positive')
    if args.output.exists() and (args.output / 'best_model').exists():
        ap.error('output already contains a model checkpoint; choose a new --output to avoid overwriting it')

    seed_everything(SEED)
    texts, y, names = load_dataset(args.db)
    train, val, test = split_indices(len(texts))
    if args.limit_train:
        train = train[:args.limit_train]
        print('DIAGNOSTIC SUBSET -- NOT A FULL-DATASET RESULT', flush=True)
    print(f'Rows: train={len(train)} validation={len(val)} test={len(test)} (TEST UNUSED), labels={len(names)}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    train_loader = DataLoader(SkillDataset(texts, y, train, tokenizer, args.max_length),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SkillDataset(texts, y, val, tokenizer, args.max_length),
                            batch_size=args.batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(names), problem_type='multi_label_classification',
        id2label={i: n for i, n in enumerate(names)},
        label2id={n: i for i, n in enumerate(names)}).to(device)
    model.gradient_checkpointing_enable()

    print(f'GPU={device}; focal gamma={args.gamma} (no positive weighting)', flush=True)

    head, encoder = [], []
    for name, parameter in model.named_parameters():
        (head if name.startswith('classifier.') else encoder).append(parameter)
    optimizer = torch.optim.AdamW([
        {'params': encoder, 'lr': args.encoder_lr},
        {'params': head, 'lr': args.head_lr},
    ], weight_decay=0.01)
    amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    args.output.mkdir(parents=True, exist_ok=True)
    history, best = [], -1
    y_val = y[val].astype(np.uint8)

    # Train-only frequency comparison, same validation partition.
    frequency = y[train].mean(axis=0)
    freq_candidates = [(score(y_val, np.broadcast_to(frequency, y_val.shape), float(t)), float(t))
                       for t in np.arange(0.1, 0.801, 0.025)]
    freq_best = max(freq_candidates, key=lambda x: x[0]['micro_f1'])
    print(f'Frequency-only validation F1: {freq_best[0]["micro_f1"]:.4f}', flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            targets = batch.pop('labels').to(device)
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
                per_label_bce = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction='none')
                pt = torch.exp(-per_label_bce)
                loss = ((1.0 - pt) ** args.gamma * per_label_bce).mean()
            running_loss += float(loss.detach())
            scaler.scale(loss / args.grad_accum).backward()
            if (i + 1) % args.grad_accum == 0 or i + 1 == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if (i + 1) % 300 == 0:
                print(f'Epoch {epoch}/{args.epochs}: batch {i+1}/{len(train_loader)}', flush=True)

        val_probs = probabilities(model, val_loader, device)
        threshold, tuned = choose_threshold(y_val, val_probs)
        fixed = score(y_val, val_probs, 0.5)
        line = {'epoch': epoch, 'training_loss_mean': running_loss / len(train_loader),
                'validation_threshold': threshold, 'validation_tuned': tuned,
                'validation_fixed_0_5': fixed}
        history.append(line)
        print(f'EPOCH {epoch}: loss={line["training_loss_mean"]:.4f} '
              f'val F1@0.5={fixed["micro_f1"]:.4f} '
              f'val F1 tuned={tuned["micro_f1"]:.4f} '
              f'P={tuned["precision"]:.4f} R={tuned["recall"]:.4f} threshold={threshold:.3f}', flush=True)
        if tuned['micro_f1'] > best:
            best = tuned['micro_f1']
            best_epoch = epoch
            best_threshold = threshold
            model.save_pretrained(args.output / 'best_model')
            tokenizer.save_pretrained(args.output / 'best_model')
            print('  Saved new best validation checkpoint.', flush=True)

    # Sensitivity check on validation only, with selected checkpoint and threshold.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.output / 'best_model', local_files_only=True).to(device)
    natural = probabilities(model, val_loader, device)
    rng = np.random.default_rng(SEED)
    shuffled_val_indices = np.asarray(val)[rng.permutation(len(val))]
    shuffled_loader = DataLoader(SkillDataset(texts, y, shuffled_val_indices, tokenizer, args.max_length),
                                 batch_size=args.batch_size, shuffle=False)
    shuffled = probabilities(model, shuffled_loader, device)
    actual_scores = score(y_val, natural, best_threshold)
    shuffled_scores = score(y_val, shuffled, best_threshold)
    result = {
        'model': MODEL_ID, 'training_only_label_weights': False,
        'loss': 'symmetric focal loss', 'focal_gamma':args.gamma, 'seed': SEED,
        'counts': {'train':len(train), 'validation':len(val), 'held_out_test_unused':len(test)},
        'encoder_lr': args.encoder_lr,
        'head_lr': args.head_lr, 'max_length': args.max_length,
        'frequency_only_validation': {'threshold':freq_best[1], **freq_best[0]},
        'epochs':history, 'selected_epoch':best_epoch,
        'selected_validation_threshold':best_threshold,
        'selected_validation_metrics':actual_scores,
        'shuffled_text_metrics_at_selected_threshold':shuffled_scores,
        'validation_f1_text_advantage':actual_scores['micro_f1'] - shuffled_scores['micro_f1'],
        'avg_absolute_probability_change_on_shuffle':float(np.abs(natural-shuffled).mean()),
        'warning':'Validation scores only; held-out test never inspected. No guarantee of official competition eligibility.'
    }
    (args.output / 'recovery_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('\nFINAL VALIDATION:', actual_scores, flush=True)
    print('SHUFFLED TEXT:', shuffled_scores, flush=True)
    print('FREQUENCY ONLY:', freq_best[0], flush=True)
    print('Saved:', args.output / 'recovery_metrics.json', flush=True)
    print('Held-out TEST NOT evaluated.', flush=True)


if __name__ == '__main__':
    main()
