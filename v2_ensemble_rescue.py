"""Validation-only independent-seed focal ensemble experiment for NLBSE corrected v2.

Runs exactly two additional full-data focal training jobs, each in a separate
process with its own seed and new output folder, then compares the 3-model
ensemble with the already published frozen 0.545 focal model. Never inspects
test records. Never overwrites checkpoints, metrics or previous experiment.
This experiment is exploratory, not official eligibility verification.
"""
from __future__ import annotations

import argparse
import gc
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from skill_classifier import SkillDataset, load_dataset, split_indices
from compare_skillscope_roberta import metrics, skill_scope_clean
import emoji

BASE = Path('runs/v2_roberta_focal/best_model')
SEEDS = (2028, 2029)
MODEL_PATHS = [BASE] + [Path(f'runs/v2_focal_seed_{seed}/best_model') for seed in SEEDS]
OUTPUT = Path('runs/v2_ensemble_rescue/validation_report.json')


def read_probabilities(checkpoint, texts, truth, labels, validation):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True)
    if model.config.num_labels != len(labels) or [model.config.id2label[i] for i in range(len(labels))] != labels:
        raise ValueError(f'Wrong skill label order for {checkpoint}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    loader = DataLoader(SkillDataset(texts, truth, validation, tokenizer, 256), batch_size=4, shuffle=False)
    result = []
    with torch.inference_mode():
        for batch in loader:
            batch.pop('labels')
            logits = model(**{key: value.to(device) for key, value in batch.items()}).logits
            result.append(torch.sigmoid(logits.float()).cpu().numpy())
    del model, loader
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return np.concatenate(result)


def baseline_validation(db, truth, train, val):
    with sqlite3.connect(db) as conn:
        df = pd.read_sql_query('SELECT "issue text", "issue description" FROM "nlbse_tool_competition_data_by_issue"', conn)
    clean = (df['issue text'].map(lambda z: skill_scope_clean(z, emoji.demojize)) + ' ' +
             df['issue description'].map(lambda z: skill_scope_clean(z, emoji.demojize))).tolist()
    vec = TfidfVectorizer(max_features=1000)
    x_train = vec.fit_transform([clean[int(i)] for i in train])
    x_val = vec.transform([clean[int(i)] for i in val])
    rf = RandomForestClassifier(random_state=42)
    rf.fit(x_train.toarray(), truth[train].astype(np.uint8))
    return metrics(truth[val].astype(np.uint8), rf.predict(x_val.toarray()).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=Path('dataset_v2/skillscope_data_v2.db'))
    parser.add_argument('--epochs', type=int, default=4)
    args = parser.parse_args()
    if not args.db.is_file() or not BASE.is_dir():
        parser.error('Missing corrected v2 database or already-published focal model')
    if args.epochs < 1:
        parser.error('epochs must be positive')
    if OUTPUT.exists():
        parser.error(f'Refusing to overwrite existing report: {OUTPUT}')
    for seed in SEEDS:
        dest = Path(f'runs/v2_focal_seed_{seed}')
        if dest.exists():
            parser.error(f'Refusing to reuse an existing experiment folder: {dest}')

    texts, truth, labels = load_dataset(args.db)
    train, validation, test = split_indices(len(texts))
    if (len(texts), len(labels), len(train), len(validation), len(test)) != (3070, 217, 2149, 307, 614):
        raise ValueError('Unexpected corrected v2 data shape; refusing experiment')
    print('VALIDATION ONLY: 2149 train / 307 validation / 614 TEST NEVER READ.', flush=True)
    print('Each seed trains from FacebookAI/roberta-base; existing models are never changed.', flush=True)
    # Check both train runs before spending time on inference.
    for seed in SEEDS:
        dest = Path(f'runs/v2_focal_seed_{seed}')
        print(f'\nTRAINING INDEPENDENT FOCAL MODEL seed={seed}, folder={dest}', flush=True)
        wrapper = ('import sys; import roberta_v2_focal as m; '
                   'm.SEED=int(sys.argv[1]); '
                   'sys.argv=["roberta_v2_focal.py"]+sys.argv[2:]; m.main()')
        command = [sys.executable, '-c', wrapper, str(seed),
                   '--db', str(args.db), '--epochs', str(args.epochs), '--output', str(dest)]
        subprocess.run(command, check=True)
        if not (dest / 'best_model' / 'model.safetensors').is_file():
            raise RuntimeError(f'Missing newly trained checkpoint: {dest}')
    rf = baseline_validation(args.db, truth, train, validation)
    print('LEAKAGE-CONTROLLED RF VALIDATION:', json.dumps(rf), flush=True)
    probs = []
    for checkpoint in MODEL_PATHS:
        print('VALIDATION INFERENCE:', checkpoint, flush=True)
        probs.append(read_probabilities(checkpoint, texts, truth, labels, validation))
    y = truth[validation].astype(np.uint8)
    frozen = metrics(y, (probs[0] >= 0.545).astype(np.uint8))
    avg = np.mean(np.stack(probs), axis=0)
    candidates = []
    for threshold in np.round(np.arange(0.30, 0.751, 0.005), 3):
        metric = metrics(y, (avg >= threshold).astype(np.uint8))
        margins = {k: metric[k] - rf[k] for k in ('precision', 'recall', 'micro_f1')}
        candidates.append({'threshold': float(threshold), 'metrics': metric,
                           'margins_vs_reproduced_rf': margins,
                           'clean_sweep_validation': all(v >= 0 for v in margins.values()) and any(v > 0 for v in margins.values())})
    winners = [x for x in candidates if x['clean_sweep_validation']]
    # Pick the largest worst-case absolute margin across three metrics (no test input).
    best = max(winners, key=lambda x: (min(x['margins_vs_reproduced_rf'].values()), x['metrics']['micro_f1'])) if winners else None
    near = max(candidates, key=lambda x: (min(x['margins_vs_reproduced_rf'].values()), x['metrics']['micro_f1']))
    print('FROZEN PUBLISHED FOCAL 0.545:', json.dumps(frozen), flush=True)
    print('ENSEMBLE VALIDATION CLEAN SWEEP THRESHOLDS:', len(winners), flush=True)
    print('ENSEMBLE MAXIMUM MINIMUM MARGIN:', json.dumps(best if best else near), flush=True)
    if best:
        print('VALIDATION SWEEP: YES (NOT AN OFFICIAL COMPETITION PASS)', flush=True)
    else:
        print('VALIDATION SWEEP: NO', flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        'experiment': 'three independently initialized focal RoBERTa models; equal-weight probability average',
        'models': [str(x) for x in MODEL_PATHS], 'seeds': [42, *SEEDS],
        'train_rows': 2149, 'validation_rows': 307, 'test_rows_untouched': 614,
        'baseline_note': 'train-only TF-IDF+RF reproduction, not official competition baseline',
        'baseline_validation': rf, 'frozen_published_model_validation': frozen,
        'clean_sweep_validation_threshold_count': len(winners), 'best_candidate': best,
        'best_near_candidate': near,
        'caution': 'Exploratory validation only; this partition was used to select checkpoints. Official eligibility cannot be inferred.'
    }, indent=2), encoding='utf-8')
    print('Saved:', OUTPUT, flush=True)
    print('No model or prediction-interface changes were made.', flush=True)

if __name__ == '__main__':
    main()
