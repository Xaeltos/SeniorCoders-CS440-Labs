"""Rebuild the NLBSE RoBERTa experiment against the corrected v2 SQLite data.

Single command: python run_v2_rebuild.py --db dataset_v2/skillscope_data_v2.db
Reuses the existing, working roberta_recovery.py training loop. Trains a fresh
v2-only checkpoint, fits a leakage-controlled SkillScope RF+TF-IDF comparator,
selects an operating threshold on VALIDATION, and evaluates the held-out v2
TEST only once after checkpoint and threshold selection.

Does NOT alter the old model, old database, or old reports. This is not an
official competition score and not an exact reproduction of SkillScope's
original pre-split MLSMOTE protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path


def checksum(path: Path):
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db', type=Path, default=Path('dataset_v2/skillscope_data_v2.db'))
    p.add_argument('--output', type=Path, default=Path('runs/v2_roberta_recovery'))
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=4, help='Evaluation batch size')
    a = p.parse_args()

    if not a.db.is_file():
        p.error(f'Corrected dataset not found at {a.db}. Extract the zip into dataset_v2 first.')
    if not (Path('roberta_recovery.py').is_file() and Path('skill_classifier.py').is_file() and
            Path('compare_skillscope_roberta.py').is_file()):
        p.error('Run from the ORIGINAL project root, next to the three existing training/comparison scripts.')
    if not (1 <= a.epochs <= 20):
        p.error('epochs must be between 1 and 20')
    if a.batch_size < 1:
        p.error('batch-size must be positive')

    # Imports happen only after path checks to give useful error messages.
    import emoji
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from compare_skillscope_roberta import metrics, skill_scope_clean
    from skill_classifier import SkillDataset, load_dataset, split_indices

    texts, y, names = load_dataset(a.db)
    if len(texts) != 3070 or len(names) != 217:
        p.error(f'Wrong v2 dataset: found {len(texts)} records / {len(names)} labels; expected 3070 / 217')
    train, val, test = split_indices(len(texts))
    if (np.intersect1d(train, val).size or np.intersect1d(train, test).size or
            np.intersect1d(val, test).size):
        raise RuntimeError('Overlapping split indices')
    print(f'V2 VERIFIED: {len(texts)} records / {len(names)} labels / '
          f'train={len(train)} / validation={len(val)} / held-out={len(test)}', flush=True)
    print('Old experiment remains untouched.', flush=True)

    checkpoint = a.output / 'best_model'
    report_path = a.output / 'v2_locked_comparison.json'
    # No overwrite/retraining of a previously trained experiment.
    if report_path.exists():
        p.error(f'Final report already exists: {report_path}. Refusing repeat test use.')
    if checkpoint.exists():
        print(f'Reusing existing v2 checkpoint: {checkpoint}', flush=True)
        if not (a.output / 'recovery_metrics.json').exists():
            p.error('Existing v2 checkpoint has no recovery_metrics.json; cannot verify selected epoch.')
    else:
        if a.output.exists() and any(a.output.iterdir()):
            p.error(f'Output {a.output} is nonempty but has no complete checkpoint; inspect it before retrying.')
        cmd = [sys.executable, 'roberta_recovery.py', '--db', str(a.db),
               '--output', str(a.output), '--epochs', str(a.epochs)]
        print('TRAINING FRESH V2 ROBERTA:', ' '.join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    recovery = json.loads((a.output / 'recovery_metrics.json').read_text(encoding='utf-8'))
    if recovery.get('counts', {}).get('train') != len(train):
        raise ValueError('Checkpoint training row count does not match v2!')

    print('Fitting leakage-controlled SkillScope-method RF on v2 training rows...', flush=True)
    with sqlite3.connect(f'file:{a.db.resolve().as_posix()}?mode=ro', uri=True) as conn:
        repo_count = conn.execute('SELECT COUNT(DISTINCT "Repo Name") FROM "nlbse_tool_competition_data_by_issue"').fetchone()[0]
        frame = pd.read_sql_query(
            'SELECT "issue text", "issue description" FROM "nlbse_tool_competition_data_by_issue"', conn)
    if len(frame) != len(texts):
        raise ValueError('RF text row count mismatch')
    cleaned = (frame['issue text'].map(lambda z: skill_scope_clean(z, emoji.demojize)) + ' ' +
               frame['issue description'].map(lambda z: skill_scope_clean(z, emoji.demojize))).tolist()
    tfidf = TfidfVectorizer(max_features=1000)
    x_train = tfidf.fit_transform([cleaned[int(i)] for i in train])
    x_val = tfidf.transform([cleaned[int(i)] for i in val])
    rf = RandomForestClassifier(random_state=42)
    rf.fit(x_train.toarray(), y[train].astype(np.uint8))
    rf_val_pred = rf.predict(x_val.toarray()).astype(np.uint8)
    rf_val = metrics(y[val].astype(np.uint8), rf_val_pred)
    print('SKILLSCOPE METHOD VALIDATION:', json.dumps(rf_val), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True)
    if model.config.num_labels != 217 or [model.config.id2label[i] for i in range(217)] != names:
        raise ValueError('V2 checkpoint label names/order do not match v2 database')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    @torch.inference_mode()
    def get_probs(indices):
        loader = DataLoader(SkillDataset(texts, y, indices, tokenizer, 256),
                            batch_size=a.batch_size, shuffle=False)
        output = []
        for j, batch in enumerate(loader, 1):
            batch.pop('labels')
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            output.append(torch.sigmoid(logits.float()).cpu().numpy())
            if j % 100 == 0:
                print(f'  Inference {j}/{len(loader)}', flush=True)
        return np.concatenate(output)

    print('Evaluating V2 VALIDATION to select threshold (TEST not evaluated)...', flush=True)
    prob_val = get_probs(val)
    candidates = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.005), 3):
        stats = metrics(y[val].astype(np.uint8), (prob_val >= threshold).astype(np.uint8))
        margins = {k: stats[k] - rf_val[k] for k in ('precision', 'recall', 'micro_f1')}
        candidates.append((float(threshold), stats, margins))
    sweeps = [v for v in candidates if all(delta >= 0 for delta in v[2].values())
              and any(delta > 0 for delta in v[2].values())]
    if sweeps:
        chosen = max(sweeps, key=lambda z: (min(z[2].values()), z[1]['micro_f1']))
        print(f'VALIDATION CLEAN SWEEP thresholds: {len(sweeps)}', flush=True)
    else:
        chosen = max(candidates, key=lambda z: z[1]['micro_f1'])
        print('NO VALIDATION CLEAN SWEEP; selecting best F1 as fallback.', flush=True)
    threshold, rb_val, val_margins = chosen
    print(f'LOCKED V2 THRESHOLD: {threshold:.3f}', flush=True)
    print('ROBERTA VALIDATION:', json.dumps(rb_val), flush=True)
    print('VALIDATION MARGINS:', json.dumps(val_margins), flush=True)

    # Only after the validation-selected checkpoint and threshold are frozen:
    print('Starting ONE locked evaluation of the v2 TEST partition...', flush=True)
    x_test = tfidf.transform([cleaned[int(i)] for i in test])
    rf_pred_test = rf.predict(x_test.toarray()).astype(np.uint8)
    rf_test = metrics(y[test].astype(np.uint8), rf_pred_test)
    prob_test = get_probs(test)
    rb_test = metrics(y[test].astype(np.uint8), (prob_test >= threshold).astype(np.uint8))
    test_margins = {k: rb_test[k] - rf_test[k] for k in ('precision', 'recall', 'micro_f1')}
    clean = (all(d >= 0 for d in test_margins.values())
             and any(d > 0 for d in test_margins.values()))
    result = {
        'dataset_version':'v2', 'database_file':str(a.db), 'database_sha256':checksum(a.db),
        'rows':len(texts), 'labels':len(names), 'repositories_in_dataset':int(repo_count),
        'seed':42, 'train_rows':len(train), 'val_rows':len(val), 'test_rows':len(test),
        'checkpoint':str(checkpoint), 'selected_epoch':recovery['selected_epoch'],
        'threshold_selected_only_on_validation':threshold,
        'baseline_description':'SkillScope RF + max_features=1000 TF-IDF, cleaning, no MLSMOTE, train-only fit',
        'reference_is_official_competition_score':False,
        'validation':{'skillscope_method':rf_val,'roberta':rb_val,'margins':val_margins,
                      'clean_sweep_thresholds_found':len(sweeps)},
        'locked_test':{'skillscope_method':rf_test,'roberta':rb_test,'margins':test_margins,
                       'clean_sweep':bool(clean)},
        'caution':'v2 held-out experiment compared to leakage-controlled SkillScope method. Not official competition qualification.'
    }
    a.output.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('V2 LOCKED TEST SKILLSCOPE:', json.dumps(rf_test), flush=True)
    print('V2 LOCKED TEST ROBERTA:', json.dumps(rb_test), flush=True)
    print('V2 ROBERTA MINUS SKILLSCOPE:', json.dumps(test_margins), flush=True)
    print('V2 CLEAN SWEEP:', 'YES' if clean else 'NO', flush=True)
    print('Saved:', report_path, flush=True)


if __name__ == '__main__':
    main()
